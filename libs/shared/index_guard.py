"""What the indexing consumers may store: never-record windows and confidential meetings.

Polish plan, Phase 0 ("never record everywhere"). `embedding-service` (Qdrant) and
`search-service` (Postgres FTS) index every final of `transcript-chunks`, so before this a
window the company said not to record was still embedded and full-text indexed, and RAG / Ask
could bring it back. Both consumers now ask `IndexGuard.check` before they store a chunk.

Never record, per chunk, cheapest first:
1. The chunk's 30-second window (`start_ms // 30000`, the agent runtime's window) was already
   marked never-record for this meeting -> skip.
2. The org's never-record topics (`libs/shared/never_record.py`): a topic or a glossary alias of
   it said outright -> skip, mark this window and the next one (the talk around a topic rarely
   names it in every sentence), and tell the caller to PURGE what it already stored for this
   window (chunks before the one that named the topic).
3. Only when ambiguous (a significant word of a topic, not the topic): one triage-tier check
   (`never_record_check@v1`, Haiku, budget-gated and metered like every model call), its answer
   cached per (org, text) in Redis so the two consumers ask once. About a topic -> as in 2.
   The check unavailable (no model wired, budget blocked, an error) -> skip this chunk: a
   chunk wrongly kept out of search costs little; one wrongly stored breaks the rule.
4. Otherwise index.
The marks live in Redis, `nexus:{org_id}:never_record:{meeting_id}:{window}`, 24 hours, shared
by both consumers. The org's topics come from `GetOrgContext` read STRICTLY: if the company
brain cannot be read, `check` raises and the consumer does not commit, so the chunk is retried
rather than indexed unvetted.

Confidential meetings (`meetings.confidential`, ADR-0039 B2): indexed WITH `confidential=true`
rather than skipped. Every retrieval path RAG and Ask use is participant-scoped already (rag-
service asks core-api for the meetings the asker attended, ADR-0033 D1, and passes them as an
allowlist to both arms), so a participant can still ask about their own confidential meeting;
the org-wide paths (`GET /search`, an unscoped FTS call, `qdrant.search_with_tenant_filter`)
exclude flagged rows. Skipping instead would have taken the meeting away from the people who
were in it. The flag comes from core-api's `GET /internal/meetings/{id}/index-policy`, cached;
when it cannot be read the chunk is treated as confidential (participant-only), never as open.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

import httpx

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext

logger = logging.getLogger(__name__)

WINDOW_MS = 30_000  # the agent runtime's window (services/agent_runtime_service/windowing.py)
TAIL_WINDOWS = 1  # windows after a hit that are treated as never-record too
MARK_TTL_S = 24 * 3600
CHECK_TTL_S = 24 * 3600
PRIVACY_TTL_S = 300.0
PROMPTS_DIR = Path(__file__).parent / "prompt_templates"
CHECK_KEY = "never_record_check"

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]
PrivacyFn = Callable[[UUID, UUID], Awaitable[bool | None]]
Reason = Literal["ok", "never_record", "never_record_window", "unvetted"]


class Classifier(Protocol):
    async def about(self, org_id: UUID, text: str, policy: NeverRecordPolicy) -> bool | None:
        """True / False, or None when the check could not run."""


@dataclass(frozen=True)
class IndexVerdict:
    index: bool
    confidential: bool
    reason: Reason
    purge: tuple[int, int] | None = None  # [start_ms, end_ms) of this meeting to remove

    @property
    def skip(self) -> bool:
        return not self.index


def mark_key(org_id: UUID, meeting_id: UUID, window: int) -> str:
    return f"nexus:{org_id}:never_record:{meeting_id}:{window}"


class IndexGuard:
    def __init__(
        self,
        *,
        redis: Any,
        org_context: OrgContextFn,
        privacy: PrivacyFn | None = None,
        classifier: Classifier | None = None,
    ) -> None:
        self._redis = redis
        self._org = org_context
        self._privacy = privacy
        self._classifier = classifier

    async def confidential(self, org_id: UUID, meeting_id: UUID) -> bool:
        if self._privacy is None:
            return True  # unknown: participant-scoped only
        try:
            flag = await self._privacy(org_id, meeting_id)
        except Exception:  # noqa: BLE001 - unknown: participant-scoped only, never open
            flag = None
        return True if flag is None else flag

    async def check(self, org_id: UUID, meeting_id: UUID, start_ms: int, text: str) -> IndexVerdict:
        confidential = await self.confidential(org_id, meeting_id)
        policy = NeverRecordPolicy.from_org(await self._org(org_id))  # raises: retry
        if policy.empty:
            return IndexVerdict(True, confidential, "ok")
        window = max(0, start_ms) // WINDOW_MS
        if await self._redis.exists(mark_key(org_id, meeting_id, window)):
            return IndexVerdict(False, confidential, "never_record_window")
        about: bool | None = policy.hit(text) is not None
        if not about and policy.ambiguous(text) is not None:
            about = await self._ask(org_id, text, policy)
            if about is None:
                return IndexVerdict(False, confidential, "unvetted")
        if not about:
            return IndexVerdict(True, confidential, "ok")
        for w in range(window, window + TAIL_WINDOWS + 1):
            await self._redis.set(mark_key(org_id, meeting_id, w), "1", ex=MARK_TTL_S)
        return IndexVerdict(
            False,
            confidential,
            "never_record",
            purge=(window * WINDOW_MS, (window + 1) * WINDOW_MS),
        )

    async def _ask(self, org_id: UUID, text: str, policy: NeverRecordPolicy) -> bool | None:
        if self._classifier is None:
            return None
        digest = hashlib.sha256(
            json.dumps([policy.topics, " ".join(text.split())]).encode()
        ).hexdigest()[:32]
        key = f"nexus:{org_id}:never_record:check:{digest}"
        cached = await self._redis.get(key)
        if cached is not None:
            return bytes(cached) == b"1" if isinstance(cached, bytes | bytearray) else cached == "1"
        try:
            about = await self._classifier.about(org_id, text, policy)
        except Exception:  # noqa: BLE001 - unvetted: skipped, never indexed
            logger.warning("never-record check failed org=%s; chunk not indexed", org_id)
            return None
        if about is not None:
            await self._redis.set(key, "1" if about else "0", ex=CHECK_TTL_S)
        return about


class RouterClassifier:
    """`never_record_check@v1` on the triage tier through `LiteLLMRouter` (budget gate and
    `usage-events` metering included). A score of 0.5 or more, or `about: true`, is about."""

    def __init__(self, router: Any, loader: Any | None = None) -> None:
        from libs.shared.prompts import PromptLoader

        self._router = router
        self._loader = loader or PromptLoader(PROMPTS_DIR)

    async def about(self, org_id: UUID, text: str, policy: NeverRecordPolicy) -> bool | None:
        from libs.shared.router import BudgetBlocked

        prompt = self._loader.load(CHECK_KEY, self._loader.resolve(CHECK_KEY, org_id))
        topics = "\n".join(
            f"- {t}" + (f" (also said as {', '.join(terms[1:6])})" if terms[1:] else "")
            for t, terms in policy.topics
        )
        try:
            out = await self._router.complete(
                tier="triage",
                system=prompt.system,
                **prompt.user_parts(topics=topics, text=text[:2000]),
                org_id=org_id,
                user_id=None,
                max_tokens=60,
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            return None
        return parse_check(out.text)


def parse_check(text: str) -> bool | None:
    """`{"about": bool, "score": x}`; unparseable is None (unvetted, so not indexed)."""
    body = text.strip().strip("`").strip()
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if start >= 0 and end > start else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    about, score = data.get("about"), data.get("score")
    if about is True:
        return True
    if isinstance(score, int | float) and not isinstance(score, bool) and score >= 0.5:
        return True
    return False if about is False else None


class MeetingPrivacyClient:
    """core-api `GET /internal/meetings/{meeting_id}/index-policy` -> `{"confidential": bool}`,
    as `service` (embedding-service / search-service) with `CORE_CALLER_TOKEN`. Cached per
    meeting for `ttl_s`; a failure returns None and is not cached."""

    def __init__(
        self,
        service: str,
        base_url: str | None = None,
        *,
        token: str | None = None,
        ttl_s: float = PRIVACY_TTL_S,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._service = service
        self._url = (base_url or os.environ.get("CORE_URL", "")).strip().rstrip("/")
        self._token = (
            token if token is not None else os.environ.get("CORE_CALLER_TOKEN", "")
        ).strip()
        self._ttl = ttl_s
        self._timeout = timeout_s
        self._clock = clock
        self._transport = transport
        self._cache: dict[tuple[UUID, UUID], tuple[float, bool]] = {}

    async def __call__(self, org_id: UUID, meeting_id: UUID) -> bool | None:
        hit = self._cache.get((org_id, meeting_id))
        now = self._clock()
        if hit is not None and now < hit[0]:
            return hit[1]
        if not self._url or not self._token:
            return None
        path = f"/internal/meetings/{meeting_id}/index-policy"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": self._service,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("core-api %s unreachable: %s (treated as confidential)", path, exc)
            return None
        if res.status_code != 200:
            logger.warning(
                "core-api %s answered %s (treated as confidential)", path, res.status_code
            )
            return None
        try:
            flag = res.json().get("confidential")
        except (ValueError, AttributeError):
            return None
        if not isinstance(flag, bool):
            return None
        self._cache[(org_id, meeting_id)] = (now + self._ttl, flag)
        return flag


def guard_from_env(service: str, redis: Any) -> IndexGuard:
    """The production guard for an indexing consumer: the company brain over `GetOrgContext`
    (`IDENTITY_GRPC_ADDR`), the meeting's confidentiality from core-api (`CORE_URL`,
    `CORE_CALLER_TOKEN`), and the triage check through a budget-gated router when
    `ANTHROPIC_API_KEY` is set (without it an ambiguous chunk is not indexed)."""
    from libs.shared.org_context import OrgContextClient

    org = OrgContextClient()
    classifier: Classifier | None = None
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        classifier = RouterClassifier(router_from_env(service, redis))
    else:
        logger.warning("ANTHROPIC_API_KEY unset: ambiguous never-record chunks are not indexed")
    return IndexGuard(
        redis=redis,
        org_context=org.get_strict,
        privacy=MeetingPrivacyClient(service),
        classifier=classifier,
    )


def router_from_env(service: str, redis: Any) -> Any:
    from confluent_kafka import Producer

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.pii import EgressRedactor
    from libs.shared.router import LiteLLMRouter
    from libs.shared.usage import UsageEmitter

    producer = Producer(
        {
            "bootstrap.servers": os.environ.get("REDPANDA_BROKERS", "localhost:9092"),
            "acks": "all",
            "enable.idempotence": True,
        }
    )
    codec = AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    return LiteLLMRouter(
        BudgetGate(redis, BillingRefresher(service=service)),
        UsageEmitter(producer, codec, producer_name=service),
        redactor=EgressRedactor(redis),
    )
