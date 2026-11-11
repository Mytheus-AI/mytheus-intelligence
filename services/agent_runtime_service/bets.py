"""Decisions as bets: after a meeting ends, propose what its decisions were expected to achieve.

"Beyond the meeting", Layer 3. When someone in the meeting said what a decision should do
("this should cut churn by 3 points by March"), DecisionAgent proposes that expectation as the
decision's bet: a metric or an event, a direction and size, a check date and a source. core-api
stores it as a `proposed` bet for the decision's owner, who accepts, edits or dismisses it. The
agent never makes a bet on its own and never scores one.

Trigger. `meeting-events` `ended`, consumed in its own group `agent-runtime-bets` (a new group
starts at the end of the topic, like the minutes group: no model calls for historic meetings).

Input. core-api's `GET /internal/meetings/{id}/minutes-source`, the minutes writer's input
(participants, finals, the decisions recorded during the meeting with their ids, the agenda),
plus its `scheduled_start` for the meeting date that relative dates ("by March") resolve from.
No recorded decision, or no transcript: nothing is called and nothing is emitted.

One model call per meeting (`decision_bet@vN`, extract tier, through the LiteLLM router: budget
gated, metered, PII redacted), with the same never-record handling as the minutes: windows that
name a never-record topic and the window after them are removed before the call, and any
proposal that names one is dropped after it.

Grounding. A proposal is kept only when its `quote` is found, word for word (whitespace and case
aside), in one transcript line: that is the guarantee the meeting actually said it. The quote's
line gives `quote_start_ms`. Proposals below `MIN_CONFIDENCE`, without a usable date, or on a
decision that does not exist are dropped. Labels become names in the text.

Output. One `agent-results` record, agent `DecisionAgent`, kind `decision_candidate`,
`window_id = bets:{generation}`, `user_id` = the host (the nil uuid when there is none),
`payload_json = {prompt_version, generation, expectations: [...]}` with each expectation's
`decision_id`. core-api's consumer routes a `decision_candidate` that carries `expectations`
to the bets store; realtime-gateway ignores it (no `candidates`). Nothing is emitted when there
is nothing to propose. Once flushed, `nexus:{org_id}:bets:{meeting_id}:{generation}` is set for
7 days so a replayed `ended` does not bill the call again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

import httpx

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import EXTRACT_MAX_TOKENS, EXTRACTION_EFFORT, LiteLLMRouter
from libs.shared.state import AgentResult
from services.agent_runtime_service.agents.company import render_company, render_ground_rules
from services.agent_runtime_service.minutes import (
    SERVICE_NAME,
    MinutesSource,
    Resolver,
    SourceUnavailable,
    _json_object,
    assign_labels,
    generation_of,
    parse_ended,
    parse_source,
    quiet_turns,
    render_record,
)

logger = logging.getLogger(__name__)

AGENT = "DecisionAgent"
KIND = "decision_candidate"
PROMPT_KEY = "decision_bet"
GROUP = "agent-runtime-bets"
TOPIC = "meeting-events"
NIL = UUID(int=0)
MIN_CONFIDENCE = 0.6
MAX_EXPECTATIONS = 10
CLAIM_TTL_S = 7 * 24 * 3600
MAX_HORIZON_DAYS = 3 * 366
DEFAULT_HORIZON_DAYS = 90
_REF = re.compile(r"^\s*D(\d+)\s*$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_KINDS = ("metric", "event")
_METRIC_DIRECTIONS = ("increase", "decrease", "reach", "hold")
_EVENT_DIRECTIONS = ("happen", "not_happen")
_SOURCES = ("crm", "metrics", "meetings", "jira")
_CATEGORIES = (
    "pricing",
    "product",
    "growth",
    "customers",
    "hiring",
    "operations",
    "engineering",
    "finance",
    "other",
)


# --- the source ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BetSource:
    source: MinutesSource
    meeting_date: date


class BetSourceClient(Protocol):
    async def source(self, org_id: UUID, meeting_id: UUID) -> BetSource | None: ...


def _date_of(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_bet_source(body: object, today: date) -> BetSource | None:
    source = parse_source(body)
    if source is None:
        return None
    assert isinstance(body, dict)
    when = _date_of(body.get("scheduled_start")) or today
    return BetSource(source=source, meeting_date=when)


class HttpBetSource:
    """core-api's minutes source plus the meeting date. `None` for an unknown meeting;
    `SourceUnavailable` for anything worth retrying."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    async def source(self, org_id: UUID, meeting_id: UUID) -> BetSource | None:
        path = f"/internal/meetings/{meeting_id}/minutes-source"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"{path}: {exc}") from None
        if res.status_code == 404:
            return None
        if res.status_code != 200:
            raise SourceUnavailable(f"{path} answered {res.status_code}")
        try:
            return parse_bet_source(res.json(), datetime.now(UTC).date())
        except ValueError:
            return None


def source_from_env() -> HttpBetSource | None:
    import os

    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpBetSource(url, token)


# --- parsing the answer ------------------------------------------------------------------------


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower().strip("\"'“”‘’.,")


def _clip(raw: object, limit: int) -> str | None:
    if not isinstance(raw, str):
        return None
    value = " ".join(raw.split())
    if not value:
        return None
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _number(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if abs(raw) <= 1e12 else None


def ground_quote(quote: str | None, source: MinutesSource) -> tuple[str, int] | None:
    """(the quote as said, the line's start) when the quote is in one transcript line."""
    if not quote:
        return None
    needle = _norm(quote)
    if len(needle) < 8:
        return None
    for turn in source.turns:
        if needle in _norm(turn.text):
            return quote, turn.start_ms
    return None


def parse_expectations(
    text: str,
    *,
    source: MinutesSource,
    resolver: Resolver,
    meeting_date: date,
) -> list[dict[str, Any]] | None:
    """The model's answer as core-api's `expectations`, grounded and filtered; None when the
    answer is not the JSON asked for."""
    raw = _json_object(text)
    if raw is None or not isinstance(raw.get("expectations"), list):
        return None
    out: list[dict[str, Any]] = []
    seen: set[UUID] = set()
    for e in raw["expectations"]:
        if not isinstance(e, dict):
            continue
        m = _REF.match(e.get("ref") or "") if isinstance(e.get("ref"), str) else None
        if not m or not 1 <= int(m.group(1)) <= len(source.decisions):
            continue
        decision = source.decisions[int(m.group(1)) - 1]
        if decision.id in seen or decision.status == "dismissed":
            continue
        confidence = _number(e.get("confidence"))
        if confidence is None or confidence < MIN_CONFIDENCE:
            continue
        grounded = ground_quote(_clip(e.get("quote"), 300), source)
        if grounded is None:
            continue
        kind = e.get("kind") if e.get("kind") in _KINDS else "metric"
        direction = e.get("direction")
        if direction not in (_METRIC_DIRECTIONS if kind == "metric" else _EVENT_DIRECTIONS):
            continue
        magnitude = _number(e.get("magnitude")) if kind == "metric" else None
        if direction == "reach" and magnitude is None:
            continue
        check = _date_of(e.get("check_by")) if isinstance(e.get("check_by"), str) else None
        if check is None or check <= meeting_date:
            check = meeting_date + timedelta(days=DEFAULT_HORIZON_DAYS)
        if check > meeting_date + timedelta(days=MAX_HORIZON_DAYS):
            continue
        statement = resolver.keep(_clip(e.get("statement"), 300))
        metric = resolver.keep(_clip(e.get("metric"), 120))
        quote = resolver.keep(grounded[0])
        if not statement or not metric or quote is None:
            continue  # empty, or names a never-record topic
        assumption = resolver.keep(_clip(e.get("assumption"), 300))
        seen.add(decision.id)
        out.append(
            {
                "decision_id": str(decision.id),
                "statement": statement,
                "kind": kind,
                "metric": metric,
                "direction": direction,
                "magnitude": magnitude,
                "unit": _clip(e.get("unit"), 20) if kind == "metric" else None,
                "check_date": check.isoformat(),
                "date_said": e.get("date_said") is not False,
                "source": e.get("source") if e.get("source") in _SOURCES else "meetings",
                "category": e.get("category") if e.get("category") in _CATEGORIES else "other",
                "assumption": assumption,
                "quote": quote,
                "quote_start_ms": grounded[1],
                "confidence": round(min(1.0, confidence), 3),
            }
        )
        if len(out) == MAX_EXPECTATIONS:
            break
    return out


# --- the proposer ------------------------------------------------------------------------------

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


@dataclass
class Proposal:
    user_id: UUID
    payload: dict[str, Any]

    def result(self, generation: str) -> AgentResult:
        n = len(self.payload["expectations"])
        return AgentResult(
            kind=KIND,
            content=f"{n} expected outcome{'s' if n != 1 else ''} proposed",
            relevance_score=1.0,
            confidence=max((e["confidence"] for e in self.payload["expectations"]), default=0.0),
            source_chunk_ids=[],
            payload=self.payload,
            window_id=f"bets:{generation}",
        )


class BetProposer:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        source: BetSourceClient,
        org_context: OrgContextFn | None = None,
    ) -> None:
        self.loader = loader
        self.router = router
        self.source = source
        self.org_context = org_context

    async def propose(self, org_id: UUID, meeting_id: UUID, generation: str) -> Proposal | None:
        """What to emit for one meeting end, or None (nothing recorded, nothing said, nothing
        kept). Raises `SourceUnavailable` / the org context's error when worth retrying."""
        found = await self.source.source(org_id, meeting_id)
        if found is None:
            return None
        source = found.source
        live = [d for d in source.decisions if d.status != "dismissed"]
        if not live:
            return None
        org = await self.org_context(org_id) if self.org_context is not None else None
        policy = NeverRecordPolicy.from_org(org)
        turns = quiet_turns(source.turns, policy)
        if not turns:
            return None
        grounded = MinutesSource(**{**source.__dict__, "turns": turns})
        labels = assign_labels(grounded)
        resolver = Resolver.build(grounded, labels, policy)
        prompt = self.loader.load(PROMPT_KEY, self.loader.resolve(PROMPT_KEY, org_id))
        values = {
            "company": render_company(org) if org is not None else "",
            "ground_rules": render_ground_rules(org) if org is not None else "",
            "meeting_record": render_record(grounded, turns, labels),
            "meeting_date": found.meeting_date.isoformat(),
        }
        out = await self.router.complete(
            tier="extract",
            system=prompt.system,
            org_id=org_id,
            user_id=None,
            max_tokens=EXTRACT_MAX_TOKENS,
            prompt_ref=prompt.ref,
            effort=EXTRACTION_EFFORT,
            **prompt.user_parts(**values),
        )
        expectations = parse_expectations(
            out.text, source=grounded, resolver=resolver, meeting_date=found.meeting_date
        )
        if expectations is None:
            logger.warning("bets answer unparseable meeting=%s", meeting_id)
            return None
        if not expectations:
            return None
        return Proposal(
            user_id=source.host_user_id or NIL,
            payload={
                "prompt_version": prompt.ref,
                "generation": generation,
                "expectations": expectations,
            },
        )


# --- the loop --------------------------------------------------------------------------------


class ResultSink(Protocol):
    async def agent_result(
        self,
        *,
        org_id: UUID,
        meeting_id: UUID,
        window_id: str,
        user_id: UUID,
        agent: str,
        result: AgentResult,
    ) -> None: ...

    def flush(self, timeout_s: float = 10.0) -> None: ...


def claim_key(org_id: UUID, meeting_id: UUID, generation: str) -> str:
    return f"nexus:{org_id}:bets:{meeting_id}:{generation}"


class BetsLoop:
    """One decoded `meeting-events` record in, at most one proposal out, flushed. A
    retryable failure is retried with backoff; after the last attempt nothing is emitted (a
    missing proposal costs nothing: people can still add the bet themselves)."""

    def __init__(
        self,
        proposer: BetProposer,
        sink: ResultSink,
        *,
        claims: Any = None,
        settle_s: float = 20.0,
        attempts: int = 3,
        backoff_s: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.proposer = proposer
        self.sink = sink
        self.claims = claims
        self.settle_s = settle_s
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self._sleep = sleep

    async def handle(self, value: dict[str, Any]) -> int:
        ended = parse_ended(value)
        if ended is None:
            return 0
        key = claim_key(ended.org_id, ended.meeting_id, ended.generation)
        if self.claims is not None:
            try:
                if await self.claims.exists(key):
                    return 0
            except Exception:  # noqa: BLE001 - no claim store: core-api dedupes per decision
                pass
        if self.settle_s > 0:
            await self._sleep(self.settle_s)
        proposal: Proposal | None = None
        for attempt in range(self.attempts):
            try:
                proposal = await self.proposer.propose(
                    ended.org_id, ended.meeting_id, ended.generation
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then given up
                logger.warning(
                    "bets for %s failed (attempt %d): %r", ended.meeting_id, attempt + 1, exc
                )
                if attempt + 1 < self.attempts:
                    await self._sleep(self.backoff_s * (2**attempt))
        if proposal is not None:
            await self.sink.agent_result(
                org_id=ended.org_id,
                meeting_id=ended.meeting_id,
                window_id=f"bets:{ended.generation}",
                user_id=proposal.user_id,
                agent=AGENT,
                result=proposal.result(ended.generation),
            )
            self.sink.flush()  # raises EmitError: the caller must not commit
        if self.claims is not None:
            try:
                await self.claims.set(key, "1", ex=CLAIM_TTL_S)
            except Exception:  # noqa: BLE001
                logger.warning("bets claim not stored for %s", ended.meeting_id)
        return 1 if proposal is not None else 0


async def run_bets_loop(
    consumer: Any, codec: Any, loop: BetsLoop, stop: asyncio.Event
) -> None:  # the confluent-kafka harness, like the minutes loop's
    from confluent_kafka import TopicPartition

    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, 1.0)
        if msg is None or msg.error():
            continue
        topic, partition, offset, raw = msg.topic(), msg.partition(), msg.offset(), msg.value()
        if topic is None or partition is None or offset is None:
            continue
        if raw is not None:
            try:
                value = codec.deserialize_for(topic, raw)
            except Exception:  # noqa: BLE001
                logger.exception("undecodable message on %s@%s; skipped", topic, offset)
            else:
                await loop.handle(value)
        await asyncio.to_thread(
            consumer.commit,
            offsets=[TopicPartition(topic, partition, offset + 1)],
            asynchronous=False,
        )


def consumer_config(brokers: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": GROUP,
        "enable.auto.commit": False,
        # A new group starts at the end: no proposals (and no model calls) for old meetings.
        "auto.offset.reset": "latest",
        "max.poll.interval.ms": 900_000,
    }


def describe(payload: dict[str, Any]) -> str:  # for logs and the eval report
    return json.dumps(payload.get("expectations", []), ensure_ascii=False)


__all__ = [
    "AGENT",
    "GROUP",
    "KIND",
    "TOPIC",
    "BetProposer",
    "BetSource",
    "BetsLoop",
    "generation_of",
    "parse_expectations",
    "run_bets_loop",
]
