"""`agent-requests` / `agent-responses`: the requester client, the responder, and their loops.

P10 contract sections 1, 5.5, 5.7 and 5.8; ADR-0013.

Requester (`A2AClient.ask`): mint the requester's bound id, register an in-process future under
a fresh `correlation_id`, publish the AgentRequest (key `org_id`, idempotency key
`areq:{correlation_id}`, `deadline` = the hop deadline), and wait for the future until the
deadline. The PEP pre-publish check happens BEFORE this call, in `ContextGapAgent`.

Response routing (`ResponseRouter`): every runtime instance consumes `agent-responses` in its own
group `agent-runtime-resp-{hostname}` (auto.offset.reset latest) and resolves futures by
`correlation_id`. An unknown id (another instance's request, a late answer) is ignored. A response
whose `responder_agent_id` is not the bound id of the user that was asked is ignored too: only the
asked agent can answer.

Responder (`Responder.handle`, group `agent-runtime-a2a` on `agent-requests`), the PEP
pre-process check, in order, fail closed:
1. verify `requester_agent_id`; bad -> `agent-requests.dlq`, log only (no derivable identity, so
   nobody to attribute an entry to);
2. derived org != `header.org_id` -> spoof: DLQ + an `org_spoof` entry under the DERIVED org;
3. responder user (from `responder_target`) not in the derived org (GetUserContext fails or
   answers another org) -> DLQ + a `foreign_org` entry under the derived org;
4. past the deadline on receipt -> dropped silently (the requester has already moved on);
5. re-authorize with the PDP (defense in depth): deny -> `success=false`,
   `{"error": "authz_denied"}`, plus an `authz_denied` entry;
6. retrieve AS THE RESPONDER (`rag.retrieve(org, user_id=<responder>, query)`), top 3 passages
   -> `{"passages": [{"text", "score", "citation": {"kind", "id", "meeting_id"?}}]}`; nothing ->
   `success=false`, `{"error": "no_answer"}`.

Requests are deadline-bound (about a second), so both consumers auto-commit: a crash loses at
most requests that would have expired anyway.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from libs.shared.rag_client import RagClient, RagPassage
from services.agent_runtime_service.a2a.collab import (
    AGENT,
    security_result,
    security_window,
)
from services.agent_runtime_service.a2a.identity import (
    IdentitySigner,
    parse_target,
    responder_target,
)
from services.agent_runtime_service.a2a.pep import (
    Pep,
    classify_query,
    grantee_role,
    resolve_scope,
)
from services.agent_runtime_service.agents.grounding import CITATION_KINDS, _citation

logger = logging.getLogger(__name__)

REQUESTS_TOPIC = "agent-requests"
RESPONSES_TOPIC = "agent-responses"
DLQ_TOPIC = "agent-requests.dlq"
RESPONDER_GROUP = "agent-runtime-a2a"
MAX_PASSAGES = 3
MAX_PASSAGE_CHARS = 600
MAX_QUERY_CHARS = 1000
ID_KEYS = {
    "transcript": "chunk_id",
    "document": "document_id",
    "decision": "decision_id",
    "action": "action_id",
}

ContextFn = Callable[[UUID, UUID], Awaitable[dict[str, Any] | None]]


def response_group(hostname: str | None = None) -> str:
    return f"agent-runtime-resp-{hostname or socket.gethostname()}"


def micros(value: object) -> int | None:
    """A timestamp-micros field: an int as produced, a datetime as fastavro decodes it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000_000)
    return None


def now_us() -> int:
    return int(time.time() * 1_000_000)


@dataclass(frozen=True)
class ScoredPassage:
    text: str
    score: float
    citation: dict[str, Any] = field(default_factory=dict)  # the stored (ADR-0033) shape


def from_rag(p: RagPassage) -> ScoredPassage:
    return ScoredPassage(
        text=p.text.strip()[:MAX_PASSAGE_CHARS], score=float(p.score), citation=_citation(p)
    )


def wire_citation(p: RagPassage) -> dict[str, Any] | None:
    ident = {
        "transcript": p.chunk_id,
        "document": p.document_id,
        "decision": p.decision_id,
        "action": p.action_id,
    }.get(p.kind)
    if ident is None:
        return None
    out: dict[str, Any] = {"kind": p.kind, "id": str(ident)}
    if p.meeting_id is not None:
        out["meeting_id"] = str(p.meeting_id)
    return out


def answer_json(passages: list[RagPassage]) -> str | None:
    """The responder's success body, or None when no passage can be cited."""
    out = []
    for p in sorted(passages, key=lambda x: -float(x.score)):
        citation = wire_citation(p)
        if citation is None or not p.text.strip():
            continue
        out.append(
            {
                "text": p.text.strip()[:MAX_PASSAGE_CHARS],
                "score": round(float(p.score), 4),
                "citation": citation,
            }
        )
        if len(out) == MAX_PASSAGES:
            break
    return json.dumps({"passages": out}) if out else None


def stored_citation(raw: object) -> dict[str, Any] | None:
    """A wire citation `{kind, id, meeting_id?}` to the stored shape QuestionAgent also uses
    (`{kind, chunk_id|document_id|decision_id|action_id, meeting_id?}`); invalid is None."""
    if not isinstance(raw, dict) or raw.get("kind") not in CITATION_KINDS:
        return None
    kind = str(raw["kind"])
    try:
        ident = str(UUID(str(raw.get("id"))))
    except ValueError:
        return None
    out: dict[str, Any] = {"kind": kind, ID_KEYS[kind]: ident}
    if raw.get("meeting_id"):
        try:
            out["meeting_id"] = str(UUID(str(raw["meeting_id"])))
        except ValueError:
            return None
    return out


def parse_answer(result: object) -> list[ScoredPassage]:
    """A responder's success body to passages; anything malformed is skipped (never raises)."""
    try:
        body = json.loads(str(result))
    except ValueError:
        return []
    raw = body.get("passages") if isinstance(body, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[ScoredPassage] = []
    for p in raw[:MAX_PASSAGES]:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            continue
        citation = stored_citation(p.get("citation"))
        score = p.get("score")
        if citation is None or isinstance(score, bool) or not isinstance(score, int | float):
            continue
        clamped = max(0.0, min(1.0, float(score))) if score == score else 0.0
        out.append(ScoredPassage(p["text"].strip()[:MAX_PASSAGE_CHARS], clamped, citation))
    return out


def error_of(result: object) -> str | None:
    try:
        body = json.loads(str(result))
    except ValueError:
        return None
    return str(body.get("error")) if isinstance(body, dict) and body.get("error") else None


# --- requester -----------------------------------------------------------------------------


class ResponseRouter:
    """In-process futures keyed by `correlation_id`, resolved by the response consumer."""

    def __init__(self) -> None:
        self._waiting: dict[str, tuple[asyncio.Future[dict[str, Any]], str]] = {}

    def expect(
        self, correlation_id: str, responder_agent_id: str
    ) -> asyncio.Future[dict[str, Any]]:
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[correlation_id] = (fut, responder_agent_id)
        return fut

    def forget(self, correlation_id: str) -> None:
        self._waiting.pop(correlation_id, None)

    @property
    def pending(self) -> int:
        return len(self._waiting)

    def resolve(self, value: dict[str, Any]) -> bool:
        """True when the response completed a waiting request; unknown ids are ignored."""
        cid = value.get("correlation_id")
        if not isinstance(cid, str):
            return False
        waiting = self._waiting.get(cid)
        if waiting is None:
            return False
        fut, expected = waiting
        if value.get("responder_agent_id") != expected:
            logger.warning("a2a: response for %s from an agent that was not asked; ignored", cid)
            return False
        self._waiting.pop(cid, None)
        if fut.done():
            return False
        fut.set_result(value)
        return True


@dataclass(frozen=True)
class AskOutcome:
    result: str  # answered | no_answer | deadline_miss | authz_denied
    correlation_id: str | None
    latency_ms: int
    passages: list[ScoredPassage] = field(default_factory=list)

    @property
    def best(self) -> float:
        return max((p.score for p in self.passages), default=0.0)


class A2AClient:
    def __init__(self, emitter: Any, router: ResponseRouter, signer: IdentitySigner) -> None:
        self._emitter = emitter
        self._router = router
        self.signer = signer

    async def ask(
        self,
        *,
        org_id: UUID,
        requester: UUID,
        responder: UUID,
        query: str,
        deadline_s: float,
    ) -> AskOutcome:
        """`deadline_s` is wall-clock epoch seconds (it travels in the envelope)."""
        start = time.monotonic()
        remaining = deadline_s - time.time()
        if remaining <= 0:
            return AskOutcome("deadline_miss", None, 0)
        cid = str(uuid.uuid4())
        fut = self._router.expect(cid, self.signer.mint(org_id, responder))
        try:
            await self._emitter.agent_request(
                org_id=org_id,
                requester_agent_id=self.signer.mint(org_id, requester),
                responder_target=responder_target(responder),
                query=query[:MAX_QUERY_CHARS],
                deadline_us=int(deadline_s * 1_000_000),
                correlation_id=cid,
            )
        except Exception:  # noqa: BLE001 - an unpublished request is simply not answered
            logger.exception("a2a: could not publish request %s", cid)
            self._router.forget(cid)
            return AskOutcome("no_answer", cid, _ms(start))
        try:
            value = await asyncio.wait_for(fut, remaining)
        except TimeoutError:
            return AskOutcome("deadline_miss", cid, _ms(start))
        finally:
            self._router.forget(cid)
        latency = _ms(start)
        if value.get("success") is True:
            passages = parse_answer(value.get("result"))
            if passages:
                return AskOutcome("answered", cid, latency, passages)
            return AskOutcome("no_answer", cid, latency)
        if error_of(value.get("result")) == "authz_denied":
            return AskOutcome("authz_denied", cid, latency)
        return AskOutcome("no_answer", cid, latency)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# --- responder -----------------------------------------------------------------------------


class Responder:
    def __init__(
        self,
        *,
        signer: IdentitySigner | None,
        pep: Pep,
        rag: RagClient | None,
        user_context: ContextFn,
        emitter: Any,
    ) -> None:
        self._signer = signer
        self._pep = pep
        self._rag = rag
        self._context = user_context
        self._emitter = emitter

    def _dlq(
        self,
        value: dict[str, Any],
        raw: bytes,
        partition: int,
        offset: int,
        failure_class: str,
        error: str,
        derived_org: UUID | None,
    ) -> None:
        header = value.get("header") if isinstance(value.get("header"), dict) else {}
        assert isinstance(header, dict)
        self._emitter.dead_letter(
            topic=REQUESTS_TOPIC,
            dlq_topic=DLQ_TOPIC,
            consumer_group=RESPONDER_GROUP,
            partition=partition,
            offset=offset,
            org_id=str(derived_org) if derived_org else str(header.get("org_id") or "unknown"),
            key_org=str(derived_org) if derived_org else None,
            idempotency_key=str(header.get("idempotency_key") or ""),
            failure_class=failure_class,
            error=error,
            raw_value=raw,
        )

    def dead_letter_undecodable(self, raw: bytes, partition: int, offset: int, error: str) -> None:
        self._dlq({}, raw, partition, offset, "DESERIALIZE", error, None)

    async def _security_entry(
        self,
        *,
        org_id: UUID,
        requester: UUID,
        responder: str,
        query: str,
        result: str,
        correlation_id: str,
        target_scope: str | None = None,
    ) -> None:
        window = security_window(org_id, correlation_id)
        await self._emitter.result(
            window=window,
            user_id=requester,
            agent=AGENT,
            result=security_result(
                window=window,
                requester=requester,
                responder=responder,
                query=query,
                result=result,
                target_scope=target_scope,
                correlation_id=correlation_id,
            ),
        )

    async def handle(
        self, value: dict[str, Any], *, raw: bytes = b"", partition: int = -1, offset: int = -1
    ) -> str:
        """Process one AgentRequest. Returns what happened (for logs and tests)."""
        start = time.monotonic()
        header = value.get("header") if isinstance(value.get("header"), dict) else {}
        assert isinstance(header, dict)
        cid = str(value.get("correlation_id") or "")
        query = str(value.get("query") or "")[:MAX_QUERY_CHARS]
        target_raw = str(value.get("responder_target") or "")[:64]

        derived = self._signer.verify(value.get("requester_agent_id")) if self._signer else None
        if derived is None:
            logger.warning("a2a: request %s with an unverifiable requester id; DLQ", cid[:64])
            self._dlq(value, raw, partition, offset, "BAD_SIGNATURE", "requester_agent_id", None)
            return "dlq_bad_signature"
        org, requester = derived.org_id, derived.user_id

        if str(header.get("org_id") or "") != str(org):
            logger.error("a2a: org spoof on %s (derived %s); DLQ", cid[:64], org)
            self._dlq(value, raw, partition, offset, "ORG_SPOOF", "header.org_id", org)
            await self._security_entry(
                org_id=org,
                requester=requester,
                responder=target_raw or "user:unknown",
                query=query,
                result="org_spoof",
                correlation_id=cid,
            )
            return "dlq_org_spoof"

        responder = parse_target(value.get("responder_target"))
        responder_ctx = await self._context(org, responder) if responder is not None else None
        if responder is None or responder_ctx is None:
            logger.error("a2a: responder %s is not in org %s; DLQ", target_raw, org)
            self._dlq(value, raw, partition, offset, "FOREIGN_ORG", "responder_target", org)
            await self._security_entry(
                org_id=org,
                requester=requester,
                responder=target_raw or "user:unknown",
                query=query,
                result="foreign_org",
                correlation_id=cid,
            )
            return "dlq_foreign_org"

        deadline = micros(value.get("deadline"))
        if deadline is None or deadline <= now_us():
            return "expired"

        requester_ctx = await self._context(org, requester)
        query_class = classify_query(query)
        scope = (
            resolve_scope(requester, requester_ctx, responder, responder_ctx)
            if requester_ctx is not None
            else None
        )
        allowed = (
            requester_ctx is not None
            and scope is not None
            and await self._pep.authorize(org, grantee_role(requester_ctx), query_class, scope)
        )
        assert self._signer is not None
        me = self._signer.mint(org, responder)
        if not allowed:
            await self._emitter.agent_response(
                org_id=org,
                correlation_id=cid,
                responder_agent_id=me,
                result=json.dumps({"error": "authz_denied"}),
                success=False,
                latency_ms=_ms(start),
            )
            await self._security_entry(
                org_id=org,
                requester=requester,
                responder=responder_target(responder),
                query=query,
                result="authz_denied",
                correlation_id=cid,
                target_scope=scope,
            )
            return "denied"

        body: str | None = None
        if self._rag is not None and query.strip():
            remaining = deadline / 1_000_000 - time.time()
            try:
                found = await asyncio.wait_for(
                    # AS the responder, never as a service identity (ADR-0033 D1).
                    self._rag.retrieve(org_id=org, user_id=responder, query=query),
                    max(0.0, remaining),
                )
            except TimeoutError:
                found = []
            body = answer_json(found)
        await self._emitter.agent_response(
            org_id=org,
            correlation_id=cid,
            responder_agent_id=me,
            result=body if body is not None else json.dumps({"error": "no_answer"}),
            success=body is not None,
            latency_ms=_ms(start),
        )
        return "answered" if body is not None else "no_answer"


# --- loops ---------------------------------------------------------------------------------


async def run_request_loop(
    consumer: Any,
    codec: Any,
    responder: Responder,
    stop: asyncio.Event,
    *,
    concurrency: int = 16,
    poll_s: float = 0.5,
) -> None:
    """The responder loop. Requests are handled concurrently (each waits on rag-service), up
    to `concurrency` at a time; `poll` runs in a thread so the event loop never blocks."""
    sem = asyncio.Semaphore(concurrency)
    tasks: set[asyncio.Task[None]] = set()

    async def one(value: dict[str, Any], raw: bytes, partition: int, offset: int) -> None:
        try:
            outcome = await responder.handle(value, raw=raw, partition=partition, offset=offset)
            logger.info("a2a request at %s@%s: %s", partition, offset, outcome)
        except Exception:  # noqa: BLE001 - one bad request must not stop the responder
            logger.exception("a2a: request at %s@%s failed", partition, offset)
        finally:
            sem.release()

    try:
        while not stop.is_set():
            msg = await asyncio.to_thread(consumer.poll, poll_s)
            if msg is None or msg.error():
                continue
            raw, partition, offset = msg.value(), msg.partition(), msg.offset()
            if raw is None:
                continue
            try:
                value = codec.deserialize_for(REQUESTS_TOPIC, raw)
            except Exception as exc:  # noqa: BLE001
                responder.dead_letter_undecodable(raw, partition, offset, repr(exc)[:200])
                continue
            await sem.acquire()
            task = asyncio.create_task(one(value, raw, partition, offset))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def run_response_loop(
    consumer: Any, codec: Any, router: ResponseRouter, stop: asyncio.Event, *, poll_s: float = 0.5
) -> None:
    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, poll_s)
        if msg is None or msg.error() or msg.value() is None:
            continue
        try:
            value = codec.deserialize_for(RESPONSES_TOPIC, msg.value())
        except Exception:  # noqa: BLE001 - a response nobody can read is a response not received
            logger.warning("a2a: undecodable agent-responses message skipped")
            continue
        router.resolve(value)


def consumer_configs(brokers: str, hostname: str | None = None) -> dict[str, dict[str, Any]]:
    common = {"bootstrap.servers": brokers, "enable.auto.commit": True}
    return {
        "requests": {**common, "group.id": RESPONDER_GROUP, "auto.offset.reset": "latest"},
        "responses": {
            **common,
            "group.id": response_group(hostname),
            "auto.offset.reset": "latest",
        },
    }
