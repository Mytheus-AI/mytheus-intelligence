"""The `agent_collaborations` entry both sides of A2A put on `agent-results` (P10 contract 5.9).

`ContextGapAgent` never writes Postgres: every round-trip, denial and block becomes a
collaboration entry inside a `context_fill` AgentResult, and core-api writes one row per entry
(insert ON CONFLICT (id) DO NOTHING) plus the audit row. Ids are deterministic,
`uuid5(CANDIDATE_NS, "ContextGapAgent:{window_id}:{requester}:{n}")`, so a replayed window or
a redelivered request yields the same rows.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid5

from libs.shared.state import AgentResult, TranscriptWindow
from services.agent_runtime_service.a2a.pep import classify_query
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS

AGENT = "ContextGapAgent"
KIND = "context_fill"
PROMPT_REF = "context_gap_agent@v1"
NIL_MEETING = UUID(int=0)
RESULTS = frozenset(
    {
        "answered",
        "no_answer",
        "deadline_miss",
        "authz_denied",
        "org_spoof",
        "foreign_org",
        "rag_hit",
        "rag_miss",
    }
)
MAX_CORRELATION_CHARS = 128


def collaboration_id(window_id: str, requester: UUID, n: int) -> str:
    return str(uuid5(CANDIDATE_NS, f"{AGENT}:{window_id}:{requester}:{n}"))


def entry(
    *,
    hop: str,
    requester: UUID,
    responder: str,
    query: str,
    query_class: str,
    target_scope: str | None,
    result: str,
    success: bool,
    latency_ms: int,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """One collaboration entry without its id (ids are numbered when the list is final)."""
    assert result in RESULTS, result
    return {
        "correlation_id": correlation_id,
        "hop": hop,
        "requester": str(requester),
        "responder": responder,
        "query": query,
        "query_class": query_class,
        "target_scope": target_scope,
        "result": result,
        "success": success,
        "latency_ms": max(0, int(latency_ms)),
    }


def number(entries: list[dict[str, Any]], window_id: str, requester: UUID) -> list[dict[str, Any]]:
    return [{"id": collaboration_id(window_id, requester, n), **e} for n, e in enumerate(entries)]


def payload(
    *,
    query: str,
    query_class: str,
    collaborations: list[dict[str, Any]],
    fill: dict[str, Any] | None,
    model_tier: str | None,
    prompt: str = PROMPT_REF,
) -> dict[str, Any]:
    return {
        "gap": {"query": query, "query_class": query_class},
        "collaborations": collaborations,
        "fill": fill,
        "prompt": prompt,
        "model_tier": model_tier,
    }


def security_window(org_id: UUID, correlation_id: str) -> TranscriptWindow:
    """The pseudo window a responder-side entry is emitted under: the derived org, the nil
    meeting (an AgentRequest does not carry the requester's meeting), and `a2a:{correlation_id}`
    as the window id so a redelivered request emits the same ids and idempotency key."""
    cid = correlation_id[:MAX_CORRELATION_CHARS]
    return TranscriptWindow(
        org_id=org_id, meeting_id=NIL_MEETING, window_id=f"a2a:{cid}", start_ms=0, end_ms=0
    )


def security_result(
    *,
    window: TranscriptWindow,
    requester: UUID,
    responder: str,
    query: str,
    result: str,
    target_scope: str | None = None,
    correlation_id: str | None = None,
) -> AgentResult:
    """A responder-side denial or block: one entry, fill null, confidence 0. The hop is
    "peer": the responder cannot tell a peer request from a related-meeting one (the
    AgentRequest schema carries neither the hop nor the requester's meeting)."""
    q = query[:1000]
    qc = classify_query(q)
    e = entry(
        hop="peer",
        requester=requester,
        responder=responder,
        query=q,
        query_class=qc,
        target_scope=target_scope,
        result=result,
        success=False,
        latency_ms=0,
        correlation_id=(correlation_id or "")[:MAX_CORRELATION_CHARS] or None,
    )
    return AgentResult(
        kind=KIND,
        content="",
        relevance_score=0.0,
        confidence=0.0,
        source_chunk_ids=[],
        payload=payload(
            query=q,
            query_class=qc,
            collaborations=number([e], window.window_id, requester),
            fill=None,
            model_tier=None,
        ),
    )
