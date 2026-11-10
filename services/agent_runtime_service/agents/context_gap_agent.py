"""`ContextGapAgent`: fills in prior context a participant lacks, by asking other users' agents.

Plan 10-13 section 3.3; mytheus-docs ADR-0013; the P10 contract section 5. One instance per
meeting (`scope = "meeting"`), and unlike every other agent it emits one result per REQUESTER
(`multi_user`), so the runtime calls `process_meeting` with every participant's context. Per
closed 30 s window:

1. **Detector** (no model call): a turn references prior context ("last time", "we decided",
   "as discussed", "the ... decision", "remember when", "previous meeting", "who owns",
   "what happened with", "per the ...", "follow up on"). No hit, nothing. After a window that
   surfaced a fill the next one is skipped (`COOLDOWN_WINDOWS` = 1).
2. **One gate per window** (triage tier, Haiku, shared by all participants):
   `context_gap_gate@vN` answers `{is_gap, query, score}`; below `CONTEXT_GAP_GATE_THRESHOLD`
   (0.5), or no standalone query, nothing.
3. **query_class**, deterministic (`a2a/pep.py`).
4. **Who lacks it**: each participant's OWN participant-scoped retrieval
   (`rag.retrieve(org, user_id=P, query)`); P lacks the context when no passage scores at least
   `CONTEXT_GAP_HAVE_SCORE`. Requesters: weakest own score first, the referencing speaker only
   if nobody else lacks it, at most `CONTEXT_GAP_MAX_REQUESTERS` (2).
5. **Fallback chain per requester**, all inside `CONTEXT_GAP_DEADLINE_MS` (1500):
   throttled budget or `CONTEXT_GAP_A2A` != 1 -> straight to RAG. Otherwise
   - `peer`: up to 3 other participants (the best-informed first), in parallel, until 60% of the
     deadline. PEP pre-publish per target (bound identity, role, class, scope, cached PDP);
     deny -> an `authz_denied` entry and no publish. Best answered response wins.
   - `related_meeting`: up to 3 users from core-api's related-attendees route, same PEP and
     request flow, until the overall deadline. Route unconfigured or failing -> hop skipped.
   - `rag`: P's own passages from step 4, if there are any.
   - nothing: silent. Never fabricate.
6. **Synthesis** (extract tier, Sonnet): `context_gap_agent@vN` over the gap, the window ("You"
   and "Participant N") and the winning passages -> `{content, confidence, rationale, sources}`.
   Bands: `< 0.4` no fill (collaborations still emitted), `0.4-0.7` `surfaced: false`, `> 0.7`
   `surfaced: true`. A fill that cites no passage it was shown is dropped: never surface uncited
   output.
7. **Output**: one `context_fill` AgentResult per requester that had any hop activity, with the
   collaboration entries core-api turns into `agent_collaborations` rows (`a2a/collab.py`).

What the model sees: transcript lines labelled "You" / "Participant N", the search query and
passage text. No user id, name or email; the router redacts PII on top. Every model call goes
through the router (budget gate, metering on `usage-events`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from uuid import UUID

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.rag_client import HttpRagClient, RagClient
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.a2a import collab
from services.agent_runtime_service.a2a.bus import A2AClient, ScoredPassage, from_rag
from services.agent_runtime_service.a2a.identity import responder_target
from services.agent_runtime_service.a2a.pep import (
    Pep,
    classify_query,
    grantee_role,
    resolve_scope,
)
from services.agent_runtime_service.a2a.related import RelatedAttendeesClient
from services.agent_runtime_service.agents.candidates import confidence_of, text_of
from services.agent_runtime_service.agents.candidates import render_window as labelled_window
from services.agent_runtime_service.agents.grounding import valid_citation
from services.agent_runtime_service.agents.personal_listener_agent import band, render_window

logger = logging.getLogger(__name__)

GATE_KEY = "context_gap_gate"
MAX_PEERS = 3
MAX_RELATED = 3
PEER_SHARE = 0.6  # the peer hop may use 60% of the deadline; related-meeting gets the rest
COOLDOWN_WINDOWS = 1
MAX_QUERY_CHARS = 300
MAX_CONTENT_CHARS = 500

DEFAULT_DEADLINE_MS = 1500
DEFAULT_GATE_THRESHOLD = 0.5
DEFAULT_HAVE_SCORE = 0.5
DEFAULT_MAX_REQUESTERS = 2

_REFERENCE = re.compile(
    r"\b(last time|we decided|we agreed|as (?:we )?discussed|the (?:[\w'-]+ ){0,3}decision|"
    r"remember when|previous meeting|last meeting|earlier meeting|who owns|"
    r"what happened (?:with|to)|per the|follow(?:ing)?[- ]up on)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

ContextFn = Callable[[UUID, UUID], Awaitable[dict[str, Any] | None]]


def detect(window: TranscriptWindow) -> Turn | None:
    """The last turn that references prior context, or None (no model call)."""
    hits = [t for t in window.turns if _REFERENCE.search(t.text)]
    return hits[-1] if hits else None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning("%s is not an integer; using %s", name, default)
        return default


@dataclass(frozen=True)
class ContextGapSettings:
    a2a: bool = True
    deadline_ms: int = DEFAULT_DEADLINE_MS
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    have_score: float = DEFAULT_HAVE_SCORE
    max_requesters: int = DEFAULT_MAX_REQUESTERS

    @classmethod
    def from_env(cls) -> ContextGapSettings:
        return cls(
            a2a=os.environ.get("CONTEXT_GAP_A2A", "1").strip() == "1",
            deadline_ms=max(1, _env_int("CONTEXT_GAP_DEADLINE_MS", DEFAULT_DEADLINE_MS)),
            gate_threshold=min(
                1.0, max(0.0, _env_float("CONTEXT_GAP_GATE_THRESHOLD", DEFAULT_GATE_THRESHOLD))
            ),
            have_score=_env_float("CONTEXT_GAP_HAVE_SCORE", DEFAULT_HAVE_SCORE),
            max_requesters=max(0, _env_int("CONTEXT_GAP_MAX_REQUESTERS", DEFAULT_MAX_REQUESTERS)),
        )


@dataclass
class ContextGapDeps:
    """What the agent talks to. No `rag` = silent agent. No `a2a` or `pep` = RAG-only.
    No `related` or `user_context` = the related-meeting hop is skipped."""

    rag: RagClient | None = None
    a2a: A2AClient | None = None
    pep: Pep | None = None
    related: RelatedAttendeesClient | None = None
    user_context: ContextFn | None = None


def rag_from_env(timeout_s: float) -> RagClient | None:
    """`HttpRagClient` against `RAG_URL` as agent-runtime-service; None without a caller token."""
    if not os.environ.get("RAG_CALLER_TOKEN", "").strip():
        return None
    return HttpRagClient(timeout_s=timeout_s)


def deps_from_env() -> ContextGapDeps:
    """The A2A-less default (RAG only), for a runtime built without the bus."""
    settings = ContextGapSettings.from_env()
    return ContextGapDeps(rag=rag_from_env(settings.deadline_ms / 1000))


@dataclass(frozen=True)
class OwnContext:
    passages: list[ScoredPassage] = field(default_factory=list)
    best: float = 0.0
    latency_ms: int = 0


@dataclass(frozen=True)
class GateVerdict:
    is_gap: bool
    query: str
    score: float
    speaker: UUID | None  # the referencing turn's speaker, when it is a user

    def passed(self, threshold: float) -> bool:
        return self.is_gap and bool(self.query) and self.score >= threshold


@dataclass(frozen=True)
class Winner:
    index: int  # into the requester's entry list
    hop: str
    responder: str  # "user:{uuid}" or "rag"
    passages: list[ScoredPassage]


def choose_requesters(
    own: dict[UUID, OwnContext], speaker: UUID | None, have_score: float, max_n: int
) -> list[UUID]:
    lacking = [u for u, o in own.items() if o.best < have_score]
    if speaker in lacking and len(lacking) > 1:
        lacking.remove(speaker)
    lacking.sort(key=lambda u: (own[u].best, str(u)))
    return lacking[:max_n]


def parse_object(text: str) -> dict[str, Any] | None:
    body = _FENCE.sub("", text.strip()).strip()
    for candidate in (body, *(m.group(0) for m in [_OBJECT.search(body)] if m)):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_gate(text: str) -> tuple[bool, str, float]:
    obj = parse_object(text) or {}
    raw_query = obj.get("query")
    query = " ".join(raw_query.split())[:MAX_QUERY_CHARS] if isinstance(raw_query, str) else ""
    score = confidence_of({"confidence": obj.get("score")}) or 0.0
    return obj.get("is_gap") is True, query, score


def render_passages(passages: list[ScoredPassage]) -> str:
    return "\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages)) or "(none)"


def clip_content(text: str) -> str:
    """At most two sentences and `MAX_CONTENT_CHARS` characters."""
    sentences = _SENTENCE_END.split(" ".join(text.split()))
    return " ".join(sentences[:2])[:MAX_CONTENT_CHARS].strip()


def cited(sources: object, passages: list[ScoredPassage]) -> list[dict[str, Any]]:
    """The citations of the passages the fill names; numbers outside the list are ignored, so a
    fill can never cite something it was not shown."""
    if not isinstance(sources, list):
        return []
    out: list[dict[str, Any]] = []
    for n in sources:
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(passages):
            citation = valid_citation(passages[n - 1].citation)
            if citation is not None and citation not in out:
                out.append(citation)
    return out


def _uuid(raw: str) -> UUID | None:
    try:
        return UUID(raw)
    except ValueError:
        return None


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class ContextGapAgent(BaseAgent):
    name: ClassVar[str] = "ContextGapAgent"
    prompt_key: ClassVar[str] = "context_gap_agent"
    scope: ClassVar[Literal["participant", "meeting"]] = "meeting"
    kind: ClassVar[str] = "context_fill"
    multi_user: ClassVar[bool] = True  # the runtime calls process_meeting, not process

    gate_prompt: Prompt | None = None

    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        deps: ContextGapDeps | None = None,
        settings: ContextGapSettings | None = None,
    ) -> None:
        super().__init__(loader, router)
        self._deps = deps if deps is not None else ContextGapDeps()
        self._settings = settings if settings is not None else ContextGapSettings.from_env()
        self._windows_seen = 0
        self._quiet_until = 0

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        await super().spawn(org_id=org_id, meeting_id=meeting_id, user_id=user_id)
        assert self.org_id is not None
        self.gate_prompt = self._loader.load(GATE_KEY, self._loader.resolve(GATE_KEY, self.org_id))

    async def process(self, state: AgentState) -> AgentResult | None:
        return None  # one result per requester: see process_meeting

    def build_graph(self) -> Any:
        """The shared per-window gate as a LangGraph node: the score goes to `relevance_score`,
        `{is_gap, query}` to `retrieved_facts[0]`."""

        async def gate(s: AgentState) -> dict[str, Any]:
            assert self.gate_prompt is not None and s.org_id is not None
            out = await self.router.complete(
                tier="triage",
                system=self.gate_prompt.system,
                user=self.gate_prompt.render_user(window=labelled_window(s.chunk)[0]),
                org_id=s.org_id,
                user_id=None,  # shared by every participant: metered to the tenant
                max_tokens=120,
                prompt_ref=self.gate_prompt.ref,
            )
            is_gap, query, score = parse_gate(out.text)
            return {
                "relevance_score": score,
                "retrieved_facts": [{"is_gap": is_gap, "query": query}],
            }

        g: Any = StateGraph(AgentState)
        g.add_node("gate", gate)
        g.set_entry_point("gate")
        g.add_edge("gate", END)
        return g.compile()

    async def process_meeting(
        self, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
    ) -> list[tuple[UUID, AgentResult]]:
        if self._graph is None or self.prompt is None or self.org_id is None:
            raise RuntimeError(f"{self.name}.process_meeting before spawn")
        if window.org_id != self.org_id:
            raise ValueError(f"{self.name}: window org_id does not match the spawned agent")
        self._windows_seen += 1
        if self._windows_seen <= self._quiet_until or self._deps.rag is None:
            return []
        people = {u: c for u, c in contexts.items() if c is not None}
        if not people:
            return []
        verdict = await self.assess(window)
        if verdict is None or not verdict.passed(self._settings.gate_threshold):
            return []
        query, score, speaker = verdict.query, verdict.score, verdict.speaker

        query_class = classify_query(query)
        own = await self._own(window.org_id, list(people), query)
        requesters = choose_requesters(
            own, speaker, self._settings.have_score, self._settings.max_requesters
        )
        if not requesters:
            return []
        throttled = await self._throttled(window.org_id)
        results = await asyncio.gather(
            *(
                self._serve(window, p, people, own, query, query_class, float(score), throttled)
                for p in requesters
            )
        )
        emitted = [(p, r) for p, r in zip(requesters, results, strict=True) if r is not None]
        if any((r.payload.get("fill") or {}).get("surfaced") for _p, r in emitted):
            self._quiet_until = self._windows_seen + COOLDOWN_WINDOWS
        return emitted

    async def assess(self, window: TranscriptWindow) -> GateVerdict | None:
        """The detector, then the shared gate. None when nothing references prior context or
        the tenant is blocked (no model call either way)."""
        assert self._graph is not None and self.prompt is not None
        hit = detect(window)
        if hit is None:
            return None
        speaker = _uuid(hit.speaker_id)
        state = AgentState(
            meeting_id=window.meeting_id,
            user_id=speaker or window.meeting_id,
            chunk=window,
            context={},
            org_id=window.org_id,
            agent_id=self.agent_id,
            prompt_version=self.prompt.version,
        )
        try:
            out = await self._graph.ainvoke(state)
        except BudgetBlocked:
            self.blocked = True
            return None
        facts = (out.get("retrieved_facts") if isinstance(out, dict) else out.retrieved_facts) or []
        score = (out.get("relevance_score") if isinstance(out, dict) else out.relevance_score) or 0
        gate = facts[0] if facts else {}
        return GateVerdict(
            is_gap=gate.get("is_gap") is True,
            query=str(gate.get("query") or ""),
            score=float(score),
            speaker=speaker,
        )

    async def _throttled(self, org_id: UUID) -> bool:
        try:
            return await self.router.budget_state(org_id) == "throttle"
        except Exception:  # noqa: BLE001 - unknown budget: take the cheap path (RAG only)
            return True

    async def _own(self, org_id: UUID, users: list[UUID], query: str) -> dict[UUID, OwnContext]:
        rag = self._deps.rag
        assert rag is not None
        timeout = self._settings.deadline_ms / 1000

        async def one(user: UUID) -> OwnContext:
            start = time.monotonic()
            try:
                # AS the participant (ADR-0033 D1): only meetings that user attended.
                found = await asyncio.wait_for(
                    rag.retrieve(org_id=org_id, user_id=user, query=query), timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - no passages is "lacks the context"
                found = []
            passages = [
                sp
                for sp in (from_rag(p) for p in found)
                if sp.text and valid_citation(sp.citation) is not None
            ]
            return OwnContext(
                passages=passages,
                best=max((p.score for p in passages), default=0.0),
                latency_ms=_ms(start),
            )

        found = await asyncio.gather(*(one(u) for u in users))
        return dict(zip(users, found, strict=True))

    async def _serve(
        self,
        window: TranscriptWindow,
        requester: UUID,
        people: dict[UUID, dict[str, Any]],
        own: dict[UUID, OwnContext],
        query: str,
        query_class: str,
        gate_score: float,
        throttled: bool,
    ) -> AgentResult | None:
        org = window.org_id
        start = time.time()
        deadline = start + self._settings.deadline_ms / 1000
        entries: list[dict[str, Any]] = []
        winner: Winner | None = None
        a2a, pep = self._deps.a2a, self._deps.pep
        if self._settings.a2a and not throttled and a2a is not None and pep is not None:
            peers = sorted(
                (u for u in people if u != requester), key=lambda u: (-own[u].best, str(u))
            )[:MAX_PEERS]
            winner = await self._hop(
                "peer",
                org,
                requester,
                people,
                [(u, people[u]) for u in peers],
                query,
                query_class,
                start + (deadline - start) * PEER_SHARE,
                entries,
            )
            related, contexts = self._deps.related, self._deps.user_context
            if winner is None and related is not None and contexts is not None:
                ids = await related.related(
                    org_id=org,
                    meeting_id=window.meeting_id,
                    user_id=requester,
                    timeout_s=deadline - time.time(),
                )
                if ids is not None:  # None: route unavailable, hop skipped without an entry
                    ids = [u for u in ids if u != requester and u not in people][:MAX_RELATED]
                    ctxs = await self._contexts(org, ids, deadline)
                    targets = [(u, c) for u, c in zip(ids, ctxs, strict=True) if c is not None]
                    winner = await self._hop(
                        "related_meeting",
                        org,
                        requester,
                        {**people, **dict(targets)},
                        targets,
                        query,
                        query_class,
                        deadline,
                        entries,
                    )
        if winner is None:
            mine = own[requester]
            entries.append(
                collab.entry(
                    hop="rag",
                    requester=requester,
                    responder="rag",
                    query=query,
                    query_class=query_class,
                    target_scope=None,
                    result="rag_hit" if mine.passages else "rag_miss",
                    success=bool(mine.passages),
                    latency_ms=mine.latency_ms,
                )
            )
            if mine.passages:
                winner = Winner(len(entries) - 1, "rag", "rag", mine.passages)
        numbered = collab.number(entries, window.window_id, requester)
        fill, tier = (None, None)
        if winner is not None:
            fill, tier = await self._synthesize(window, requester, query, winner, numbered)
        return AgentResult(
            kind=self.kind,
            content=fill["content"] if fill else "",
            relevance_score=gate_score,
            confidence=float(fill["confidence"]) if fill else 0.0,
            source_chunk_ids=window.chunk_ids,
            payload=collab.payload(
                query=query,
                query_class=query_class,
                collaborations=numbered,
                fill=fill,
                model_tier=tier,
            ),
        )

    async def _contexts(
        self, org_id: UUID, ids: list[UUID], deadline: float
    ) -> list[dict[str, Any] | None]:
        fetch = self._deps.user_context
        assert fetch is not None

        async def one(user: UUID) -> dict[str, Any] | None:
            try:
                return await asyncio.wait_for(fetch(org_id, user), max(0.0, deadline - time.time()))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - unknown user: nobody to ask
                return None

        return list(await asyncio.gather(*(one(u) for u in ids)))

    async def _hop(
        self,
        hop: str,
        org_id: UUID,
        requester: UUID,
        contexts: dict[UUID, dict[str, Any]],
        targets: list[tuple[UUID, dict[str, Any]]],
        query: str,
        query_class: str,
        deadline: float,
        entries: list[dict[str, Any]],
    ) -> Winner | None:
        a2a, pep = self._deps.a2a, self._deps.pep
        assert a2a is not None and pep is not None
        # PEP pre-publish: the identity is DERIVED from the bound id, never declared.
        bound = a2a.signer.verify(a2a.signer.mint(org_id, requester))
        requester_ctx = contexts[requester]
        role = grantee_role(requester_ctx)

        async def one(target: UUID, target_ctx: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            scope = resolve_scope(requester, requester_ctx, target, target_ctx)
            start = time.monotonic()
            allowed = (
                bound is not None
                and bound.org_id == org_id
                and bound.user_id == requester
                and await pep.authorize(org_id, role, query_class, scope)
            )
            if not allowed:
                denied = collab.entry(
                    hop=hop,
                    requester=requester,
                    responder=responder_target(target),
                    query=query,
                    query_class=query_class,
                    target_scope=scope,
                    result="authz_denied",
                    success=False,
                    latency_ms=_ms(start),
                )
                return denied, None
            outcome = await a2a.ask(
                org_id=org_id,
                requester=requester,
                responder=target,
                query=query,
                deadline_s=deadline,
            )
            asked = collab.entry(
                hop=hop,
                requester=requester,
                responder=responder_target(target),
                query=query,
                query_class=query_class,
                target_scope=scope,
                result=outcome.result,
                success=outcome.result == "answered",
                latency_ms=outcome.latency_ms,
                correlation_id=outcome.correlation_id,
            )
            return asked, outcome if outcome.result == "answered" else None

        best: tuple[float, Winner] | None = None
        for e, outcome in await asyncio.gather(*(one(t, c) for t, c in targets)):
            entries.append(e)
            if outcome is not None and (best is None or outcome.best > best[0]):
                best = (
                    outcome.best,
                    Winner(len(entries) - 1, hop, e["responder"], outcome.passages),
                )
        return best[1] if best else None

    async def _synthesize(
        self,
        window: TranscriptWindow,
        requester: UUID,
        query: str,
        winner: Winner,
        numbered: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        assert self.prompt is not None
        try:
            out = await self.router.complete(
                tier="extract",
                system=self.prompt.system,
                user=self.prompt.render_user(
                    query=query,
                    window=render_window(window, requester),
                    passages=render_passages(winner.passages),
                ),
                org_id=window.org_id,
                user_id=requester,
                max_tokens=400,
                prompt_ref=self.prompt.ref,
            )
        except BudgetBlocked:
            self.blocked = True
            return None, None
        obj = parse_object(out.text) or {}
        content, confidence = text_of(obj, "content"), confidence_of(obj)
        if content is None or confidence is None or band(confidence) == "dropped":
            return None, out.tier
        citations = cited(obj.get("sources"), winner.passages)
        if not citations:
            return None, out.tier  # never surface uncited output
        return {
            "id": numbered[winner.index]["id"],
            "content": clip_content(content),
            "confidence": confidence,
            "surfaced": band(confidence) == "surfaced",
            "rationale": text_of(obj, "rationale") or "",
            "hop": winner.hop,
            "responder": winner.responder,
            "citations": citations,
        }, out.tier
