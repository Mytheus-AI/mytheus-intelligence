"""`RAGService`: expand -> retrieve -> rerank -> assemble -> cite (mytheus-docs doc 09, frozen
shape), with the ADR-0033 additions inside the stages.

Per query:

0. Scope. The asker's meetings, from core-api (D1). No scope, no retrieval: an unknown user
   gets an empty answer, and core-api being down fails the request rather than widening it.
   Then the answer cache, `nexus:{org_id}:rag:{user_id}:{query_hash}`, 10 minutes.
1. Expand (D2). One Haiku call: variants plus filters. Raw-query retrieval starts at the same
   moment, so expansion is off the critical path. Skipped for short queries and for a tenant
   that is throttled or blocked (degrade, never fail).
2. Retrieve. Per query text, the vector arm (transcripts, documents, confirmed decisions, D3)
   and the FTS arm in parallel; the graph arm once, seeded by the person and topics. Every
   transcript and decision candidate must sit in an allowed meeting; that is checked again
   here after the arms, whatever the arms did.
3. RRF (k=60) to 50, hydrate text (transcripts and documents with neighbours, D4; decisions and
   actions from core-api's confirmed rows), rerank against the user's own query.
4. Assemble: best first, near-duplicates dropped, at most `per_meeting_cap` passages from one
   meeting, inside the token budget (halved when the tenant is past 80% of its quota).
5. Cite: every passage binds a citation of its kind, or is dropped. With `generate`, a checked
   answer (D6), after at most one corrective retry without date/person filters.

Each addition sits behind a flag in `RagConfig` so the eval harness can switch it off (D7).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from libs.shared.router import BudgetBlocked
from services.rag_service.answer import AnswerWriter, abstain
from services.rag_service.contracts import (
    AnswerPassage,
    AppliedFilters,
    Citation,
    RagQueryRequest,
    RagResponse,
)
from services.rag_service.domain import (
    Caller,
    Candidate,
    Filters,
    Scope,
    allowed_meetings,
)
from services.rag_service.expand import Expander, Expansion, should_expand
from services.rag_service.fusion import rrf
from services.rag_service.ports import (
    Embedder,
    GraphSearch,
    LexicalSearch,
    Reranker,
    ScopeProvider,
    VectorSearch,
)

logger = logging.getLogger(__name__)

CACHE_TTL_S = 600
BudgetState = Literal["ok", "throttle", "block"]


class BudgetProbe(Protocol):
    async def state(self, org_id: UUID) -> tuple[BudgetState, float]:
        """(decision, fraction of the token quota used)."""
        ...


class AnswerCache(Protocol):
    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: str, ex: int) -> Any: ...


@dataclass(frozen=True)
class RagConfig:
    expand: bool = True
    filters: bool = True
    graph: bool = True
    decisions: bool = True
    rerank: bool = True
    small_to_big: bool = True
    corrective_retry: bool = True
    arm_limit: int = 20
    fused_top: int = 50
    top_k: int = 10
    token_budget: int = 3000
    per_meeting_cap: int = 4
    abstain_below: float = 0.05


@dataclass
class _Run:
    passages: list[AnswerPassage]
    best_score: float
    filters: Filters
    variants: list[str] = field(default_factory=list)


def query_hash(req: RagQueryRequest) -> str:
    norm = " ".join(req.query.lower().split())
    raw = f"{int(req.generate)}|{req.meeting_id or ''}|{norm}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def cache_key(org_id: UUID, user_id: UUID, req: RagQueryRequest) -> str:
    return f"nexus:{org_id}:rag:{user_id}:{query_hash(req)}"


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _near_duplicate(a: str, b: str) -> bool:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.9


def _citation(c: Candidate) -> Citation | None:
    """Bind provenance, or None: a candidate without its required ids is never returned."""
    if c.kind == "transcript" and c.chunk_id and c.meeting_id:
        return Citation(
            kind="transcript",
            meeting_id=c.meeting_id,
            chunk_id=c.chunk_id,
            speaker=c.speaker,
            start_ms=c.start_ms,
        )
    if c.kind == "document" and c.document_id and c.element_id:
        return Citation(
            kind="document",
            document_id=c.document_id,
            element_id=c.element_id,
            element_type=c.element_type,
        )
    if c.kind == "decision" and c.decision_id and c.meeting_id:
        return Citation(kind="decision", decision_id=c.decision_id, meeting_id=c.meeting_id)
    if c.kind == "action" and c.action_id and c.meeting_id:
        return Citation(kind="action", action_id=c.action_id, meeting_id=c.meeting_id)
    return None


class RAGService:
    def __init__(
        self,
        *,
        scope: ScopeProvider,
        embedder: Embedder,
        vector: VectorSearch,
        lexical: LexicalSearch,
        graph: GraphSearch,
        reranker: Reranker,
        expander: Expander | None,
        answerer: AnswerWriter | None,
        budget: BudgetProbe,
        cache: AnswerCache | None = None,
        config: RagConfig | None = None,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
    ) -> None:
        self._scope = scope
        self._embedder = embedder
        self._vector = vector
        self._lexical = lexical
        self._graph = graph
        self._reranker = reranker
        self._expander = expander
        self._answerer = answerer
        self._budget = budget
        self._cache = cache
        self.config = config or RagConfig()
        self._today = today

    def with_config(self, config: RagConfig) -> RAGService:
        clone = RAGService.__new__(RAGService)
        clone.__dict__.update(self.__dict__)
        clone.config = config
        return clone

    async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse:
        scope = await self._scope.scope(caller)
        if scope is None:
            return RagResponse(answer_passages=[], answer=abstain() if req.generate else None)

        key = cache_key(caller.org_id, scope.user_id, req)
        if self._cache is not None:
            try:
                hit = await self._cache.get(key)
            except Exception:
                hit = None
            if hit:
                cached = RagResponse.model_validate_json(hit)
                return cached.model_copy(update={"cache": "hit"})

        state, used = await self._budget.state(caller.org_id)
        degraded = state != "ok"
        budget = self.config.token_budget // 2 if used >= 0.8 else self.config.token_budget

        run, expansion = await self._expand_and_run(caller, scope, req, degraded, budget)

        answer = None
        if req.generate and not degraded and self._answerer is not None:
            low = self.config.rerank and run.best_score < self.config.abstain_below
            if low and self.config.corrective_retry and run.filters.narrows:
                retry = await self._run(
                    caller,
                    scope,
                    req,
                    [req.query, *run.variants],
                    run.filters.relaxed(),
                    expansion.topics,
                    budget,
                )
                if retry.best_score > run.best_score:
                    run = retry
                low = run.best_score < self.config.abstain_below
            if low or not run.passages:
                answer = abstain()
            else:
                try:
                    answer = await self._answerer.write(
                        req.query, run.passages, org_id=caller.org_id, user_id=scope.user_id
                    )
                except BudgetBlocked:
                    degraded, answer = True, None

        f = run.filters
        response = RagResponse(
            answer_passages=run.passages,
            answer=answer,
            expanded_queries=run.variants,
            filters=AppliedFilters(
                date_from=f.date_from.isoformat() if f.date_from else None,
                date_to=f.date_to.isoformat() if f.date_to else None,
                person_ids=list(f.person_ids),
                source=f.source,
            ),
            degraded=degraded,
        )
        if self._cache is not None:
            try:
                await self._cache.set(key, response.model_dump_json(), ex=CACHE_TTL_S)
            except Exception:
                logger.warning("rag: answer cache write failed")
        return response

    async def _expand_and_run(
        self, caller: Caller, scope: Scope, req: RagQueryRequest, degraded: bool, budget: int
    ) -> tuple[_Run, Expansion]:
        expand = (
            self.config.expand
            and not degraded
            and self._expander is not None
            and should_expand(req.query)
        )
        raw_lists = asyncio.create_task(
            self._retrieve_lists(caller, scope, req, [req.query], Filters(), [])
        )
        expansion = Expansion()
        if expand:
            assert self._expander is not None
            try:
                expansion = await self._expander.expand(
                    req.query,
                    org_id=caller.org_id,
                    user_id=scope.user_id,
                    people=scope.people,
                    today=self._today(),
                )
            except BudgetBlocked:
                expansion = Expansion()
            except Exception:
                logger.exception("rag: expansion failed; continuing with the raw query")
                expansion = Expansion()
        filters = expansion.filters if self.config.filters else Filters()
        topics = expansion.topics if self.config.graph else []
        lists, pool = await raw_lists
        if filters.narrows:
            # The raw query ran before the filters existed: hold its hits to them now.
            allowed, kinds, speakers = self._admission(scope, req, filters)
            lists = [
                [k for k in ranked if self._admissible(pool[k], allowed, kinds, speakers)]
                for ranked in lists
            ]
        extra = list(expansion.variants)
        if filters.narrows:
            extra = [req.query, *extra]  # the raw query again, now with the filters pushed down
        if extra or topics:
            more_lists, more_pool = await self._retrieve_lists(
                caller, scope, req, extra, filters, topics
            )
            lists += more_lists
            for k, c in more_pool.items():
                pool.setdefault(k, c)
        run = await self._finish(caller, scope, req, lists, pool, filters, budget)
        run.variants = expansion.variants
        return run, expansion

    async def _run(
        self,
        caller: Caller,
        scope: Scope,
        req: RagQueryRequest,
        texts: Sequence[str],
        filters: Filters,
        topics: Sequence[str],
        budget: int,
    ) -> _Run:
        lists, pool = await self._retrieve_lists(caller, scope, req, texts, filters, topics)
        return await self._finish(caller, scope, req, lists, pool, filters, budget)

    async def _retrieve_lists(
        self,
        caller: Caller,
        scope: Scope,
        req: RagQueryRequest,
        texts: Sequence[str],
        filters: Filters,
        topics: Sequence[str],
    ) -> tuple[list[list[str]], dict[str, Candidate]]:
        org = caller.org_id
        allowed, kinds, speakers = self._admission(scope, req, filters)
        meetings = sorted(allowed, key=str)
        limit = self.config.arm_limit

        vectors: list[list[float]] = []
        if texts:
            try:
                vectors = await self._embedder.embed(org, list(texts)[:3])
            except Exception:
                logger.warning("rag: query embedding failed; vector arm skipped")
        calls: list[Any] = []
        for v in vectors:
            calls.append(
                self._vector.search(
                    org, v, meeting_ids=meetings, speaker_ids=speakers, kinds=kinds, limit=limit
                )
            )
        if "transcript" in kinds:
            for t in texts:
                calls.append(
                    self._lexical.search(
                        org, t, meeting_ids=meetings, speaker_ids=speakers, limit=limit
                    )
                )
        if self.config.graph and "decision" in kinds and (filters.person_ids or topics):
            calls.append(
                self._graph.decisions(
                    org, person_ids=filters.person_ids, topics=topics, limit=limit
                )
            )
        results = await asyncio.gather(*calls, return_exceptions=True)

        lists: list[list[str]] = []
        pool: dict[str, Candidate] = {}
        for res in results:
            if isinstance(res, BaseException):
                logger.warning("rag: an arm failed: %r", res)
                continue
            ranked: list[str] = []
            for c in res:
                if not self._admissible(c, allowed, kinds, speakers):
                    continue
                ranked.append(c.key)
                if c.key not in pool or (c.text and not pool[c.key].text):
                    pool[c.key] = c
            lists.append(ranked)
        return lists, pool

    def _admission(
        self, scope: Scope, req: RagQueryRequest, filters: Filters
    ) -> tuple[set[UUID], set[str], list[str]]:
        """(allowed meetings, wanted kinds, speaker ids) for this scope and these filters."""
        allowed = set(allowed_meetings(scope, filters, req.meeting_id))
        kinds = {"transcript"}
        if not (filters.date_from or filters.date_to or filters.person_ids or req.meeting_id):
            kinds.add("document")  # documents carry no date, speaker or meeting
        if self.config.decisions:
            kinds.add("decision")
        kinds = {k for k in kinds if filters.wants(k)}
        return allowed, kinds, [str(p) for p in filters.person_ids]

    @staticmethod
    def _admissible(c: Candidate, allowed: set[UUID], kinds: set[str], speakers: list[str]) -> bool:
        """The scope rule, applied after the arms whatever they did (ADR-0033 D1)."""
        if c.kind not in kinds and not (c.kind == "action" and "decision" in kinds):
            return False
        if c.kind in ("transcript", "decision", "action"):
            if c.meeting_id is None or c.meeting_id not in allowed:
                return False
        if c.kind == "transcript" and speakers and c.speaker not in speakers:
            return False
        return True

    async def _finish(
        self,
        caller: Caller,
        scope: Scope,
        req: RagQueryRequest,
        lists: list[list[str]],
        pool: dict[str, Candidate],
        filters: Filters,
        budget: int,
    ) -> _Run:
        fused = rrf(lists, top=self.config.fused_top)
        candidates = [replace(pool[k], score=s) for k, s in fused if k in pool]
        hydrated = await self._hydrate(caller, scope, candidates)
        if not hydrated:
            return _Run(passages=[], best_score=0.0, filters=filters)

        if self.config.rerank:
            try:
                order = await self._reranker.rerank(
                    caller.org_id, req.query, [c.text for c in hydrated], len(hydrated)
                )
                ranked = [replace(hydrated[i], score=s) for i, s in order]
            except Exception:
                logger.warning("rag: rerank failed; keeping the fused order")
                ranked = hydrated
        else:
            ranked = hydrated
        ranked = ranked[: self.config.top_k]
        best = ranked[0].score if ranked else 0.0
        return _Run(
            passages=self._assemble(scope, ranked, budget), best_score=best, filters=filters
        )

    async def _hydrate(
        self, caller: Caller, scope: Scope, candidates: list[Candidate]
    ) -> list[Candidate]:
        neighbors = 1 if self.config.small_to_big else 0
        text_kinds = [c for c in candidates if c.kind in ("transcript", "document")]
        decisions = [c.decision_id for c in candidates if c.kind == "decision" and c.decision_id]
        actions = [c.action_id for c in candidates if c.kind == "action" and c.action_id]
        texts, records = await asyncio.gather(
            self._lexical.passages(caller.org_id, text_kinds, neighbors),
            self._scope.records(caller.org_id, scope.user_id, decisions, actions),
        )
        out: list[Candidate] = []
        for c in candidates:
            got = texts.get(c.key) or records.get(c.key)
            if got is None or not got.text:
                continue  # cannot be hydrated, so cannot be cited: dropped
            if (
                c.kind in ("transcript", "decision", "action")
                and got.meeting_id not in scope.meetings
            ):
                continue
            out.append(
                replace(
                    got,
                    score=c.score,
                    before=got.before if neighbors else "",
                    after=got.after if neighbors else "",
                )
            )
        return out

    def _assemble(self, scope: Scope, ranked: list[Candidate], budget: int) -> list[AnswerPassage]:
        used = 0
        per_meeting: dict[UUID, int] = {}
        kept: list[AnswerPassage] = []
        texts: list[str] = []
        for c in ranked:
            citation = _citation(c)
            if citation is None:
                continue
            if any(_near_duplicate(c.text, t) for t in texts):
                continue
            if c.meeting_id is not None:
                if per_meeting.get(c.meeting_id, 0) >= self.config.per_meeting_cap:
                    continue
            cost = _tokens(" ".join((c.before, c.text, c.after)))
            if kept and used + cost > budget:
                continue
            used += cost
            if c.meeting_id is not None:
                per_meeting[c.meeting_id] = per_meeting.get(c.meeting_id, 0) + 1
            info = scope.meetings.get(c.meeting_id) if c.meeting_id else None
            texts.append(c.text)
            kept.append(
                AnswerPassage(
                    n=len(kept) + 1,
                    text=c.text,
                    rerank_score=round(c.score, 4),
                    citation=citation,
                    context_before=c.before or None,
                    context_after=c.after or None,
                    meeting_title=info.title if info else None,
                    occurred_at=info.started_at if info else None,
                )
            )
        return kept
