"""RAG eval harness and ablation (mytheus-docs ADR-0033 D7; doc 08 section 6.3).

A fixed corpus (`tests/eval/rag_corpus.json`: six meetings, one private to two other people,
one document, five confirmed decisions) and a golden set (`tests/eval/rag_golden.json`:
answerable questions with gold passages, permission traps, no-answer questions).

What is real: BGE-small-en query and passage embeddings, the BGE-reranker-base cross-encoder,
and Postgres FTS (`search-service`'s own `fts_search` / `get_passages` over a seeded tenant).
What is in-process: exact cosine search over the same BGE vectors in place of Qdrant (identical
ranking at this size), the asker's scope and the confirmed decisions (core-api's job), and the
graph arm (decisions in meetings a person attended, the `decisions_by_person` template).

Offline rows run with `pytest -m eval` (needs Postgres and the two models). The expansion and
answer rows need `EVAL_LIVE=1` and an Anthropic key, as for the agent evals.

Metrics, answerable questions: Recall@10, MRR, nDCG@10 over gold passage ids. Traps: any
passage from the private meeting is a leak, and every row must have zero. With answers:
answer rate on answerable questions, abstain rate on no-answer questions, and citation hit
(the answer cites at least one gold passage). Results go to `tests/eval/rag_ablation.json`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from services.rag_service.clients import rebind
from services.rag_service.contracts import RagQueryRequest, RagResponse
from services.rag_service.domain import (
    Caller,
    Candidate,
    MeetingInfo,
    Person,
    Scope,
    decision_key,
    document_key,
    transcript_key,
)
from services.rag_service.pipeline import RagConfig, RAGService
from services.search_service.fts import fts_search, get_passages
from tests.conftest import APP_URL

pytestmark = pytest.mark.eval

EVAL = Path(__file__).parent / "eval"
CORPUS = json.loads((EVAL / "rag_corpus.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((EVAL / "rag_golden.json").read_text(encoding="utf-8"))
NS = uuid.UUID("0b3a3f7e-5f7a-4f39-9d8a-3c1f6a2b9e11")
LIVE = os.environ.get("EVAL_LIVE") == "1"

_OFF = {"expand": False, "graph": False}
OFFLINE_ROWS: list[tuple[str, RagConfig]] = [
    (
        "vector only",
        RagConfig(
            lexical=False,
            rerank=False,
            decisions=False,
            small_to_big=False,
            rerank_context=False,
            skip_headings=False,
            **_OFF,
        ),
    ),
    (
        "+ FTS and RRF",
        RagConfig(
            rerank=False,
            decisions=False,
            small_to_big=False,
            rerank_context=False,
            skip_headings=False,
            **_OFF,
        ),
    ),
    (
        "+ rerank (plain)",
        RagConfig(
            decisions=False,
            small_to_big=False,
            rerank_context=False,
            skip_headings=False,
            rerank_fusion=False,
            **_OFF,
        ),
    ),
    (
        "+ rerank with context",
        RagConfig(
            decisions=False, small_to_big=False, skip_headings=False, rerank_fusion=False, **_OFF
        ),
    ),
    (
        "+ skip headings",
        RagConfig(decisions=False, small_to_big=False, rerank_fusion=False, **_OFF),
    ),
    ("+ fuse rerank with first stage", RagConfig(decisions=False, small_to_big=False, **_OFF)),
    ("+ decisions source", RagConfig(small_to_big=False, **_OFF)),
    ("+ small-to-big", RagConfig(**_OFF)),
]
LIVE_ROWS: list[tuple[str, RagConfig]] = [("+ expansion, filters, graph", RagConfig())]


# --- the world ---------------------------------------------------------------------------


@dataclass
class World:
    org: UUID
    scope: Scope
    vectors: list[tuple[Candidate, list[float]]]
    decisions: dict[str, Candidate]
    attended: dict[UUID, set[UUID]]
    local: dict[str, str]  # candidate key -> corpus id ("m1#2", "d1#e3", "dec1")
    private_meeting: UUID
    sf: Any


def _uid(org: UUID, local: str) -> UUID:
    return uuid.uuid5(NS, f"{org}:{local}")


@pytest.fixture(scope="module")
def world(migrated_db) -> World:
    import asyncio

    from services.embedding_service.model import embed_batch

    org = uuid.uuid4()
    people = {name: _uid(org, f"person:{name}") for name in CORPUS["people"]}
    asker = people[CORPUS["asker"]]
    meetings: dict[str, UUID] = {m["id"]: _uid(org, m["id"]) for m in CORPUS["meetings"]}
    info: dict[UUID, MeetingInfo] = {}
    attended: dict[UUID, set[UUID]] = {p: set() for p in people.values()}
    rows: list[dict[str, Any]] = []
    cands: list[Candidate] = []
    local: dict[str, str] = {}
    for m in CORPUS["meetings"]:
        mid = meetings[m["id"]]
        info[mid] = MeetingInfo(
            m["title"], datetime.fromisoformat(m["date"]).replace(hour=10, tzinfo=UTC)
        )
        for name in m["participants"]:
            attended[people[name]].add(mid)
        for i, (speaker, body) in enumerate(m["chunks"]):
            cid = _uid(org, f"{m['id']}#{i}")
            c = Candidate(
                kind="transcript",
                key=transcript_key(cid),
                chunk_id=cid,
                meeting_id=mid,
                speaker=str(people[speaker]),
                start_ms=i * 15000,
                text=body,
            )
            cands.append(c)
            local[c.key] = f"{m['id']}#{i}"
            rows.append(
                {
                    "org_id": str(org),
                    "meeting_id": str(mid),
                    "chunk_id": str(cid),
                    "speaker_id": c.speaker,
                    "start_ms": c.start_ms,
                    "end_ms": c.start_ms + 14000,
                    "text": body,
                }
            )
    docs: list[dict[str, Any]] = []
    for d in CORPUS["documents"]:
        did = _uid(org, d["id"])
        for pos, (el, etype, body) in enumerate(d["elements"]):
            c = Candidate(
                kind="document",
                key=document_key(did, el),
                document_id=did,
                element_id=el,
                element_type=etype,
                text=body,
            )
            cands.append(c)
            local[c.key] = f"{d['id']}#{el}"
            docs.append(
                {"org_id": str(org), "d": str(did), "el": el, "t": etype, "x": body, "p": pos}
            )
    decisions: dict[str, Candidate] = {}
    for dec in CORPUS["decisions"]:
        did = _uid(org, dec["id"])
        c = Candidate(
            kind="decision",
            key=decision_key(did),
            decision_id=did,
            meeting_id=meetings[dec["meeting"]],
            text=dec["content"],
        )
        decisions[c.key] = c
        cands.append(c)
        local[c.key] = dec["id"]

    async def seed() -> None:
        engine = create_async_engine(APP_URL, pool_pre_ping=True)
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)}
            )
            await conn.execute(
                text(
                    "INSERT INTO intelligence.transcript_chunk_search (org_id, meeting_id, "
                    "chunk_id, speaker_id, start_ms, end_ms, text) VALUES (:org_id, "
                    ":meeting_id, :chunk_id, :speaker_id, :start_ms, :end_ms, :text)"
                ),
                rows,
            )
            await conn.execute(
                text(
                    "INSERT INTO intelligence.document_chunks (org_id, document_id, element_id, "
                    "element_type, text, position) VALUES (:org_id, :d, :el, :t, :x, :p)"
                ),
                docs,
            )
        await engine.dispose()

    asyncio.run(seed())
    vecs, _ = embed_batch([c.text for c in cands])
    scope = Scope(
        user_id=asker,
        meetings={m: info[m] for m in attended[asker]},
        people=tuple(Person(people[n], e) for n, e in CORPUS["people"].items()),
    )
    return World(
        org=org,
        scope=scope,
        vectors=list(zip(cands, vecs, strict=True)),
        decisions=decisions,
        attended=attended,
        local=local,
        private_meeting=meetings[GOLDEN["private_meeting"]],
        sf=None,
    )


# --- the arms ---------------------------------------------------------------------------


class ScopeArm:
    def __init__(self, w: World) -> None:
        self.w = w

    async def scope(self, caller: Caller) -> Scope:
        return self.w.scope

    async def records(self, org_id, user_id, decision_ids, action_ids):
        allowed = set(self.w.scope.meetings)
        return {
            decision_key(d): self.w.decisions[decision_key(d)]
            for d in decision_ids
            if decision_key(d) in self.w.decisions
            and self.w.decisions[decision_key(d)].meeting_id in allowed
        }


class LocalEmbedder:
    async def embed(self, org_id, queries):
        from services.embedding_service.model import embed_batch

        vecs, _ = embed_batch(list(queries))
        return vecs


class ExactVector:
    """Exact cosine over the BGE vectors, with Qdrant's filters: transcripts and decisions in
    the allowed meetings (and speakers), documents org-wide."""

    def __init__(self, w: World) -> None:
        self.w = w

    async def search(self, org_id, vector, *, meeting_ids, speaker_ids, kinds, limit):
        allowed = set(meeting_ids)
        scored = []
        for c, v in self.w.vectors:
            if c.kind not in kinds:
                continue
            if c.kind in ("transcript", "decision") and c.meeting_id not in allowed:
                continue
            if c.kind == "transcript" and speaker_ids and c.speaker not in speaker_ids:
                continue
            scored.append((sum(a * b for a, b in zip(vector, v, strict=True)), c))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [replace(c, text="") for _, c in scored[:limit]]


class PostgresLexical:
    def __init__(self, w: World) -> None:
        self.w = w

    async def search(self, org_id, query, *, meeting_ids, speaker_ids, limit):
        rows = await fts_search(
            session_factory=self.w.sf,
            org_id=org_id,
            q=query,
            meeting_id=None,
            limit=limit,
            meeting_ids=list(meeting_ids),
            speaker_ids=list(speaker_ids) or None,
            match_any=True,
        )
        return [
            Candidate(
                kind="transcript",
                key=transcript_key(r.chunk_id),
                chunk_id=UUID(r.chunk_id),
                meeting_id=UUID(r.meeting_id),
                speaker=r.speaker_id,
                start_ms=r.start_ms,
                text=r.passage,
            )
            for r in rows
        ]

    async def passages(self, org_id, candidates, neighbors):
        chunk_ids = [c.chunk_id for c in candidates if c.kind == "transcript" and c.chunk_id]
        docs = [
            (c.document_id, c.element_id)
            for c in candidates
            if c.kind == "document" and c.document_id and c.element_id
        ]
        ts, ds = await get_passages(
            session_factory=self.w.sf,
            org_id=org_id,
            chunk_ids=chunk_ids,
            neighbors=neighbors,
            documents=docs,
        )
        out: dict[str, Candidate] = {}
        for t in ts:
            out[transcript_key(t.chunk_id)] = Candidate(
                kind="transcript",
                key=transcript_key(t.chunk_id),
                chunk_id=UUID(t.chunk_id),
                meeting_id=UUID(t.meeting_id),
                speaker=t.speaker_id,
                start_ms=t.start_ms,
                text=t.text,
                before=t.before,
                after=t.after,
            )
        for d in ds:
            key = document_key(d.document_id, d.element_id)
            out[key] = Candidate(
                kind="document",
                key=key,
                document_id=UUID(d.document_id),
                element_id=d.element_id,
                element_type=d.element_type,
                text=d.text,
            )
        return out


class AttendedGraph:
    def __init__(self, w: World) -> None:
        self.w = w

    async def decisions(self, org_id, *, person_ids, topics, limit):
        meetings = set().union(*(self.w.attended.get(p, set()) for p in person_ids))
        return [replace(c, text="") for c in self.w.decisions.values() if c.meeting_id in meetings][
            :limit
        ]


class LocalReranker:
    async def rerank(self, org_id, query, passages: Sequence[str], top_k):
        from services.reranker_service.model import rerank_batch

        ranked, _ = rerank_batch(query, list(passages), top_k=top_k)
        return rebind(passages, ranked)


class OkBudget:
    async def state(self, org_id):
        return "ok", 0.0


def _load_dotenv() -> None:
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def build_service(w: World, config: RagConfig) -> RAGService:
    expander = answerer = None
    if LIVE:
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("EVAL_LIVE=1 needs ANTHROPIC_API_KEY (env or .env)")
        from libs.shared.prompts import PromptLoader
        from libs.shared.router import LiteLLMRouter
        from services.knowledge_graph_service.taxonomy import distinct_topics
        from services.rag_service.answer import AnswerWriter
        from services.rag_service.expand import Expander
        from services.rag_service.main import PROMPTS_DIR

        class OpenGate:
            async def check(self, org_id, est):
                return "ok"

        class NullSink:
            async def usage(self, **kw):
                return None

        router = LiteLLMRouter(OpenGate(), NullSink())  # type: ignore[arg-type]
        prompts = PromptLoader(PROMPTS_DIR)
        expander = Expander(router, prompts, distinct_topics())
        answerer = AnswerWriter(router, prompts)
    return RAGService(
        scope=ScopeArm(w),
        embedder=LocalEmbedder(),
        vector=ExactVector(w),
        lexical=PostgresLexical(w),
        graph=AttendedGraph(w),
        reranker=LocalReranker(),
        expander=expander,
        answerer=answerer,
        budget=OkBudget(),
        config=config,
        today=lambda: date.fromisoformat(CORPUS["today"]),
    )


# --- metrics ----------------------------------------------------------------------------


def _ids(w: World, res: RagResponse) -> list[str]:
    out = []
    for p in res.answer_passages:
        c = p.citation
        if c.kind == "transcript":
            key = transcript_key(str(c.chunk_id))
        elif c.kind == "document":
            key = document_key(str(c.document_id), c.element_id or "")
        elif c.kind == "decision":
            key = decision_key(str(c.decision_id))
        else:
            key = f"action:{c.action_id}"
        out.append(w.local.get(key, key))
    return out


def _leaks(w: World, res: RagResponse) -> int:
    return sum(1 for p in res.answer_passages if p.citation.meeting_id == w.private_meeting)


def score_row(w: World, results: list[tuple[dict[str, Any], RagResponse, float]]) -> dict:
    recalls, rrs, ndcgs, latencies = [], [], [], []
    leaks = 0
    answered = abstained_ok = cite_hits = answerable_n = no_answer_n = 0
    for item, res, secs in results:
        latencies.append(secs)
        leaks += _leaks(w, res)
        got = _ids(w, res)[:10]
        if item["kind"] == "answerable":
            answerable_n += 1
            gold = set(item["gold"])
            recalls.append(len(gold & set(got)) / len(gold))
            first = next((i for i, g in enumerate(got, 1) if g in gold), None)
            rrs.append(1 / first if first else 0.0)
            dcg = sum(1 / math.log2(i + 1) for i, g in enumerate(got, 1) if g in gold)
            ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
            ndcgs.append(dcg / ideal)
            if res.answer is not None and not res.answer.abstained:
                answered += 1
                cited = {got[n - 1] for n in res.answer.cited if 0 < n <= len(got)}
                cite_hits += bool(cited & gold)
        elif item["kind"] == "no_answer":
            no_answer_n += 1
            abstained_ok += bool(res.answer is not None and res.answer.abstained)
    row = {
        "recall_at_10": round(statistics.mean(recalls), 3),
        "mrr": round(statistics.mean(rrs), 3),
        "ndcg_at_10": round(statistics.mean(ndcgs), 3),
        "trap_leaks": leaks,
        "p95_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
    }
    if any(res.answer is not None for _, res, _ in results):
        row |= {
            "answer_rate": round(answered / answerable_n, 3),
            "abstain_rate_no_answer": round(abstained_ok / max(no_answer_n, 1), 3),
            "citation_hit": round(cite_hits / max(answered, 1), 3),
        }
    return row


async def run_row(w: World, config: RagConfig, generate: bool) -> dict:
    svc = build_service(w, config)
    caller = Caller(org_id=w.org, user_id=w.scope.user_id)
    results = []
    for item in GOLDEN["questions"]:
        started = time.perf_counter()
        res = await svc.query(caller, RagQueryRequest(query=item["q"], generate=generate))
        results.append((item, res, time.perf_counter() - started))
        if os.environ.get("RAG_EVAL_DEBUG"):
            top = max((p.rerank_score for p in res.answer_passages), default=0.0)
            print(item["kind"], round(top, 4), _ids(w, res)[:5], item["q"][:40])
            if generate:
                print("  FILTERS", res.filters.model_dump(), res.expanded_queries)
    return score_row(w, results)


# --- the gate ---------------------------------------------------------------------------


async def test_rag_ablation(world: World):
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    world.sf = async_sessionmaker(engine, expire_on_commit=False)
    rows: list[dict[str, Any]] = []
    for name, config in OFFLINE_ROWS:
        rows.append({"row": name, **await run_row(world, config, generate=False)})
    if LIVE:
        for name, config in LIVE_ROWS:
            rows.append({"row": name, **await run_row(world, config, generate=True)})

    print("\nRAG ablation")
    for r in rows:
        print(json.dumps(r))
    (EVAL / "rag_ablation.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    assert all(r["trap_leaks"] == 0 for r in rows), "a private meeting leaked into an answer"
    baseline = json.loads((EVAL / "rag_baseline.json").read_text(encoding="utf-8"))
    final_offline = rows[len(OFFLINE_ROWS) - 1]
    assert final_offline["recall_at_10"] >= baseline["min_recall_at_10"]
    assert final_offline["ndcg_at_10"] >= baseline["min_ndcg_at_10"]
    assert final_offline["mrr"] >= baseline["min_mrr"]
    # The shipped configuration must not be worse than plain vector search on any metric.
    for metric in ("recall_at_10", "mrr", "ndcg_at_10"):
        assert final_offline[metric] >= rows[0][metric], metric
    if LIVE:
        live = rows[-1]
        assert live["recall_at_10"] >= baseline["live_min_recall_at_10"]
        assert live["citation_hit"] >= baseline["live_min_citation_hit"]
        assert live["abstain_rate_no_answer"] >= baseline["live_min_abstain_rate_no_answer"]
