"""rag-service's `RAGService`: the five stages and the ADR-0033 additions, with every arm faked.

No infra. The arms record what they were asked, so the tests can prove the scope and filters
reached each one; one fake arm misbehaves on purpose to prove the pipeline re-checks scope
whatever an arm returns.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

import pytest

from libs.shared.prompts import PromptLoader
from services.rag_service.answer import ABSTAIN_TEXT, AnswerWriter, validate
from services.rag_service.clients import rebind
from services.rag_service.contracts import RagQueryRequest
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
from services.rag_service.expand import Expander, parse_expansion, resolve_person
from services.rag_service.fusion import rrf
from services.rag_service.main import PROMPTS_DIR
from services.rag_service.pipeline import RagConfig, RAGService, cache_key

ORG = uuid.uuid4()
PRIYA, SAM = uuid.uuid4(), uuid.uuid4()
M_OLD, M_NEW, M_PRIVATE = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
DOC = uuid.uuid4()
DEC = uuid.uuid4()
TODAY = date(2026, 11, 9)

SCOPE = Scope(
    user_id=PRIYA,
    meetings={
        M_OLD: MeetingInfo("Pricing kickoff", datetime(2026, 10, 1, tzinfo=UTC)),
        M_NEW: MeetingInfo("Pricing review", datetime(2026, 11, 4, tzinfo=UTC)),
    },
    people=(Person(PRIYA, "priya.nair@example.com"), Person(SAM, "sam@example.com")),
)


def tc(meeting: UUID, text: str, speaker: UUID = PRIYA, start: int = 0) -> Candidate:
    cid = uuid.uuid4()
    return Candidate(
        kind="transcript",
        key=transcript_key(cid),
        chunk_id=cid,
        meeting_id=meeting,
        speaker=str(speaker),
        start_ms=start,
        text=text,
    )


C_PRICE = tc(M_NEW, "We locked the Pro seat price at 25 dollars per seat.")
C_OLD = tc(M_OLD, "Pricing options were discussed but nothing settled.", SAM)
C_SECRET = tc(M_PRIVATE, "Secret pricing from a meeting Priya was not in.", SAM)
D_DOC = Candidate(
    kind="document",
    key=document_key(DOC, "el_1"),
    document_id=DOC,
    element_id="el_1",
    element_type="NarrativeText",
    text="Tax handling is deferred to Stripe Tax.",
)
D_DEC = Candidate(
    kind="decision",
    key=decision_key(DEC),
    decision_id=DEC,
    meeting_id=M_NEW,
    text="Pro is 25 dollars per seat.",
)


class FakeScope:
    def __init__(self, scope: Scope | None = SCOPE) -> None:
        self._scope = scope
        self.record_calls: list[tuple[list[UUID], list[UUID]]] = []

    async def scope(self, caller: Caller) -> Scope | None:
        return self._scope

    async def records(self, org_id, user_id, decision_ids, action_ids):
        self.record_calls.append((list(decision_ids), list(action_ids)))
        return {D_DEC.key: D_DEC} if DEC in decision_ids else {}


class FakeEmbedder:
    async def embed(self, org_id, queries):
        return [[float(len(q))] for q in queries]


@dataclass
class FakeVector:
    hits: list[Candidate]
    calls: list[dict[str, Any]] = field(default_factory=list)
    delay: float = 0.0

    async def search(self, org_id, vector, *, meeting_ids, speaker_ids, kinds, limit):
        self.calls.append(
            {"meeting_ids": list(meeting_ids), "speaker_ids": list(speaker_ids), "kinds": kinds}
        )
        await asyncio.sleep(self.delay)
        # Strip text: the vector arm returns ids only, like Qdrant payloads.
        return [replace(h, text="") for h in self.hits if h.kind in kinds]


@dataclass
class FakeLexical:
    hits: list[Candidate]
    store: dict[str, Candidate]
    calls: list[dict[str, Any]] = field(default_factory=list)
    delay: float = 0.0

    async def search(self, org_id, query, *, meeting_ids, speaker_ids, limit):
        self.calls.append({"meeting_ids": list(meeting_ids), "speaker_ids": list(speaker_ids)})
        await asyncio.sleep(self.delay)
        return list(self.hits)

    async def passages(self, org_id, candidates, neighbors):
        out = {}
        for c in candidates:
            if c.key in self.store:
                s = self.store[c.key]
                out[c.key] = replace(
                    s, before="(before)" if neighbors else "", after="(after)" if neighbors else ""
                )
        return out


@dataclass
class FakeGraph:
    hits: list[Candidate] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def decisions(self, org_id, *, person_ids, topics, limit):
        self.calls.append({"person_ids": list(person_ids), "topics": list(topics)})
        return [replace(h, text="") for h in self.hits]


class FakeReranker:
    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.scores = scores or {}
        self.default = default

    async def rerank(self, org_id, query, passages: Sequence[str], top_k):
        scored = [(i, self.scores.get(p, self.default)) for i, p in enumerate(passages)]
        return sorted(scored, key=lambda s: s[1], reverse=True)[:top_k]


@dataclass
class Done:
    text: str


class FakeLlm:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw):
        self.calls.append(kw)
        return Done(self.replies.pop(0) if self.replies else "{}")


class FakeBudget:
    def __init__(self, state: str = "ok", used: float = 0.1) -> None:
        self._s = (state, used)

    async def state(self, org_id):
        return self._s


class FakeCache:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex):
        self.data[key] = value


def build(
    *,
    vector_hits=(C_PRICE, C_OLD, D_DOC, D_DEC),
    lexical_hits=(C_PRICE,),
    graph_hits=(),
    scores=None,
    llm_replies=(),
    budget=None,
    scope=SCOPE,
    config=None,
    delay=0.0,
):
    store = {c.key: c for c in (C_PRICE, C_OLD, C_SECRET, D_DOC)}
    llm = FakeLlm(list(llm_replies))
    prompts = PromptLoader(PROMPTS_DIR)
    parts = {
        "scope": FakeScope(scope),
        "vector": FakeVector(list(vector_hits), delay=delay),
        "lexical": FakeLexical(list(lexical_hits), store, delay=delay),
        "graph": FakeGraph(list(graph_hits)),
        "llm": llm,
        "cache": FakeCache(),
    }
    svc = RAGService(
        scope=parts["scope"],
        embedder=FakeEmbedder(),
        vector=parts["vector"],
        lexical=parts["lexical"],
        graph=parts["graph"],
        reranker=FakeReranker(scores),
        expander=Expander(llm, prompts, ["pricing", "roadmap"]),
        answerer=AnswerWriter(llm, prompts),
        budget=budget or FakeBudget(),
        cache=parts["cache"],
        config=config,
        today=lambda: TODAY,
    )
    return svc, parts


CALLER = Caller(org_id=ORG, user_id=PRIYA)


def q(text: str, **kw) -> RagQueryRequest:
    return RagQueryRequest(query=text, **kw)


# --- the five stages ---------------------------------------------------------------------


async def test_every_returned_passage_is_cited_and_numbered():
    svc, _ = build(llm_replies=["{}"])
    res = await svc.query(CALLER, q("what did we decide about the pro seat price"))
    assert res.answer_passages, "expected passages"
    assert [p.n for p in res.answer_passages] == list(range(1, len(res.answer_passages) + 1))
    kinds = {p.citation.kind for p in res.answer_passages}
    assert kinds <= {"transcript", "document", "decision"}
    for p in res.answer_passages:
        if p.citation.kind == "transcript":
            assert p.citation.chunk_id and p.citation.meeting_id
            assert p.meeting_title in ("Pricing kickoff", "Pricing review")


async def test_a_candidate_that_cannot_be_hydrated_is_dropped():
    ghost = tc(M_NEW, "never stored")
    svc, _ = build(vector_hits=(ghost, C_PRICE), lexical_hits=())
    res = await svc.query(CALLER, q("seat"))
    texts = [p.text for p in res.answer_passages]
    assert "never stored" not in texts and C_PRICE.text in texts


# --- ADR-0033 D1: participant scope -----------------------------------------------------


async def test_arms_receive_only_the_askers_meetings_and_a_leaky_arm_is_filtered():
    svc, parts = build(vector_hits=(C_SECRET, C_PRICE), lexical_hits=(C_SECRET,))
    res = await svc.query(CALLER, q("pricing"))
    assert all(set(c["meeting_ids"]) == {M_OLD, M_NEW} for c in parts["vector"].calls)
    assert all(set(c["meeting_ids"]) == {M_OLD, M_NEW} for c in parts["lexical"].calls)
    assert C_SECRET.text not in [p.text for p in res.answer_passages]


async def test_an_unknown_user_gets_nothing():
    svc, parts = build(scope=None)
    res = await svc.query(CALLER, q("pricing", generate=True))
    assert res.answer_passages == [] and res.answer and res.answer.abstained
    assert parts["vector"].calls == []


async def test_meeting_id_narrows_inside_scope_never_outside():
    svc, parts = build()
    await svc.query(CALLER, q("pricing", meeting_id=M_PRIVATE))
    assert all(c["meeting_ids"] == [] for c in parts["vector"].calls)


# --- ADR-0033 D2: expansion and filters -------------------------------------------------


async def test_expansion_filters_reach_every_arm():
    reply = (
        '{"variants": ["seat price decision"], "date_from": "2026-11-02", '
        '"date_to": "2026-11-08", "person": "Priya", "source": null, "topics": ["pricing"]}'
    )
    svc, parts = build(llm_replies=[reply], graph_hits=(D_DEC,))
    res = await svc.query(CALLER, q("what did Priya say about pricing last week"))
    assert res.expanded_queries == ["seat price decision"]
    assert res.filters.date_from == "2026-11-02" and res.filters.person_ids == [PRIYA]
    filtered = [c for c in parts["vector"].calls if c["speaker_ids"]]
    assert filtered and all(c["meeting_ids"] == [M_NEW] for c in filtered)
    assert all("document" not in c["kinds"] for c in filtered)
    assert parts["graph"].calls == [{"person_ids": [PRIYA], "topics": ["pricing"]}]
    # The raw-query results that break the filters (M_OLD, Sam) are gone.
    assert C_OLD.text not in [p.text for p in res.answer_passages]


async def test_short_queries_and_degraded_tenants_skip_expansion():
    svc, parts = build()
    await svc.query(CALLER, q("Pro seat price"))
    assert parts["llm"].calls == []
    svc, parts = build(budget=FakeBudget("throttle"))
    res = await svc.query(CALLER, q("what did we decide about the pro seat price", generate=True))
    assert parts["llm"].calls == [] and res.degraded and res.answer is None
    assert res.answer_passages, "a throttled tenant still gets cited passages"


async def test_malformed_expansion_degrades_to_the_raw_query():
    svc, _ = build(llm_replies=["not json at all"])
    res = await svc.query(CALLER, q("what did we decide about the pro seat price"))
    assert res.answer_passages and res.expanded_queries == []


def test_parse_expansion_rejects_unknown_topics_and_swaps_dates():
    e = parse_expansion(
        '{"variants": ["a", "b", "c"], "date_from": "2026-11-08", "date_to": "2026-11-01", '
        '"source": "nonsense", "topics": ["pricing", "made-up"]}',
        "q",
        [],
        ["pricing"],
    )
    assert e.variants == ["a", "b"] and e.topics == ["pricing"] and e.filters.source is None
    assert e.filters.date_from == date(2026, 11, 1) and e.filters.date_to == date(2026, 11, 8)


def test_resolve_person_by_email_local_part():
    people = [Person(PRIYA, "priya.nair@x.com"), Person(SAM, "sam@x.com")]
    assert resolve_person("Priya", people) == (PRIYA,)
    assert resolve_person("Priya Nair", people) == (PRIYA,)
    assert resolve_person("Nobody", people) == ()


# --- ADR-0033 D3, D4: decisions as a source, small-to-big -------------------------------


async def test_a_decision_is_cited_as_a_decision():
    svc, parts = build(vector_hits=(D_DEC,), lexical_hits=())
    res = await svc.query(CALLER, q("seat"))
    [p] = res.answer_passages
    assert p.citation.kind == "decision" and p.citation.decision_id == DEC
    assert p.citation.meeting_id == M_NEW
    assert parts["scope"].record_calls == [([DEC], [])]


async def test_small_to_big_adds_context_and_can_be_switched_off():
    svc, _ = build(lexical_hits=(), vector_hits=(C_PRICE,))
    on = await svc.query(CALLER, q("seat"))
    assert on.answer_passages[0].context_before == "(before)"
    off = await svc.with_config(RagConfig(small_to_big=False)).query(CALLER, q("seat price"))
    assert off.answer_passages[0].context_before is None


# --- stage 4: assembly -------------------------------------------------------------------


async def test_near_duplicates_and_the_per_meeting_cap():
    dup = tc(M_NEW, "We locked the Pro seat price at 25 dollars per seat!")
    many = [tc(M_NEW, f"distinct point number {i} about seats") for i in range(6)]
    svc, parts = build(vector_hits=(C_PRICE, dup, *many), lexical_hits=())
    parts["lexical"].store.update({c.key: c for c in (dup, *many)})
    res = await svc.query(CALLER, q("seat"))
    texts = [p.text for p in res.answer_passages]
    assert not (C_PRICE.text in texts and dup.text in texts)
    assert sum(1 for p in res.answer_passages if p.citation.meeting_id == M_NEW) <= 4


# --- ADR-0033 D6: the grounded answer ----------------------------------------------------


async def test_answer_keeps_only_cited_sentences():
    reply = "Pro is 25 dollars per seat [1]. It was a great meeting. Tax goes to Stripe. [9]"
    svc, _ = build(llm_replies=["{}", reply], scores={C_PRICE.text: 0.9})
    res = await svc.query(CALLER, q("what did we decide about the pro seat price", generate=True))
    assert res.answer and not res.answer.abstained
    assert res.answer.text == "Pro is 25 dollars per seat [1]."
    assert res.answer.cited == [1]


async def test_low_scores_abstain_without_calling_the_answer_model():
    svc, parts = build(scores={}, llm_replies=["{}"])
    svc = svc.with_config(RagConfig(abstain_below=0.9))
    res = await svc.query(CALLER, q("what is the capital of France today", generate=True))
    assert res.answer and res.answer.abstained and res.answer.text == ABSTAIN_TEXT
    assert [c["tier"] for c in parts["llm"].calls] == ["triage"]  # expansion only


async def test_corrective_retry_drops_date_and_person_filters():
    reply = '{"variants": [], "person": "Sam", "date_from": "2026-11-02"}'
    svc, parts = build(
        llm_replies=[reply, "Options were discussed [1]."],
        scores={C_OLD.text: 0.8},
        vector_hits=(C_OLD,),
        lexical_hits=(),
    )
    svc = svc.with_config(RagConfig(abstain_below=0.3))
    res = await svc.query(CALLER, q("what did Sam say about pricing last week", generate=True))
    # C_OLD is from October: the filtered run finds nothing, the relaxed retry finds it.
    assert res.answer and not res.answer.abstained
    assert any(
        c["speaker_ids"] == [] and set(c["meeting_ids"]) == {M_OLD, M_NEW}
        for c in parts["vector"].calls
    )


def test_validate_moves_trailing_markers_and_rejects_out_of_range():
    assert validate("We ship Friday. [1]", 2).text == "We ship Friday [1]."
    assert validate("We ship Friday [3].", 2).abstained
    assert validate("NOT_FOUND", 2).abstained
    assert validate("A [1]. B [2][1].", 2).cited == [1, 2]


# --- cache, parallelism, fusion ---------------------------------------------------------


async def test_cache_hit_is_per_user():
    svc, parts = build()
    first = await svc.query(CALLER, q("seat"))
    assert first.cache == "miss"
    calls = len(parts["vector"].calls)
    second = await svc.query(CALLER, q("seat"))
    assert second.cache == "hit" and len(parts["vector"].calls) == calls
    assert cache_key(ORG, PRIYA, q("seat")) != cache_key(ORG, SAM, q("seat"))
    assert cache_key(ORG, PRIYA, q("seat")).startswith(f"nexus:{ORG}:rag:{PRIYA}:")


async def test_arms_run_in_parallel():
    svc, _ = build(delay=0.2)
    started = time.perf_counter()
    await svc.query(CALLER, q("seat"))
    assert time.perf_counter() - started < 0.35  # two 0.2 s arms, not 0.4 s in series


def test_rrf_rewards_agreement_across_lists():
    fused = rrf([["a", "b", "c"], ["b", "a"], ["b"]], k=60)
    assert [k for k, _ in fused][:2] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


def test_rerank_rebinds_duplicate_texts_to_distinct_indexes():
    assert rebind(["x", "y", "x"], [("x", 0.9), ("x", 0.8), ("y", 2.0)]) == [
        (0, 0.9),
        (2, 0.8),
        (1, pytest.approx(0.8808, abs=1e-3)),
    ]
