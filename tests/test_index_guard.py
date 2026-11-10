"""Never record in the indexes, and confidential meetings out of org-wide search.

1. `IndexGuard`: a chunk naming a never-record topic (or alias) is not indexed, its window is
   purged and the next one is marked; a chunk in a marked window is not indexed; an ambiguous
   chunk costs one cached triage check and is not indexed when the check cannot run; an
   unreadable company brain raises (retry, never index unvetted); an unknown meeting privacy
   is confidential.
2. embedding-service: the same, against an in-memory Qdrant: purged points, no new points, a
   confidential payload flag the org-wide search path excludes.
3. search-service: the same, against the real Postgres read model (skipped without it).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import fakeredis
import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.consumer import ConsumedEvent
from libs.shared.index_guard import (
    IndexGuard,
    MeetingPrivacyClient,
    RouterClassifier,
    mark_key,
    parse_check,
)
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, GlossaryTerm, OrgContextUnavailable
from libs.shared.qdrant import (
    LOGICAL_TRANSCRIPT_CHUNKS,
    physical_collection,
    search_with_tenant_filter,
)
from services.embedding_service import consumer as embedding_consumer
from services.embedding_service.consumer import EmbeddingConsumer
from services.search_service.consumer import TranscriptChunkSearchConsumer
from services.search_service.fts import fts_search

ORG = uuid.uuid4()
MEETING, SECRET_MEETING = uuid.uuid4(), uuid.uuid4()
BRAIN = CompanyContext(
    never_record_topics=("Project Falcon", "salaries"),
    glossary=(GlossaryTerm("Project Falcon", "codename", "", ("Falcon", "the bird deal")),),
)


class Classifier:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def about(self, org_id: uuid.UUID, text_: str, policy: NeverRecordPolicy) -> bool | None:
        self.calls.append(text_)
        return self.answer


def guard(
    classifier: Any = None, brain: CompanyContext = BRAIN, redis: Any = None, private: Any = False
) -> IndexGuard:
    async def org(_org: uuid.UUID) -> CompanyContext:
        return brain

    async def privacy(_org: uuid.UUID, meeting: uuid.UUID) -> bool | None:
        if isinstance(private, Exception):
            raise private
        return meeting == SECRET_MEETING or private

    return IndexGuard(
        redis=redis or fakeredis.aioredis.FakeRedis(),
        org_context=org,
        privacy=privacy,
        classifier=classifier,
    )


# --- 1. the guard ------------------------------------------------------------------------------


def test_the_policy_matches_topics_and_aliases_on_word_boundaries_with_plurals():
    policy = NeverRecordPolicy.from_org(BRAIN)
    assert policy.hit("the falcon numbers came back") == "Project Falcon"
    assert policy.hit("THE BIRD DEAL is close") == "Project Falcon"
    assert policy.hit("our salary review") == "salaries"
    assert policy.hit("falconry is a hobby") is None and policy.hit("") is None
    assert policy.ambiguous("the project plan slipped") is None  # "project" is a stop word
    assert policy.ambiguous("the deal with the bird people") == "Project Falcon"
    assert NeverRecordPolicy.from_org(None).empty


async def test_a_chunk_naming_a_topic_is_skipped_its_window_purged_and_the_next_marked():
    redis = fakeredis.aioredis.FakeRedis()
    g = guard(redis=redis)
    clean = await g.check(ORG, MEETING, 31_000, "Let's review the roadmap.")
    assert clean.index and clean.reason == "ok" and clean.confidential is False
    hit = await g.check(ORG, MEETING, 45_000, "And the Falcon term sheet arrived.")
    assert hit.skip and hit.reason == "never_record" and hit.purge == (30_000, 60_000)
    assert await redis.exists(mark_key(ORG, MEETING, 1)) and await redis.exists(
        mark_key(ORG, MEETING, 2)
    )
    after = await g.check(ORG, MEETING, 70_000, "The number they offered was lower.")
    assert after.skip and after.reason == "never_record_window" and after.purge is None
    other = await g.check(ORG, uuid.uuid4(), 45_000, "Unrelated meeting, same window.")
    assert other.index  # marks are per meeting


async def test_an_ambiguous_chunk_costs_one_cached_check_and_unvetted_is_not_indexed():
    redis = fakeredis.aioredis.FakeRedis()
    yes = Classifier(True)
    g = guard(yes, redis=redis)
    assert (await g.check(ORG, MEETING, 0, "the bird people called back")).reason == "never_record"
    g2 = guard(yes, redis=redis)
    assert (await g2.check(ORG, uuid.uuid4(), 0, "the bird people called back")).skip
    assert len(yes.calls) == 1  # the second consumer reads the cached answer
    no = guard(Classifier(False))
    assert (await no.check(ORG, MEETING, 0, "a bird landed on the deal sheet")).index
    for unavailable in (Classifier(None), None):
        v = await guard(unavailable).check(ORG, MEETING, 0, "a bird landed on the deal sheet")
        assert v.skip and v.reason == "unvetted"


async def test_no_topics_index_everything_and_an_unreadable_brain_raises_for_a_retry():
    assert (await guard(brain=CompanyContext()).check(ORG, MEETING, 0, "Falcon")).index

    async def down(_org: uuid.UUID) -> CompanyContext:
        raise OrgContextUnavailable(str(_org))

    g = IndexGuard(redis=fakeredis.aioredis.FakeRedis(), org_context=down)
    with pytest.raises(OrgContextUnavailable):
        await g.check(ORG, MEETING, 0, "anything")


async def test_an_unknown_meeting_privacy_is_confidential():
    assert (await guard().check(ORG, SECRET_MEETING, 0, "hi")).confidential is True
    assert (await guard(private=None).check(ORG, MEETING, 0, "hi")).confidential is True
    assert (await guard(private=RuntimeError("x")).check(ORG, MEETING, 0, "hi")).confidential
    no_client = IndexGuard(redis=fakeredis.aioredis.FakeRedis(), org_context=guard()._org)
    assert await no_client.confidential(ORG, MEETING) is True


def test_the_check_answer_parses_strictly():
    assert parse_check('{"about": true, "score": 0.9}') is True
    assert parse_check('```json\n{"about": false, "score": 0.2}\n```') is False
    assert parse_check('{"about": false, "score": 0.7}') is True  # the score wins upward
    assert parse_check("I think so") is None


async def test_the_router_classifier_runs_the_registered_triage_prompt():
    calls: list[dict[str, Any]] = []

    class Router:
        async def complete(self, **kw: Any) -> Any:
            calls.append(kw)
            return type("C", (), {"text": json.dumps({"about": True, "score": 0.8})})()

    policy = NeverRecordPolicy.from_org(BRAIN)
    assert await RouterClassifier(Router()).about(ORG, "the bird people", policy) is True
    [call] = calls
    assert call["tier"] == "triage" and call["prompt_ref"] == "never_record_check@v1"
    assert "Project Falcon (also said as Falcon, the bird deal)" in call["user"]
    assert len(call["system"]) > 2400  # past the 512-token cache minimum


async def test_the_privacy_client_reads_core_api_caches_and_never_caches_a_failure():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith(f"{SECRET_MEETING}/index-policy"):
            return httpx.Response(200, json={"confidential": True})
        return httpx.Response(503)

    client = MeetingPrivacyClient(
        "search-service", "http://core", token="t" * 32, transport=httpx.MockTransport(handler)
    )
    assert await client(ORG, SECRET_MEETING) is True
    assert await client(ORG, SECRET_MEETING) is True and len(seen) == 1
    assert seen[0].headers["X-Mytheus-Service"] == "search-service"
    assert seen[0].headers["X-Mytheus-Org"] == str(ORG)
    assert await client(ORG, MEETING) is None and await client(ORG, MEETING) is None
    assert len(seen) == 3


# --- 2. embedding-service ------------------------------------------------------------------------


class _Session:
    async def execute(self, *_a: Any, **_k: Any) -> None:
        return None


def chunk(meeting: uuid.UUID, start_ms: int, text_: str) -> ConsumedEvent:
    cid = uuid.uuid4()
    return ConsumedEvent(
        topic="transcript-chunks",
        org_id=ORG,
        idempotency_key=f"chunk:{cid}",
        header={},
        body={
            "chunk_id": str(cid),
            "meeting_id": str(meeting),
            "speaker_id": "spk",
            "start_ms": start_ms,
            "is_final": True,
            "text": text_,
        },
        partition=0,
        offset=0,
    )


async def test_embedding_never_stores_a_never_record_window_and_flags_confidential(monkeypatch):
    monkeypatch.setattr(
        embedding_consumer, "embed_batch", lambda pieces: ([[0.1] * 384 for _ in pieces], 0.0)
    )

    async def no_outbox(*_a: Any, **_k: Any) -> None:
        return None

    qdrant = AsyncQdrantClient(location=":memory:")
    consumer = EmbeddingConsumer(
        qdrant=qdrant,
        guard=guard(),
        session_factory=None,  # type: ignore[arg-type]
        redis=fakeredis.aioredis.FakeRedis(),
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,
    )
    monkeypatch.setattr(consumer, "emit_outbox", no_outbox)
    name = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, ORG)

    async def count() -> int:
        return (await qdrant.count(name)).count

    await consumer.apply(chunk(MEETING, 30_500, "Roadmap first."), _Session())  # type: ignore[arg-type]
    await consumer.apply(chunk(MEETING, 31_000, "Then the numbers."), _Session())  # type: ignore[arg-type]
    assert await count() == 2
    await consumer.apply(chunk(MEETING, 32_000, "Salaries go up 4%."), _Session())  # type: ignore[arg-type]
    assert await count() == 0  # the window is purged and the naming chunk never stored
    await consumer.apply(chunk(MEETING, 61_000, "for the leads only"), _Session())  # type: ignore[arg-type]
    assert await count() == 0  # the next window is marked
    await consumer.apply(chunk(SECRET_MEETING, 0, "Board prep."), _Session())  # type: ignore[arg-type]
    [point] = (await qdrant.scroll(name))[0]
    assert point.payload["confidential"] is True
    assert (
        await search_with_tenant_filter(qdrant, LOGICAL_TRANSCRIPT_CHUNKS, ORG, [0.1] * 384) == []
    )


# --- 3. search-service ---------------------------------------------------------------------------


async def test_search_never_stores_a_never_record_window_and_hides_confidential_org_wide(
    migrated_db,
):
    from tests.conftest import APP_URL

    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org = uuid.uuid4()
    consumer = TranscriptChunkSearchConsumer(
        guard=guard(),
        session_factory=factory,
        redis=fakeredis.aioredis.FakeRedis(),  # type: ignore[arg-type]
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,
    )

    async def apply(meeting: uuid.UUID, start_ms: int, text_: str) -> None:
        evt = chunk(meeting, start_ms, text_)
        evt = ConsumedEvent(**{**evt.__dict__, "org_id": org})
        evt.body.update(end_ms=start_ms + 500)
        async with factory() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)}
            )
            await consumer.apply(evt, session)

    async def find(q: str, allow: list[uuid.UUID] | None = None) -> list[str]:
        rows = await fts_search(
            session_factory=factory, org_id=org, q=q, meeting_id=None, limit=10, meeting_ids=allow
        )
        return [r.passage for r in rows]

    try:
        await apply(MEETING, 30_500, "The roadmap review covers Falconer onboarding.")
        assert await find("roadmap") == ["The roadmap review covers Falconer onboarding."]
        await apply(MEETING, 31_000, "The salary bands for the roadmap leads.")
        assert await find("roadmap") == []  # purged with its window
        await apply(MEETING, 61_000, "roadmap numbers for the leads only")
        assert await find("roadmap") == []  # the next window is marked
        await apply(SECRET_MEETING, 0, "The board roadmap draft.")
        assert await find("board") == []  # org-wide: never a confidential meeting
        assert await find("board", [SECRET_MEETING]) == ["The board roadmap draft."]
    finally:
        async with factory() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)}
            )
            await session.execute(
                text("DELETE FROM intelligence.transcript_chunk_search WHERE org_id = :o"),
                {"o": str(org)},
            )
        await engine.dispose()
