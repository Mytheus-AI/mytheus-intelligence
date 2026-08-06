"""`GET /search` end to end through a real FastAPI `TestClient`, against real Postgres (no
mocked infra). Boots the app the same way `main.py` does (`create_app` + `include_router`).

The central property under test, per the task brief: RLS (plus the route's own explicit
`WHERE org_id = :org_id`) must constrain the CANDIDATE SET a query considers, not merely filter
the page of results returned at the end. A design that ran a global `ORDER BY rank LIMIT` first
and filtered by tenant afterward would never return a foreign ROW, but it could still silently
STARVE a tenant's own true matches whenever another tenant has more, or higher-ranked, matches
for the same shared vocabulary — a real leak of information (the presence/ranking of another
org's content) even with zero foreign rows ever returned. `test_asymmetric_candidate_set_...`
below is built specifically to surface that failure mode: two orgs share a distinctive phrase,
one org has far more/stronger matches than the other, and a small `limit` is used so a
global-then-filter design would starve the smaller org down to zero results if it existed here.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from libs.shared.app import create_app
from services.search_service.consumer import TranscriptChunkSearchConsumer
from services.search_service.routes import create_search_router
from tests.conftest import APP_URL, auth_headers

_INSERT = text("""
    INSERT INTO intelligence.transcript_chunk_search
        (org_id, meeting_id, chunk_id, speaker_id, start_ms, end_ms, text)
    VALUES (:org_id, :meeting_id, :chunk_id, :speaker_id, :start_ms, :end_ms, :text)
    """)


@pytest.fixture()
def engine(migrated_db):
    return create_async_engine(APP_URL, pool_pre_ping=True)


@pytest.fixture()
def client(migrated_db):
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app("search-service")
    app.include_router(create_search_router(session_factory=session_factory))
    # `with TestClient(...)`, not a bare instance — pins one event loop for the client's whole
    # lifetime so pooled asyncpg connections stay valid across calls (ingestion-service's
    # test_ingestion_routes.py established why this matters for this repo's async engines).
    with TestClient(app) as c:
        yield c


async def _insert_chunk(
    engine,
    *,
    org_id: uuid.UUID,
    meeting_id: uuid.UUID,
    chunk_id: uuid.UUID,
    text_: str,
    speaker: str = "spk-1",
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await conn.execute(
            _INSERT,
            {
                "org_id": str(org_id),
                "meeting_id": str(meeting_id),
                "chunk_id": str(chunk_id),
                "speaker_id": speaker,
                "start_ms": 0,
                "end_ms": 1000,
                "text": text_,
            },
        )


async def _delete_org(engine, org_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await conn.execute(
            text("DELETE FROM intelligence.transcript_chunk_search WHERE org_id = :o"),
            {"o": str(org_id)},
        )


def test_healthz_still_works_alongside_search_router(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "search-service"}


def test_get_search_missing_token_is_401(client):
    resp = client.get("/search", params={"q": "hello"})
    assert resp.status_code == 401


def test_get_search_malformed_token_is_401(client):
    resp = client.get(
        "/search",
        params={"q": "hello"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_search_with_another_orgs_id_in_the_header_but_no_valid_token_is_rejected(
    client, engine
):
    """The exploit this whole fix closes: `X-Org-Id` carrying a victim org's id used to be
    trusted outright, letting anyone with network reachability read another tenant's
    transcript index. A request that supplies it but presents no valid bearer token must now
    be rejected with 401 before the search query ever runs."""
    victim_org, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    try:
        await _insert_chunk(
            engine,
            org_id=victim_org,
            meeting_id=meeting_id,
            chunk_id=chunk_id,
            text_="a secret only the victim org should be able to find",
        )
        resp = client.get(
            "/search",
            params={"q": "secret"},
            headers={"X-Org-Id": str(victim_org)},  # the old, now-untrusted header
        )
        assert resp.status_code == 401
    finally:
        await _delete_org(engine, victim_org)


def test_get_search_missing_q_is_422(client):
    resp = client.get("/search", headers=auth_headers(uuid.uuid4()))
    assert resp.status_code == 422


def test_get_search_malformed_meeting_id_is_400(client):
    resp = client.get(
        "/search",
        params={"q": "hello", "meeting_id": "not-a-uuid"},
        headers=auth_headers(uuid.uuid4()),
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_search_returns_ranked_hits_shaped_for_rerank(client, engine):
    org_id, meeting_id = uuid.uuid4(), uuid.uuid4()
    chunk_low, chunk_high = uuid.uuid4(), uuid.uuid4()
    try:
        # `chunk_high` repeats the query term, so its ts_rank is strictly higher.
        await _insert_chunk(
            engine,
            org_id=org_id,
            meeting_id=meeting_id,
            chunk_id=chunk_low,
            text_="the roadmap was mentioned once",
        )
        await _insert_chunk(
            engine,
            org_id=org_id,
            meeting_id=meeting_id,
            chunk_id=chunk_high,
            text_="roadmap roadmap roadmap: the quarterly roadmap review",
        )

        resp = client.get("/search", params={"q": "roadmap"}, headers=auth_headers(org_id))
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "roadmap"
        assert len(body["results"]) == 2
        # Shaped to feed POST /rerank with zero adapter: field is `passage`, mirroring
        # RerankedPassage.passage exactly (see routes.py's module docstring).
        first = body["results"][0]
        assert set(first.keys()) == {
            "passage",
            "rank",
            "chunk_id",
            "meeting_id",
            "speaker_id",
            "start_ms",
            "end_ms",
        }
        # Rank-ordered: the chunk mentioning "roadmap" repeatedly ranks first.
        assert first["chunk_id"] == str(chunk_high)
        assert body["results"][0]["rank"] >= body["results"][1]["rank"]
    finally:
        await _delete_org(engine, org_id)
        await engine.dispose()


@pytest.mark.asyncio
async def test_meeting_id_filter_narrows_results(client, engine):
    org_id = uuid.uuid4()
    meeting_a, meeting_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()
    try:
        await _insert_chunk(
            engine,
            org_id=org_id,
            meeting_id=meeting_a,
            chunk_id=chunk_a,
            text_="synchronization plan",
        )
        await _insert_chunk(
            engine,
            org_id=org_id,
            meeting_id=meeting_b,
            chunk_id=chunk_b,
            text_="synchronization plan",
        )

        resp = client.get(
            "/search",
            params={"q": "synchronization", "meeting_id": str(meeting_a)},
            headers=auth_headers(org_id),
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["chunk_id"] == str(chunk_a)
    finally:
        await _delete_org(engine, org_id)
        await engine.dispose()


@pytest.mark.asyncio
async def test_asymmetric_candidate_set_org_with_fewer_matches_is_not_starved(client, engine):
    """The team lead's specific ask: two orgs share a distinctive phrase. Org B has FAR more
    matching rows (and higher-term-frequency, higher-ranking ones) than org A. A small `limit`
    is used, so if tenant filtering happened only on the FINAL page (global rank -> LIMIT ->
    filter by org) rather than on the candidate set, org A's true matches would be crowded out
    entirely by org B's stronger ones and org A would see zero or too-few results. Correct
    behaviour: org A still gets its own true match count, and NEVER sees an org B chunk_id."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    meeting_a, meeting_b = uuid.uuid4(), uuid.uuid4()
    shared_term = "synchronization"
    org_a_chunks = [uuid.uuid4(), uuid.uuid4()]  # only 2 matches, weak (single mention)
    # org B: 20 matches, each with the term repeated so ts_rank ranks them ABOVE org A's.
    org_b_chunks = [uuid.uuid4() for _ in range(20)]
    try:
        for cid in org_a_chunks:
            await _insert_chunk(
                engine,
                org_id=org_a,
                meeting_id=meeting_a,
                chunk_id=cid,
                text_=f"a brief mention of {shared_term} today",
            )
        for cid in org_b_chunks:
            await _insert_chunk(
                engine,
                org_id=org_b,
                meeting_id=meeting_b,
                chunk_id=cid,
                text_=f"{shared_term} {shared_term} {shared_term} dominates this whole agenda",
            )

        # A LIMIT smaller than org B's match count: if candidate-set filtering were broken, a
        # naive "rank globally then filter" query would return org B's top matches only, and
        # org A (queried separately, under its own RLS session) would get its own correct
        # answer regardless of org B's data ONLY if isolation is real per-request. The
        # meaningful assertion is what org A sees when IT queries: its own 2, never org B's 20.
        resp_a = client.get(
            "/search",
            params={"q": shared_term, "limit": 5},
            headers=auth_headers(org_a),
        )
        assert resp_a.status_code == 200
        results_a = resp_a.json()["results"]
        assert len(results_a) == 2  # org A's true match count, not starved to fewer or zero
        returned_ids_a = {r["chunk_id"] for r in results_a}
        assert returned_ids_a == {str(c) for c in org_a_chunks}
        assert returned_ids_a.isdisjoint({str(c) for c in org_b_chunks})

        resp_b = client.get(
            "/search",
            params={"q": shared_term, "limit": 5},
            headers=auth_headers(org_b),
        )
        assert resp_b.status_code == 200
        results_b = resp_b.json()["results"]
        assert len(results_b) == 5  # clipped by limit, all org B's own
        returned_ids_b = {r["chunk_id"] for r in results_b}
        assert returned_ids_b.issubset({str(c) for c in org_b_chunks})
        assert returned_ids_b.isdisjoint({str(c) for c in org_a_chunks})
    finally:
        await _delete_org(engine, org_a)
        await _delete_org(engine, org_b)
        await engine.dispose()


@pytest.mark.asyncio
async def test_rls_alone_isolates_even_without_the_routes_explicit_where(client, engine):
    """Direct-DB proof that RLS is a real, independent backstop — not merely decorative next to
    the route's own `WHERE org_id = :org_id`. Runs the identical SELECT the route issues, minus
    the explicit org filter, as `mytheus_ai_app` with only `app.current_org` set."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    meeting_a, meeting_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()
    term = "roadmapsynchronizationadversarialtest"
    try:
        await _insert_chunk(
            engine, org_id=org_a, meeting_id=meeting_a, chunk_id=chunk_a, text_=f"{term} for org a"
        )
        await _insert_chunk(
            engine, org_id=org_b, meeting_id=meeting_b, chunk_id=chunk_b, text_=f"{term} for org b"
        )

        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_a)}
            )
            # Deliberately NO "WHERE org_id = ..." — RLS alone must do the isolating here.
            rows = (
                await conn.execute(
                    text(
                        "SELECT chunk_id FROM intelligence.transcript_chunk_search "
                        "WHERE tsv @@ plainto_tsquery('english', :q)"
                    ),
                    {"q": term},
                )
            ).all()
        visible = {str(r.chunk_id) for r in rows}
        assert visible == {str(chunk_a)}
        assert str(chunk_b) not in visible
    finally:
        await _delete_org(engine, org_a)
        await _delete_org(engine, org_b)
        await engine.dispose()


@pytest.mark.asyncio
async def test_end_to_end_via_real_consumer_then_real_route(
    client, engine, redis_client, avro_codec
):
    """The full arm, proven together: a real `transcript-chunks` Avro event -> the real
    `TranscriptChunkSearchConsumer` -> `intelligence.transcript_chunk_search` -> a real
    `GET /search` call finds it. `test_search_consumer.py` proves the consumer half and
    `test_search_migration.py` proves the table half in isolation; this is the one test that
    wires them together end to end, the way `rag-service` eventually will."""
    from confluent_kafka import Producer

    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    consumer = TranscriptChunkSearchConsumer(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=Producer({"bootstrap.servers": "localhost:9092"}),
    )
    value = {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": f"chunk:{chunk_id}",
            "org_id": str(org_id),
            "schema_version": 1,
            "producer": "transcription-service",
            "occurred_at": 0,
            "correlation_id": None,
            "causation_id": None,
        },
        "meeting_id": str(meeting_id),
        "chunk_id": str(chunk_id),
        "speaker_id": "spk-1",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "the launch retrospective covered onboarding friction",
        "is_final": True,
        "confidence": 0.95,
        "redacted": False,
    }
    raw = avro_codec.serialize_for("transcript-chunks", value)
    try:
        outcome = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=1
        )
        assert outcome == "applied"

        resp = client.get(
            "/search",
            params={"q": "onboarding friction"},
            headers=auth_headers(org_id),
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["chunk_id"] == str(chunk_id)
        assert results[0]["passage"] == value["text"]
    finally:
        await _delete_org(engine, org_id)
        await engine.dispose()
