"""search-service's scoped FTS and passage lookup against real Postgres (ADR-0033 D1, D4).

`fts_search` with a meeting allowlist and a speaker filter, and `get_passages` with neighbours,
both as `mytheus_ai_app` under RLS. Skips without a database unless `MYTHEUS_REQUIRE_DB=1`.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from services.search_service.fts import fts_search, get_passages
from tests.conftest import APP_URL

_INSERT = text("""
    INSERT INTO intelligence.transcript_chunk_search
        (org_id, meeting_id, chunk_id, speaker_id, start_ms, end_ms, text)
    VALUES (:org_id, :meeting_id, :chunk_id, :speaker_id, :start_ms, :end_ms, :text)
    """)


@pytest.fixture()
def factory(migrated_db):
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _chunk(engine, org, meeting, start_ms, body, speaker="u1") -> uuid.UUID:
    cid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)})
        await conn.execute(
            _INSERT,
            {
                "org_id": str(org),
                "meeting_id": str(meeting),
                "chunk_id": str(cid),
                "speaker_id": speaker,
                "start_ms": start_ms,
                "end_ms": start_ms + 1000,
                "text": body,
            },
        )
    return cid


async def test_allowlist_and_speaker_filter(factory):
    engine, sf = factory
    org, mine, theirs = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _chunk(engine, org, mine, 0, "pricing tier locked at twenty five", "priya")
    await _chunk(engine, org, mine, 2000, "pricing for enterprise later", "sam")
    await _chunk(engine, org, theirs, 0, "pricing secret from a private meeting", "sam")

    allowed = await fts_search(
        session_factory=sf, org_id=org, q="pricing", meeting_id=None, limit=10, meeting_ids=[mine]
    )
    assert {r.meeting_id for r in allowed} == {str(mine)}
    assert len(allowed) == 2

    by_priya = await fts_search(
        session_factory=sf,
        org_id=org,
        q="pricing",
        meeting_id=None,
        limit=10,
        meeting_ids=[mine],
        speaker_ids=["priya"],
    )
    assert [r.speaker_id for r in by_priya] == ["priya"]

    none = await fts_search(
        session_factory=sf, org_id=org, q="pricing", meeting_id=None, limit=10, meeting_ids=[]
    )
    assert none == []
    await engine.dispose()


async def test_passages_carry_neighbours_from_the_same_meeting_only(factory):
    engine, sf = factory
    org, meeting, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _chunk(engine, org, meeting, 0, "Should we ship on the second?")
    hit = await _chunk(engine, org, meeting, 1000, "Yes, let's do that.")
    await _chunk(engine, org, meeting, 2000, "Great, I will tell sales.")
    await _chunk(engine, org, meeting, 3000, "Unrelated later remark.")
    await _chunk(engine, org, other, 500, "Another meeting entirely.")

    transcripts, docs = await get_passages(
        session_factory=sf, org_id=org, chunk_ids=[hit], neighbors=1, documents=[]
    )
    assert docs == []
    [p] = transcripts
    assert p.text == "Yes, let's do that."
    assert p.before == "Should we ship on the second?"
    assert p.after == "Great, I will tell sales."

    # Another tenant asking for the same chunk id gets nothing.
    foreign, _ = await get_passages(
        session_factory=sf, org_id=uuid.uuid4(), chunk_ids=[hit], neighbors=1, documents=[]
    )
    assert foreign == []
    await engine.dispose()


async def test_document_passages_by_element(factory):
    engine, sf = factory
    org, doc = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)})
        await conn.execute(
            text(
                "INSERT INTO intelligence.document_chunks "
                "(org_id, document_id, element_id, element_type, text, position) "
                "VALUES (:o, :d, 'el_1', 'NarrativeText', 'Tax goes to Stripe Tax.', 1)"
            ),
            {"o": str(org), "d": str(doc)},
        )
    _, docs = await get_passages(
        session_factory=sf, org_id=org, chunk_ids=[], neighbors=0, documents=[(doc, "el_1")]
    )
    assert [(d.element_id, d.text) for d in docs] == [("el_1", "Tax goes to Stripe Tax.")]
    await engine.dispose()
