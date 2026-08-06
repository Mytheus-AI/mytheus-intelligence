"""Proves migration `d415f023220e` against the REAL shared Postgres:
- `intelligence.transcript_chunk_search` exists, not `public.transcript_chunks` (this repo's own
  read model, never the table `transcription-service` owns — see `consumer.py`'s docstring).
- `mytheus_ai_app` can read/write it WITHOUT any new GRANT in this migration (inherited from
  migration `9c2b7e4a1d03`'s `ALTER DEFAULT PRIVILEGES IN SCHEMA intelligence` — proving the
  inheritance actually happened, matching task 9's own discipline for its tables).
- `tenant_isolation` RLS is fail-closed and cross-tenant-isolating.
- `mytheus_ai_relay` has no grant on this table (it is not outbox-drained).
- The GIN-indexed generated `tsv` column actually auto-populates and matches, at the SQL layer,
  independent of the HTTP route (`test_search_routes.py` proves the route on top of this).
- `UNIQUE (org_id, chunk_id)` is the mechanism `consumer.py`'s `ON CONFLICT DO NOTHING` idempotent
  redelivery relies on.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import APP_URL, RELAY_URL

pytestmark = pytest.mark.asyncio

_INSERT = text("""
    INSERT INTO intelligence.transcript_chunk_search
        (org_id, meeting_id, chunk_id, speaker_id, start_ms, end_ms, text)
    VALUES (:org_id, :meeting_id, :chunk_id, 'spk-1', 0, 1000, :text)
    """)


async def _engine(url: str):
    return create_async_engine(url, pool_pre_ping=True)


async def _set_org(conn, org_id: uuid.UUID) -> None:
    await conn.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)})


async def test_table_lives_in_intelligence_schema_not_public(migrated_db):
    engine = await _engine(migrated_db)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT table_schema FROM information_schema.tables "
                    "WHERE table_name = 'transcript_chunk_search'"
                )
            )
        ).all()
    await engine.dispose()
    schemas = {r.table_schema for r in rows}
    assert schemas == {"intelligence"}


async def test_mytheus_ai_app_inherited_grants_with_no_new_grant_statement(migrated_db):
    engine = await _engine(APP_URL)
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await _set_org(conn, org_id)
            await conn.execute(
                _INSERT,
                {
                    "org_id": str(org_id),
                    "meeting_id": str(meeting_id),
                    "chunk_id": str(chunk_id),
                    "text": "hello world",
                },
            )
            visible = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM intelligence.transcript_chunk_search "
                        "WHERE chunk_id = :c"
                    ),
                    {"c": str(chunk_id)},
                )
            ).scalar_one()
        assert visible == 1
    finally:
        async with engine.begin() as conn:
            await _set_org(conn, org_id)
            await conn.execute(
                text("DELETE FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
                {"c": str(chunk_id)},
            )
        await engine.dispose()


async def test_rls_fails_closed_without_app_current_org(migrated_db):
    engine = await _engine(APP_URL)
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        with pytest.raises(sa.exc.DBAPIError, match="row-level security"):
            await conn.execute(
                _INSERT,
                {
                    "org_id": str(org_id),
                    "meeting_id": str(meeting_id),
                    "chunk_id": str(chunk_id),
                    "text": "no org set",
                },
            )
    await engine.dispose()


async def test_cross_tenant_isolation(migrated_db):
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    meeting_a, meeting_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()
    engine = await _engine(APP_URL)
    try:
        async with engine.begin() as conn:
            await _set_org(conn, org_a)
            await conn.execute(
                _INSERT,
                {
                    "org_id": str(org_a),
                    "meeting_id": str(meeting_a),
                    "chunk_id": str(chunk_a),
                    "text": "org a secret",
                },
            )
        async with engine.begin() as conn:
            await _set_org(conn, org_b)
            await conn.execute(
                _INSERT,
                {
                    "org_id": str(org_b),
                    "meeting_id": str(meeting_b),
                    "chunk_id": str(chunk_b),
                    "text": "org b secret",
                },
            )
        async with engine.begin() as conn:
            await _set_org(conn, org_a)
            visible = (
                (
                    await conn.execute(
                        text("SELECT chunk_id FROM intelligence.transcript_chunk_search")
                    )
                )
                .scalars()
                .all()
            )
        visible_strs = {str(v) for v in visible}
        assert str(chunk_a) in visible_strs
        assert str(chunk_b) not in visible_strs
    finally:
        for org, chunk in ((org_a, chunk_a), (org_b, chunk_b)):
            async with engine.begin() as conn:
                await _set_org(conn, org)
                await conn.execute(
                    text("DELETE FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
                    {"c": str(chunk)},
                )
        await engine.dispose()


async def test_mytheus_ai_relay_has_no_grant(migrated_db):
    engine = await _engine(RELAY_URL)
    try:
        with pytest.raises(sa.exc.DBAPIError, match="permission denied"):
            async with engine.begin() as conn:
                await conn.execute(
                    text("SELECT count(*) FROM intelligence.transcript_chunk_search")
                )
    finally:
        await engine.dispose()


async def test_org_chunk_uniqueness_enables_idempotent_reinsert(migrated_db):
    engine = await _engine(APP_URL)
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    insert_on_conflict = text("""
        INSERT INTO intelligence.transcript_chunk_search
            (org_id, meeting_id, chunk_id, speaker_id, start_ms, end_ms, text)
        VALUES (:org_id, :meeting_id, :chunk_id, 'spk-1', 0, 1000, 'v1')
        ON CONFLICT (org_id, chunk_id) DO NOTHING
        """)
    try:
        async with engine.begin() as conn:
            await _set_org(conn, org_id)
            params = {
                "org_id": str(org_id),
                "meeting_id": str(meeting_id),
                "chunk_id": str(chunk_id),
            }
            await conn.execute(insert_on_conflict, params)
            await conn.execute(insert_on_conflict, params)  # redelivery
            count = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM intelligence.transcript_chunk_search "
                        "WHERE chunk_id = :c"
                    ),
                    {"c": str(chunk_id)},
                )
            ).scalar_one()
        assert count == 1
    finally:
        async with engine.begin() as conn:
            await _set_org(conn, org_id)
            await conn.execute(
                text("DELETE FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
                {"c": str(chunk_id)},
            )
        await engine.dispose()


async def test_generated_tsv_column_and_gin_index_are_real(migrated_db):
    """The plan's literal ask (`GENERATED ALWAYS AS (to_tsvector(...)) STORED` + a GIN index),
    proven at the SQL layer: the column auto-populates on insert (no application code writes
    `tsv` directly — `consumer.py` never mentions it) and a `tsv @@ plainto_tsquery(...)` match
    actually finds the row."""
    engine = await _engine(APP_URL)
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    try:
        async with engine.begin() as conn:
            index_rows = (
                await conn.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'intelligence' "
                        "AND tablename = 'transcript_chunk_search' "
                        "AND indexname = 'ix_intelligence_transcript_chunk_search_tsv'"
                    )
                )
            ).all()
            assert len(index_rows) == 1
            assert "gin" in index_rows[0].indexdef.lower()

            await _set_org(conn, org_id)
            await conn.execute(
                _INSERT,
                {
                    "org_id": str(org_id),
                    "meeting_id": str(meeting_id),
                    "chunk_id": str(chunk_id),
                    "text": "the quarterly roadmap synchronization meeting",
                },
            )
            matched = (
                await conn.execute(
                    text(
                        "SELECT tsv IS NOT NULL AS has_tsv, "
                        "tsv @@ plainto_tsquery('english', 'roadmap synchronization') AS matches "
                        "FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"
                    ),
                    {"c": str(chunk_id)},
                )
            ).one()
            assert matched.has_tsv is True
            assert matched.matches is True
    finally:
        async with engine.begin() as conn:
            await _set_org(conn, org_id)
            await conn.execute(
                text("DELETE FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
                {"c": str(chunk_id)},
            )
        await engine.dispose()
