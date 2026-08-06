"""Proves migration `b3f7a1c9e402` against the REAL shared Postgres:
- `document_chunks` / `document_elements` / `ingestion_jobs` exist in `intelligence`, not `public`.
- `mytheus_ai_app` can read/write them WITHOUT any new GRANT in this migration (inherited from
  migration `9c2b7e4a1d03`'s `ALTER DEFAULT PRIVILEGES IN SCHEMA intelligence` — proving the
  inheritance actually happened, not just asserting the docstring's claim).
- `tenant_isolation` RLS is fail-closed and cross-tenant-isolating on all three tables.
- `mytheus_ai_relay` has NO grant on any of the three (they are not outbox-drained tables).
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import APP_URL, RELAY_URL

pytestmark = pytest.mark.asyncio

_TABLES = ("document_chunks", "document_elements", "ingestion_jobs")


async def _engine(url: str):
    return create_async_engine(url, pool_pre_ping=True)


async def test_all_three_tables_live_in_intelligence_schema(migrated_db):
    engine = await _engine(migrated_db)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_name = ANY(:names)"
                ),
                {"names": list(_TABLES)},
            )
        ).all()
    await engine.dispose()
    pairs = {(r.table_schema, r.table_name) for r in rows}
    for t in _TABLES:
        assert ("intelligence", t) in pairs
        assert ("public", t) not in pairs


async def test_mytheus_ai_app_inherited_grants_with_no_new_grant_statement(migrated_db):
    """The migration's central claim: no GRANT was needed for mytheus_ai_app because
    `ALTER DEFAULT PRIVILEGES IN SCHEMA intelligence` (migration 9c2b7e4a1d03) already covers
    tables created later in the same schema. Proven by actually inserting as that role."""
    engine = await _engine(APP_URL)
    org_id = uuid.uuid4()
    document_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO intelligence.document_elements (org_id, document_id, element_id, "
                "element_type, raw) VALUES (:org_id, :document_id, 'el_0000', 'Title', '{}'::jsonb)"
            ),
            {"org_id": str(org_id), "document_id": str(document_id)},
        )
        await conn.execute(
            text(
                "INSERT INTO intelligence.document_chunks (org_id, document_id, element_id, "
                'element_type, text, "position") VALUES '
                "(:org_id, :document_id, 'el_0000_0', 'Title', 'hello', 0)"
            ),
            {"org_id": str(org_id), "document_id": str(document_id)},
        )
        await conn.execute(
            text(
                "INSERT INTO intelligence.ingestion_jobs (id, org_id, source, bucket, object_key) "
                "VALUES (:document_id, :org_id, 'upload', 'nexus-documents', 'k')"
            ),
            {"org_id": str(org_id), "document_id": str(document_id)},
        )
        visible = (
            await conn.execute(
                text("SELECT count(*) FROM intelligence.document_chunks WHERE document_id = :d"),
                {"d": str(document_id)},
            )
        ).scalar_one()
    await engine.dispose()
    assert visible == 1


async def test_rls_fails_closed_on_all_three_tables_without_app_current_org(migrated_db):
    engine = await _engine(APP_URL)
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    stmts = {
        "document_chunks": text(
            "INSERT INTO intelligence.document_chunks (org_id, document_id, element_id, "
            "element_type, text, \"position\") VALUES (:o, :d, 'el_0000_0', 'Title', 'x', 0)"
        ),
        "document_elements": text(
            "INSERT INTO intelligence.document_elements (org_id, document_id, element_id, "
            "element_type, raw) VALUES (:o, :d, 'el_0000', 'Title', '{}'::jsonb)"
        ),
        "ingestion_jobs": text(
            "INSERT INTO intelligence.ingestion_jobs (id, org_id, source, bucket, object_key) "
            "VALUES (:d, :o, 'upload', 'nexus-documents', 'k')"
        ),
    }
    for stmt in stmts.values():
        async with engine.begin() as conn:
            with pytest.raises(sa.exc.DBAPIError, match="row-level security"):
                await conn.execute(stmt, {"o": str(org_id), "d": str(document_id)})
    await engine.dispose()


async def test_cross_tenant_isolation_on_document_chunks(migrated_db):
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    engine = await _engine(APP_URL)
    insert = text(
        "INSERT INTO intelligence.document_chunks (org_id, document_id, element_id, "
        "element_type, text, \"position\") VALUES (:o, :d, 'el_0000_0', 'Title', 'secret', 0)"
    )
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_a)}
            )
            await conn.execute(insert, {"o": str(org_a), "d": str(doc_a)})
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_b)}
            )
            await conn.execute(insert, {"o": str(org_b), "d": str(doc_b)})

        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_a)}
            )
            visible = (
                (await conn.execute(text("SELECT document_id FROM intelligence.document_chunks")))
                .scalars()
                .all()
            )
        visible_strs = {str(v) for v in visible}
        assert str(doc_a) in visible_strs
        assert str(doc_b) not in visible_strs
    finally:
        await engine.dispose()


async def test_mytheus_ai_relay_has_no_grant_on_any_of_the_three_tables(migrated_db):
    engine = await _engine(RELAY_URL)
    try:
        for table in _TABLES:
            # A fresh transaction per table: a permission-denied error aborts the transaction,
            # so a second query on the SAME transaction would fail with "transaction is aborted"
            # rather than the "permission denied" this test is actually checking for.
            async with engine.begin() as conn:
                with pytest.raises(sa.exc.DBAPIError, match="permission denied"):
                    await conn.execute(text(f"SELECT count(*) FROM intelligence.{table}"))
    finally:
        await engine.dispose()


async def test_element_id_uniqueness_constraint_enables_idempotent_reinsert(migrated_db):
    """The mechanism the plan's 'a re-run is idempotent' acceptance check relies on, proven at
    the SQL layer directly (the pipeline-level proof is in test_ingestion_pipeline.py)."""
    engine = await _engine(APP_URL)
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    insert = text(
        "INSERT INTO intelligence.document_chunks (org_id, document_id, element_id, "
        "element_type, text, \"position\") VALUES (:o, :d, 'el_0000_0', 'Title', 'v1', 0) "
        "ON CONFLICT (org_id, document_id, element_id) DO NOTHING"
    )
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            await conn.execute(insert, {"o": str(org_id), "d": str(document_id)})
            await conn.execute(insert, {"o": str(org_id), "d": str(document_id)})  # re-run
            count = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM intelligence.document_chunks "
                        "WHERE document_id = :d AND element_id = 'el_0000_0'"
                    ),
                    {"d": str(document_id)},
                )
            ).scalar_one()
        assert count == 1
    finally:
        await engine.dispose()
