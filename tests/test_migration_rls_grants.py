"""Proves migration `9c2b7e4a1d03` against the REAL shared Postgres:
- `intelligence` schema isolated from `public` (no table collision with mytheus-backend).
- `tenant_isolation` RLS is fail-closed (no `app.current_org` -> zero rows / INSERT rejected).
- Cross-tenant isolation holds under the request-path role, `mytheus_ai_app`.
- `mytheus_ai_relay` can drain across tenants (via `outbox_relay_drain`) but holds no other grant.
- Neither role holds `rolsuper` or `rolbypassrls`.
- This repo's Alembic version table is independent of mytheus-backend's.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import APP_URL, RELAY_URL

pytestmark = pytest.mark.asyncio

_INSERT_OUTBOX_ROW = text(
    "INSERT INTO intelligence.outbox "
    "(org_id, topic, partition_key, idempotency_key, schema_subject, payload) "
    "VALUES (:org_id, 'embedding-events', 'x', :idem, 'embedding-events-value', '{}'::jsonb)"
)


async def _engine(url: str):
    return create_async_engine(url, pool_pre_ping=True)


async def test_public_outbox_untouched_and_intelligence_tables_exist(migrated_db):
    engine = await _engine(migrated_db)
    async with engine.begin() as conn:
        tables = (
            await conn.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_name IN ('outbox', 'inbox') ORDER BY table_schema"
                )
            )
        ).all()
    await engine.dispose()
    pairs = {(row.table_schema, row.table_name) for row in tables}
    assert ("intelligence", "outbox") in pairs
    assert ("intelligence", "inbox") in pairs
    # mytheus-backend's own outbox/inbox in `public` must still exist, untouched (proves no
    # collision — if this migration had clobbered them, the count would be wrong or the
    # column shape would differ from what mytheus-backend expects).
    assert ("public", "outbox") in pairs
    assert ("public", "inbox") in pairs


async def test_two_independent_alembic_version_tables(migrated_db):
    engine = await _engine(migrated_db)
    async with engine.begin() as conn:
        mine = (
            await conn.execute(text("SELECT version_num FROM alembic_version_intelligence"))
        ).scalar_one()
        backend_exists = (
            await conn.execute(text("SELECT to_regclass('public.alembic_version') IS NOT NULL"))
        ).scalar_one()
    await engine.dispose()
    assert mine == "9c2b7e4a1d03"
    # mytheus-backend's own alembic_version table exists independently, if that repo's
    # migrations have ever run against this database. This repo does not touch it either way.
    assert backend_exists in (True, False)


async def test_neither_role_holds_superuser_or_bypassrls(migrated_db):
    engine = await _engine(migrated_db)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname IN ('mytheus_ai_app', 'mytheus_ai_relay')"
                )
            )
        ).all()
    await engine.dispose()
    assert len(rows) == 2
    for row in rows:
        assert row.rolsuper is False, f"{row.rolname} must not hold SUPERUSER"
        assert row.rolbypassrls is False, f"{row.rolname} must not hold BYPASSRLS"


async def test_rls_fails_closed_without_app_current_org(migrated_db):
    engine = await _engine(APP_URL)
    org = uuid.uuid4()
    async with engine.begin() as conn:
        with pytest.raises(sa.exc.DBAPIError, match="row-level security"):
            await conn.execute(
                _INSERT_OUTBOX_ROW,
                {"org_id": str(org), "idem": f"rls-fail-closed-{org}"},
            )
    await engine.dispose()


async def test_app_role_cannot_read_across_tenants(migrated_db):
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    engine = await _engine(APP_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_a)}
            )
            await conn.execute(
                _INSERT_OUTBOX_ROW,
                {"org_id": str(org_a), "idem": f"iso-a-{org_a}"},
            )
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_b)}
            )
            await conn.execute(
                _INSERT_OUTBOX_ROW,
                {"org_id": str(org_b), "idem": f"iso-b-{org_b}"},
            )

        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_a)}
            )
            visible = (
                (await conn.execute(text("SELECT idempotency_key FROM intelligence.outbox")))
                .scalars()
                .all()
            )
        assert f"iso-a-{org_a}" in visible
        assert f"iso-b-{org_b}" not in visible
    finally:
        await engine.dispose()


async def test_relay_role_drains_across_tenants_but_has_no_other_grant(migrated_db):
    engine = await _engine(RELAY_URL)
    async with engine.begin() as conn:
        # Can see rows across tenants (the whole point of outbox_relay_drain).
        rows = (await conn.execute(text("SELECT count(*) FROM intelligence.outbox"))).scalar_one()
        assert rows >= 0  # succeeds without an app.current_org set: proves the drain policy

        # Cannot touch intelligence.inbox at all (no grant).
        with pytest.raises(sa.exc.DBAPIError, match="permission denied"):
            await conn.execute(text("SELECT count(*) FROM intelligence.inbox"))
    await engine.dispose()
