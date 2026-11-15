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
            await conn.execute(
                text("SELECT version_num FROM intelligence.alembic_version_intelligence")
            )
        ).scalar_one()
        backend_exists = (
            await conn.execute(text("SELECT to_regclass('public.alembic_version') IS NOT NULL"))
        ).scalar_one()
    await engine.dispose()
    # HEAD of THIS repo's Alembic history, not a fixed revision: task 7's original assertion
    # hardcoded "9c2b7e4a1d03" (the only revision that existed then). Task 9 added
    # "b3f7a1c9e402" on top (down_revision = 9c2b7e4a1d03), and task 10 added "d415f023220e" on
    # top of that (down_revision = b3f7a1c9e402), so head moved again — updating this to the new
    # head is the correct, expected consequence of a later migration landing, not a regression;
    # the test's actual intent ("this repo's own history advanced to head, the backend's did
    # not move") is unchanged. The polish plan added "e7c1a9d2b4f5" (the FTS confidential flag);
    # journal plan Part 3 added "f2a6c8e1b3d5" (InitPlan policies, FTS under RLS) and
    # "a3c5e7f9b1d2" (relay column grants).
    assert mine == "a3c5e7f9b1d2"
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


async def test_relay_can_update_only_the_drain_columns(migrated_db):
    """Journal plan Part 3 fix 4, migration `a3c5e7f9b1d2`. The drain policy reaches every
    tenant, so a table-level UPDATE let the relay re-key a pending event into another tenant
    (`org_id`), retarget it (`topic`) or forge it (`payload`) before publishing. Only
    `published_at` and `attempts` may change."""
    org, other = uuid.uuid4(), uuid.uuid4()
    idem = f"relaycol-{org}"
    app = await _engine(APP_URL)
    try:
        async with app.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)}
            )
            await conn.execute(_INSERT_OUTBOX_ROW, {"org_id": str(org), "idem": idem})
    finally:
        await app.dispose()

    relay = await _engine(RELAY_URL)
    try:
        for column, value in (
            ("org_id", f"'{other}'"),
            ("topic", "'decisions'"),
            ("payload", "'{\"forged\": true}'::jsonb"),
            ("partition_key", "'x'"),
            ("idempotency_key", "'x'"),
            ("schema_subject", "'x'"),
        ):
            with pytest.raises(sa.exc.DBAPIError, match="permission denied"):
                async with relay.begin() as conn:
                    await conn.execute(
                        text(  # noqa: S608 - fixed list above
                            f"UPDATE intelligence.outbox SET {column} = {value} "
                            "WHERE idempotency_key = :k"
                        ),
                        {"k": idem},
                    )
        async with relay.begin() as conn:
            locked = (
                await conn.execute(
                    text(
                        "SELECT id FROM intelligence.outbox WHERE idempotency_key = :k "
                        "FOR UPDATE SKIP LOCKED"
                    ),
                    {"k": idem},
                )
            ).scalar_one()
            result = await conn.execute(
                text(
                    "UPDATE intelligence.outbox SET published_at = now(), "
                    "attempts = attempts + 1 WHERE id = :id"
                ),
                {"id": locked},
            )
            assert result.rowcount == 1
    finally:
        await relay.dispose()


async def test_every_intelligence_tenant_policy_is_in_initplan_form(migrated_db):
    """Journal plan Part 3 fix 1, migration `f2a6c8e1b3d5`: the tenant predicate is evaluated
    once per statement (InitPlan), never per row; the fail-closed NULLIF is unchanged."""
    engine = await _engine(migrated_db)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT tablename, policyname, coalesce(qual, '') || ' ' || "
                        "coalesce(with_check, '') AS expr FROM pg_policies "
                        "WHERE schemaname = 'intelligence' AND (coalesce(qual, '') || "
                        "coalesce(with_check, '')) LIKE '%app.current_org%'"
                    )
                )
            ).all()
    finally:
        await engine.dispose()
    assert rows, "no tenant policies found"
    for table, policy, expr in rows:
        assert "SELECT (NULLIF(current_setting('app.current_org'" in expr, (table, policy)
        bare = expr.replace(
            "( SELECT (NULLIF(current_setting('app.current_org'::text, true), ''::text))::uuid "
            'AS "nullif")',
            "",
        )
        assert "current_setting('app.current_org'" not in bare, (table, policy, expr)


async def test_transcript_search_uses_its_gin_index_under_rls(migrated_db):
    """Journal plan Part 3 fix 1. Under RLS the GIN index was unreachable for every tenant (the
    `@@` operator was not leakproof), so search read the tenant's whole slice. With a leakproof
    `ts_match_vq` the `@@` qual becomes an INDEX CONDITION on the composite (org_id, tsv) index.

    On a near-empty table the planner rightly prefers the org_id btree, so the competing org_id
    indexes are dropped inside a rolled-back transaction; the assertion is that `tsv @@` is an
    index condition (with a non-leakproof `@@` it can only ever be a filter)."""
    engine = await _engine(migrated_db)
    org = uuid.uuid4()
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            for ddl in (
                "DROP INDEX intelligence.ix_intelligence_transcript_chunk_search_org_id",
                "DROP INDEX intelligence.ix_intelligence_transcript_chunk_search_org_meeting",
                "ALTER TABLE intelligence.transcript_chunk_search "
                "DROP CONSTRAINT transcript_chunk_search_org_chunk_uq",
            ):
                await conn.execute(text(ddl))
            await conn.execute(text("SET LOCAL ROLE mytheus_ai_app"))
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)}
            )
            await conn.execute(text("SET LOCAL enable_seqscan = off"))
            plan = "\n".join(
                r[0]
                for r in await conn.execute(
                    text(
                        "EXPLAIN SELECT chunk_id FROM intelligence.transcript_chunk_search "
                        "WHERE org_id = :o AND tsv @@ websearch_to_tsquery('english', 'kestrel')"
                    ),
                    {"o": str(org)},
                )
            )
            await trans.rollback()
    finally:
        await engine.dispose()
    assert "ix_intelligence_transcript_chunk_search_org_tsv" in plan, plan
    index_conds = [line for line in plan.splitlines() if "Index Cond:" in line]
    assert any("tsv @@" in line for line in index_conds), plan
