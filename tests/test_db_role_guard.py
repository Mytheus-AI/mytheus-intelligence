"""Journal plan Part 3 fix 5: the process-wide pool guard (`libs/shared/db_guard.py`) refuses a
role that makes row-level security inert. The rest of this suite runs with the guard off (its
fixtures connect as the schema owner on purpose); every test here turns it back on."""

from __future__ import annotations

import os
import uuid

import psycopg2
import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import Pool

import libs.shared.tenancy  # noqa: F401 - importing it installs the guard
from libs.shared import db_guard
from libs.shared.db_guard import PrivilegedRoleError, verify_role, violations
from tests.conftest import APP_URL, BOOTSTRAP_URL, RELAY_URL

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(db_guard.GUARD_ENV, "enforce")


async def test_importing_the_db_modules_installs_the_guard_for_every_pool() -> None:
    assert event.contains(Pool, "connect", db_guard._check)


async def test_violations_names_every_problem() -> None:
    assert violations([("mytheus_ai_app", False, False, [])]) == []
    assert violations([("supabase_admin", True, True, [])]) == [
        "supabase_admin is a superuser",
        "supabase_admin holds BYPASSRLS",
    ]


@pytest.mark.parametrize("url", [APP_URL, RELAY_URL], ids=["mytheus_ai_app", "mytheus_ai_relay"])
async def test_guard_admits_the_least_privilege_roles(migrated_db, url) -> None:
    engine = create_async_engine(url)
    try:
        await verify_role(engine)
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        await engine.dispose()


async def test_guard_refuses_the_schema_owner(migrated_db) -> None:
    engine = create_async_engine(BOOTSTRAP_URL)
    try:
        with pytest.raises(PrivilegedRoleError, match="refusing to use this database"):
            await verify_role(engine)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("grant", ["BYPASSRLS", "pg_read_all_data", "pg_write_all_data"])
async def test_guard_refuses_each_way_of_bypassing_isolation(migrated_db, grant) -> None:
    su_url = os.environ.get("SUPERUSER_DATABASE_URL") or BOOTSTRAP_URL
    conn = psycopg2.connect(su_url.replace("+asyncpg", ""))
    conn.autocommit = True
    role = f"guard_probe_{uuid.uuid4().hex[:8]}"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
            if not cur.fetchone()[0]:
                pytest.skip("needs SUPERUSER_DATABASE_URL to create a privileged probe role")
            cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'probe'")
            if grant == "BYPASSRLS":
                cur.execute(f"ALTER ROLE {role} BYPASSRLS")
            else:
                cur.execute(f"GRANT {grant} TO {role}")
        scheme, _, rest = BOOTSTRAP_URL.partition("://")
        engine = create_async_engine(f"{scheme}://{role}:probe@{rest.rpartition('@')[2]}")
        try:
            with pytest.raises(PrivilegedRoleError):
                await verify_role(engine)
        finally:
            await engine.dispose()
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP ROLE IF EXISTS {role}")
        conn.close()
