"""Alembic environment for `mytheus-intelligence`'s OWN migration history.

This repo had NO outbox/inbox Alembic migrations at all before this task (routed here per the
task brief: "you are the first task to touch this repo, so this is yours to establish"). It is
a SEPARATE Alembic history from `mytheus-backend/migrations/` — different repo, different
`script_location`, no shared revision graph — that happens to point at the same physical
Postgres database (`mytheus`, `localhost:5443` in dev).

Two collision risks that a second history against the same database creates, both handled:

1. Version-tracking table collision. `mytheus-backend` uses the default `alembic_version`.
   This history renames its own via `version_table` below, following the EXACT precedent set
   by `mytheus-realtime`'s `transcription-service/migrations/env.py`
   (`alembic_version_transcription_service`) for a second Alembic history against this same
   database — reviewed and endorsed there, reused verbatim here rather than inventing a
   second dialect.

2. Table-name collision. `mytheus-backend` already owns `public.outbox` / `public.inbox`.
   Rather than colliding on those names, every table this repo owns lives in its own Postgres
   schema, `intelligence` (see `libs/shared/db.py` for the full rationale). This env.py creates
   that schema in migration 0001 before anything else.

Runs as the SCHEMA OWNER (`BOOTSTRAP_DATABASE_URL`, falling back to `DATABASE_URL`), never as
the least-privilege `mytheus_ai_app` role the application uses — that role deliberately holds
no DDL/CREATE ROLE privilege (mirrors `mytheus-backend`'s `mytheus_app` split).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Raw SQL migrations (schema/role/RLS DDL is not autogenerate-friendly); no ORM metadata here.
target_metadata = None

VERSION_TABLE = "alembic_version_intelligence"


def _database_url() -> str:
    url = (
        os.environ.get("BOOTSTRAP_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql+asyncpg://postgres:postgres@localhost:5443/mytheus"
    )
    return url.replace("+asyncpg", "")


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


VERSION_SCHEMA = "intelligence"


def _prepare_version_table(connection: Connection) -> None:
    """Keep this history's version table out of `public`.

    It is created by the schema-owner role (`supabase_admin`), and a `public` table owned by that
    role makes its image-installed default ACL live, which mytheus-backend's grant assertions
    correctly reject. An existing `public` table is moved, not recreated, so history survives.
    """
    connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {VERSION_SCHEMA}"))
    connection.execute(text(f"""
            DO $$
            BEGIN
              IF to_regclass('public.{VERSION_TABLE}') IS NOT NULL
                 AND to_regclass('{VERSION_SCHEMA}.{VERSION_TABLE}') IS NULL THEN
                ALTER TABLE public.{VERSION_TABLE} SET SCHEMA {VERSION_SCHEMA};
              END IF;
            END $$;
            """))
    connection.commit()


def do_run_migrations(connection: Connection) -> None:
    _prepare_version_table(connection)
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        version_table_schema=VERSION_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online_sync() -> None:
    from sqlalchemy import create_engine

    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        do_run_migrations(connection)
    connectable.dispose()


async def run_migrations_online_async() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = "postgresql+asyncpg://" + _database_url().split("://", 1)[1]
    connectable = async_engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # Sync engine is simpler and avoids an event-loop-in-event-loop hazard when `command.upgrade`
    # is invoked from within an already-running asyncio loop (test fixtures). psycopg2 is
    # already a dependency for exactly this reason.
    run_migrations_online_sync()
