"""Postgres session factory + this repo's schema constant.

`mytheus-intelligence` writes to the SAME physical Postgres as `mytheus-backend` (Plan 5-6
task 7 scope note). `mytheus-backend` already owns `public.outbox` / `public.inbox`, so this
repo's outbox/inbox tables live in their own Postgres schema, `intelligence`, rather than
colliding with those names in `public`. This is a deliberate bounded-context boundary, not
just a naming dodge: `ALTER DEFAULT PRIVILEGES IN SCHEMA public` (`mytheus-backend` migration
`a1c4e7f20b18`) is schema-scoped, so a table of ours living in `public` would silently inherit
DML grants for `mytheus_app`/`mytheus_admin` — roles that belong to a different repo's request
path. A dedicated schema means our grants are 100% explicit (`migrations/versions/
0001_intelligence_schema_outbox_inbox_roles.py`) and nothing is inherited by accident. This is
the convention established here for Plan 5-6 tasks 8-10 (`ingestion-service`,
`search-service`): any table this repo owns lives in the `intelligence` schema, never `public`.

Alembic version table. This repo's Alembic history is independent of `mytheus-backend`'s, so
its version table is renamed to `alembic_version_intelligence` (see `migrations/env.py` and
`alembic.ini`), following the exact precedent set by `mytheus-realtime`'s
`transcription-service` (`alembic_version_transcription_service`) for a second Alembic history
against the same physical database.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

SCHEMA = "intelligence"


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Expected postgresql+asyncpg://mytheus_ai_app:...@host/mytheus "
            "(the least-privilege request-path role created by migration "
            "0001_intelligence_schema_outbox_inbox_roles)."
        )
    return url


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    return create_async_engine(database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)
