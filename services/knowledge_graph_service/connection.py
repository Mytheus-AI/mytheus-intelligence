"""asyncpg connection factory for the AGE Postgres, prepared for Cypher on every open."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import asyncpg

from libs.shared.age import AgeConnection, age_database_url, prepare_session


@asynccontextmanager
async def age_connection(dsn: str | None = None) -> AsyncIterator[AgeConnection]:
    raw = await asyncpg.connect(dsn or age_database_url())
    conn = cast(AgeConnection, raw)
    try:
        await prepare_session(conn)
        yield conn
    finally:
        await raw.close()
