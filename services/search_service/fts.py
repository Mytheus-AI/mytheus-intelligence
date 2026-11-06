"""The one FTS query, shared by the REST route (`routes.py`) and the internal gRPC arm
(`grpc_server.py`). Tenant isolation (RLS set BEFORE the SELECT, plus an explicit
`WHERE org_id`) is documented in `routes.py`'s module docstring and lives here so both
transports get it identically."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from libs.shared.tenancy import set_rls

MAX_LIMIT = 50


@dataclass(frozen=True)
class FtsRow:
    passage: str
    rank: float
    chunk_id: str
    meeting_id: str
    speaker_id: str
    start_ms: int
    end_ms: int


async def fts_search(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    org_id: UUID,
    q: str,
    meeting_id: UUID | None,
    limit: int,
) -> list[FtsRow]:
    # websearch_to_tsquery: safe for raw user input (never a syntax error), supports phrases.
    sql = """
        SELECT chunk_id, meeting_id, speaker_id, start_ms, end_ms, text,
               ts_rank(tsv, websearch_to_tsquery('english', :q)) AS rank
        FROM intelligence.transcript_chunk_search
        WHERE org_id = :org_id
          AND tsv @@ websearch_to_tsquery('english', :q)
    """
    params: dict[str, object] = {"org_id": str(org_id), "q": q, "limit": limit}
    if meeting_id is not None:
        sql += " AND meeting_id = :meeting_id"
        params["meeting_id"] = str(meeting_id)
    sql += " ORDER BY rank DESC LIMIT :limit"

    async with session_factory() as session, session.begin():
        await set_rls(session, org_id)
        rows = (await session.execute(text(sql), params)).all()
    return [
        FtsRow(
            passage=r.text,
            rank=float(r.rank),
            chunk_id=str(r.chunk_id),
            meeting_id=str(r.meeting_id),
            speaker_id=r.speaker_id,
            start_ms=r.start_ms,
            end_ms=r.end_ms,
        )
        for r in rows
    ]
