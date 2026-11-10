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
    meeting_ids: list[UUID] | None = None,
    speaker_ids: list[str] | None = None,
    match_any: bool = False,
) -> list[FtsRow]:
    """`meeting_ids` is an allowlist when given (rag-service's participant scope, mytheus-docs
    ADR-0033 D1): an empty list means the asker has no meetings and the answer is empty, never
    "unfiltered"."""
    if meeting_ids is not None and not meeting_ids:
        return []
    # `websearch_to_tsquery` ANDs every term, which suits a user typing keywords into a search
    # box. A RAG question ("how much did we decide to charge for Pro?") almost never has every
    # word in one utterance, so rag-service asks for any-term matching ranked by ts_rank: the
    # same lexemes, joined with OR (ADR-0033 D7 ablation: AND contributed nothing).
    tsq = (
        "to_tsquery('english', replace(plainto_tsquery('english', :q)::text, ' & ', ' | '))"
        if match_any
        else "websearch_to_tsquery('english', :q)"
    )
    # websearch_to_tsquery: safe for raw user input (never a syntax error), supports phrases.
    sql = f"""
        SELECT chunk_id, meeting_id, speaker_id, start_ms, end_ms, text,
               ts_rank(tsv, {tsq}) AS rank
        FROM intelligence.transcript_chunk_search
        WHERE org_id = :org_id
          AND tsv @@ {tsq}
    """
    params: dict[str, object] = {"org_id": str(org_id), "q": q, "limit": limit}
    if meeting_id is not None:
        sql += " AND meeting_id = :meeting_id"
        params["meeting_id"] = str(meeting_id)
    if meeting_ids is not None:
        sql += " AND meeting_id = ANY(CAST(:meeting_ids AS uuid[]))"
        params["meeting_ids"] = [str(m) for m in meeting_ids]
    else:
        # Org-wide (no participant allowlist): never a confidential meeting's chunk. Those are
        # reachable only through rag-service's participant scope (libs/shared/index_guard.py).
        sql += " AND NOT confidential"
    if speaker_ids:
        sql += " AND speaker_id = ANY(CAST(:speaker_ids AS text[]))"
        params["speaker_ids"] = list(speaker_ids)
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


MAX_PASSAGES = 50
MAX_NEIGHBORS = 3


@dataclass(frozen=True)
class TranscriptPassageRow:
    chunk_id: str
    meeting_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    before: str
    after: str


@dataclass(frozen=True)
class DocumentPassageRow:
    document_id: str
    element_id: str
    element_type: str
    text: str


# Small-to-big (ADR-0033 D4): the hit plus up to :n chunks either side in the same meeting.
_TRANSCRIPT_PASSAGES = """
    SELECT h.chunk_id, h.meeting_id, h.speaker_id, h.start_ms, h.end_ms, h.text,
      COALESCE((SELECT string_agg(b.text, ' ' ORDER BY b.start_ms) FROM (
          SELECT p.text, p.start_ms FROM intelligence.transcript_chunk_search p
          WHERE p.org_id = h.org_id AND p.meeting_id = h.meeting_id AND p.start_ms < h.start_ms
          ORDER BY p.start_ms DESC LIMIT :n) b), '') AS before,
      COALESCE((SELECT string_agg(a.text, ' ' ORDER BY a.start_ms) FROM (
          SELECT p.text, p.start_ms FROM intelligence.transcript_chunk_search p
          WHERE p.org_id = h.org_id AND p.meeting_id = h.meeting_id AND p.start_ms > h.start_ms
          ORDER BY p.start_ms ASC LIMIT :n) a), '') AS after
    FROM intelligence.transcript_chunk_search h
    WHERE h.org_id = :org_id AND h.chunk_id = ANY(CAST(:ids AS uuid[]))
"""


async def get_passages(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    org_id: UUID,
    chunk_ids: list[UUID],
    neighbors: int,
    documents: list[tuple[UUID, str]],
) -> tuple[list[TranscriptPassageRow], list[DocumentPassageRow]]:
    """Text for passages found by id. Same tenant isolation as `fts_search`: RLS first, plus
    an explicit `org_id` predicate. Ids of another tenant simply return nothing."""
    transcripts: list[TranscriptPassageRow] = []
    docs: list[DocumentPassageRow] = []
    async with session_factory() as session, session.begin():
        await set_rls(session, org_id)
        if chunk_ids:
            rows = (
                await session.execute(
                    text(_TRANSCRIPT_PASSAGES),
                    {"org_id": str(org_id), "ids": [str(c) for c in chunk_ids], "n": neighbors},
                )
            ).all()
            transcripts = [
                TranscriptPassageRow(
                    chunk_id=str(r.chunk_id),
                    meeting_id=str(r.meeting_id),
                    speaker_id=r.speaker_id,
                    start_ms=r.start_ms,
                    end_ms=r.end_ms,
                    text=r.text,
                    before=r.before if neighbors else "",
                    after=r.after if neighbors else "",
                )
                for r in rows
            ]
        if documents:
            rows = (
                await session.execute(
                    text(
                        "SELECT d.document_id, d.element_id, d.element_type, d.text "
                        "FROM intelligence.document_chunks d "
                        "JOIN unnest(CAST(:doc_ids AS uuid[]), CAST(:el_ids AS text[])) "
                        "  AS want(document_id, element_id) "
                        "  ON want.document_id = d.document_id AND want.element_id = d.element_id "
                        "WHERE d.org_id = :org_id"
                    ),
                    {
                        "org_id": str(org_id),
                        "doc_ids": [str(d) for d, _ in documents],
                        "el_ids": [e for _, e in documents],
                    },
                )
            ).all()
            docs = [
                DocumentPassageRow(
                    document_id=str(r.document_id),
                    element_id=r.element_id,
                    element_type=r.element_type,
                    text=r.text,
                )
                for r in rows
            ]
    return transcripts, docs
