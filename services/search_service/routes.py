"""`GET /search` — plan 3.10 step 3's user-facing surface ("in-page transcript search") and, for
now, the same REST contract the internal gRPC FTS arm for `rag-service` would use once
`proto/rag.proto` grows past its `RagQueryPlaceholder` (identical REST-now/gRPC-later reasoning
tasks 7-9 already established for this repo's other "internal gRPC only" contracts — see e.g.
`reranker_service/rerank.py`'s module docstring).

`X-Org-Id` header, fail-closed `UUID` parse — the same interim REST-authn convention
`embedding-service`'s `query_embed.py`, `reranker-service`'s `rerank.py`, and
`ingestion-service`'s `routes.py` already established for this repo (no JWT/gRPC-metadata authn
middleware exists anywhere in this tier yet).

--- Response shape: built to feed `POST /rerank` with zero adapter ---

Each hit's text field is named `passage`, mirroring `reranker-service`'s
`RerankedPassage.passage` field name exactly (`services/reranker_service/rerank.py`), so a caller
can do `passages=[h.passage for h in response.results]` and pass that straight into a
`RerankRequest` — no renaming, no reshaping. The remaining fields (`chunk_id`, `meeting_id`,
`speaker_id`, `start_ms`, `end_ms`) are NOT part of `/rerank`'s own request or response (which
only ever carries bare passage text in and scored passage text out) — they ride along so a caller
can re-bind a reranked passage back to its provenance for the RAG pipeline's `cite` stage
(`system.md` section 5: "every passage binds provenance... transcript -> meeting_id / chunk_id /
speaker"). Re-binding a `/rerank` result back to one of these hits (by passage text, since
`/rerank` does not echo an id) is a `rag-service` concern for a later task in this program, not
solved here — flagged rather than reached past.

--- Tenant isolation: RLS constrains the candidate set, not just the final page ---

`set_rls` (transaction-local `app.current_org`) is applied BEFORE the `SELECT`, so a tenant's row
set is restricted at the query-planning level, before `ORDER BY rank LIMIT` ever clips it — a
weaker design that fetched globally-ranked rows first and filtered by `org_id` afterward could
silently starve a real match (a tenant with true, correct matches could get fewer than `limit`
results, or none, purely because another tenant's stronger matches for the same shared vocabulary
consumed the `LIMIT` first) even without ever returning a foreign row. An explicit
`WHERE org_id = :org_id` is ALSO present, as a second, independent layer — not because RLS is
trusted less, but because a bug in the transaction-local GUC (e.g. `set_rls` never called, or
called against the wrong session) would otherwise depend on RLS alone to fail safely; the
explicit filter, and RLS, must each independently be wrong for a cross-tenant row to leak. See
`tests/test_search_routes.py` for the adversarial proof (two orgs sharing vocabulary, one with
far more matching rows than the other).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from libs.shared.tenancy import set_rls

_MAX_LIMIT = 50


class SearchHit(BaseModel):
    passage: str  # mirrors RerankedPassage.passage — see module docstring
    rank: float
    chunk_id: str
    meeting_id: str
    speaker_id: str
    start_ms: int
    end_ms: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


def _parse_org_id(x_org_id: str) -> UUID:
    try:
        return UUID(x_org_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Org-Id must be a valid UUID") from None


def _parse_meeting_id(meeting_id: str) -> UUID:
    try:
        return UUID(meeting_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="meeting_id must be a valid UUID") from None


def create_search_router(*, session_factory: async_sessionmaker) -> APIRouter:  # type: ignore[type-arg]
    router = APIRouter()

    @router.get("/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., min_length=1),
        meeting_id: str | None = Query(None),
        limit: int = Query(20, ge=1, le=_MAX_LIMIT),
        x_org_id: str = Header(..., alias="X-Org-Id"),
    ) -> SearchResponse:
        org_id = _parse_org_id(x_org_id)
        meeting_filter = _parse_meeting_id(meeting_id) if meeting_id is not None else None

        # websearch_to_tsquery (not to_tsquery/plainto_tsquery): the one PostgreSQL tsquery
        # constructor designed for raw, adversarial user search-box input — it never raises a
        # syntax error on unbalanced quotes or stray operator characters the way `to_tsquery`
        # would, and unlike `plainto_tsquery` it still understands quoted phrases and `-exclude`.
        sql = """
            SELECT chunk_id, meeting_id, speaker_id, start_ms, end_ms, text,
                   ts_rank(tsv, websearch_to_tsquery('english', :q)) AS rank
            FROM intelligence.transcript_chunk_search
            WHERE org_id = :org_id
              AND tsv @@ websearch_to_tsquery('english', :q)
        """
        params: dict[str, object] = {"org_id": str(org_id), "q": q, "limit": limit}
        if meeting_filter is not None:
            sql += " AND meeting_id = :meeting_id"
            params["meeting_id"] = str(meeting_filter)
        sql += " ORDER BY rank DESC LIMIT :limit"

        async with session_factory() as session, session.begin():
            await set_rls(session, org_id)
            rows = (await session.execute(text(sql), params)).all()

        return SearchResponse(
            query=q,
            results=[
                SearchHit(
                    passage=row.text,
                    rank=float(row.rank),
                    chunk_id=str(row.chunk_id),
                    meeting_id=str(row.meeting_id),
                    speaker_id=row.speaker_id,
                    start_ms=row.start_ms,
                    end_ms=row.end_ms,
                )
                for row in rows
            ],
        )

    return router
