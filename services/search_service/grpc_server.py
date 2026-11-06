"""search-service's internal gRPC FTS arm for `rag-service` (`proto/rag.proto` `FtsService`).

REST `GET /search` (`routes.py`) is unchanged and stays the user-facing surface; this is the
service-to-service transport for hybrid retrieval, over the SAME query (`fts.fts_search`), so
tenant isolation (RLS before SELECT + explicit `WHERE org_id`) is identical on both.

Auth: `org_id` is never in the message. The caller sends `authorization: Bearer <Clerk JWT>` in
gRPC metadata and this server verifies it with the same `libs.shared.auth` chain REST uses
(RS256/JWKS, `nbf` required, internal-org claim must be a UUID). Missing/invalid token ->
UNAUTHENTICATED; IdP outage -> UNAVAILABLE. Bad `meeting_id` / empty query / limit out of range
-> INVALID_ARGUMENT. Tests inject `verify` to avoid a live JWKS.

Run: python -m services.search_service.grpc_main   (GRPC_PORT, default 50103)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import grpc
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from libs.shared import rag_proto
from libs.shared.auth import org_id_from, verify_token
from services.search_service.fts import MAX_LIMIT, fts_search

DEFAULT_LIMIT = 20

VerifyFn = Callable[[str], UUID]


def verify_bearer(token: str) -> UUID:
    """Default verifier: the same JWT chain as REST. Raises HTTPException on any failure."""
    claims = verify_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    return org_id_from(claims)


def _bearer_from(context: grpc.aio.ServicerContext) -> str | None:
    for key, value in context.invocation_metadata() or ():
        if key.lower() == "authorization":
            v = value if isinstance(value, str) else value.decode()
            scheme, _, token = v.partition(" ")
            if scheme.lower() == "bearer" and token:
                return token
    return None


class FtsServicer:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        verify: VerifyFn = verify_bearer,
    ) -> None:
        self._session_factory = session_factory
        self._verify = verify

    async def FtsSearch(  # noqa: N802 - gRPC method name
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        token = _bearer_from(context)
        if token is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing bearer token")
        try:
            org_id = self._verify(token)  # type: ignore[arg-type]
        except HTTPException as exc:
            code = (
                grpc.StatusCode.UNAVAILABLE
                if exc.status_code == 503
                else grpc.StatusCode.UNAUTHENTICATED
            )
            await context.abort(code, str(exc.detail))

        if not request.query.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must be non-empty")
        limit = request.limit or DEFAULT_LIMIT
        if not 1 <= limit <= MAX_LIMIT:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"limit must be 1..{MAX_LIMIT}")
        meeting_id: UUID | None = None
        if request.meeting_id:
            try:
                meeting_id = UUID(request.meeting_id)
            except ValueError:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "meeting_id must be a valid UUID"
                )

        rows = await fts_search(
            session_factory=self._session_factory,
            org_id=org_id,
            q=request.query,
            meeting_id=meeting_id,
            limit=limit,
        )
        return rag_proto.FtsSearchResponse(
            query=request.query,
            results=[
                rag_proto.FtsHit(
                    passage=r.passage,
                    rank=r.rank,
                    chunk_id=r.chunk_id,
                    meeting_id=r.meeting_id,
                    speaker_id=r.speaker_id,
                    start_ms=r.start_ms,
                    end_ms=r.end_ms,
                )
                for r in rows
            ],
        )


def add_fts_servicer(servicer: FtsServicer, server: grpc.aio.Server) -> None:
    handler = grpc.unary_unary_rpc_method_handler(
        servicer.FtsSearch,
        request_deserializer=rag_proto.FtsSearchRequest.FromString,
        response_serializer=rag_proto.FtsSearchResponse.SerializeToString,
    )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(rag_proto.SERVICE_NAME, {"FtsSearch": handler}),)
    )
