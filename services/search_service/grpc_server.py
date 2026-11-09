"""search-service's internal gRPC FTS arm for `rag-service` (`proto/rag.proto` `FtsService`).

REST `GET /search` (`routes.py`) is unchanged and stays the user-facing surface; this is the
service-to-service transport for hybrid retrieval, over the SAME query (`fts.fts_search`), so
tenant isolation (RLS before SELECT + explicit `WHERE org_id`) is identical on both.

Auth: `org_id` is never in the message. Either the caller sends `authorization: Bearer <Clerk
JWT>` in gRPC metadata and this server verifies it with the same `libs.shared.auth` chain REST
uses (RS256/JWKS, `nbf` required, internal-org claim must be a UUID), or rag-service sends its
service credentials (`x-mytheus-service`, `x-mytheus-service-token`, `x-mytheus-org`; secrets in
`MYTHEUS_SEARCH_CALLER_TOKENS`). rag-service authenticates the asker itself and may be asking
for an agent, which has no user token, so the service path is the one it uses. `GetPassages`
(text by id, for the vector arm's hits) accepts the service path only. Missing/invalid
credentials -> UNAUTHENTICATED; IdP outage -> UNAVAILABLE. Bad ids / empty query / limit out of
range -> INVALID_ARGUMENT. Tests inject `verify` to avoid a live JWKS.

Scope (mytheus-docs ADR-0033 D1): with `scoped = true`, `meeting_ids` is an allowlist and an
empty one returns nothing.

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
from libs.shared.service_auth import SERVICE_HEADER, ServiceAuthError, authenticate
from services.search_service.fts import (
    MAX_LIMIT,
    MAX_NEIGHBORS,
    MAX_PASSAGES,
    fts_search,
    get_passages,
)

CALLER_TOKENS_ENV = "MYTHEUS_SEARCH_CALLER_TOKENS"
ALLOWED_CALLERS = frozenset({"rag-service"})

DEFAULT_LIMIT = 20

VerifyFn = Callable[[str], UUID]


def verify_bearer(token: str) -> UUID:
    """Default verifier: the same JWT chain as REST. Raises HTTPException on any failure."""
    claims = verify_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    return org_id_from(claims)


def _metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in context.invocation_metadata() or ():
        out[key.lower()] = value if isinstance(value, str) else value.decode()
    return out


def _bearer_from(md: dict[str, str]) -> str | None:
    scheme, _, token = md.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return None


def _uuids(raw: Any, field: str) -> list[UUID]:
    try:
        return [UUID(v) for v in raw]
    except ValueError:
        raise ValueError(f"{field} must hold valid UUIDs") from None


class FtsServicer:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        verify: VerifyFn = verify_bearer,
    ) -> None:
        self._session_factory = session_factory
        self._verify = verify

    async def _service_org(self, md: dict[str, str], context: grpc.aio.ServicerContext) -> UUID:
        try:
            return authenticate(md, env_name=CALLER_TOKENS_ENV, allowed=ALLOWED_CALLERS).org_id
        except ServiceAuthError:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "unauthenticated")
            raise  # unreachable: abort raises

    async def _org(self, md: dict[str, str], context: grpc.aio.ServicerContext) -> UUID:
        if SERVICE_HEADER in md:
            return await self._service_org(md, context)
        token = _bearer_from(md)
        if token is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing bearer token")
        try:
            return self._verify(token)  # type: ignore[arg-type]
        except HTTPException as exc:
            code = (
                grpc.StatusCode.UNAVAILABLE
                if exc.status_code == 503
                else grpc.StatusCode.UNAUTHENTICATED
            )
            await context.abort(code, str(exc.detail))
            raise  # unreachable: abort raises

    async def FtsSearch(  # noqa: N802 - gRPC method name
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        org_id = await self._org(_metadata(context), context)

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

        allow: list[UUID] | None = None
        if request.scoped:
            try:
                allow = _uuids(request.meeting_ids, "meeting_ids")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        rows = await fts_search(
            session_factory=self._session_factory,
            org_id=org_id,
            q=request.query,
            meeting_id=meeting_id,
            limit=limit,
            meeting_ids=allow,
            speaker_ids=list(request.speaker_ids) or None,
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

    async def GetPassages(  # noqa: N802 - gRPC method name
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        org_id = await self._service_org(_metadata(context), context)
        if len(request.chunk_ids) > MAX_PASSAGES or len(request.documents) > MAX_PASSAGES:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"at most {MAX_PASSAGES} of each kind"
            )
        if not 0 <= request.neighbors <= MAX_NEIGHBORS:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"neighbors must be 0..{MAX_NEIGHBORS}"
            )
        chunk_ids: list[UUID] = []
        documents: list[tuple[UUID, str]] = []
        try:
            chunk_ids = _uuids(request.chunk_ids, "chunk_ids")
            documents = [
                (UUID(d.document_id), d.element_id) for d in request.documents if d.element_id
            ]
        except ValueError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ids must be valid UUIDs")
        transcripts, docs = await get_passages(
            session_factory=self._session_factory,
            org_id=org_id,
            chunk_ids=chunk_ids,
            neighbors=request.neighbors,
            documents=documents,
        )
        return rag_proto.GetPassagesResponse(
            transcripts=[rag_proto.TranscriptPassage(**vars(t)) for t in transcripts],
            documents=[rag_proto.DocumentPassage(**vars(d)) for d in docs],
        )


def add_fts_servicer(servicer: FtsServicer, server: grpc.aio.Server) -> None:
    search = grpc.unary_unary_rpc_method_handler(
        servicer.FtsSearch,
        request_deserializer=rag_proto.FtsSearchRequest.FromString,
        response_serializer=rag_proto.FtsSearchResponse.SerializeToString,
    )
    passages = grpc.unary_unary_rpc_method_handler(
        servicer.GetPassages,
        request_deserializer=rag_proto.GetPassagesRequest.FromString,
        response_serializer=rag_proto.GetPassagesResponse.SerializeToString,
    )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                rag_proto.SERVICE_NAME, {"FtsSearch": search, "GetPassages": passages}
            ),
        )
    )
