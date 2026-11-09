"""`POST /graph/query/{template}` and `POST /graph/cypher`: internal-only, read-only.

Never routed by `api-gateway`. Callers are services (`bff` for the lineage screen, `rag-service`
for the graph arm, `agent-runtime-service` for ProactiveAgent), authenticated with the service
token scheme in `libs/shared/service_auth.py`; the tenant is the `X-Mytheus-Org` header and is
the only `org_id` any Cypher ever binds. Secrets: `MYTHEUS_KG_CALLER_TOKENS`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from libs.shared.age import AgeConnection, AgeQueryRejected, age_database_url, prepare_session
from libs.shared.service_auth import InternalCaller, ServiceAuthError, authenticate
from services.knowledge_graph_service.templates import (
    TemplateNotFound,
    run_adhoc,
    run_template,
)

CALLER_TOKENS_ENV = "MYTHEUS_KG_CALLER_TOKENS"
ALLOWED_CALLERS = frozenset({"bff", "rag-service", "agent-runtime-service"})

router = APIRouter(prefix="/graph", tags=["knowledge-graph"])


def internal_caller(request: Request) -> InternalCaller:
    headers = {k.lower(): v for k, v in request.headers.items()}
    try:
        return authenticate(headers, env_name=CALLER_TOKENS_ENV, allowed=ALLOWED_CALLERS)
    except ServiceAuthError:
        raise HTTPException(status_code=401, detail="unauthenticated") from None


async def graph_connection(request: Request) -> AsyncIterator[AgeConnection]:
    pool: asyncpg.Pool | None = getattr(request.app.state, "age_pool", None)
    if pool is None:
        pool = await asyncpg.create_pool(age_database_url(), init=prepare_session, max_size=10)
        request.app.state.age_pool = pool
    async with pool.acquire() as conn:
        yield conn


class TemplateRequest(BaseModel):
    params: dict[str, str | int] = Field(default_factory=dict)


class NodeOut(BaseModel):
    label: str
    id: str
    role: str
    depth: int
    properties: dict[str, Any]


class TemplateResponse(BaseModel):
    template: str
    nodes: list[NodeOut]


class CypherRequest(BaseModel):
    cypher: str = Field(max_length=4000)
    params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=lambda: ["result"], min_length=1, max_length=8)


class CypherResponse(BaseModel):
    rows: list[list[Any]]


Caller = Annotated[InternalCaller, Depends(internal_caller)]
Conn = Annotated[AgeConnection, Depends(graph_connection)]


@router.post("/query/{template}", response_model=TemplateResponse)
async def query_template(
    template: str, body: TemplateRequest, caller: Caller, conn: Conn
) -> TemplateResponse:
    try:
        nodes = await run_template(conn, caller.org_id, template, body.params)
    except TemplateNotFound:
        raise HTTPException(status_code=404, detail=f"unknown template {template!r}") from None
    except AgeQueryRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return TemplateResponse(template=template, nodes=[NodeOut(**n.as_dict()) for n in nodes])


@router.post("/cypher", response_model=CypherResponse)
async def query_cypher(body: CypherRequest, caller: Caller, conn: Conn) -> CypherResponse:
    try:
        rows = await run_adhoc(conn, caller.org_id, body.cypher, body.params, body.columns)
    except AgeQueryRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return CypherResponse(rows=rows)
