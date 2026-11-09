"""knowledge-graph-service's internal gRPC surface (`proto/knowledge_graph.proto`).

Same templates and the same authentication as the HTTP routes: the caller's service name and
secret, and the tenant, ride in invocation metadata (`x-mytheus-service`,
`x-mytheus-service-token`, `x-mytheus-org`). Missing or wrong credentials -> UNAUTHENTICATED;
unknown template -> NOT_FOUND; bad parameters -> INVALID_ARGUMENT.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import grpc

from libs.shared import kg_proto
from libs.shared.age import AgeConnection, AgeQueryRejected
from libs.shared.service_auth import ServiceAuthError, authenticate
from services.knowledge_graph_service.routes import ALLOWED_CALLERS, CALLER_TOKENS_ENV
from services.knowledge_graph_service.templates import TemplateNotFound, run_template

GraphConnect = Callable[[], AbstractAsyncContextManager[AgeConnection]]


def _metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in context.invocation_metadata() or ():
        out[key.lower()] = value if isinstance(value, str) else value.decode()
    return out


class KnowledgeGraphServicer:
    def __init__(self, *, graph_connect: GraphConnect) -> None:
        self._graph_connect = graph_connect

    async def QueryTemplate(  # noqa: N802 - gRPC method name
        self, request: Any, context: grpc.aio.ServicerContext
    ) -> Any:
        try:
            caller = authenticate(
                _metadata(context), env_name=CALLER_TOKENS_ENV, allowed=ALLOWED_CALLERS
            )
        except ServiceAuthError:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "unauthenticated")
            raise  # unreachable: abort raises
        params = {p.name: p.value for p in request.params}
        try:
            async with self._graph_connect() as conn:
                nodes = await run_template(conn, caller.org_id, request.template, params)
        except TemplateNotFound:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown template {request.template}")
            raise
        except AgeQueryRejected as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        return kg_proto.QueryTemplateResponse(
            template=request.template,
            nodes=[
                kg_proto.GraphNode(
                    label=n.label,
                    id=n.id,
                    role=n.role,
                    depth=n.depth,
                    properties_json=json.dumps(n.properties, sort_keys=True),
                )
                for n in nodes
            ],
        )


def add_knowledge_graph_servicer(servicer: KnowledgeGraphServicer, server: grpc.aio.Server) -> None:
    handler = grpc.unary_unary_rpc_method_handler(
        servicer.QueryTemplate,
        request_deserializer=kg_proto.QueryTemplateRequest.FromString,
        response_serializer=kg_proto.QueryTemplateResponse.SerializeToString,
    )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(kg_proto.SERVICE_NAME, {"QueryTemplate": handler}),)
    )
