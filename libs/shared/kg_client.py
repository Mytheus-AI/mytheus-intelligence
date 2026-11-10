"""A gRPC client for `knowledge-graph-service`'s `QueryTemplate` (`proto/knowledge_graph.proto`).

`ProactiveAgent` (P11) reads the AGE `nexus_knowledge` graph through it: `decision_lineage` for
the conflict check, `related_decisions_for_topic` for the missing-stakeholder check. Calls carry
the same three metadata keys the server checks (`grpc_server.py`): `x-mytheus-service`
(`agent-runtime-service`), `x-mytheus-service-token` (`KG_CALLER_TOKEN`, listed on the server as
`agent-runtime-service=<secret>` in `MYTHEUS_KG_CALLER_TOKENS`) and `x-mytheus-org`. The org
comes from the verified event header, never from a request body.

Hot degradation (R1, plan 10-13 section 3.6, doc 14 section 5.3): every call has a hard
deadline (`KG_TIMEOUT_MS`, default 800 ms). A deadline miss, an unreachable server or any other
RPC error raises `KgUnavailable`, which the caller turns into "use the Qdrant / rag-service
path only". A bad answer is never guessed at.

Env: `KG_GRPC_ADDR` (host:port of `services.knowledge_graph_service.grpc_main`, which binds
`KG_GRPC_BIND`, default port 50104) and `KG_CALLER_TOKEN`. Without both there is no client and
every KG-backed step takes its fallback.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import grpc

from libs.shared import kg_proto

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
DEFAULT_TIMEOUT_MS = 800


class KgUnavailable(Exception):
    """knowledge-graph-service did not answer in time, or answered with an error."""


@dataclass(frozen=True)
class KgNode:
    label: str
    id: str
    role: str
    depth: int = 0
    properties: dict[str, Any] = field(default_factory=dict)


class KgClient(Protocol):
    async def query(
        self, org_id: UUID, template: str, params: Mapping[str, str]
    ) -> list[KgNode]: ...


def _timeout_ms() -> int:
    raw = os.environ.get("KG_TIMEOUT_MS", "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_TIMEOUT_MS
    except ValueError:
        logger.warning("KG_TIMEOUT_MS=%r is not an integer; using %s", raw, DEFAULT_TIMEOUT_MS)
        return DEFAULT_TIMEOUT_MS


def _node(raw: Any) -> KgNode:
    try:
        props = json.loads(raw.properties_json) if raw.properties_json else {}
    except ValueError:
        props = {}
    return KgNode(
        label=str(raw.label),
        id=str(raw.id),
        role=str(raw.role),
        depth=int(raw.depth),
        properties=props if isinstance(props, dict) else {},
    )


class GrpcKgClient:
    def __init__(self, address: str, token: str, *, timeout_ms: int | None = None) -> None:
        self._address = address
        self._token = token
        self._timeout = (timeout_ms if timeout_ms is not None else _timeout_ms()) / 1000
        self._channel: grpc.aio.Channel | None = None

    def _call(self) -> Any:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)  # mesh-internal
        return self._channel.unary_unary(
            kg_proto.QUERY_TEMPLATE_METHOD,
            request_serializer=kg_proto.QueryTemplateRequest.SerializeToString,
            response_deserializer=kg_proto.QueryTemplateResponse.FromString,
        )

    async def query(self, org_id: UUID, template: str, params: Mapping[str, str]) -> list[KgNode]:
        request = kg_proto.QueryTemplateRequest(
            template=template,
            params=[kg_proto.TemplateParam(name=k, value=str(v)) for k, v in params.items()],
        )
        try:
            reply = await self._call()(
                request,
                metadata=(
                    ("x-mytheus-service", SERVICE_NAME),
                    ("x-mytheus-service-token", self._token),
                    ("x-mytheus-org", str(org_id)),
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            code = exc.code() if hasattr(exc, "code") else None
            logger.warning("knowledge-graph %s failed org=%s: %s", template, org_id, code)
            raise KgUnavailable(f"{template}: {code}") from exc
        return [_node(n) for n in reply.nodes]

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()


def kg_client_from_env() -> GrpcKgClient | None:
    address = os.environ.get("KG_GRPC_ADDR", "").strip()
    token = os.environ.get("KG_CALLER_TOKEN", "").strip()
    if not address or not token:
        return None
    return GrpcKgClient(address, token)
