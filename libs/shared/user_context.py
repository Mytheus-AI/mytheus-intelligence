"""`get_user_context(user_id)` over gRPC to `identity-tenant-service`, and the A2A PDP
`Authorize` (ADR-0013, P10) on the same service.

The contract is `mytheus-backend/proto/identity.proto` (`mytheus.identity.v1.IdentityService`
`GetUserContext`), with `org_id` in gRPC metadata under `org-id`, never in the message body.
The server caches at `nexus:{org_id}:user_ctx:{user_id}` (5 min), so this client does not.

Same no-protoc approach as `libs/shared/rag_proto.py`: the messages are built at import time
from a descriptor that mirrors the .proto field for field; `tests/test_agent_runtime.py`
parses the backend's .proto and asserts the two match.

A failure returns None rather than raising. The caller then skips that user for the window:
without a role there is nothing to score relevance against, and skipping costs no tokens.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import UUID

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

logger = logging.getLogger(__name__)

PACKAGE = "mytheus.identity.v1"
METHOD = f"/{PACKAGE}.IdentityService/GetUserContext"
AUTHORIZE_METHOD = f"/{PACKAGE}.IdentityService/Authorize"
_F = descriptor_pb2.FieldDescriptorProto

# (name, number, type, repeated) - mirrors identity.proto
FIELDS: dict[str, list[tuple[str, int, int, bool]]] = {
    "GetUserContextRequest": [("user_id", 1, _F.TYPE_STRING, False)],
    "UserContext": [
        ("user_id", 1, _F.TYPE_STRING, False),
        ("org_id", 2, _F.TYPE_STRING, False),
        ("email", 3, _F.TYPE_STRING, False),
        ("role", 4, _F.TYPE_STRING, False),
        ("skills", 5, _F.TYPE_STRING, True),
        ("bio", 6, _F.TYPE_STRING, False),
        ("avatar_url", 7, _F.TYPE_STRING, False),
        ("rbac_tier", 8, _F.TYPE_STRING, False),
        ("team_ids", 9, _F.TYPE_STRING, True),
        ("department_id", 10, _F.TYPE_STRING, False),
        ("cache_expires_at", 11, _F.TYPE_INT64, False),
    ],
    "AuthorizeRequest": [
        ("org_id", 1, _F.TYPE_STRING, False),
        ("grantee_role", 2, _F.TYPE_STRING, False),
        ("query_class", 3, _F.TYPE_STRING, False),
        ("target_scope", 4, _F.TYPE_STRING, False),
    ],
    "AuthorizeResponse": [("decision", 1, _F.TYPE_STRING, False)],
}


def _build() -> dict[str, Any]:
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "mytheus_identity_v1_mirror.proto"
    fdp.package = PACKAGE
    fdp.syntax = "proto3"
    for msg_name, fields in FIELDS.items():
        m = fdp.message_type.add()
        m.name = msg_name
        for name, number, ftype, repeated in fields:
            f = m.field.add()
            f.name, f.number, f.type = name, number, ftype
            f.label = _F.LABEL_REPEATED if repeated else _F.LABEL_OPTIONAL
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {
        n: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{PACKAGE}.{n}"))
        for n in FIELDS
    }


_classes = _build()
GetUserContextRequest: Any = _classes["GetUserContextRequest"]
UserContext: Any = _classes["UserContext"]
AuthorizeRequest: Any = _classes["AuthorizeRequest"]
AuthorizeResponse: Any = _classes["AuthorizeResponse"]


class UserContextClient:
    def __init__(self, address: str | None = None, timeout_s: float = 2.0) -> None:
        self._address = address or os.environ.get("IDENTITY_GRPC_ADDR", "localhost:50051")
        self._timeout = timeout_s
        self._channel: grpc.aio.Channel | None = None

    def _call(self) -> Any:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)
        return self._channel.unary_unary(
            METHOD,
            request_serializer=GetUserContextRequest.SerializeToString,
            response_deserializer=UserContext.FromString,
        )

    async def get(self, org_id: UUID, user_id: UUID) -> dict[str, Any] | None:
        try:
            reply = await self._call()(
                GetUserContextRequest(user_id=str(user_id)),
                metadata=(("org-id", str(org_id)),),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            logger.warning("get_user_context failed org=%s user=%s: %s", org_id, user_id, exc)
            return None
        if reply.org_id and reply.org_id != str(org_id):  # defense in depth: never cross a tenant
            logger.error("get_user_context returned another tenant's user; dropping")
            return None
        return {
            "user_id": reply.user_id,
            "email": reply.email,
            "role": reply.role,
            "skills": list(reply.skills),
            "bio": reply.bio,
            "team_ids": list(reply.team_ids),
            "department_id": reply.department_id,
            # The RBAC tier is the A2A grantee_role (ADR-0013). Never rendered into a prompt.
            "rbac_tier": reply.rbac_tier,
        }

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()


class PdpClient:
    """The agent-to-agent PDP, `IdentityService.Authorize` (ADR-0013, P10 contract section 3).

    Returns "allow" or "deny" from the server, or None when the call failed (unreachable,
    UNIMPLEMENTED, timeout). The PEP treats None as deny and does not cache it."""

    def __init__(self, address: str | None = None, timeout_s: float = 0.5) -> None:
        self._address = address or os.environ.get("IDENTITY_GRPC_ADDR", "localhost:50051")
        self._timeout = timeout_s
        self._channel: grpc.aio.Channel | None = None

    def _call(self) -> Any:
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)
        return self._channel.unary_unary(
            AUTHORIZE_METHOD,
            request_serializer=AuthorizeRequest.SerializeToString,
            response_deserializer=AuthorizeResponse.FromString,
        )

    async def authorize(
        self, org_id: UUID, grantee_role: str, query_class: str, target_scope: str
    ) -> str | None:
        try:
            reply = await self._call()(
                AuthorizeRequest(
                    org_id=str(org_id),
                    grantee_role=grantee_role,
                    query_class=query_class,
                    target_scope=target_scope,
                ),
                metadata=(("org-id", str(org_id)),),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            logger.warning("Authorize failed org=%s: %s (deny)", org_id, exc)
            return None
        decision = str(reply.decision)
        return decision if decision in ("allow", "deny") else None

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
