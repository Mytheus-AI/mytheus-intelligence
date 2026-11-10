"""`GetOrgContext` over gRPC to `identity-tenant-service`: the company brain (ADR-0039 A9).

The contract is `mytheus-backend/proto/identity.proto` (`mytheus.identity.v1.IdentityService`
`GetOrgContext`), with `org_id` in gRPC metadata under `org-id`, never in the message body, the
same as `GetUserContext` (`user_context.py`). The server caches at `nexus:{org_id}:org_ctx` for
5 minutes and evicts on every company-brain write; this client adds a short in-process cache
(`ORG_CONTEXT_TTL_S`, default 60 s) because every agent of every window asks for it, and shares
one call between concurrent misses on the same org.

Same no-protoc approach as `user_context.py`: the messages are built at import time from a
descriptor that mirrors the .proto field for field; `tests/test_company_brain.py` parses
`proto/identity.proto` (this repo's copy of the backend's) and asserts the two match.

A failure (unreachable, UNIMPLEMENTED on an older identity server, UNAVAILABLE, a timeout) is
the EMPTY context, cached for `FAILURE_TTL_S` so a missing server is not asked every window.
Everything in ADR-0039 is opt-in by data: an empty context renders nothing in any prompt, and
the agents behave exactly as they did before the company brain existed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

logger = logging.getLogger(__name__)

PACKAGE = "mytheus.identity.v1"
METHOD = f"/{PACKAGE}.IdentityService/GetOrgContext"
_F = descriptor_pb2.FieldDescriptorProto
DEFAULT_TTL_S = 60.0
FAILURE_TTL_S = 15.0

# (name, number, type, repeated, message type or None) - mirrors identity.proto
FIELDS: dict[str, list[tuple[str, int, int, bool, str | None]]] = {
    "GetOrgContextRequest": [],
    "GlossaryTerm": [
        ("term", 1, _F.TYPE_STRING, False, None),
        ("kind", 2, _F.TYPE_STRING, False, None),
        ("definition", 3, _F.TYPE_STRING, False, None),
        ("aliases", 4, _F.TYPE_STRING, True, None),
    ],
    "TopicOwner": [
        ("topic", 1, _F.TYPE_STRING, False, None),
        ("owner_user_id", 2, _F.TYPE_STRING, False, None),
    ],
    "Goal": [
        ("id", 1, _F.TYPE_STRING, False, None),
        ("title", 2, _F.TYPE_STRING, False, None),
        ("period", 3, _F.TYPE_STRING, False, None),
    ],
    "TopicWatch": [
        ("user_id", 1, _F.TYPE_STRING, False, None),
        ("topic", 2, _F.TYPE_STRING, False, None),
    ],
    "OrgContext": [
        ("description", 1, _F.TYPE_STRING, False, None),
        ("industry", 2, _F.TYPE_STRING, False, None),
        ("products", 3, _F.TYPE_STRING, True, None),
        ("customers", 4, _F.TYPE_STRING, True, None),
        ("glossary", 5, _F.TYPE_MESSAGE, True, "GlossaryTerm"),
        ("topic_owners", 6, _F.TYPE_MESSAGE, True, "TopicOwner"),
        ("goals", 7, _F.TYPE_MESSAGE, True, "Goal"),
        ("never_record_topics", 8, _F.TYPE_STRING, True, None),
        ("topic_watches", 9, _F.TYPE_MESSAGE, True, "TopicWatch"),
    ],
}


def _build() -> dict[str, Any]:
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "mytheus_identity_v1_org_mirror.proto"
    fdp.package = PACKAGE
    fdp.syntax = "proto3"
    for msg_name, fields in FIELDS.items():
        m = fdp.message_type.add()
        m.name = msg_name
        for name, number, ftype, repeated, type_name in fields:
            f = m.field.add()
            f.name, f.number, f.type = name, number, ftype
            f.label = _F.LABEL_REPEATED if repeated else _F.LABEL_OPTIONAL
            if type_name is not None:
                f.type_name = f".{PACKAGE}.{type_name}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {
        n: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{PACKAGE}.{n}"))
        for n in FIELDS
    }


_classes = _build()
GetOrgContextRequest: Any = _classes["GetOrgContextRequest"]
OrgContextMessage: Any = _classes["OrgContext"]


# --- the Python side --------------------------------------------------------------------------


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    kind: str = "other"
    definition: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicOwner:
    topic: str
    owner_user_id: UUID


@dataclass(frozen=True)
class Goal:
    id: str
    title: str
    period: str = ""


@dataclass(frozen=True)
class TopicWatch:
    user_id: UUID
    topic: str


@dataclass(frozen=True)
class CompanyContext:
    """The company brain one org's agents see. Every field may be empty."""

    description: str = ""
    industry: str = ""
    products: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()
    glossary: tuple[GlossaryTerm, ...] = ()
    topic_owners: tuple[TopicOwner, ...] = ()
    goals: tuple[Goal, ...] = ()
    never_record_topics: tuple[str, ...] = ()
    topic_watches: tuple[TopicWatch, ...] = ()

    @property
    def empty(self) -> bool:
        return self == EMPTY


EMPTY = CompanyContext()


def _flat(raw: object) -> str:
    return " ".join(str(raw).split()) if isinstance(raw, str) else ""


def _strings(raw: Any) -> tuple[str, ...]:
    out: list[str] = []
    for value in raw:
        text = _flat(value)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _uuid(raw: object) -> UUID | None:
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def from_message(reply: Any) -> CompanyContext:
    """The reply as a `CompanyContext`, dropping malformed entries one by one."""
    glossary = tuple(
        GlossaryTerm(
            term=_flat(t.term),
            kind=_flat(t.kind) or "other",
            definition=_flat(t.definition),
            aliases=_strings(t.aliases),
        )
        for t in reply.glossary
        if _flat(t.term)
    )
    owners: list[TopicOwner] = []
    for o in reply.topic_owners:
        uid = _uuid(o.owner_user_id)
        if uid is not None and _flat(o.topic):
            owners.append(TopicOwner(_flat(o.topic), uid))
    watches: list[TopicWatch] = []
    for w in reply.topic_watches:
        uid = _uuid(w.user_id)
        if uid is not None and _flat(w.topic):
            watches.append(TopicWatch(uid, _flat(w.topic)))
    return CompanyContext(
        description=" ".join(reply.description.split()) if reply.description else "",
        industry=_flat(reply.industry),
        products=_strings(reply.products),
        customers=_strings(reply.customers),
        glossary=glossary,
        topic_owners=tuple(owners),
        goals=tuple(
            Goal(id=_flat(g.id), title=_flat(g.title), period=_flat(g.period))
            for g in reply.goals
            if _flat(g.id) and _flat(g.title)
        ),
        never_record_topics=_strings(reply.never_record_topics),
        topic_watches=tuple(watches),
    )


class OrgContextUnavailable(Exception):
    """`get_strict`: the company brain could not be read (the caller must not guess)."""


class OrgContextClient:
    """`get(org_id)` never raises and never returns None: a failure is `EMPTY`.
    `get_strict(org_id)` raises `OrgContextUnavailable` instead, for a caller that must not
    treat "unknown" as "no never-record topics" (the indexing consumers retry rather than
    index a window they cannot vet)."""

    def __init__(
        self,
        address: str | None = None,
        *,
        timeout_s: float = 1.0,
        ttl_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        call: Any = None,
    ) -> None:
        self._address = address or os.environ.get("IDENTITY_GRPC_ADDR", "localhost:50051")
        self._timeout = timeout_s
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get("ORG_CONTEXT_TTL_S", "") or DEFAULT_TTL_S)
            except ValueError:
                ttl_s = DEFAULT_TTL_S
        self._ttl = max(0.0, ttl_s)
        self._clock = clock
        self._channel: grpc.aio.Channel | None = None
        self._stub = call  # tests inject the unary call
        self._cache: dict[UUID, tuple[float, CompanyContext, bool]] = {}
        self._inflight: dict[UUID, asyncio.Future[tuple[CompanyContext, bool]]] = {}

    def _call(self) -> Any:
        if self._stub is not None:
            return self._stub
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(self._address)
        return self._channel.unary_unary(
            METHOD,
            request_serializer=GetOrgContextRequest.SerializeToString,
            response_deserializer=OrgContextMessage.FromString,
        )

    async def get(self, org_id: UUID) -> CompanyContext:
        return (await self._get(org_id, strict=False))[0]

    async def get_strict(self, org_id: UUID) -> CompanyContext:
        ctx, ok = await self._get(org_id, strict=True)
        if not ok:
            raise OrgContextUnavailable(str(org_id))
        return ctx

    async def _get(self, org_id: UUID, *, strict: bool) -> tuple[CompanyContext, bool]:
        hit = self._cache.get(org_id)
        if hit is not None and self._clock() < hit[0] and (hit[2] or not strict):
            return hit[1], hit[2]
        flight = self._inflight.get(org_id)
        if flight is None:
            flight = asyncio.ensure_future(self._fetch(org_id))
            self._inflight[org_id] = flight
            flight.add_done_callback(lambda _f: self._inflight.pop(org_id, None))
        return await asyncio.shield(flight)

    async def _fetch(self, org_id: UUID) -> tuple[CompanyContext, bool]:
        try:
            reply = await self._call()(
                GetOrgContextRequest(),
                metadata=(("org-id", str(org_id)),),
                timeout=self._timeout,
            )
            ctx, ok = from_message(reply), True
            ttl = self._ttl
        except grpc.RpcError as exc:
            logger.warning("get_org_context failed org=%s: %s (no company context)", org_id, exc)
            ctx, ok, ttl = EMPTY, False, min(self._ttl, FAILURE_TTL_S)
        except Exception:  # noqa: BLE001 - a bad reply must not stop a window
            logger.exception("get_org_context reply unusable org=%s (no company context)", org_id)
            ctx, ok, ttl = EMPTY, False, min(self._ttl, FAILURE_TTL_S)
        if ttl > 0:
            self._cache[org_id] = (self._clock() + ttl, ctx, ok)
        return ctx, ok

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
