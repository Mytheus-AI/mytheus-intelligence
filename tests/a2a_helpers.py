"""Shared fakes for the ContextGapAgent / A2A tests: no broker, no PDP, no rag-service, no model.

`Loopback` is an in-memory Redpanda for the REAL `Emitter`: a JSON codec instead of Avro, and a
producer that hands `agent-requests` to the real `Responder` and `agent-responses` to the real
`ResponseRouter` after a configurable delay, so the requester, the PEP and the responder run
exactly as in production, only the transport is faked.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import fakeredis

from libs.shared.budget import BudgetGate
from libs.shared.rag_client import RagPassage
from services.agent_runtime_service.a2a.bus import (
    A2AClient,
    Responder,
    ResponseRouter,
)
from services.agent_runtime_service.a2a.identity import IdentitySigner
from services.agent_runtime_service.a2a.pep import Pep
from services.agent_runtime_service.emitter import Emitter

SECRET = "test-agent-identity-secret-0123456789abcdef"


class JsonCodec:
    def serialize_for(self, topic: str, value: dict[str, Any]) -> bytes:
        return json.dumps(value).encode()

    def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(raw)
        return out


@dataclass
class Produced:
    topic: str
    key: Any
    value: bytes

    def json(self) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(self.value)
        return out


class LoopbackProducer:
    def __init__(self) -> None:
        self.produced: list[Produced] = []
        self.on_request: Callable[[bytes], None] | None = None
        self.on_response: Callable[[bytes], None] | None = None

    def produce(self, topic: str, key: Any = None, value: bytes = b"", on_delivery: Any = None):
        self.produced.append(Produced(topic, key, value))
        if topic == "agent-requests" and self.on_request:
            self.on_request(value)
        elif topic == "agent-responses" and self.on_response:
            self.on_response(value)

    def poll(self, _t: float) -> int:
        return 0

    def flush(self, _t: float = 0) -> int:
        return 0

    def on(self, topic: str) -> list[dict[str, Any]]:
        return [p.json() for p in self.produced if p.topic == topic]


def rag_passage(text: str, score: float, *, kind: str = "transcript", **ids: Any) -> RagPassage:
    meeting = ids.pop("meeting_id", uuid.uuid4())
    defaults = {
        "transcript": {"chunk_id": uuid.uuid4()},
        "decision": {"decision_id": uuid.uuid4()},
        "action": {"action_id": uuid.uuid4()},
        "document": {"document_id": uuid.uuid4()},
    }[kind]
    return RagPassage(
        n=1,
        text=text,
        score=score,
        kind=kind,  # type: ignore[arg-type]
        meeting_id=meeting if kind != "document" else None,
        **{**defaults, **ids},
    )


@dataclass
class ScriptedRag:
    """Canned passages per user; `delay` is fixed seconds or a callable returning seconds."""

    by_user: dict[UUID, list[RagPassage]] = field(default_factory=dict)
    delay: float | Callable[[], float] = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def retrieve(
        self, *, org_id: UUID, user_id: UUID, query: str, meeting_id: UUID | None = None
    ) -> list[RagPassage]:
        self.calls.append({"org_id": org_id, "user_id": user_id, "query": query})
        d = self.delay() if callable(self.delay) else self.delay
        if d:
            await asyncio.sleep(d)
        return list(self.by_user.get(user_id, []))


class FakePdp:
    """Allows exactly the (role, query_class, scope) triples in `grants`, `None` = PDP down."""

    def __init__(self, grants: set[tuple[str, str, str]] | None = None, down: bool = False):
        self.grants = grants if grants is not None else set()
        self.down = down
        self.calls: list[tuple[UUID, str, str, str]] = []

    async def authorize(self, org_id: UUID, role: str, qc: str, scope: str) -> str | None:
        self.calls.append((org_id, role, qc, scope))
        if self.down:
            return None
        return "allow" if (role, qc, scope) in self.grants else "deny"


def allow_all() -> set[tuple[str, str, str]]:
    return {
        (r, q, s)
        for r in ("owner", "admin", "member")
        for q in ("meeting_context", "decision_lineage", "action_status")
        for s in ("self", "same_team", "department", "org")
    }


class FakeRelated:
    def __init__(self, ids: list[UUID] | None) -> None:
        self.ids = ids
        self.calls: list[dict[str, Any]] = []

    async def related(
        self, *, org_id: UUID, meeting_id: UUID, user_id: UUID, timeout_s: float
    ) -> list[UUID] | None:
        self.calls.append({"org_id": org_id, "meeting_id": meeting_id, "user_id": user_id})
        return None if self.ids is None else list(self.ids)


def budget(consumed: int = 0, quota: int = 10**6, cap: int = 10**7) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": quota,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


class Loopback:
    """Requester and responder wired through the real Emitter over an in-memory bus."""

    def __init__(
        self,
        *,
        org_contexts: dict[UUID, dict[UUID, dict[str, Any]]],
        rag: Any,
        pdp: FakePdp,
        hop_delay: float | Callable[[], float] = 0.0,
        responder_rag: Any = None,
        redis: Any = None,
    ) -> None:
        self.signer = IdentitySigner(SECRET)
        self.producer = LoopbackProducer()
        self.emitter = Emitter(self.producer, JsonCodec())
        self.router = ResponseRouter()
        self.redis = redis or fakeredis.aioredis.FakeRedis()
        self.pdp = pdp
        self.pep = Pep(self.redis, pdp)
        self.org_contexts = org_contexts
        self.client = A2AClient(self.emitter, self.router, self.signer)
        self.responder = Responder(
            signer=self.signer,
            pep=self.pep,
            rag=responder_rag if responder_rag is not None else rag,
            user_context=self.user_context,
            emitter=self.emitter,
        )
        self.hop_delay = hop_delay
        self.outcomes: list[str] = []
        self.tasks: set[asyncio.Task[Any]] = set()
        self.producer.on_request = self._deliver_request
        self.producer.on_response = self._deliver_response

    async def user_context(self, org_id: UUID, user_id: UUID) -> dict[str, Any] | None:
        return self.org_contexts.get(org_id, {}).get(user_id)

    def _delay(self) -> float:
        return self.hop_delay() if callable(self.hop_delay) else self.hop_delay

    def _deliver_request(self, raw: bytes) -> None:
        async def go() -> None:
            await asyncio.sleep(self._delay())
            self.outcomes.append(await self.responder.handle(json.loads(raw), raw=raw))

        task = asyncio.get_running_loop().create_task(go())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _deliver_response(self, raw: bytes) -> None:
        async def go() -> None:
            await asyncio.sleep(self._delay())
            self.router.resolve(json.loads(raw))

        task = asyncio.get_running_loop().create_task(go())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def drain(self) -> None:
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)


def lognormal_delay(rng: random.Random, median_s: float, sigma: float, cap_s: float) -> float:
    import math

    return min(cap_s, median_s * math.exp(rng.gauss(0, sigma)))
