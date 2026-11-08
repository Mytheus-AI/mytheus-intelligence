"""Everything `agent-runtime-service` publishes: `usage-events`, `agent-stream`,
`agent-results`, `notifications`.

Produced straight to Redpanda with an idempotent producer (`acks=all`,
`enable.idempotence`), NOT through a transactional outbox. The plan's section 3.2 step 10 says
"via the transactional outbox", but an outbox exists to make an emit atomic with a database
write, and this service makes none (its acceptance check is "writes zero Postgres rows").
At-least-once comes from ordering instead: the consumer flushes this producer before it
commits the input offsets, so a crash replays the window rather than losing its output.

Duplicates on replay are made harmless by the idempotency keys:
- `agent-results`: `agentresult:{meeting_id}:{window_id}:{user_id}:{agent}`, deduped by
  `core-api`. A replayed window republishes the same key.
- `usage-events`: `usage:{org_id}:agent_token:{occurred_at}:{call_id}`, one per real model
  call. A replay makes real calls again, which are real cost, so they are metered again.
- `agent-stream` deltas are throwaway (shed first under backpressure); `notifications`
  carry `notif:{source_event_id}:{channel}:{user_id}`.

Partition keys: `org_id` everywhere except `agent-stream`, which is `{org_id}:{meeting_id}`
(ADR-0017).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from libs.shared.avro_codec import AvroCodec
from libs.shared.events import EventEnvelope
from libs.shared.state import AgentResult, TranscriptWindow

logger = logging.getLogger(__name__)

PRODUCER = "agent-runtime-service"


class EmitError(RuntimeError):
    """A produced message was not acknowledged. The window must not be committed."""


class Emitter:
    def __init__(self, producer: Any, codec: AvroCodec) -> None:
        self._producer = producer
        self._codec = codec
        self._errors: list[str] = []

    def _produce(self, envelope: EventEnvelope, key: str) -> None:
        value = self._codec.serialize_for(envelope.topic, envelope.to_value())

        def on_delivery(err: Any, _msg: Any) -> None:
            if err is not None:
                self._errors.append(f"{envelope.topic}: {err}")

        self._producer.produce(envelope.topic, key=key, value=value, on_delivery=on_delivery)
        self._producer.poll(0)

    def flush(self, timeout_s: float = 10.0) -> None:
        remaining = self._producer.flush(timeout_s)
        errors, self._errors = self._errors, []
        if remaining or errors:
            raise EmitError(f"{remaining} undelivered, errors={errors}")

    async def usage(
        self, *, org_id: UUID, user_id: UUID | None, tokens: int, model_tier: str, call_id: str
    ) -> None:
        env = EventEnvelope(
            topic="usage-events",
            org_id=str(org_id),
            producer=PRODUCER,
            idempotency_key="",
            payload={
                "user_id": str(user_id) if user_id else None,
                "event_type": "agent_token",
                "quantity": float(tokens),
                "source_id": call_id,
                "model_tier": model_tier,
            },
        )
        env.idempotency_key = f"usage:{org_id}:agent_token:{env.occurred_at}:{call_id}"
        self._produce(env, key=str(org_id))

    async def delta(
        self,
        *,
        window: TranscriptWindow,
        user_id: UUID,
        seq: int,
        text: str,
        done: bool = False,
    ) -> None:
        env = EventEnvelope(
            topic="agent-stream",
            org_id=str(window.org_id),
            producer=PRODUCER,
            idempotency_key=f"agentstream:{window.window_id}:{user_id}:{seq}",
            payload={
                "meeting_id": str(window.meeting_id),
                "user_id": str(user_id),
                "window_id": window.window_id,
                "seq": seq,
                "delta": text,
                "done": done,
            },
        )
        self._produce(env, key=f"{window.org_id}:{window.meeting_id}")

    async def result(
        self, *, window: TranscriptWindow, user_id: UUID, agent: str, result: AgentResult
    ) -> None:
        env = EventEnvelope(
            topic="agent-results",
            org_id=str(window.org_id),
            producer=PRODUCER,
            idempotency_key=f"agentresult:{window.meeting_id}:{window.window_id}:{user_id}:{agent}",
            payload={
                "meeting_id": str(window.meeting_id),
                "window_id": window.window_id,
                "user_id": str(user_id),
                "agent": agent,
                "kind": result.kind,
                "relevance_score": float(result.relevance_score),
                "confidence": float(result.confidence),
                "content": result.content,
                "payload_json": json.dumps(result.payload),
                "source_chunk_ids": list(result.source_chunk_ids),
            },
        )
        self._produce(env, key=str(window.org_id))

    async def notification(
        self, *, org_id: UUID, user_id: UUID, source_event_id: str, title: str, body: str
    ) -> None:
        env = EventEnvelope(
            topic="notifications",
            org_id=str(org_id),
            producer=PRODUCER,
            idempotency_key=f"notif:{source_event_id}:in_app:{user_id}",
            payload={
                "source_event_id": source_event_id,
                "channel": "in_app",
                "user_id": str(user_id),
                "priority": 1,
                "title": title,
                "body": body,
                "link": None,
            },
        )
        self._produce(env, key=str(org_id))
