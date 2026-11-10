"""Everything `agent-runtime-service` publishes: `usage-events`, `agent-stream`,
`agent-results`, `notifications`, and for agent-to-agent collaboration (P10) `agent-requests`,
`agent-responses` and the JSON dead letters on `agent-requests.dlq`.

Produced straight to Redpanda with an idempotent producer (`acks=all`,
`enable.idempotence`), NOT through a transactional outbox. The plan's section 3.2 step 10 says
"via the transactional outbox", but an outbox exists to make an emit atomic with a database
write, and this service makes none (its acceptance check is "writes zero Postgres rows").
At-least-once comes from ordering instead: the consumer flushes this producer before it
commits the input offsets, so a crash replays the window rather than losing its output.

Duplicates on replay are made harmless by the idempotency keys:
- `agent-results`: `agentresult:{meeting_id}:{window_id}:{user_id}:{agent}`, deduped by
  `core-api`. A replayed window republishes the same key. ProactiveAgent alerts use
  `window_id = proactive:{alert_type}:{subject}`, so a replayed trigger does too.
- `usage-events`: `usage:{org_id}:agent_token:{occurred_at}:{call_id}`, one per real model
  call. A replay makes real calls again, which are real cost, so they are metered again.
- `agent-requests` / `agent-responses`: `areq:{correlation_id}` / `aresp:{correlation_id}`,
  partition key `org_id`. Both are deadline-bound (about 1.5 s): nobody replays them.
- `agent-stream` deltas are throwaway (shed first under backpressure); `notifications`
  carry `notif:{source_event_id}:{channel}:{user_id}`.

Partition keys: `org_id` everywhere except `agent-stream`, which is `{org_id}:{meeting_id}`
(ADR-0017).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from libs.shared.events import EventEnvelope
from libs.shared.metrics import AGENT_TIME_TO_CARD
from libs.shared.state import AgentResult, TranscriptWindow

logger = logging.getLogger(__name__)

PRODUCER = "agent-runtime-service"
MAX_TIME_TO_CARD_S = 600.0  # a replay of old finals is not a latency


def top_item(result: AgentResult) -> dict[str, Any]:
    """The item a result's `content` is about: the note's own payload, or the candidate /
    suggestion matching the content (else the most confident). Its moment fields are copied to
    the record's top level for consumers that do not read `payload_json`."""
    payload = result.payload or {}
    for key, text_key in (("candidates", "content"), ("suggestions", "question")):
        items = [i for i in payload.get(key) or [] if isinstance(i, dict)]
        if items:
            same = [i for i in items if i.get(text_key) == result.content]
            return (same or sorted(items, key=lambda i: -float(i.get("confidence") or 0.0)))[0]
    return payload


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


class EmitError(RuntimeError):
    """A produced message was not acknowledged. The window must not be committed."""


class Emitter:
    """`on_error`: when given, delivery errors go to it instead of the buffer `flush` raises
    from. The A2A emitter is never flushed (its traffic is deadline-bound and not tied to a
    window commit), so it logs instead of buffering errors forever."""

    def __init__(
        self, producer: Any, codec: Any, *, on_error: Callable[[str], None] | None = None
    ) -> None:
        self._producer = producer
        self._codec = codec
        self._errors: list[str] = []
        self._on_error = on_error

    def _delivery_error(self, error: str) -> None:
        if self._on_error is not None:
            self._on_error(error)
        else:
            self._errors.append(error)

    def _produce(self, envelope: EventEnvelope, key: str) -> None:
        value = self._codec.serialize_for(envelope.topic, envelope.to_value())

        def on_delivery(err: Any, _msg: Any) -> None:
            if err is not None:
                self._delivery_error(f"{envelope.topic}: {err}")

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
        await self.agent_result(
            org_id=window.org_id,
            meeting_id=window.meeting_id,
            window_id=result.window_id or window.window_id,
            user_id=user_id,
            agent=agent,
            result=result,
            window=window,
        )
        if window.last_chunk_at:
            # speech-to-card: the window's last final was produced -> this result is out
            latency = time.time() - window.last_chunk_at
            if 0.0 <= latency <= MAX_TIME_TO_CARD_S:
                AGENT_TIME_TO_CARD.labels(agent, str(window.org_id)).observe(latency)

    async def agent_result(
        self,
        *,
        org_id: UUID,
        meeting_id: UUID,
        window_id: str,
        user_id: UUID,
        agent: str,
        result: AgentResult,
        window: TranscriptWindow | None = None,
    ) -> None:
        """One `agent-results` record. `result()` is this for a window; ProactiveAgent's
        event-driven alerts (no window) call it directly.

        Polish (one card per moment): the optional top-level fields `moment_id`, `why`,
        `updates_candidate_id` (the top item's, `top_item`) and `window_start_ms` /
        `window_end_ms` (meeting time) ride beside `payload_json`, where every item keeps its
        own. A Schema Registry still on the previous `agent-results` version simply drops them
        (the serializer writes the registered fields only)."""
        top = top_item(result)
        env = EventEnvelope(
            topic="agent-results",
            org_id=str(org_id),
            producer=PRODUCER,
            idempotency_key=f"agentresult:{meeting_id}:{window_id}:{user_id}:{agent}",
            payload={
                "meeting_id": str(meeting_id),
                "window_id": window_id,
                "user_id": str(user_id),
                "agent": agent,
                "kind": result.kind,
                "relevance_score": float(result.relevance_score),
                "confidence": float(result.confidence),
                "content": result.content,
                "payload_json": json.dumps(result.payload),
                "source_chunk_ids": list(result.source_chunk_ids),
                "moment_id": _opt_str(top.get("moment_id")),
                "why": _opt_str(top.get("why")),
                "updates_candidate_id": _opt_str(top.get("updates_candidate_id")),
                "window_start_ms": int(window.start_ms) if window is not None else None,
                "window_end_ms": int(window.end_ms) if window is not None else None,
            },
        )
        self._produce(env, key=str(org_id))

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

    async def agent_request(
        self,
        *,
        org_id: UUID,
        requester_agent_id: str,
        responder_target: str,
        query: str,
        deadline_us: int,
        correlation_id: str,
    ) -> None:
        env = EventEnvelope(
            topic="agent-requests",
            org_id=str(org_id),
            producer=PRODUCER,
            idempotency_key=f"areq:{correlation_id}",
            correlation_id=correlation_id,
            payload={
                "requester_agent_id": requester_agent_id,
                "responder_target": responder_target,
                "query": query,
                "deadline": int(deadline_us),
                "correlation_id": correlation_id,
            },
        )
        self._produce(env, key=str(org_id))

    async def agent_response(
        self,
        *,
        org_id: UUID,
        correlation_id: str,
        responder_agent_id: str,
        result: str,
        success: bool,
        latency_ms: int,
    ) -> None:
        env = EventEnvelope(
            topic="agent-responses",
            org_id=str(org_id),
            producer=PRODUCER,
            idempotency_key=f"aresp:{correlation_id}",
            correlation_id=correlation_id,
            payload={
                "correlation_id": correlation_id,
                "responder_agent_id": responder_agent_id,
                "result": result,
                "success": success,
                "latency_ms": int(latency_ms),
            },
        )
        self._produce(env, key=str(org_id))

    def dead_letter(
        self,
        *,
        topic: str,
        dlq_topic: str,
        consumer_group: str,
        partition: int,
        offset: int,
        org_id: str | None,
        key_org: str | None,
        idempotency_key: str,
        failure_class: str,
        error: str,
        raw_value: bytes,
    ) -> None:
        """A `DlqEnvelope`-shaped JSON dead letter, the same shape `libs/shared/consumer.py`
        writes (not Avro: the message is here because it failed to conform). `key_org` is the
        partition key: the DERIVED org when there is one, never the untrusted envelope org.
        Never flushed here: this runs on the event loop, and a blocking flush would stall every
        A2A round trip; `poll(0)` serves delivery callbacks without waiting."""
        envelope = {
            "original_topic": topic,
            "original_partition": partition,
            "original_offset": offset,
            "org_id": org_id or "unknown",
            "idempotency_key": idempotency_key,
            "consumer_group": consumer_group,
            "failure_class": failure_class,
            "error": error,
            "attempts": 1,
            "first_seen_at": int(time.time() * 1_000_000),
            "raw_value_b64": base64.b64encode(raw_value).decode("ascii"),
        }
        self._producer.produce(
            dlq_topic,
            key=key_org.encode() if key_org else None,
            value=json.dumps(envelope).encode(),
            on_delivery=self._dlq_delivery,
        )
        self._producer.poll(0)

    def _dlq_delivery(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self._delivery_error(f"dlq: {err}")
