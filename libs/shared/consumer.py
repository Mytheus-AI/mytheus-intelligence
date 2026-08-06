"""`BaseConsumer` — the canonical at-least-once consumer for this repo.

`embedding-service`'s `EmbeddingConsumer` (Plan 5-6 task 7) is the FIRST event-sourced writer
in `mytheus-intelligence` and the flagship instance of this class; tasks 8-10
(`ingestion-service`, `search-service`, `agent-runtime-service`) subclass it rather than
re-implement dedupe/DLQ/offset handling. Mirrors the delivery contract documented in
`mytheus-events` CLAUDE.md ("The delivery contract") and `mytheus-backend`'s inbox/outbox
pattern, adapted to a base class because this repo (unlike the backend) has several consumers
sharing the same dedupe/DLQ machinery from day one.

Contract a subclass implements:
  group: ClassVar[str]              # consumer group name, e.g. "embedding-service.embed-on-ingest"
  topics: ClassVar[list[str]]       # topics this group subscribes to
  async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
      ...                           # the business effect; call `self.emit_outbox(...)` to ack

Delivery guarantees `process_raw` enforces before `apply` ever runs:
  1. Deserialize failure -> DLQ (`failure_class=DESERIALIZE`), never raises into the caller.
  2. Missing or unparseable `org_id` -> DLQ (`failure_class=BUSINESS_INVALID`). NEVER guessed,
     NEVER defaulted — this is Risk checkpoint R6 from the task brief.
  3. Redis dedupe cache hit, or durable inbox hit (cache-evicted case) -> short-circuit as a
     duplicate. `apply` does not run again.
  4. `apply` and the inbox `record_consumed` insert run in ONE transaction, transaction-locally
     scoped by `SET app.current_org` (this repo's own `intelligence.outbox`/`intelligence.inbox`
     tables carry the same `tenant_isolation` RLS policy shape as the rest of the program), so a
     crash between "effect applied" and "fact recorded" cannot happen — they commit together.
  5. Any OTHER exception from `apply` (a transient failure: Qdrant unreachable, DB deadlock, ...)
     propagates to the caller uncaught. The caller (the real Kafka wiring, `run_forever` below)
     does NOT commit the offset in that case, so the broker redelivers — never block or skip a
     partition on a transient fault; only a classified poison message goes to DLQ.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from uuid import UUID

from confluent_kafka import Consumer, Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from libs.shared.avro_codec import AvroCodec
from libs.shared.events import EventEnvelope
from libs.shared.inbox import already_recorded, is_duplicate, record_consumed
from libs.shared.outbox import enqueue as outbox_enqueue
from libs.shared.tenancy import parse_org_id, set_rls

logger = logging.getLogger(__name__)

ProcessOutcome = Literal["applied", "duplicate", "dlq"]


@dataclass
class ConsumedEvent:
    topic: str
    org_id: UUID
    idempotency_key: str
    header: dict[str, Any]
    body: dict[str, Any]  # decoded Avro value with `header` removed
    partition: int
    offset: int


class BaseConsumer(ABC):
    group: ClassVar[str]
    topics: ClassVar[list[str]]
    producer_name: ClassVar[str] = (
        "embedding-service"  # kebab-case owning service, for EventEnvelope
    )

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        # Not `Redis[bytes]`: this installed redis-py version's stubs do not declare `Redis`
        # as Generic, so the subscript fails mypy strict even though it works at runtime
        # (mytheus-backend's own `libs/shared/inbox.py` uses `Redis[bytes]` against whatever
        # redis-py version IT resolved; this repo's `pyproject.toml` only pins `>=5`, and pip
        # resolved a newer one here).
        redis: Redis,
        codec: AvroCodec,
        dlq_producer: Producer,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis
        self._codec = codec
        self._dlq_producer = dlq_producer

    @abstractmethod
    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None: ...

    async def emit_outbox(
        self,
        session: AsyncSession,
        evt: ConsumedEvent,
        *,
        topic: str,
        idem: str,
        payload: dict[str, Any],
    ) -> None:
        """The one sanctioned way a subclass's `apply` acknowledges its work. Same transaction
        as the inbox insert (both run inside `process_raw`'s `session.begin()`)."""
        # The decoded Avro header carries native Python types for logicalType fields (`UUID`
        # for `event_id`/`org_id`, `datetime` for `occurred_at`) — fastavro's deserializer
        # projects logical types, it does not keep them as wire-format strings/ints. Stringify
        # here so `EventEnvelope.to_value()` stays JSON-serializable for the outbox's `payload`
        # JSONB column.
        correlation_id = evt.header.get("correlation_id")
        causation_id = evt.header.get("event_id")
        envelope = EventEnvelope(
            topic=topic,
            payload=payload,
            org_id=str(evt.org_id),
            producer=self.producer_name,
            idempotency_key=idem,
            correlation_id=str(correlation_id) if correlation_id is not None else None,
            causation_id=str(causation_id) if causation_id is not None else None,
        )
        await outbox_enqueue(session, envelope)

    async def process_raw(
        self,
        *,
        topic: str,
        raw_value: bytes,
        partition: int,
        offset: int,
    ) -> ProcessOutcome:
        try:
            decoded = self._codec.deserialize_for(topic, raw_value)
        except Exception as exc:  # noqa: BLE001 - any decode failure is poison, not fatal
            logger.error("consumer %s: deserialize failure on %s: %s", self.group, topic, exc)
            self._to_dlq(
                topic=topic,
                partition=partition,
                offset=offset,
                org_id="unknown",
                idempotency_key="unknown",
                failure_class="DESERIALIZE",
                error=str(exc),
                raw_value=raw_value,
            )
            return "dlq"

        header = dict(decoded["header"])
        org_id = parse_org_id(header.get("org_id"))
        idem = header.get("idempotency_key")

        if org_id is None or not idem:
            # R6: never guess or default a tenant. Straight to DLQ.
            logger.error(
                "consumer %s: missing/unparseable org_id or idempotency_key on %s offset %d",
                self.group,
                topic,
                offset,
            )
            self._to_dlq(
                topic=topic,
                partition=partition,
                offset=offset,
                org_id=str(header.get("org_id")),
                idempotency_key=str(idem),
                failure_class="BUSINESS_INVALID",
                error="missing or unparseable org_id / idempotency_key in event header",
                raw_value=raw_value,
            )
            return "dlq"

        body = {k: v for k, v in decoded.items() if k != "header"}
        evt = ConsumedEvent(
            topic=topic,
            org_id=org_id,
            idempotency_key=str(idem),
            header=header,
            body=body,
            partition=partition,
            offset=offset,
        )

        if await is_duplicate(self._redis, str(org_id), topic, evt.idempotency_key):
            logger.info("consumer %s: duplicate (redis) %s", self.group, evt.idempotency_key)
            return "duplicate"

        async with self._session_factory() as session, session.begin():
            await set_rls(session, org_id)
            if await already_recorded(session, self.group, str(org_id), evt.idempotency_key):
                logger.info(
                    "consumer %s: duplicate (inbox, cache-evicted) %s",
                    self.group,
                    evt.idempotency_key,
                )
                return "duplicate"
            await self.apply(evt, session)
            await record_consumed(session, self.group, str(org_id), evt.idempotency_key)
        return "applied"

    def _to_dlq(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        org_id: str,
        idempotency_key: str,
        failure_class: str,
        error: str,
        raw_value: bytes,
    ) -> None:
        """Publish a `DlqEnvelope`-shaped JSON message to `{topic}.dlq`. Deliberately not
        Avro-encoded (matches `mytheus-backend`'s relay `_dead_letter`: the message is here
        because it already failed to conform, so re-encoding it would fail the same way; no
        `.dlq` subject is registered in the Schema Registry either, confirming this is the
        intended shape)."""
        envelope = {
            "original_topic": topic,
            "original_partition": partition,
            "original_offset": offset,
            "org_id": org_id,
            "idempotency_key": idempotency_key,
            "consumer_group": self.group,
            "failure_class": failure_class,
            "error": error,
            "attempts": 1,
            "first_seen_at": int(time.time() * 1_000_000),
            "raw_value_b64": base64.b64encode(raw_value).decode("ascii"),
        }
        self._dlq_producer.produce(
            topic=f"{topic}.dlq",
            key=org_id.encode() if org_id != "unknown" else None,
            value=json.dumps(envelope).encode(),
        )
        self._dlq_producer.flush(5)

    def run_forever(self, *, brokers: str, poll_timeout: float = 1.0) -> None:
        """The real confluent-kafka wiring: subscribe, poll, `process_raw`, THEN commit —
        manual offset commit after the effect, never before (mirrors
        `transcription-service`'s `meeting_room_consumer.py` note on why: committing before the
        effect and then crashing silently skips that message forever, since the broker never
        redelivers a committed offset).

        Not exercised in unit tests (it blocks forever); `process_raw` above carries all the
        tested business logic. Kept here, not left unwritten, because the plan calls for the
        embed-on-ingest consumer to actually run, and the wiring around it is a few lines once
        `process_raw` exists.
        """
        consumer = Consumer(
            {
                "bootstrap.servers": brokers,
                "group.id": self.group,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe(self.topics)
        try:
            while True:
                msg = consumer.poll(poll_timeout)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("consumer %s: broker error: %s", self.group, msg.error())
                    continue
                import asyncio

                # confluent-kafka's stubs type these Optional even though a message with no
                # `.error()` always carries them; the `assert`s make that guarantee explicit
                # for mypy rather than silencing the check.
                topic, raw_value = msg.topic(), msg.value()
                partition, offset = msg.partition(), msg.offset()
                assert topic is not None
                assert raw_value is not None
                assert partition is not None
                assert offset is not None

                asyncio.run(
                    self.process_raw(
                        topic=topic, raw_value=raw_value, partition=partition, offset=offset
                    )
                )
                consumer.commit(message=msg, asynchronous=False)
        finally:
            consumer.close()
