"""The agent runtime's input loop: `transcript-chunks` + `meeting-room-events` in, agent output
out. Run: `uv run python -m services.agent_runtime_service.consumer`.

Delivery (at-least-once, zero Postgres):
1. Each message's (topic, partition, offset) is tracked as pending.
2. A `transcript-chunks` final joins its meeting's open window and stays pending until that
   window is processed; every other message is done as soon as it is applied.
3. When windows close, they are run through the agents, the emitter is FLUSHED (every
   produced message acknowledged), and only then are their offsets marked done.
4. The committed offset per partition is the lowest still-pending offset, so a crash replays
   from the oldest unprocessed window, never past it.

A malformed message (no tenant, not a UUID meeting, not a final) is logged and skipped: there
is nothing durable to protect here, and the producer's own DLQ handling is where poison is
kept. A transient failure (Redpanda unacknowledged, an unexpected error in an agent) stops the
loop without committing, so the process restarts from the last safe offset.

Windows are processed one at a time on this loop. Within a window, users run concurrently.
That is enough for this project's scale; partitions are the unit of horizontal scale.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.windowing import Chunk, ClosedWindow, Windower

logger = logging.getLogger(__name__)

GROUP = "agent-runtime-service"
TOPICS = ["transcript-chunks", "meeting-room-events"]
PROMPTS_DIR = Path(__file__).parent / "prompts"


class OffsetTracker:
    """Lowest-pending-offset commit tracking per (topic, partition)."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, int], set[int]] = defaultdict(set)
        self._high: dict[tuple[str, int], int] = {}

    def received(self, topic: str, partition: int, offset: int) -> None:
        tp = (topic, partition)
        self._pending[tp].add(offset)
        self._high[tp] = max(self._high.get(tp, -1), offset)

    def done(self, topic: str, partition: int, offset: int) -> None:
        self._pending[(topic, partition)].discard(offset)

    def committable(self) -> dict[tuple[str, int], int]:
        """Next offset to commit (Kafka semantics: the first offset NOT yet processed)."""
        return {
            tp: (min(pending) if pending else self._high[tp] + 1)
            for tp, pending in self._pending.items()
            if tp in self._high
        }


def parse_chunk(value: dict[str, Any], partition: int, offset: int) -> Chunk | None:
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    try:
        meeting_id = UUID(str(value.get("meeting_id")))
    except ValueError:
        return None
    if org_id is None or not value.get("is_final", True):
        return None
    return Chunk(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=str(value["chunk_id"]),
        speaker_id=str(value["speaker_id"]),
        start_ms=int(value["start_ms"]),
        end_ms=int(value["end_ms"]),
        text=str(value["text"]),
        partition=partition,
        offset=offset,
    )


class Loop:
    """The testable core: decoded messages in, windows processed, offsets tracked."""

    def __init__(self, runtime: AgentRuntime, emitter: Any, windower: Windower) -> None:
        self.runtime = runtime
        self.emitter = emitter
        self.windower = windower
        self.offsets = OffsetTracker()

    async def handle(self, topic: str, partition: int, offset: int, value: dict[str, Any]) -> None:
        self.offsets.received(topic, partition, offset)
        closed: list[ClosedWindow] = []
        if topic == "transcript-chunks":
            chunk = parse_chunk(value, partition, offset)
            if chunk is None:
                logger.warning("transcript-chunks: skipping malformed message at %s", offset)
                self.offsets.done(topic, partition, offset)
            else:
                closed = self.windower.add(chunk)
        elif topic == "meeting-room-events":
            closed = await self._room_event(value)
            self.offsets.done(topic, partition, offset)
        await self.run_closed(closed)

    async def _room_event(self, value: dict[str, Any]) -> list[ClosedWindow]:
        org_id = parse_org_id((value.get("header") or {}).get("org_id"))
        try:
            meeting_id = UUID(str(value.get("meeting_id")))
        except ValueError:
            return []
        if org_id is None:
            return []
        event = str(value.get("event_type"))
        await self.runtime.on_room_event(org_id, meeting_id, event, value.get("participant_id"))
        if event == "left" and not self.runtime.roster(org_id, meeting_id):
            closed = self.windower.close_meeting(org_id, meeting_id)
            await self.run_closed(closed)
            await self.runtime.end_meeting(org_id, meeting_id)
        return []

    async def tick(self) -> None:
        """Close windows of meetings that went quiet."""
        await self.run_closed(self.windower.expire_idle())

    async def run_closed(self, closed: list[ClosedWindow]) -> None:
        for cw in closed:
            await self.runtime.process_window(cw.window)
        if closed:
            self.emitter.flush()  # raises EmitError: nothing below runs, nothing is committed
            for cw in closed:
                for partition, offset in cw.offsets:
                    self.offsets.done("transcript-chunks", partition, offset)


async def run() -> None:  # pragma: no cover - the confluent-kafka harness
    from confluent_kafka import Consumer, Producer, TopicPartition
    from redis.asyncio import from_url

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.metrics import serve_metrics, set_service
    from libs.shared.prompts import PromptLoader
    from libs.shared.router import LiteLLMRouter
    from libs.shared.user_context import UserContextClient
    from services.agent_runtime_service.agents import registered_agents
    from services.agent_runtime_service.emitter import Emitter

    logging.basicConfig(level=logging.INFO)
    set_service(GROUP)
    serve_metrics()
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    codec = AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    producer = Producer(
        {"bootstrap.servers": brokers, "acks": "all", "enable.idempotence": True, "linger.ms": 5}
    )
    emitter = Emitter(producer, codec)
    redis = from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
    router = LiteLLMRouter(BudgetGate(redis, BillingRefresher()), emitter)
    loader = PromptLoader(PROMPTS_DIR)
    contexts = UserContextClient()
    runtime = AgentRuntime(
        sink=emitter,
        user_context=contexts.get,
        agents=registered_agents(loader, router),
        max_concurrency=int(os.environ.get("AGENT_MAX_CONCURRENCY", "8")),
    )
    loop = Loop(runtime, emitter, Windower(idle_s=float(os.environ.get("WINDOW_IDLE_S", "15"))))
    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": GROUP,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe(TOPICS)
    committed: dict[tuple[str, int], int] = {}
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is not None and not msg.error():
                topic, partition, offset = msg.topic(), msg.partition(), msg.offset()
                raw = msg.value()
                if topic is None or partition is None or offset is None or raw is None:
                    continue
                try:
                    value = codec.deserialize_for(topic, raw)
                except Exception:
                    logger.exception("undecodable message on %s@%s; skipped", topic, offset)
                    loop.offsets.received(topic, partition, offset)
                    loop.offsets.done(topic, partition, offset)
                else:
                    await loop.handle(topic, partition, offset, value)
            await loop.tick()
            due = {tp: o for tp, o in loop.offsets.committable().items() if committed.get(tp) != o}
            if due:
                consumer.commit(
                    offsets=[TopicPartition(t, p, o) for (t, p), o in due.items()],
                    asynchronous=False,
                )
                committed.update(due)
    finally:
        consumer.close()
        await contexts.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run())
