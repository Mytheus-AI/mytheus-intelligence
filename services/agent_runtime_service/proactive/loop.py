"""The `decisions` / `actions` consumer for `ProactiveAgent` (P11 contract section 2).

Consumer group `agent-runtime-proactive`, topics `decisions` and `actions` (the confirmed facts
core-api publishes; the frozen `Decision` / `Action` Avro records). It runs beside the window
loop in `consumer.run`, on the same event loop, with its own consumer and its own producer, so a
delivery failure here never fails a window commit and vice versa.

Per message, at-least-once and in order per partition:
1. decode (an undecodable or malformed record is logged and skipped: nothing durable to protect,
   and core-api's own DLQ is where poison is kept);
2. `ProactiveAgent.check_decision` (conflict) or `.check_action` (workload, dependency);
3. every alert result goes to `agent-results` through the producer, which is FLUSHED;
4. only then is the offset committed. A flush failure raises `EmitError` and the loop stops
   without committing, so the process restarts from that message. The replay re-emits the same
   alert ids and idempotency keys (deterministic), which downstream dedupes.

An exception inside a check (a bug, a provider outage) is logged and the message is committed:
re-running it on every restart would re-bill model calls and block the partition.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol
from uuid import UUID

from libs.shared.state import AgentResult
from services.agent_runtime_service.agents.proactive_agent import (
    AGENT,
    Alert,
    ProactiveAgent,
    parse_action,
    parse_decision,
)

logger = logging.getLogger(__name__)

GROUP = "agent-runtime-proactive"
TOPICS = ["decisions", "actions"]


class AlertSink(Protocol):
    async def agent_result(
        self,
        *,
        org_id: UUID,
        meeting_id: UUID,
        window_id: str,
        user_id: UUID,
        agent: str,
        result: AgentResult,
    ) -> None: ...

    def flush(self, timeout_s: float = 10.0) -> None: ...


class ProactiveLoop:
    """The testable core: one decoded record in, alerts out, flushed. Returns the number of
    `agent-results` records emitted."""

    def __init__(self, agent: ProactiveAgent, sink: AlertSink) -> None:
        self.agent = agent
        self.sink = sink

    async def handle(self, topic: str, value: dict[str, Any]) -> int:
        alerts: list[Alert] = []
        org: UUID | None = None
        try:
            if topic == "decisions":
                decision = parse_decision(value)
                if decision is None:
                    logger.warning("decisions: skipping a malformed record")
                    return 0
                org = decision.org_id
                alerts = await self.agent.check_decision(decision)
            elif topic == "actions":
                action = parse_action(value)
                if action is None:
                    logger.warning("actions: skipping a malformed record")
                    return 0
                org = action.org_id
                alerts = await self.agent.check_action(action)
            else:
                return 0
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - see module docstring: skip, never crash-loop
            logger.exception("ProactiveAgent failed on a %s record; skipped", topic)
            return 0
        emitted = 0
        for alert in alerts:
            assert org is not None
            for user, result in alert.results():
                await self.sink.agent_result(
                    org_id=org,
                    meeting_id=alert.meeting_id,
                    window_id=alert.window_id,
                    user_id=user,
                    agent=AGENT,
                    result=result,
                )
                emitted += 1
        if emitted:
            self.sink.flush()  # raises EmitError: the caller must not commit
        return emitted


async def run_proactive_loop(
    consumer: Any, codec: Any, loop: ProactiveLoop, stop: asyncio.Event
) -> None:  # the confluent-kafka harness; covered by the live broker test
    from confluent_kafka import TopicPartition

    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, 1.0)
        if msg is None or msg.error():
            continue
        topic, partition, offset, raw = msg.topic(), msg.partition(), msg.offset(), msg.value()
        if topic is None or partition is None or offset is None:
            continue
        if raw is not None:
            try:
                value = codec.deserialize_for(topic, raw)
            except Exception:  # noqa: BLE001
                logger.exception("undecodable message on %s@%s; skipped", topic, offset)
            else:
                await loop.handle(topic, value)  # EmitError propagates: no commit
        await asyncio.to_thread(
            consumer.commit,
            offsets=[TopicPartition(topic, partition, offset + 1)],
            asynchronous=False,
        )


def consumer_config(brokers: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": GROUP,
        "enable.auto.commit": False,
        # A new group starts at the end: replaying every historic decision on first deploy
        # would flood users with stale alerts and bill a model call per old decision.
        "auto.offset.reset": "latest",
    }
