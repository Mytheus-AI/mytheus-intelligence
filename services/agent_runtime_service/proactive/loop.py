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

"Decisions that know why" (`decision_why.py`, a4 contract section 1): after the conflict check,
a confirmed decision also goes to `DecisionWhy.on_decision` (the `decision_why` record and the
assumption watcher's trigger A). Its records ride the same producer and the same flush. It runs
in its own guard: a failure there never stops the conflict alerts, and a failing conflict check
does not stop it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol
from uuid import UUID

from libs.shared.state import AgentResult
from services.agent_runtime_service import decision_why
from services.agent_runtime_service.agents.proactive_agent import (
    AGENT,
    Alert,
    ConfirmedDecision,
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

    def __init__(
        self, agent: ProactiveAgent, sink: AlertSink, why: decision_why.DecisionWhy | None = None
    ) -> None:
        self.agent = agent
        self.sink = sink
        self.why = why

    async def handle(self, topic: str, value: dict[str, Any]) -> int:
        alerts: list[Alert] = []
        org: UUID | None = None
        decision: ConfirmedDecision | None = None
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
            alerts = []
        records = await self._why(decision, value) if decision is not None else []
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
        emitted += await decision_why.emit(self.sink, records)
        if emitted:
            self.sink.flush()  # raises EmitError: the caller must not commit
        return emitted

    async def _why(
        self, decision: ConfirmedDecision, value: dict[str, Any]
    ) -> list[decision_why.Record]:
        if self.why is None:
            return []
        try:
            return await self.why.on_decision(decision, decision_why.event_date(value))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never stops the conflict alerts
            logger.exception("decision why failed on a decisions record; skipped")
            return []


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
