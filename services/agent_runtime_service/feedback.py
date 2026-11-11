"""The `agent-feedback` listener (polish program wave 2, agents that learn).

core-api publishes one `agent-feedback` event (outbox, schema `AgentFeedback`) per confirm,
dismiss, edit, ask or save a person makes on something an agent proposed, after recording it and
recalibrating the thresholds. The runtime keeps no learned state of its own (it never writes
Postgres, and an in-process store would vanish with the pod): the event only tells it that
core-api's answers changed, so it drops the org's cached policy and the person's cached guidance
and memory (`LearningClient.invalidate`). The next window fetches the recalibrated threshold,
the new examples and the updated memory instead of waiting out the 60 s cache, which keeps the
live `surfaced` and core-api's reload in step.

Every runtime instance holds its own caches, so every instance must see every event: the group
is per host (`agent-runtime-feedback-{hostname}`, like the A2A response router), offsets start
at `latest` (an old event has nothing left to invalidate) and auto-commit (a lost event costs at
most one cache TTL of staleness). A record that cannot be decoded or names no tenant is skipped.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, Protocol
from uuid import UUID

from libs.shared.tenancy import parse_org_id

logger = logging.getLogger(__name__)

TOPIC = "agent-feedback"


class Invalidator(Protocol):
    def invalidate(self, org_id: UUID, user_id: UUID | None = None) -> None: ...


def feedback_group(hostname: str | None = None) -> str:
    return f"agent-runtime-feedback-{hostname or socket.gethostname()}"


def consumer_config(brokers: str, hostname: str | None = None) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": feedback_group(hostname),
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }


class FeedbackListener:
    def __init__(self, learning: Invalidator) -> None:
        self._learning = learning
        self.applied = 0

    def apply(self, value: dict[str, Any]) -> bool:
        """Drop the caches this feedback makes stale. False for a record naming no tenant."""
        org_id = parse_org_id((value.get("header") or {}).get("org_id"))
        if org_id is None:
            return False
        try:
            user_id: UUID | None = UUID(str(value.get("user_id")))
        except ValueError:
            user_id = None
        self._learning.invalidate(org_id, user_id)
        self.applied += 1
        return True


async def run_feedback_loop(
    consumer: Any,
    codec: Any,
    listener: FeedbackListener,
    stop: asyncio.Event,
    *,
    poll_s: float = 0.5,
) -> None:  # pragma: no cover - the confluent-kafka harness; `apply` is tested
    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, poll_s)
        if msg is None or msg.error() or msg.value() is None:
            continue
        try:
            value = codec.deserialize_for(TOPIC, msg.value())
        except Exception:  # noqa: BLE001 - a stale cache for one TTL beats a dead loop
            logger.warning("agent-feedback: undecodable message skipped")
            continue
        listener.apply(value)
