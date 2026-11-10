"""`ConfirmedFactsConsumer`: the edge-on-confirmation writer (Plan 7-9 section 3.7 step 2).

Subscribes to:
- `decisions` (producer `core-api`, on confirm only): `Decision` -PRODUCED-> `Meeting`, and
  -FOLLOWS_FROM-> the decision named in `supersedes` (ADR-0031 D2).
- `actions` (producer `core-api`, on confirm only): `Action` -PRODUCED-> `Meeting`, and
  -ASSIGNED_TO-> `Person` when `assignee_id` is set.
- `meeting-room-events` `joined` (producer `meeting-service`): `Person` -ATTENDED-> `Meeting`
  when `participant_id` is a user id (ADR-0031 D7). Non-UUID identities (the audio tap, egress)
  and `left` / `recording_ready` write nothing.

`BaseConsumer` has already rejected a missing or malformed header `org_id` to the DLQ and
deduped redeliveries (Redis cache, then the durable inbox), so a redelivered event never reaches
`apply`. The graph writes themselves are MERGEs, so even a replay that slipped both guards is a
no-op. `org_id` is always the envelope header's, never a payload field.

Atomicity, as in `consumer.py` (Topic seeding): the graph is a different database from the
inbox, so the graph transaction commits first, then the inbox row and the `graph-events` outbox
rows commit together. A crash between the two redelivers the event and re-runs MERGEs that
create nothing. The reverse order could lose an edge.

`graph-events` carries one record per edge written, keyed `graph:{edge_label}:{from}:{to}`: the
outbox's unique `(topic, idempotency_key)` makes a re-application emit nothing new.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, date, datetime
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.age import AgeConnection
from libs.shared.consumer import BaseConsumer, ConsumedEvent
from services.knowledge_graph_service.confirmed import (
    ConfirmedAction,
    ConfirmedDecision,
    GraphFact,
    apply_action,
    apply_attended,
    apply_decision,
    meeting_or_none,
)

logger = logging.getLogger(__name__)

GraphConnect = Callable[[], AbstractAsyncContextManager[AgeConnection]]


def _uuid(value: Any) -> UUID | None:
    if value is None or value == "":
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError:
        return None


def _required_uuid(body: dict[str, Any], name: str) -> UUID:
    value = _uuid(body.get(name))
    if value is None:
        raise ValueError(f"{name} is not a UUID")
    return value


def occurred_at_iso(header: dict[str, Any]) -> str:
    """The header's `occurred_at` as ISO-8601 UTC. The deserializer hands back a datetime; a
    hand-built event may carry the raw timestamp-micros int."""
    value = header.get("occurred_at")
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
    elif isinstance(value, int):
        dt = datetime.fromtimestamp(value / 1_000_000, tz=UTC)
    else:
        dt = datetime.now(UTC)
    return dt.astimezone(UTC).isoformat()


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, int):  # Avro `date`: days since the epoch
        return date.fromordinal(date(1970, 1, 1).toordinal() + value).isoformat()
    return str(value)


def decision_from(evt: ConsumedEvent) -> ConfirmedDecision:
    b = evt.body
    return ConfirmedDecision(
        decision_id=_required_uuid(b, "decision_id"),
        meeting_id=_required_uuid(b, "meeting_id"),
        content=str(b["content"]),
        confidence=float(b["confidence"]),
        confirmed_at=occurred_at_iso(evt.header),
        supersedes=_uuid(b.get("supersedes")),
    )


def action_from(evt: ConsumedEvent) -> ConfirmedAction:
    b = evt.body
    return ConfirmedAction(
        action_id=_required_uuid(b, "action_id"),
        # the nil uuid is a personal task created outside any meeting: no PRODUCED edge
        meeting_id=meeting_or_none(_required_uuid(b, "meeting_id")),
        content=str(b["content"]),
        confidence=float(b["confidence"]),
        confirmed_at=occurred_at_iso(evt.header),
        assignee_id=_uuid(b.get("assignee_id")),
        due_date=_iso_date(b.get("due_date")),
    )


class ConfirmedFactsConsumer(BaseConsumer):
    group: ClassVar[str] = "knowledge-graph-service.confirmed-facts"
    topics: ClassVar[list[str]] = ["decisions", "actions", "meeting-room-events"]
    producer_name: ClassVar[str] = "knowledge-graph-service"

    def __init__(self, *, graph_connect: GraphConnect, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph_connect = graph_connect

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        try:
            facts = await self.write_graph(evt)
        except (KeyError, ValueError) as exc:
            # A payload that cannot describe a graph fact (a non-UUID id) is poison: raising
            # would redeliver it forever and block the partition. Record it consumed, write
            # nothing. Transient faults (AGE unreachable) are not ValueError and still raise.
            logger.error("kg: unusable %s %s: %s", evt.topic, evt.idempotency_key, exc)
            return
        for fact in facts:
            await self.emit_outbox(
                session,
                evt,
                topic="graph-events",
                idem=f"graph:{fact.edge_label}:{fact.from_id}:{fact.to_id}",
                payload={
                    "event_type": fact.event_type,
                    "edge_label": fact.edge_label,
                    "from_id": fact.from_id,
                    "to_id": fact.to_id,
                },
            )

    async def write_graph(self, evt: ConsumedEvent) -> list[GraphFact]:
        if evt.topic == "decisions":
            if evt.body.get("status") != "confirmed":  # unrepresentable today; fail closed anyway
                logger.error("kg: refusing a non-confirmed decision %s", evt.idempotency_key)
                return []
            decision = decision_from(evt)
            async with self._graph_connect() as conn:
                return await apply_decision(conn, evt.org_id, decision)
        if evt.topic == "actions":
            if evt.body.get("status") != "confirmed":
                logger.error("kg: refusing a non-confirmed action %s", evt.idempotency_key)
                return []
            action = action_from(evt)
            async with self._graph_connect() as conn:
                return await apply_action(conn, evt.org_id, action)
        if evt.topic == "meeting-room-events":
            if evt.body.get("event_type") != "joined":
                return []
            user_id = _uuid(evt.body.get("participant_id"))
            meeting_id = _uuid(evt.body.get("meeting_id"))
            if user_id is None or meeting_id is None:
                return []  # not a user identity (audio tap, egress): no Person to attend
            async with self._graph_connect() as conn:
                return await apply_attended(
                    conn, evt.org_id, user_id=user_id, meeting_id=meeting_id
                )
        logger.warning("kg: unexpected topic %s", evt.topic)
        return []
