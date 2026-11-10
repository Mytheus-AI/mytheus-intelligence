"""A confirmed action with no meeting (a personal task created outside any meeting).

core-api may now publish a confirmed `actions` event whose `meeting_id` is the nil uuid. It
means "no meeting": knowledge-graph-service writes the `Action` node and its `ASSIGNED_TO` edge
and no `Meeting` node or `PRODUCED` edge, and nothing crashes or dead-letters (the graph
consumer, the rebuild, and ProactiveAgent's loop, which skips it: its alerts would have no
meeting to be delivered into).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from libs.shared.consumer import ConsumedEvent
from services.agent_runtime_service.agents.proactive_agent import parse_action
from services.agent_runtime_service.proactive.loop import ProactiveLoop
from services.knowledge_graph_service.confirmed import (
    NIL_MEETING,
    ConfirmedAction,
    GraphFact,
    apply_action,
)
from services.knowledge_graph_service.facts_consumer import ConfirmedFactsConsumer, action_from
from services.knowledge_graph_service.rebuild import parse_row

ORG = uuid.uuid4()
NIL = "00000000-0000-0000-0000-000000000000"


class Cursor:
    """An AGE connection that records every Cypher call and answers each with one row."""

    def __init__(self) -> None:
        self.cypher: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if "cypher(" in query:
            self.cypher.append((query, json.loads(args[0])))
        return [("{}",)]


def evt(meeting: str, assignee: uuid.UUID | None) -> ConsumedEvent:
    action = uuid.uuid4()
    return ConsumedEvent(
        topic="actions",
        org_id=ORG,
        idempotency_key=f"action:{action}:confirmed",
        header={"occurred_at": 1_700_000_000_000_000},
        body={
            "action_id": str(action),
            "meeting_id": meeting,
            "content": "Renew the domain",
            "assignee_id": str(assignee) if assignee else None,
            "due_date": 20_420,
            "confidence": 0.9,
            "status": "confirmed",
        },
        partition=0,
        offset=0,
    )


def test_the_nil_meeting_reads_as_no_meeting_everywhere_it_is_parsed():
    assert action_from(evt(NIL, None)).meeting_id is None
    real = uuid.uuid4()
    assert action_from(evt(str(real), None)).meeting_id == real
    row = {
        "kind": "action",
        "status": "confirmed",
        "action_id": str(uuid.uuid4()),
        "content": "x",
        "confidence": 0.9,
        "confirmed_at": "2026-11-11T00:00:00+00:00",
    }
    for meeting in (NIL, None):
        parsed = parse_row({**row, "meeting_id": meeting})
        assert isinstance(parsed, ConfirmedAction) and parsed.meeting_id is None
    assert NIL_MEETING == uuid.UUID(NIL)


async def test_a_personal_task_writes_the_action_and_its_assignee_but_no_meeting():
    cur = Cursor()
    person = uuid.uuid4()
    action = action_from(evt(NIL, person))
    facts = await apply_action(cur, ORG, action)  # type: ignore[arg-type]
    assert facts == [GraphFact("edge_created", "ASSIGNED_TO", str(action.action_id), str(person))]
    text = " ".join(q for q, _ in cur.cypher)
    assert "Meeting" not in text and "PRODUCED" not in text
    assert all("meeting_id" not in params for _q, params in cur.cypher)
    assert all(params["org_id"] == str(ORG) for _q, params in cur.cypher)


async def test_a_meeting_action_still_writes_produced():
    cur = Cursor()
    meeting = uuid.uuid4()
    facts = await apply_action(cur, ORG, action_from(evt(str(meeting), None)))  # type: ignore[arg-type]
    assert [f.edge_label for f in facts] == ["PRODUCED"] and facts[0].to_id == str(meeting)


async def test_the_graph_consumer_applies_it_without_raising_or_dead_lettering(monkeypatch):
    cur = Cursor()

    @asynccontextmanager
    async def connect() -> AsyncIterator[Cursor]:
        yield cur

    consumer = ConfirmedFactsConsumer(
        graph_connect=connect,  # type: ignore[arg-type]
        session_factory=None,  # type: ignore[arg-type]
        redis=None,  # type: ignore[arg-type]
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,
    )
    emitted: list[dict[str, Any]] = []

    async def outbox(_session: Any, _evt: Any, **kw: Any) -> None:
        emitted.append(kw)

    monkeypatch.setattr(consumer, "emit_outbox", outbox)
    person = uuid.uuid4()
    await consumer.apply(evt(NIL, person), None)  # type: ignore[arg-type]
    assert [e["payload"]["edge_label"] for e in emitted] == ["ASSIGNED_TO"]
    await consumer.apply(evt(NIL, None), None)  # type: ignore[arg-type]
    assert len(emitted) == 1  # no assignee, no meeting: the node only, no edge event


async def test_the_proactive_loop_skips_a_personal_task_quietly():
    value = {
        "header": {"org_id": str(ORG)},
        "action_id": str(uuid.uuid4()),
        "meeting_id": NIL,
        "content": "Renew the domain",
    }
    assert parse_action(value) is None

    class Agent:
        async def check_action(self, _a: Any) -> list[Any]:
            raise AssertionError("never called for a personal task")

    class AlertSink:
        def flush(self) -> None:
            raise AssertionError("nothing to flush")

    loop = ProactiveLoop(Agent(), AlertSink())  # type: ignore[arg-type]
    assert await loop.handle("actions", value) == 0
