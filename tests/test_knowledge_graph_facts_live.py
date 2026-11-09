"""Plan 8 knowledge graph, end to end against real infrastructure.

Fixture events are real Avro values encoded against the registered `decisions`, `actions` and
`meeting-room-events` schemas (the frozen `Decision` / `Action`, with ADR-0031's `supersedes`),
fed through the real `ConfirmedFactsConsumer.process_raw`: real Schema Registry decode, real
Redis dedupe, real `intelligence.inbox` / `intelligence.outbox`, real Apache AGE.

Covers the plan 3.7 acceptance checks:
- a confirmed decision / action creates the right nodes and edges (and `graph-events`);
- a redelivered event is a no-op (Redis path and durable-inbox path);
- a cross-tenant traversal returns nothing;
- the lineage template walks a supersession chain, whatever order the events arrive in;
- the graph is rebuildable from confirmed rows (same result as the event-built graph).

Needs the AGE container (`AGE_DATABASE_URL`, default port 5434) plus the shared stack; skips
loudly when AGE is unreachable unless `MYTHEUS_REQUIRE_AGE=1`.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import pytest
from confluent_kafka import Producer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.age import AgeConnection, AgeQueryRejected, agtype_to_int, run_cypher
from libs.shared.inbox import dedupe_key
from services.knowledge_graph_service.confirmed import (
    ConfirmedAction,
    ConfirmedDecision,
    apply_action,
    apply_decision,
    purge_org,
    soft_delete,
)
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.facts_consumer import ConfirmedFactsConsumer
from services.knowledge_graph_service.graph import ensure_graph
from services.knowledge_graph_service.rebuild import parse_row
from services.knowledge_graph_service.templates import run_adhoc, run_template
from tests.conftest import APP_URL, BOOTSTRAP_URL

pytestmark = pytest.mark.asyncio

_DSN = os.environ.get(
    "AGE_DATABASE_URL", "postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph"
)


def _reachable() -> bool:
    u = urlparse(_DSN)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_age() -> None:
    if _reachable():
        return
    msg = (
        f"LIVE AGE TEST NOT RUN: no Apache AGE Postgres at {_DSN}. The Plan 8 edge-on-"
        "confirmation, redelivery, cross-tenant and lineage checks are UNVERIFIED."
    )
    if os.environ.get("MYTHEUS_REQUIRE_AGE") == "1":
        raise RuntimeError(msg)
    pytest.skip(msg)


@asynccontextmanager
async def _graph() -> AsyncIterator[AgeConnection]:
    async with age_connection(_DSN) as conn:
        yield conn


@pytest.fixture()
async def conn() -> AsyncIterator[AgeConnection]:
    async with age_connection(_DSN) as c:
        await ensure_graph(c)
        yield c


@pytest.fixture()
def consumer(migrated_db, redis_client, avro_codec) -> ConfirmedFactsConsumer:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    return ConfirmedFactsConsumer(
        graph_connect=_graph,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=Producer({"bootstrap.servers": "localhost:9092"}),
    )


def _header(org: uuid.UUID, idem: str, producer: str) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "idempotency_key": idem,
        "org_id": str(org),
        "schema_version": 1,
        "producer": producer,
        "occurred_at": int(time.time() * 1_000_000),
        "correlation_id": None,
        "causation_id": None,
    }


def decision_value(
    org: uuid.UUID,
    decision_id: uuid.UUID,
    meeting_id: uuid.UUID,
    *,
    content: str = "Ship the graph on AGE",
    supersedes: uuid.UUID | None = None,
) -> dict[str, Any]:
    return {
        "header": _header(org, f"decision:{decision_id}:confirmed", "core-api"),
        "decision_id": str(decision_id),
        "meeting_id": str(meeting_id),
        "content": content,
        "rationale": "because",
        "confidence": 0.9,
        "status": "confirmed",
        "confirmed_by": str(uuid.uuid4()),
        "supersedes": str(supersedes) if supersedes else None,
    }


def action_value(
    org: uuid.UUID, action_id: uuid.UUID, meeting_id: uuid.UUID, assignee: uuid.UUID | None
) -> dict[str, Any]:
    return {
        "header": _header(org, f"action:{action_id}:confirmed", "core-api"),
        "action_id": str(action_id),
        "meeting_id": str(meeting_id),
        "content": "Write the lineage page",
        "assignee_id": str(assignee) if assignee else None,
        "due_date": 20_420,  # Avro date: days since 1970-01-01
        "confidence": 0.8,
        "status": "confirmed",
    }


def joined_value(org: uuid.UUID, meeting_id: uuid.UUID, participant: str) -> dict[str, Any]:
    return {
        "header": _header(org, f"room:{meeting_id}:{participant}:joined", "meeting-service"),
        "meeting_id": str(meeting_id),
        "room": f"mtg-{meeting_id}",
        "event_type": "joined",
        "participant_id": participant,
    }


async def _feed(consumer: ConfirmedFactsConsumer, topic: str, value: dict[str, Any]) -> str:
    raw = consumer._codec.serialize_for(topic, value)
    return await consumer.process_raw(topic=topic, raw_value=raw, partition=0, offset=1)


async def _count(conn: AgeConnection, org: uuid.UUID, pattern: str) -> int:
    rows = await run_cypher(conn, org, f"MATCH {pattern} RETURN count(*)", columns=("n",))
    return agtype_to_int(rows[0][0])


async def _graph_events(org: uuid.UUID) -> list[tuple[str, str]]:
    engine = create_async_engine(BOOTSTRAP_URL)
    try:
        async with engine.connect() as c:
            rows = await c.execute(
                text(
                    "SELECT payload->>'edge_label', payload->>'event_type' "
                    "FROM intelligence.outbox "
                    "WHERE topic = 'graph-events' AND org_id = :o ORDER BY id"
                ),
                {"o": str(org)},
            )
            return [(r[0], r[1]) for r in rows]
    finally:
        await engine.dispose()


async def test_confirmed_decision_creates_nodes_edges_and_graph_events(consumer, conn):
    org, d, m = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert await _feed(consumer, "decisions", decision_value(org, d, m)) == "applied"

    assert await _count(conn, org, "(:Decision {org_id: $org_id})") == 1
    assert await _count(conn, org, "(:Meeting {org_id: $org_id})") == 1
    produced = "(:Decision {org_id: $org_id})-[:PRODUCED {org_id: $org_id}]->(:Meeting)"
    assert await _count(conn, org, produced) == 1
    rows = await run_cypher(
        conn,
        org,
        "MATCH (x:Decision {org_id: $org_id, decision_id: $d}) RETURN x.status, x.content",
        {"d": str(d)},
        columns=("s", "c"),
    )
    assert [str(v) for v in rows[0]] == ['"confirmed"', '"Ship the graph on AGE"']
    assert await _graph_events(org) == [("PRODUCED", "edge_created")]


async def test_a_redelivered_decision_is_a_no_op(consumer, conn, redis_client):
    org, d, m = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    value = decision_value(org, d, m)
    assert await _feed(consumer, "decisions", value) == "applied"
    # Redis cache warm.
    assert await _feed(consumer, "decisions", value) == "duplicate"
    # Redis cache evicted: the durable inbox is the authority.
    await redis_client.delete(
        dedupe_key(str(org), "decisions", ConfirmedFactsConsumer.group, f"decision:{d}:confirmed")
    )
    assert await _feed(consumer, "decisions", value) == "duplicate"
    # Even a replay past both guards (a rebuild) creates nothing: everything is MERGE.
    await consumer.write_graph(
        await _decode(consumer, "decisions", value),
    )
    assert await _count(conn, org, "(:Decision {org_id: $org_id})") == 1
    assert await _count(conn, org, "()-[:PRODUCED {org_id: $org_id}]->()") == 1
    assert len(await _graph_events(org)) == 1


async def _decode(consumer: ConfirmedFactsConsumer, topic: str, value: dict[str, Any]) -> Any:
    from libs.shared.consumer import ConsumedEvent

    decoded = consumer._codec.deserialize_for(topic, consumer._codec.serialize_for(topic, value))
    header = dict(decoded["header"])
    return ConsumedEvent(
        topic=topic,
        org_id=uuid.UUID(str(header["org_id"])),
        idempotency_key=header["idempotency_key"],
        header=header,
        body={k: v for k, v in decoded.items() if k != "header"},
        partition=0,
        offset=0,
    )


async def test_confirmed_action_writes_produced_and_assigned_to(consumer, conn):
    org, a, m, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert await _feed(consumer, "actions", action_value(org, a, m, user)) == "applied"
    assert await _count(conn, org, "(:Action {org_id: $org_id})-[:PRODUCED]->(:Meeting)") == 1
    assigned = "(:Action {org_id: $org_id})-[:ASSIGNED_TO {org_id: $org_id}]->(:Person)"
    assert await _count(conn, org, assigned) == 1
    rows = await run_cypher(
        conn, org, "MATCH (x:Action {org_id: $org_id}) RETURN x.due_date", columns=("d",)
    )
    assert str(rows[0][0]) == '"2025-11-28"'
    assert sorted(await _graph_events(org)) == [
        ("ASSIGNED_TO", "edge_created"),
        ("PRODUCED", "edge_created"),
    ]

    # No assignee: no Person, no ASSIGNED_TO.
    org2 = uuid.uuid4()
    assert await _feed(consumer, "actions", action_value(org2, uuid.uuid4(), m, None)) == "applied"
    assert await _count(conn, org2, "(:Person {org_id: $org_id})") == 0


async def test_a_join_writes_attended_only_for_a_user_identity(consumer, conn):
    org, m, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert (
        await _feed(consumer, "meeting-room-events", joined_value(org, m, str(user))) == "applied"
    )
    attended = "(:Person {org_id: $org_id})-[:ATTENDED {org_id: $org_id}]->(:Meeting)"
    assert await _count(conn, org, attended) == 1
    tap = joined_value(org, m, "audio-tap-worker")
    assert await _feed(consumer, "meeting-room-events", tap) == "applied"
    assert await _count(conn, org, "(:Person {org_id: $org_id})") == 1


async def test_the_lineage_template_walks_a_supersession_chain_in_any_order(consumer, conn):
    org, m = uuid.uuid4(), uuid.uuid4()
    d1, d2, d3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    # Out of order: the newest decision arrives first, so d2 starts life as a stub.
    for value in (
        decision_value(org, d3, m, content="v3", supersedes=d2),
        decision_value(org, d1, m, content="v1"),
        decision_value(org, d2, m, content="v2", supersedes=d1),
    ):
        assert await _feed(consumer, "decisions", value) == "applied"

    for subject in (d1, d2, d3):
        chain = await run_template(conn, org, "decision_lineage", {"decision_id": str(subject)})
        assert [n.properties["content"] for n in chain] == ["v1", "v2", "v3"]
        assert [n.properties["status"] for n in chain] == ["superseded", "superseded", "confirmed"]
        assert next(n.role for n in chain if n.id == str(subject)) == "subject"

    chain = await run_template(conn, org, "decision_lineage", {"decision_id": str(d2)})
    assert [(n.role, n.depth) for n in chain] == [
        ("ancestor", 1),
        ("subject", 0),
        ("descendant", 1),
    ]
    events = await _graph_events(org)
    assert events.count(("FOLLOWS_FROM", "lineage_updated")) == 2


async def test_a_cross_tenant_traversal_returns_nothing(consumer, conn):
    org_a, org_b, m = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    d1, d2, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _feed(consumer, "decisions", decision_value(org_a, d1, m))
    await _feed(consumer, "decisions", decision_value(org_a, d2, m, supersedes=d1))
    await _feed(consumer, "meeting-room-events", joined_value(org_a, m, str(user)))
    await _feed(consumer, "actions", action_value(org_a, uuid.uuid4(), m, user))

    # Org B asks about org A's ids: every template is empty.
    for name, params in (
        ("decision_lineage", {"decision_id": str(d2)}),
        ("decisions_by_person", {"user_id": str(user)}),
        ("actions_blocked_on_decision", {"decision_id": str(d1)}),
        ("related_decisions_for_topic", {"topic": "Architecture"}),
    ):
        assert await run_template(conn, org_b, name, params) == [], name
    # Even a hostile ad-hoc read must bind $org_id, which is always the caller's own tenant.
    rows = await run_adhoc(
        conn, org_b, "MATCH (d:Decision {org_id: $org_id}) RETURN d.decision_id", {}, ["d"]
    )
    assert rows == []
    # The same queries as org A see its data.
    assert len(await run_template(conn, org_a, "decision_lineage", {"decision_id": str(d2)})) == 2
    by_person = await run_template(conn, org_a, "decisions_by_person", {"user_id": str(user)})
    assert {n.id for n in by_person} == {str(d1), str(d2)}
    blocked = await run_template(
        conn, org_a, "actions_blocked_on_decision", {"decision_id": str(d1)}
    )
    assert len(blocked) == 1 and blocked[0].label == "Action"


async def test_soft_deleted_decisions_drop_out_of_templates(conn):
    org, m = uuid.uuid4(), uuid.uuid4()
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    now = "2026-11-09T10:00:00+00:00"
    await apply_decision(conn, org, ConfirmedDecision(d1, m, "v1", 0.9, now))
    await apply_decision(conn, org, ConfirmedDecision(d2, m, "v2", 0.9, now, supersedes=d1))
    assert await soft_delete(conn, org, label="Decision", node_id=str(d1), deleted_at=now)
    chain = await run_template(conn, org, "decision_lineage", {"decision_id": str(d2)})
    assert [n.id for n in chain] == [str(d2)]
    assert await run_template(conn, org, "decision_lineage", {"decision_id": str(d1)}) == []
    # Another tenant cannot soft-delete this one's node.
    assert not await soft_delete(
        conn, uuid.uuid4(), label="Decision", node_id=str(d2), deleted_at=now
    )


async def test_adhoc_cypher_is_read_only_and_must_bind_org_id(conn):
    org = uuid.uuid4()
    with pytest.raises(AgeQueryRejected):
        await run_adhoc(conn, org, "MATCH (d:Decision) RETURN d", {}, ["d"])
    with pytest.raises(AgeQueryRejected):
        await run_adhoc(
            conn, org, "MATCH (d:Decision {org_id: $org_id}) DETACH DELETE d RETURN 1", {}, ["d"]
        )


async def test_the_graph_rebuilds_from_confirmed_rows(consumer, conn):
    org, m, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    d1, d2, a = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _feed(consumer, "decisions", decision_value(org, d1, m, content="v1"))
    await _feed(consumer, "decisions", decision_value(org, d2, m, content="v2", supersedes=d1))
    await _feed(consumer, "actions", action_value(org, a, m, user))

    async def snapshot() -> tuple[int, int, list[str]]:
        nodes = await _count(conn, org, "(n {org_id: $org_id})")
        edges = await _count(conn, org, "()-[e {org_id: $org_id}]->()")
        chain = await run_template(conn, org, "decision_lineage", {"decision_id": str(d2)})
        return nodes, edges, [f"{n.id}:{n.properties.get('status')}" for n in chain]

    before = await snapshot()
    await purge_org(conn, org)
    assert await _count(conn, org, "(n {org_id: $org_id})") == 0

    now = "2026-11-09T10:00:00+00:00"
    rows = [  # as exported from core-api's confirmed rows, newest first
        {"kind": "decision", "decision_id": str(d2), "meeting_id": str(m), "content": "v2",
         "confidence": 0.9, "confirmed_at": now, "supersedes": str(d1), "status": "confirmed"},
        {"kind": "decision", "decision_id": str(d1), "meeting_id": str(m), "content": "v1",
         "confidence": 0.9, "confirmed_at": now, "status": "superseded"},
        {"kind": "action", "action_id": str(a), "meeting_id": str(m), "content": "x",
         "confidence": 0.8, "confirmed_at": now, "assignee_id": str(user),
         "due_date": "2025-11-28", "status": "confirmed"},
    ]  # fmt: skip
    for row in rows:
        fact = parse_row(row)
        if isinstance(fact, ConfirmedDecision):
            await apply_decision(conn, org, fact)
        else:
            assert isinstance(fact, ConfirmedAction)
            await apply_action(conn, org, fact)
    assert await snapshot() == before

    with pytest.raises(ValueError):
        parse_row({**rows[0], "status": "proposed"})
