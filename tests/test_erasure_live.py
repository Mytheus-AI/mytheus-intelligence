"""Admin data erasure against real stores: Apache AGE, Qdrant, and the Postgres FTS read model
(with the real Schema Registry codec, Redis and `intelligence.inbox` / `outbox`).

Each part skips when its store is unreachable, like the rest of the suite (`MYTHEUS_REQUIRE_AGE`
/ `MYTHEUS_REQUIRE_QDRANT` / `MYTHEUS_REQUIRE_DB` turn a skip into a failure). The offline
counterpart, which always runs, is `tests/test_erasure_consumers.py`.

Every case seeds the SAME meeting / person id in a second org and proves it survives.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import pytest
from confluent_kafka import Producer
from qdrant_client.models import PointStruct
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.age import AgeConnection, run_cypher
from libs.shared.erasure import ERASED_SPEAKER, completion_key
from libs.shared.qdrant import (
    LOGICAL_DECISIONS,
    LOGICAL_TRANSCRIPT_CHUNKS,
    ensure_collection,
    physical_collection,
)
from services.embedding_service.erasure_consumer import (
    erase_meeting_vectors,
    erase_person_vectors,
)
from services.knowledge_graph_service.confirmed import (
    ConfirmedAction,
    ConfirmedDecision,
    apply_action,
    apply_attended,
    apply_decision,
    purge_org,
)
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.erasure_consumer import (
    erase_meeting_graph,
    erase_person_graph,
)
from services.knowledge_graph_service.graph import ensure_graph
from services.search_service.erasure_consumer import SearchErasureConsumer
from tests.conftest import APP_URL

pytestmark = pytest.mark.asyncio

_AGE_DSN = os.environ.get(
    "AGE_DATABASE_URL", "postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph"
)


def _age_reachable() -> bool:
    u = urlparse(_AGE_DSN)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture()
async def age() -> AsyncIterator[AgeConnection]:
    if not _age_reachable():
        if os.environ.get("MYTHEUS_REQUIRE_AGE") == "1":
            raise RuntimeError(f"MYTHEUS_REQUIRE_AGE=1 but no AGE at {_AGE_DSN}")
        pytest.skip(f"no Apache AGE Postgres at {_AGE_DSN}: graph erasure UNVERIFIED live")
    async with age_connection(_AGE_DSN) as conn:
        await ensure_graph(conn)
        yield conn


async def _nodes(conn: AgeConnection, org: uuid.UUID, cypher: str, **params: Any) -> list[Any]:
    rows = await run_cypher(conn, org, cypher, params, columns=("n",))
    return [json.loads(str(r[0]).split("::", 1)[0]) for r in rows]


async def test_graph_erasure_live(age: AgeConnection) -> None:
    org, other = uuid.uuid4(), uuid.uuid4()
    meeting, kept_meeting = uuid.uuid4(), uuid.uuid4()
    person, decision, action = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = "2026-11-11T10:00:00+00:00"
    try:
        for o in (org, other):
            await apply_decision(
                age, o, ConfirmedDecision(decision, meeting, "Ship on AGE", 0.9, now)
            )
            await apply_attended(age, o, user_id=person, meeting_id=meeting)
        await apply_action(
            age, org, ConfirmedAction(action, meeting, "Write the runbook", 0.8, now, person)
        )
        await apply_decision(
            age, org, ConfirmedDecision(uuid.uuid4(), kept_meeting, "Other meeting", 0.9, now)
        )

        assert await erase_meeting_graph(age, org, meeting) == {
            "meeting_nodes_deleted": 1,
            "edges_deleted": 3,  # ATTENDED, and PRODUCED from the decision and the action
            "decisions_kept": 1,
            "actions_kept": 1,
        }
        m = "MATCH (m:Meeting {org_id: $org_id, meeting_id: $id}) RETURN m"
        assert await _nodes(age, org, m, id=str(meeting)) == []
        assert len(await _nodes(age, org, m, id=str(kept_meeting))) == 1
        [d] = await _nodes(
            age,
            org,
            "MATCH (d:Decision {org_id: $org_id, decision_id: $id}) RETURN d",
            id=str(decision),
        )
        props = d["properties"]
        assert props["content"] == "Ship on AGE" and props["status"] == "confirmed"
        assert "meeting_id" not in props and props["meeting_erased"] is True
        assigned = (
            "MATCH (a:Action {org_id: $org_id, action_id: $id})-[:ASSIGNED_TO]->(p:Person) "
            "RETURN p"
        )
        assert len(await _nodes(age, org, assigned, id=str(action))) == 1
        # The other org's identical ids are untouched.
        assert len(await _nodes(age, other, m, id=str(meeting))) == 1
        assert await erase_meeting_graph(age, org, meeting) == {
            "meeting_nodes_deleted": 0,
            "edges_deleted": 0,
            "decisions_kept": 0,
            "actions_kept": 0,
        }

        assert await erase_person_graph(age, org, person) == {
            "person_nodes_deleted": 1,
            "edges_deleted": 1,  # ASSIGNED_TO (ATTENDED went with the meeting)
        }
        p = "MATCH (p:Person {org_id: $org_id, user_id: $id}) RETURN p"
        assert await _nodes(age, org, p, id=str(person)) == []
        assert await _nodes(age, org, assigned, id=str(action)) == []
        [a] = await _nodes(
            age,
            org,
            "MATCH (a:Action {org_id: $org_id, action_id: $id}) RETURN a",
            id=str(action),
        )
        assert a["properties"]["content"] == "Write the runbook"
        assert len(await _nodes(age, other, p, id=str(person))) == 1
        assert await erase_person_graph(age, org, person) == {
            "person_nodes_deleted": 0,
            "edges_deleted": 0,
        }
    finally:
        await purge_org(age, org)
        await purge_org(age, other)


async def test_vector_erasure_live(qdrant_client) -> None:
    org, other = uuid.uuid4(), uuid.uuid4()
    meeting, person = str(uuid.uuid4()), str(uuid.uuid4())

    def pt(o: uuid.UUID, **payload: Any) -> PointStruct:
        return PointStruct(
            id=str(uuid.uuid4()), vector=[0.1] * 384, payload={"org_id": str(o), **payload}
        )

    names = []
    try:
        for o in (org, other):
            chunks = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, o)
            decisions = physical_collection(LOGICAL_DECISIONS, o)
            names += [chunks, decisions]
            for n in (chunks, decisions):
                await ensure_collection(qdrant_client, n)
            await qdrant_client.upsert(
                chunks,
                [
                    pt(o, meeting_id=meeting, speaker="1"),
                    pt(o, meeting_id=str(uuid.uuid4()), speaker=person),
                ],
            )
            await qdrant_client.upsert(
                decisions, [pt(o, decision_id="d1", meeting_id=meeting, status="confirmed")]
            )

        assert await erase_meeting_vectors(qdrant_client, org, uuid.UUID(meeting)) == {
            "transcript_points_deleted": 1,
            "decision_points_deleted": 0,
            "decision_points_anonymized": 1,
        }
        assert await erase_person_vectors(qdrant_client, org, uuid.UUID(person)) == {
            "transcript_points_anonymized": 1
        }

        async def payloads(name: str) -> list[dict[str, Any]]:
            points, _ = await qdrant_client.scroll(name, limit=10, with_payload=True)
            return [dict(p.payload or {}) for p in points]

        [left] = await payloads(physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org))
        assert left["speaker"] == ERASED_SPEAKER
        [dec] = await payloads(physical_collection(LOGICAL_DECISIONS, org))
        assert "meeting_id" not in dec and dec["meeting_erased"] is True
        assert len(await payloads(physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, other))) == 2
        [other_dec] = await payloads(physical_collection(LOGICAL_DECISIONS, other))
        assert other_dec["meeting_id"] == meeting
    finally:
        for n in names:
            await qdrant_client.delete_collection(n)


async def test_search_erasure_live(migrated_db, redis_client, avro_codec) -> None:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    consumer = SearchErasureConsumer(
        session_factory=factory,
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=Producer({"bootstrap.servers": "localhost:9092"}),
    )
    org, other = uuid.uuid4(), uuid.uuid4()
    meeting, person = str(uuid.uuid4()), str(uuid.uuid4())

    async def put(o: uuid.UUID, m: str, speaker: str, words: str) -> None:
        async with factory() as s, s.begin():
            await s.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(o)})
            await s.execute(
                text(
                    "INSERT INTO intelligence.transcript_chunk_search (org_id, meeting_id, "
                    "chunk_id, speaker_id, start_ms, end_ms, text) "
                    "VALUES (:o, :m, :c, :s, 0, 1, :t)"
                ),
                {"o": str(o), "m": m, "c": str(uuid.uuid4()), "s": speaker, "t": words},
            )

    async def rows(o: uuid.UUID) -> list[tuple[str, str]]:
        async with factory() as s, s.begin():
            await s.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(o)})
            res = await s.execute(
                text(
                    "SELECT speaker_id, text FROM intelligence.transcript_chunk_search "
                    "ORDER BY text"
                )
            )
            return [(r[0], r[1]) for r in res]

    def request(kind: str, target: str, erasure_id: str) -> bytes:
        value = {
            "header": {
                "event_id": str(uuid.uuid4()),
                "idempotency_key": f"admin:erasure_requested:{kind}:{target}:{uuid.uuid4()}",
                "org_id": str(org),
                "schema_version": 1,
                "producer": "admin-ops-service",
                "occurred_at": int(time.time() * 1_000_000),
                "correlation_id": None,
                "causation_id": None,
            },
            "event_type": "erasure_requested",
            "actor_id": str(uuid.uuid4()),
            "target_type": kind,
            "target_id": target,
            "detail_json": json.dumps({"erasure_id": erasure_id, "scope": kind}),
        }
        return avro_codec.serialize_for("admin-events", value)

    async def feed(raw: bytes) -> str:
        return str(
            await consumer.process_raw(topic="admin-events", raw_value=raw, partition=0, offset=0)
        )

    try:
        for o in (org, other):
            await put(o, meeting, "1", "a meeting line")
            await put(o, str(uuid.uuid4()), person, "b their line")
        e1, e2 = str(uuid.uuid4()), str(uuid.uuid4())
        meeting_raw = request("meeting", meeting, e1)
        assert await feed(meeting_raw) == "applied"
        assert await feed(meeting_raw) == "duplicate"
        assert await feed(request("person", person, e2)) == "applied"
        assert await rows(org) == [(ERASED_SPEAKER, "b their line")]
        assert await rows(other) == [("1", "a meeting line"), (person, "b their line")]

        async with factory() as s, s.begin():
            await s.execute(text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org)})
            res = await s.execute(
                text(
                    "SELECT idempotency_key, payload FROM intelligence.outbox "
                    "WHERE topic = 'admin-events' ORDER BY idempotency_key"
                )
            )
            outbox = {r[0]: r[1] for r in res}
        assert set(outbox) == {
            completion_key(e1, "search_index"),
            completion_key(e2, "search_index"),
        }
        detail = json.loads(outbox[completion_key(e1, "search_index")]["detail_json"])
        assert detail["counts"] == {"deleted": 1} and detail["store"] == "search_index"
    finally:
        for o in (org, other):
            async with factory() as s, s.begin():
                await s.execute(
                    text("SELECT set_config('app.current_org', :o, true)"), {"o": str(o)}
                )
                for table in ("transcript_chunk_search", "outbox", "inbox"):
                    await s.execute(
                        text(f"DELETE FROM intelligence.{table} WHERE org_id = :o"),  # noqa: S608
                        {"o": str(o)},
                    )
        await engine.dispose()
