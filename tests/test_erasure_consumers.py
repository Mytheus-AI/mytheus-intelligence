"""Admin data erasure in this tier's stores, offline: in-memory Qdrant, fake Redis, a fake AGE
cursor and a fake relational session that keeps `intelligence.inbox` / `intelligence.outbox` in
memory. The live counterparts (real Postgres FTS rows, real AGE) are
`tests/test_erasure_live.py`.

Covers `libs/shared/erasure.py` and the three consumers:
1. Parsing: only `erasure_requested` for a person or a meeting acts; a non-UUID target is
   dead-lettered, never applied.
2. Vectors: a meeting's transcript points go, its kept (confirmed) decision points lose the
   meeting; a person's transcript points lose their attribution; another org's collection and
   the same meeting / person id there are untouched.
3. Through `process_raw`: the same event twice applies once (Redis, then the durable inbox when
   the Redis key is gone); one `erasure_completed` per store, encodable against the proposed
   `AdminEvent` v2; a failure releases the dedupe mark so the redelivery applies.
4. Late arrivals: the tombstone keeps a lagging indexer from storing an erased meeting's chunk,
   and strips a since-erased person from a chunk spoken before their erasure only.
5. Search and graph: the statements are org-scoped and do what the rule says.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from typing import Any

import fakeredis
import fastavro
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from libs.shared.consumer import ConsumedEvent
from libs.shared.erasure import (
    COMPLETED,
    ERASED_SPEAKER,
    UnusableErasure,
    completion_key,
    erased_at,
    parse_request,
    tombstone_key,
)
from libs.shared.inbox import dedupe_key
from libs.shared.qdrant import (
    LOGICAL_DECISIONS,
    LOGICAL_TRANSCRIPT_CHUNKS,
    ensure_collection,
    physical_collection,
)
from services.embedding_service import consumer as embedding_consumer
from services.embedding_service.erasure_consumer import (
    ErasureAwareEmbeddingConsumer,
    VectorErasureConsumer,
)
from services.knowledge_graph_service.confirmed import GraphFact
from services.knowledge_graph_service.erasure_consumer import (
    ErasureAwareFactsConsumer,
    GraphErasureConsumer,
)
from services.knowledge_graph_service.facts_consumer import ConfirmedFactsConsumer
from services.search_service.erasure_consumer import (
    ErasureAwareSearchConsumer,
    SearchErasureConsumer,
)

pytestmark = pytest.mark.asyncio

ORG, OTHER = uuid.uuid4(), uuid.uuid4()

ADMIN_EVENT_V2 = {
    "type": "record",
    "name": "AdminEvent",
    "namespace": "io.nexus.events",
    "fields": [
        {
            "name": "header",
            "type": {
                "type": "record",
                "name": "EventEnvelopeHeader",
                "fields": [
                    {"name": "event_id", "type": {"type": "string", "logicalType": "uuid"}},
                    {"name": "idempotency_key", "type": "string"},
                    {"name": "org_id", "type": {"type": "string", "logicalType": "uuid"}},
                    {"name": "schema_version", "type": "int", "default": 1},
                    {"name": "producer", "type": "string"},
                    {
                        "name": "occurred_at",
                        "type": {"type": "long", "logicalType": "timestamp-micros"},
                    },
                    {"name": "correlation_id", "type": ["null", "string"], "default": None},
                    {"name": "causation_id", "type": ["null", "string"], "default": None},
                ],
            },
        },
        {
            "name": "event_type",
            "type": {
                "type": "enum",
                "name": "AdminEventType",
                "symbols": [
                    "flag_changed",
                    "tenant_suspended",
                    "erasure_requested",
                    "erasure_completed",
                ],
            },
        },
        {"name": "actor_id", "type": "string"},
        {"name": "target_type", "type": "string"},
        {"name": "target_id", "type": "string"},
        {"name": "detail_json", "type": "string"},
    ],
}


def now_us() -> int:
    return int(time.time() * 1_000_000)


def admin_value(
    org: uuid.UUID,
    *,
    kind: str,
    target: str,
    event_type: str = "erasure_requested",
    erasure_id: str | None = None,
    at_us: int | None = None,
) -> dict[str, Any]:
    return {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": f"admin:{event_type}:{kind}:{target}:{uuid.uuid4()}",
            "org_id": str(org),
            "schema_version": 1,
            "producer": "admin-ops-service",
            "occurred_at": at_us if at_us is not None else now_us(),
            "correlation_id": None,
            "causation_id": None,
        },
        "event_type": event_type,
        "actor_id": str(uuid.uuid4()),
        "target_type": kind,
        "target_id": target,
        "detail_json": json.dumps({"erasure_id": erasure_id or str(uuid.uuid4()), "scope": kind}),
    }


def consumed(value: dict[str, Any], topic: str = "admin-events") -> ConsumedEvent:
    header = value["header"]
    return ConsumedEvent(
        topic=topic,
        org_id=uuid.UUID(header["org_id"]),
        idempotency_key=header["idempotency_key"],
        header=header,
        body={k: v for k, v in value.items() if k != "header"},
        partition=0,
        offset=0,
    )


# --- fakes ---------------------------------------------------------------------------------------


class JsonCodec:
    def serialize_for(self, topic: str, value: dict[str, Any]) -> bytes:
        return json.dumps(value).encode()

    def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
        return dict(json.loads(raw))


class Dlq:
    def __init__(self) -> None:
        self.produced: list[dict[str, Any]] = []

    def produce(self, *, topic: str, key: bytes | None, value: bytes) -> None:
        self.produced.append({"topic": topic, "value": json.loads(value)})

    def flush(self, timeout: float = 5) -> int:
        return 0


class _Result:
    def __init__(self, row: Any = None, rowcount: int = 0) -> None:
        self._row, self.rowcount = row, rowcount

    def first(self) -> Any:
        return self._row

    def scalar_one(self) -> Any:
        return self._row[0]


class Db:
    """`intelligence.inbox` / `intelligence.outbox` in memory, plus every other statement
    recorded (with the `app.current_org` it ran under) and answered with `rowcount`."""

    def __init__(self, rowcount: int = 0) -> None:
        self.inbox: set[tuple[str, str, str]] = set()
        self.outbox: dict[tuple[str, str], dict[str, Any]] = {}
        self.statements: list[tuple[str, dict[str, Any], str | None]] = []
        self.rowcount = rowcount

    def __call__(self) -> Session:
        return Session(self)


class Session:
    def __init__(self, db: Db) -> None:
        self.db = db
        self.current_org: str | None = None
        self._inbox: set[tuple[str, str, str]] = set()
        self._outbox: dict[tuple[str, str], dict[str, Any]] = {}

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def begin(self) -> Session:
        return _Tx(self)  # type: ignore[return-value]

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _Result:
        sql, p = " ".join(str(stmt).split()), dict(params or {})
        if "set_config('app.current_org'" in sql:
            self.current_org = p["org_id"]
            return _Result()
        if "FROM intelligence.inbox" in sql:
            key = (p["consumer_group"], p["org_id"], p["idempotency_key"])
            return _Result((1,) if key in self.db.inbox | self._inbox else None)
        if "INSERT INTO intelligence.inbox" in sql:
            self._inbox.add((p["consumer_group"], p["org_id"], p["idempotency_key"]))
            return _Result(rowcount=1)
        if "INSERT INTO intelligence.outbox" in sql:
            key = (p["topic"], p["idempotency_key"])
            if key in self.db.outbox or key in self._outbox:
                return _Result()
            self._outbox[key] = {**p, "payload": json.loads(p["payload"])}
            return _Result((len(self.db.outbox) + len(self._outbox),))
        if "SELECT id FROM intelligence.outbox" in sql:
            return _Result((1,))
        self.db.statements.append((sql, p, self.current_org))
        return _Result(rowcount=self.db.rowcount)


class _Tx:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, *_: Any) -> None:
        if exc_type is None:
            self.session.db.inbox |= self.session._inbox
            self.session.db.outbox.update(self.session._outbox)


def vector_consumer(qdrant: Any, redis: Any, db: Db | None = None) -> VectorErasureConsumer:
    return VectorErasureConsumer(
        qdrant=qdrant,
        session_factory=db or Db(),  # type: ignore[arg-type]
        redis=redis,
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )


async def feed(consumer: Any, value: dict[str, Any]) -> str:
    raw = json.dumps(value).encode()
    return str(
        await consumer.process_raw(topic="admin-events", raw_value=raw, partition=0, offset=1)
    )


def point(payload: dict[str, Any]) -> PointStruct:
    return PointStruct(id=str(uuid.uuid4()), vector=[0.1] * 384, payload=payload)


async def seed(qdrant: AsyncQdrantClient, logical: str, org: uuid.UUID, *payloads: Any) -> str:
    name = physical_collection(logical, org)
    await ensure_collection(qdrant, name)
    await qdrant.upsert(name, [point({"org_id": str(org), **p}) for p in payloads])
    return name


async def payloads(qdrant: AsyncQdrantClient, name: str) -> list[dict[str, Any]]:
    points, _ = await qdrant.scroll(name, limit=100, with_payload=True)
    return sorted((dict(p.payload or {}) for p in points), key=lambda p: json.dumps(p))


# --- 1. parsing ----------------------------------------------------------------------------------


def test_only_a_person_or_meeting_erasure_request_acts():
    m = str(uuid.uuid4())
    req = parse_request(consumed(admin_value(ORG, kind="meeting", target=m, erasure_id=None)))
    assert req is not None and req.kind == "meeting" and req.target == m and req.org_id == ORG
    assert parse_request(consumed(admin_value(ORG, kind="tenant", target=str(ORG)))) is None
    for other in ("flag_changed", "tenant_suspended", COMPLETED):
        value = admin_value(ORG, kind="meeting", target=m, event_type=other)
        assert parse_request(consumed(value)) is None
    with pytest.raises(UnusableErasure):
        parse_request(consumed(admin_value(ORG, kind="person", target="alice@example.com")))
    no_id = admin_value(ORG, kind="meeting", target=m)
    no_id["detail_json"] = "{}"
    fallback = parse_request(consumed(no_id))
    assert fallback is not None and fallback.erasure_id == no_id["header"]["idempotency_key"]


# --- 2. vectors ----------------------------------------------------------------------------------


async def test_meeting_erasure_deletes_its_transcript_points_and_unhooks_kept_decisions():
    qdrant = AsyncQdrantClient(location=":memory:")
    meeting, kept_meeting = str(uuid.uuid4()), str(uuid.uuid4())
    chunks = await seed(
        qdrant,
        LOGICAL_TRANSCRIPT_CHUNKS,
        ORG,
        {"meeting_id": meeting, "chunk_id": "a", "speaker": "1"},
        {"meeting_id": meeting, "chunk_id": "b", "speaker": "2"},
        {"meeting_id": kept_meeting, "chunk_id": "c", "speaker": "1"},
    )
    decisions = await seed(
        qdrant,
        LOGICAL_DECISIONS,
        ORG,
        {"decision_id": "d1", "meeting_id": meeting, "status": "confirmed"},
        {"decision_id": "d0", "meeting_id": meeting, "status": "superseded"},
        {"decision_id": "dx", "meeting_id": meeting, "status": "proposed"},
        {"decision_id": "d2", "meeting_id": kept_meeting, "status": "confirmed"},
    )
    other_chunks = await seed(
        qdrant, LOGICAL_TRANSCRIPT_CHUNKS, OTHER, {"meeting_id": meeting, "chunk_id": "z"}
    )
    consumer = vector_consumer(qdrant, fakeredis.aioredis.FakeRedis())
    req = parse_request(consumed(admin_value(ORG, kind="meeting", target=meeting)))
    assert req is not None

    counts = await consumer.erase(req, None)  # type: ignore[arg-type]
    assert counts == {
        "transcript_points_deleted": 2,
        "decision_points_deleted": 1,
        "decision_points_anonymized": 2,
    }
    assert [p["chunk_id"] for p in await payloads(qdrant, chunks)] == ["c"]
    after = {p["decision_id"]: p for p in await payloads(qdrant, decisions)}
    assert set(after) == {"d0", "d1", "d2"}
    for d in ("d0", "d1"):
        assert "meeting_id" not in after[d] and after[d]["meeting_erased"] is True
    assert after["d1"]["status"] == "confirmed" and after["d0"]["status"] == "superseded"
    assert after["d2"]["meeting_id"] == kept_meeting and "meeting_erased" not in after["d2"]
    assert [p["chunk_id"] for p in await payloads(qdrant, other_chunks)] == ["z"]

    assert await consumer.erase(req, None) == {  # type: ignore[arg-type]
        "transcript_points_deleted": 0,
        "decision_points_deleted": 0,
        "decision_points_anonymized": 0,
    }


async def test_person_erasure_removes_their_speaker_attribution_only_in_this_org():
    qdrant = AsyncQdrantClient(location=":memory:")
    person, colleague = uuid.uuid4(), str(uuid.uuid4())
    chunks = await seed(
        qdrant,
        LOGICAL_TRANSCRIPT_CHUNKS,
        ORG,
        {"chunk_id": "a", "speaker": str(person)},
        {"chunk_id": "b", "speaker": str(person).upper()},
        {"chunk_id": "c", "speaker": colleague},
        {"chunk_id": "d", "speaker": "1"},
    )
    other = await seed(
        qdrant, LOGICAL_TRANSCRIPT_CHUNKS, OTHER, {"chunk_id": "z", "speaker": str(person)}
    )
    consumer = vector_consumer(qdrant, fakeredis.aioredis.FakeRedis())
    req = parse_request(consumed(admin_value(ORG, kind="person", target=str(person))))
    counts = await consumer.erase(req, None)  # type: ignore[arg-type]
    assert counts == {"transcript_points_anonymized": 2}
    speakers = {p["chunk_id"]: p["speaker"] for p in await payloads(qdrant, chunks)}
    assert speakers == {"a": ERASED_SPEAKER, "b": ERASED_SPEAKER, "c": colleague, "d": "1"}
    assert [p["speaker"] for p in await payloads(qdrant, other)] == [str(person)]


async def test_an_org_with_no_collections_has_nothing_to_erase():
    consumer = vector_consumer(
        AsyncQdrantClient(location=":memory:"), fakeredis.aioredis.FakeRedis()
    )
    for kind in ("meeting", "person"):
        req = parse_request(consumed(admin_value(ORG, kind=kind, target=str(uuid.uuid4()))))
        assert set((await consumer.erase(req, None)).values()) == {0}  # type: ignore[arg-type]


# --- 3. through process_raw ----------------------------------------------------------------------


async def test_the_same_event_twice_applies_once_and_completes_once():
    qdrant = AsyncQdrantClient(location=":memory:")
    meeting = str(uuid.uuid4())
    chunks = await seed(
        qdrant, LOGICAL_TRANSCRIPT_CHUNKS, ORG, {"meeting_id": meeting, "chunk_id": "a"}
    )
    redis, db = fakeredis.aioredis.FakeRedis(), Db()
    consumer = vector_consumer(qdrant, redis, db)
    erasure_id = str(uuid.uuid4())
    value = admin_value(ORG, kind="meeting", target=meeting, erasure_id=erasure_id)

    assert await feed(consumer, value) == "applied"
    assert await payloads(qdrant, chunks) == []
    assert await feed(consumer, value) == "duplicate"
    idem = value["header"]["idempotency_key"]
    await redis.delete(dedupe_key(str(ORG), "admin-events", consumer.group, idem))
    assert await feed(consumer, value) == "duplicate"  # the durable inbox

    [(topic, key)] = db.outbox
    assert (topic, key) == ("admin-events", completion_key(erasure_id, "vectors"))
    row = db.outbox[(topic, key)]
    assert row["org_id"] == str(ORG) and row["partition_key"] == str(ORG)
    assert row["schema_subject"] == "admin-events-value"
    out = row["payload"]
    assert out["event_type"] == COMPLETED and out["target_type"] == "meeting"
    assert out["target_id"] == meeting and out["actor_id"] == value["actor_id"]
    assert out["header"]["producer"] == "embedding-service"
    assert out["header"]["causation_id"] == value["header"]["event_id"]
    assert json.loads(out["detail_json"]) == {
        "erasure_id": erasure_id,
        "store": "vectors",
        "service": "embedding-service",
        "counts": {
            "transcript_points_deleted": 1,
            "decision_points_deleted": 0,
            "decision_points_anonymized": 0,
        },
    }
    fastavro.schemaless_writer(io.BytesIO(), fastavro.parse_schema(ADMIN_EVENT_V2), out)
    assert await erased_at(redis, ORG, "meeting", meeting) == value["header"]["occurred_at"]
    assert await redis.ttl(tombstone_key(ORG, "meeting", meeting)) > 0


class FlakyQdrant:
    def __init__(self, real: AsyncQdrantClient, failures: int) -> None:
        self.real, self.failures = real, failures

    def __getattr__(self, name: str) -> Any:
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("qdrant unreachable")
        return getattr(self.real, name)


async def test_a_failed_attempt_releases_the_dedupe_mark_so_the_redelivery_applies():
    qdrant = AsyncQdrantClient(location=":memory:")
    meeting = str(uuid.uuid4())
    chunks = await seed(
        qdrant, LOGICAL_TRANSCRIPT_CHUNKS, ORG, {"meeting_id": meeting, "chunk_id": "a"}
    )
    redis, db = fakeredis.aioredis.FakeRedis(), Db()
    consumer = vector_consumer(FlakyQdrant(qdrant, failures=1), redis, db)
    value = admin_value(ORG, kind="meeting", target=meeting)
    with pytest.raises(ConnectionError):
        await feed(consumer, value)
    idem = value["header"]["idempotency_key"]
    assert not await redis.exists(dedupe_key(str(ORG), "admin-events", consumer.group, idem))
    assert db.outbox == {} and db.inbox == set()
    assert await feed(consumer, value) == "applied"
    assert await payloads(qdrant, chunks) == [] and len(db.outbox) == 1


async def test_other_admin_events_complete_nothing_and_a_bad_target_is_dead_lettered():
    redis, db = fakeredis.aioredis.FakeRedis(), Db()
    consumer = vector_consumer(AsyncQdrantClient(location=":memory:"), redis, db)
    m = str(uuid.uuid4())
    flag = admin_value(ORG, kind="meeting", target=m, event_type="flag_changed")
    assert await feed(consumer, flag) == "applied"
    assert await feed(consumer, admin_value(ORG, kind="tenant", target=str(ORG))) == "applied"
    assert await feed(consumer, admin_value(ORG, kind="meeting", target="nope")) == "applied"
    assert db.outbox == {}
    assert await redis.keys("nexus:*:erasure:*") == []
    [dead] = consumer._dlq_producer.produced  # type: ignore[attr-defined]
    assert dead["topic"] == "admin-events.embedding-service.erasure.dlq"
    assert dead["value"]["failure_class"] == "BUSINESS_INVALID"


# --- 4. late arrivals ----------------------------------------------------------------------------


def chunk_event(meeting: str, speaker: str, at_us: int) -> ConsumedEvent:
    return ConsumedEvent(
        topic="transcript-chunks",
        org_id=ORG,
        idempotency_key=f"chunk:{uuid.uuid4()}",
        header={"occurred_at": at_us},
        body={
            "chunk_id": str(uuid.uuid4()),
            "meeting_id": meeting,
            "speaker_id": speaker,
            "start_ms": 0,
            "end_ms": 900,
            "is_final": True,
            "redacted": False,
            "text": "late words",
        },
        partition=0,
        offset=0,
    )


async def test_a_lagging_indexer_honours_the_tombstones(monkeypatch):
    monkeypatch.setattr(
        embedding_consumer, "embed_batch", lambda pieces: ([[0.1] * 384 for _ in pieces], 0.0)
    )
    qdrant = AsyncQdrantClient(location=":memory:")
    redis = fakeredis.aioredis.FakeRedis()
    erasure = vector_consumer(qdrant, redis)
    meeting, person = str(uuid.uuid4()), str(uuid.uuid4())
    t = now_us()
    for kind, target in (("meeting", meeting), ("person", person)):
        value = admin_value(ORG, kind=kind, target=target, at_us=t)
        await erasure.apply(consumed(value), Session(Db()))  # type: ignore[arg-type]

    indexer = ErasureAwareEmbeddingConsumer(
        qdrant=qdrant,
        session_factory=None,  # type: ignore[arg-type]
        redis=redis,
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,
    )

    async def no_outbox(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(indexer, "emit_outbox", no_outbox)
    other_meeting = str(uuid.uuid4())
    await indexer.apply(chunk_event(meeting, "1", t - 5), None)  # type: ignore[arg-type]
    await indexer.apply(chunk_event(other_meeting, person, t - 5), None)  # type: ignore[arg-type]
    await indexer.apply(chunk_event(other_meeting, person, t + 5_000_000), None)  # type: ignore[arg-type]
    stored = await payloads(qdrant, physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, ORG))
    assert all(p["meeting_id"] == other_meeting for p in stored)
    assert sorted(p["speaker"] for p in stored) == sorted([ERASED_SPEAKER, person])


# --- 5. search and graph -------------------------------------------------------------------------


async def test_search_erasure_is_org_scoped_and_keeps_a_persons_words():
    db = Db(rowcount=3)
    consumer = SearchErasureConsumer(
        session_factory=db,  # type: ignore[arg-type]
        redis=fakeredis.aioredis.FakeRedis(),
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )
    meeting, person = str(uuid.uuid4()), str(uuid.uuid4())
    assert await feed(consumer, admin_value(ORG, kind="meeting", target=meeting)) == "applied"
    assert await feed(consumer, admin_value(ORG, kind="person", target=person)) == "applied"
    (m_sql, m_p, m_org), (p_sql, p_p, p_org) = db.statements
    assert m_sql.startswith("DELETE FROM intelligence.transcript_chunk_search")
    assert "org_id = :org_id AND meeting_id = :target" in m_sql
    assert m_p == {"org_id": str(ORG), "target": meeting} and m_org == str(ORG)
    assert p_sql.startswith("UPDATE intelligence.transcript_chunk_search SET speaker_id = :erased")
    assert "text" not in p_sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert p_p == {"org_id": str(ORG), "target": person, "erased": ERASED_SPEAKER}
    assert p_org == str(ORG)
    counts = sorted(
        json.dumps(json.loads(r["payload"]["detail_json"])["counts"]) for r in db.outbox.values()
    )
    assert counts == ['{"anonymized": 3}', '{"deleted": 3}']


async def test_the_search_indexer_drops_a_late_chunk_of_an_erased_meeting():
    redis = fakeredis.aioredis.FakeRedis()
    meeting = str(uuid.uuid4())
    await redis.set(tombstone_key(ORG, "meeting", meeting), str(now_us()))
    db = Db()
    indexer = ErasureAwareSearchConsumer(
        session_factory=db,  # type: ignore[arg-type]
        redis=redis,
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,
    )
    await indexer.apply(chunk_event(meeting, "1", now_us()), Session(db))  # type: ignore[arg-type]
    assert db.statements == []
    await indexer.apply(chunk_event(str(uuid.uuid4()), "1", now_us()), Session(db))  # type: ignore[arg-type]
    assert db.statements[0][0].startswith("INSERT INTO intelligence.transcript_chunk_search")


class Cursor:
    """A fake asyncpg connection for AGE (as in `test_knowledge_graph_writer.py`)."""

    def __init__(self, count: int = 2) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.count = count

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append(("execute", query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.calls.append(("fetch", query, args))
        return [(str(self.count),)]

    def cypher(self) -> list[tuple[str, dict[str, Any]]]:
        return [(q, json.loads(a[0])) for k, q, a in self.calls if k == "fetch"]


def graph_connect(cursor: Cursor) -> Any:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def connect() -> Any:
        yield cursor

    return connect


async def test_graph_meeting_erasure_deletes_the_meeting_and_keeps_confirmed_work():
    cursor = Cursor(count=2)
    consumer = GraphErasureConsumer(
        graph_connect=graph_connect(cursor),
        session_factory=Db(),  # type: ignore[arg-type]
        redis=fakeredis.aioredis.FakeRedis(),
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )
    meeting = str(uuid.uuid4())
    req = parse_request(consumed(admin_value(ORG, kind="meeting", target=meeting)))
    assert await consumer.erase(req, None) == {  # type: ignore[arg-type]
        "meeting_nodes_deleted": 2,
        "edges_deleted": 2,
        "decisions_kept": 2,
        "actions_kept": 2,
    }
    executes = [q for k, q, _ in cursor.calls if k == "execute"]
    assert executes[0] == "BEGIN" and "pg_advisory_xact_lock" in executes[1]
    assert executes[-1] == "COMMIT"
    statements = cursor.cypher()
    assert all(p == {"org_id": str(ORG), "meeting_id": meeting} for _, p in statements)
    text_ = " ".join(q for q, _ in statements)
    assert "SET d.meeting_id = null, d.meeting_erased = true" in text_
    assert "SET a.meeting_id = null, a.meeting_erased = true" in text_
    assert "MATCH (m:Meeting {org_id: $org_id, meeting_id: $meeting_id}) DETACH DELETE m" in text_
    assert "DELETE d" not in text_ and "DELETE a" not in text_


async def test_graph_person_erasure_detaches_the_person_only():
    cursor = Cursor(count=1)
    consumer = GraphErasureConsumer(
        graph_connect=graph_connect(cursor),
        session_factory=Db(),  # type: ignore[arg-type]
        redis=fakeredis.aioredis.FakeRedis(),
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )
    person = str(uuid.uuid4())
    assert await feed(consumer, admin_value(ORG, kind="person", target=person)) == "applied"
    statements = cursor.cypher()
    assert all(p == {"org_id": str(ORG), "user_id": person} for _, p in statements)
    assert any("DETACH DELETE p" in q for q, _ in statements)
    assert not any("Action" in q or "Decision" in q for q, _ in statements)


async def test_late_graph_facts_about_erased_targets_are_erased_again_and_not_announced(
    monkeypatch,
):
    redis = fakeredis.aioredis.FakeRedis()
    meeting, person, action = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    t = now_us()
    await redis.set(tombstone_key(ORG, "meeting", meeting), str(t))
    await redis.set(tombstone_key(ORG, "person", person), str(t))
    facts = [
        GraphFact("edge_created", "PRODUCED", action, meeting),
        GraphFact("edge_created", "ASSIGNED_TO", action, person),
    ]

    async def written(self: Any, evt: ConsumedEvent) -> list[GraphFact]:
        return list(facts)

    monkeypatch.setattr(ConfirmedFactsConsumer, "write_graph", written)
    cursor = Cursor(count=1)
    consumer = ErasureAwareFactsConsumer(
        graph_connect=graph_connect(cursor),
        session_factory=Db(),  # type: ignore[arg-type]
        redis=redis,
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )
    late = ConsumedEvent(
        topic="actions",
        org_id=ORG,
        idempotency_key="a",
        header={"occurred_at": t - 1},
        body={"action_id": action, "meeting_id": meeting, "assignee_id": person},
        partition=0,
        offset=0,
    )
    assert await consumer.write_graph(late) == []
    params = [p for _, p in cursor.cypher()]
    assert {"org_id": str(ORG), "meeting_id": meeting} in params
    assert {"org_id": str(ORG), "user_id": person} in params

    # After the erasure, the same person assigned work in a live meeting is attributed normally.
    cursor.calls.clear()
    fresh = ConsumedEvent(**{**late.__dict__, "header": {"occurred_at": t + 1}})
    fresh.body["meeting_id"] = str(uuid.uuid4())
    facts[0] = GraphFact("edge_created", "PRODUCED", action, fresh.body["meeting_id"])
    assert await consumer.write_graph(fresh) == facts
    assert cursor.calls == []
