"""Connected handoff: `integration-events` -> ingestion-service -> `document-chunks` ->
embedding-service -> Qdrant, through the REAL entrypoints (`BaseConsumer.run_forever`,
`libs.shared.relay.relay_once`) with only the BROKER faked.

Existing tests call `process_raw` / `ingest_document` directly inside one loop, so nothing ran the
ingestion output through the relay into the embedding consumer's real poll/commit loop. Here:

  in-memory broker  <- test produces an Avro `integration-events` message (real codec)
  IntegrationEventConsumer.run_forever  (real loop; confluent `Consumer` swapped for an
        in-memory one) -> real Postgres rows + `intelligence.outbox` rows
  relay_once (real) -> in-memory producer (real Avro serialize) -> broker `document-chunks`
  EmbeddingConsumer.run_forever -> real BGE embed -> real Qdrant upsert -> `embedding-events` outbox

Everything except the broker is real, so this needs the dev stack (Postgres, Redis, Qdrant,
MinIO, Schema Registry) and skips/fails-loud like the other infra tests. NOT RUN in the session
that wrote it (infra was down): UNVERIFIED until executed against the stack.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Any

import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import libs.shared.consumer as consumer_mod
from libs.shared.object_storage import BUCKET_ATTACHMENTS, attachment_key, put_object
from libs.shared.qdrant import LOGICAL_DOCUMENT_CHUNKS, physical_collection, write_client
from libs.shared.relay import relay_once
from services.embedding_service.consumer import EmbeddingConsumer
from services.embedding_service.relay_main import _KafkaAvroProducer
from services.ingestion_service.consumer import IntegrationEventConsumer
from tests.conftest import APP_URL, RELAY_URL
from tests.fixtures import sample_documents as docs

pytestmark = pytest.mark.asyncio


class _Drained(Exception):
    """Raised by the fake consumer when its subscribed backlog is exhausted (ends run_forever)."""


class InMemoryBroker:
    """topic -> ordered list of (key, value). The only fake in this test."""

    def __init__(self) -> None:
        self.topics: dict[str, list[tuple[bytes | None, bytes]]] = defaultdict(list)
        self.committed: dict[tuple[str, str], int] = {}

    # confluent_kafka.Producer surface used by BaseConsumer._to_dlq and the relay adapter
    def produce(self, topic: str, key: bytes | None = None, value: bytes = b"") -> None:
        self.topics[topic].append((key, value))

    def flush(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return 0


class _Msg:
    def __init__(self, topic: str, offset: int, key: bytes | None, value: bytes) -> None:
        self._t, self._o, self._k, self._v = topic, offset, key, value

    def error(self) -> None:
        return None

    def topic(self) -> str:
        return self._t

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return self._o

    def key(self) -> bytes | None:
        return self._k

    def value(self) -> bytes:
        return self._v


def _install_fake_kafka_consumer(monkeypatch, broker: InMemoryBroker, key_prefix: bytes):
    """Swap `confluent_kafka.Consumer` inside libs.shared.consumer so the REAL run_forever loop
    (poll -> process_raw -> manual commit) reads from the in-memory broker. `key_prefix` stands
    in for partition selection so leftover outbox rows from other tests/orgs are not delivered."""

    class FakeConsumer:
        def __init__(self, conf: dict[str, Any]) -> None:
            self.group = conf["group.id"]
            assert conf["enable.auto.commit"] is False  # the at-least-once contract
            self.pending: list[_Msg] = []
            self.commits = 0

        def subscribe(self, topics: list[str]) -> None:
            for t in topics:
                for off, (k, v) in enumerate(broker.topics[t]):
                    if k is not None and k.startswith(key_prefix):
                        self.pending.append(_Msg(t, off, k, v))

        def poll(self, timeout: float) -> _Msg | None:  # noqa: ARG002
            if not self.pending:
                raise _Drained
            return self.pending.pop(0)

        def commit(self, message: _Msg, asynchronous: bool) -> None:
            assert asynchronous is False
            broker.committed[(self.group, message.topic())] = message.offset() + 1
            self.commits += 1

        def close(self) -> None:
            pass

    monkeypatch.setattr(consumer_mod, "Consumer", FakeConsumer)


async def _run_loop(consumer: Any, brokers: str = "in-memory") -> None:
    """run_forever is a blocking sync loop that owns its own event loop: run it in a thread."""

    def _go() -> None:
        try:
            consumer.run_forever(brokers=brokers, poll_timeout=0.01)
        except _Drained:
            return

    await asyncio.to_thread(_go)


def _integration_event(org_id: uuid.UUID, meeting_id: uuid.UUID, key: str) -> dict[str, Any]:
    return {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": f"intevt:{uuid.uuid4()}",
            "org_id": str(org_id),
            "schema_version": 1,
            "producer": "integration-service",
            "occurred_at": int(time.time() * 1_000_000),
            "correlation_id": None,
            "causation_id": None,
        },
        "provider": "google_calendar",
        "event_type": "attachment_to_ingest",
        "external_id": f"ext-{uuid.uuid4()}",
        "payload_json": json.dumps(
            {"meeting_id": str(meeting_id), "object_key": key, "filename": "notes.docx"}
        ),
    }


def _fresh_consumer_kwargs(broker: InMemoryBroker, avro_codec) -> dict[str, Any]:
    # Fresh engine/Redis per consumer: they bind to run_forever's private event loop.
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    return {
        "session_factory": async_sessionmaker(engine, expire_on_commit=False),
        "redis": Redis.from_url(os.environ["REDIS_URL"]),
        "codec": avro_codec,
        "dlq_producer": broker,
    }


async def _scalar(org_id: uuid.UUID, sql: str) -> int:
    engine = create_async_engine(APP_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            return int((await conn.execute(text(sql), {"o": str(org_id)})).scalar_one())
    finally:
        await engine.dispose()


async def test_calendar_attachment_flows_through_relay_into_qdrant(
    migrated_db, s3_client, qdrant_client, avro_codec, monkeypatch
):
    org_id, meeting_id = uuid.uuid4(), uuid.uuid4()
    key = attachment_key(org_id, meeting_id, uuid.uuid4(), ".docx")
    put_object(
        s3_client,
        bucket=BUCKET_ATTACHMENTS,
        key=key,
        body=docs.make_docx(),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    broker = InMemoryBroker()
    org_key = str(org_id).encode()
    broker.topics["integration-events"].append(
        (
            org_key,
            avro_codec.serialize_for(
                "integration-events", _integration_event(org_id, meeting_id, key)
            ),
        )
    )

    # 1. ingestion-service's real loop consumes the integration event.
    _install_fake_kafka_consumer(monkeypatch, broker, org_key)
    ingestion = IntegrationEventConsumer(
        s3_client=s3_client, **_fresh_consumer_kwargs(broker, avro_codec)
    )
    await _run_loop(ingestion)
    assert broker.committed[(ingestion.group, "integration-events")] == 1
    assert not any(t.endswith(".dlq") for t in broker.topics if broker.topics[t])

    n_chunks = await _scalar(
        org_id, "SELECT count(*) FROM intelligence.document_chunks WHERE org_id = :o"
    )
    assert n_chunks > 0
    assert (
        await _scalar(
            org_id,
            "SELECT count(*) FROM intelligence.outbox WHERE org_id = :o "
            "AND topic = 'document-chunks'",
        )
        == n_chunks
    )

    # 2. The real relay drains intelligence.outbox to the (in-memory) broker as Avro.
    relay_engine = create_async_engine(RELAY_URL, pool_pre_ping=True)
    relay_producer = _KafkaAvroProducer(broker, avro_codec)  # type: ignore[arg-type]
    async with async_sessionmaker(relay_engine, expire_on_commit=False)() as s, s.begin():
        await relay_once(s, relay_producer)
    await relay_engine.dispose()
    mine = [(k, v) for k, v in broker.topics["document-chunks"] if k == org_key]
    assert len(mine) == n_chunks

    # 3. embedding-service's real loop consumes document-chunks and writes Qdrant.
    embedding = EmbeddingConsumer(
        qdrant=write_client(), **_fresh_consumer_kwargs(broker, avro_codec)
    )
    await _run_loop(embedding)
    assert broker.committed[(embedding.group, "document-chunks")] == n_chunks
    assert not any(t.endswith(".dlq") for t in broker.topics if broker.topics[t])

    collection = physical_collection(LOGICAL_DOCUMENT_CHUNKS, org_id)
    points, _ = await qdrant_client.scroll(collection, limit=100, with_payload=True)
    assert len(points) >= n_chunks
    assert {p.payload["org_id"] for p in points} == {str(org_id)}
    assert len({p.payload["document_id"] for p in points}) == 1
    assert (
        await _scalar(
            org_id,
            "SELECT count(*) FROM intelligence.outbox WHERE org_id = :o "
            "AND topic = 'embedding-events'",
        )
        == n_chunks
    )

    # 4. At-least-once: a full redelivery of the same document-chunks changes nothing.
    count_before = (await qdrant_client.count(collection, exact=True)).count
    redelivery = EmbeddingConsumer(
        qdrant=write_client(), **_fresh_consumer_kwargs(broker, avro_codec)
    )
    await _run_loop(redelivery)
    assert (await qdrant_client.count(collection, exact=True)).count == count_before
    assert (
        await _scalar(
            org_id,
            "SELECT count(*) FROM intelligence.outbox WHERE org_id = :o "
            "AND topic = 'embedding-events'",
        )
        == n_chunks
    )
