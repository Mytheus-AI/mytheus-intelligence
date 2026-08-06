"""Proves the relay path for real: an `intelligence.outbox` row for `embedding-events` is
Avro-serialized against the REAL registered `embedding-events-value` subject, produced to REAL
Redpanda, and readable back by a real consumer — the mechanism `embedding-service`'s consumer
relies on to actually deliver its ack (the plan's "embedding-events ack is emitted per
processed message" acceptance check, proven end-to-end through the wire, not just the DB row).

Uses `libs/shared/relay.py`'s real `relay_once`, unmodified from the mytheus-backend-derived
implementation, with a small adapter satisfying its `RelayProducer` protocol via
`confluent_kafka.Producer` + `libs/shared/avro_codec.AvroCodec` — the same pairing
`services/embedding_service/relay_main.py` uses to run for real.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from confluent_kafka import Consumer, Producer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.relay import relay_once
from tests.conftest import APP_URL, RELAY_URL

pytestmark = pytest.mark.asyncio


class _KafkaAvroProducer:
    def __init__(self, producer: Producer, codec) -> None:
        self._producer = producer
        self._codec = codec

    def serialize(self, subject: str, value: dict) -> bytes:
        topic = subject.removesuffix("-value")
        return self._codec.serialize_for(topic, value)

    def produce(self, *, topic: str, key: bytes, value: bytes) -> None:
        self._producer.produce(topic=topic, key=key, value=value)

    def flush(self) -> int:
        return self._producer.flush(10)


async def test_relay_publishes_embedding_events_row_to_real_redpanda(migrated_db, avro_codec):
    org_id = uuid.uuid4()
    idem = f"emb:transcript_chunks:relay-test-{uuid.uuid4()}"
    collection = f"transcript_chunks__{org_id.hex}"

    # Insert as mytheus_ai_app (the role this repo's consumers actually write outbox rows as),
    # with app.current_org set — mytheus_ai_relay deliberately holds no INSERT grant on
    # intelligence.outbox (SELECT, UPDATE only, per migration 9c2b7e4a1d03), so writing the
    # fixture row through the relay role would be testing a permission the real system never
    # uses either.
    app_engine = create_async_engine(APP_URL, pool_pre_ping=True)
    async with app_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await conn.execute(
            text(
                "INSERT INTO intelligence.outbox "
                "(org_id, topic, partition_key, idempotency_key, schema_subject, payload) "
                "VALUES (:org_id, 'embedding-events', :pk, :idem, 'embedding-events-value', "
                "CAST(:payload AS jsonb))"
            ),
            {
                "org_id": str(org_id),
                "pk": str(org_id),
                "idem": idem,
                "payload": json.dumps(
                    {
                        "header": {
                            "event_id": str(uuid.uuid4()),
                            "idempotency_key": idem,
                            "org_id": str(org_id),
                            "schema_version": 1,
                            "producer": "embedding-service",
                            "occurred_at": int(time.time() * 1_000_000),
                            "correlation_id": None,
                            "causation_id": None,
                        },
                        "collection": collection,
                        "point_count": 1,
                        "meeting_id": None,
                    }
                ),
            },
        )
    await app_engine.dispose()

    engine = create_async_engine(RELAY_URL, pool_pre_ping=True)
    kafka_producer = Producer({"bootstrap.servers": "localhost:9092"})
    relay_producer = _KafkaAvroProducer(kafka_producer, avro_codec)

    real_consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "group.id": f"test-relay-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
        }
    )
    real_consumer.subscribe(["embedding-events"])
    real_consumer.poll(0)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session, session.begin():
        published = await relay_once(session, relay_producer)

    await engine.dispose()
    assert published >= 1

    msg = None
    for _ in range(20):
        msg = real_consumer.poll(1.0)
        if msg is not None and not msg.error():
            decoded = avro_codec.deserialize_for("embedding-events", msg.value())
            if str(decoded["header"]["idempotency_key"]) == idem:
                break
            msg = None
    real_consumer.close()

    assert msg is not None, "expected the relayed embedding-events message on the real topic"
    decoded = avro_codec.deserialize_for("embedding-events", msg.value())
    assert decoded["collection"] == collection
    assert decoded["point_count"] == 1
    assert str(decoded["header"]["org_id"]) == str(org_id)
