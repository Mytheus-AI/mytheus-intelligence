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

    # Subscribe and pin every assigned partition's start offset to its CURRENT high watermark
    # BEFORE the outbox row is even inserted, not merely before producing. `embedding-events` is
    # a long-lived topic that both a live ambient relay worker (per the shared dev stack) and
    # this repo's own embedding-consumer tests keep appending to across every run, all day -
    # scanning from `earliest` is wrong on any topic with history: with a growing backlog spread
    # across partitions, a bounded scan from the beginning can exhaust its budget before ever
    # reaching the message this run just produced, and a bigger fixed budget only pushes the
    # same failure out to a bigger topic. Pinning to "whatever is newest right now, per
    # partition" means this test's correctness never depends on how much history the topic
    # already has - it holds on an empty topic and on one with a million messages the same way,
    # because it only ever reads messages produced AFTER this point, never the backlog. The pin
    # must happen before the INSERT below, not just before this test's own `relay_once` call:
    # the ambient relay worker can drain a newly inserted row before this test ever gets to
    # subscribing, and if the pin happened after the INSERT, that worker's produce could land
    # before the pin and be missed entirely.
    real_consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "group.id": f"test-relay-{uuid.uuid4()}",
            "auto.offset.reset": "latest",  # defensive fallback; the on_assign below is what
            # actually pins the offset, so this only matters if that somehow does not run
        }
    )
    assigned_partitions: list[int] = []

    def _pin_to_latest(consumer: Consumer, partitions: list) -> None:
        for p in partitions:
            _, high_watermark = consumer.get_watermark_offsets(p, timeout=10, cached=False)
            p.offset = high_watermark
        consumer.assign(partitions)
        assigned_partitions.extend(p.partition for p in partitions)

    real_consumer.subscribe(["embedding-events"], on_assign=_pin_to_latest)

    # Block until the subscription above has actually run (on_assign fires from inside poll()),
    # so "pinned to latest" is resolved to real offsets before anything is produced.
    assign_deadline = time.monotonic() + 15
    while not assigned_partitions and time.monotonic() < assign_deadline:
        real_consumer.poll(0.5)
    assert assigned_partitions, (
        "consumer group never received a partition assignment for 'embedding-events' within "
        "15s - cannot prove the round trip without one"
    )

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

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session, session.begin():
        published = await relay_once(session, relay_producer)

    await engine.dispose()
    assert published >= 1

    # Consume the message back and confirm it round-tripped. Deadline-based, not a fixed poll
    # count: a fixed count still silently assumes how long delivery/visibility takes, which is
    # exactly the assumption that breaks once the topic grows - a wall-clock budget is what
    # "search until it's really not showing up, then say so accurately" actually means.
    found = None
    scanned = 0
    partitions_seen: set[int] = set()
    search_deadline = time.monotonic() + 30
    while found is None and time.monotonic() < search_deadline:
        msg = real_consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        scanned += 1
        partitions_seen.add(msg.partition())
        decoded = avro_codec.deserialize_for("embedding-events", msg.value())
        if str(decoded["header"]["idempotency_key"]) == idem:
            found = decoded
    real_consumer.close()

    # A message-not-found failure here used to read "relayed message not found on the bus",
    # which points at the relay - and is exactly why this test's real failure mode (the search
    # window exhausted, not the message ever being absent) got misread as a broker flake. Say
    # what this test actually knows.
    assert found is not None, (
        f"did not find idempotency_key={idem!r} on 'embedding-events' within the 30s search "
        f"window - scanned {scanned} message(s) across {len(partitions_seen)} partition(s) "
        f"({sorted(partitions_seen)}) of {len(assigned_partitions)} assigned. The search window "
        "was exhausted; this does NOT mean the message is absent from the topic - relay_once() "
        "reported published >= 1 above, so the produce itself succeeded."
    )
    assert found["collection"] == collection
    assert found["point_count"] == 1
    assert str(found["header"]["org_id"]) == str(org_id)
