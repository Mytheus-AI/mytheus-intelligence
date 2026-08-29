"""Sustained load against `embedding-service`, and the consumer-lag SLO it has never met.

Closes the measurement half of status-overview gap 18's sibling, recorded in the Plan 5-6 band
file as gap 7: "No sustained-load test exists for `embedding-service`'s sub-5 s consumer-lag
SLO. Only single-message tests were run."

Every existing embedding test calls `process_raw` directly with a hand-built message. That
proves the handler, and it cannot prove the SLO: it never subscribes, never joins a consumer
group, never commits an offset, and never lets a backlog form. Consumer lag is a property of
the group against the broker's high-watermark, so it is only observable through the real loop.

This test runs the real `run_forever` against real Redpanda with a real consumer group and a
real backlog, then reads the lag from the broker itself rather than inferring it.

WHAT IT ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
It asserts the properties that hold on any hardware: the backlog is fully drained, every
message produces its vector, offsets advance to the high-watermark, and lag returns to zero.

It does NOT assert lag < 5 s. On CPU that assertion would fail, and the honest reason is
recorded rather than the threshold relaxed until it passes: `embedding-service`'s own band
record measures batch-32 embedding at ~3536 ms on CPU against a sub-100 ms GPU target, so a
sustained backlog cannot clear inside 5 s on this hardware and no amount of consumer tuning
changes that. The SLO is GPU-dependent, which is exactly what status-overview gap 18 says is
unbuilt. The measured drain rate is printed so the number is on the record instead of a claim.

Marked `slow`: it embeds real text with a real model. `make test-fast` skips it.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import pytest
from confluent_kafka import Consumer, Producer, TopicPartition
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.qdrant import LOGICAL_TRANSCRIPT_CHUNKS, physical_collection
from services.embedding_service.consumer import EmbeddingConsumer
from tests.conftest import APP_URL

pytestmark = pytest.mark.slow

TOPIC = "transcript-chunks"
BACKLOG = 12  # enough to form a real backlog without making the suite unusable on CPU
DRAIN_TIMEOUT_S = 420.0


def _chunk(
    *, org_id: uuid.UUID, meeting_id: uuid.UUID, chunk_id: uuid.UUID, i: int
) -> dict[str, Any]:
    return {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": f"chunk:{chunk_id}",
            "org_id": str(org_id),
            "schema_version": 1,
            "producer": "transcription-service",
            "occurred_at": int(time.time() * 1_000_000),
            "correlation_id": None,
            "causation_id": None,
        },
        "meeting_id": str(meeting_id),
        "chunk_id": str(chunk_id),
        "speaker_id": f"spk-{i % 3}",
        "start_ms": i * 2000,
        "end_ms": (i + 1) * 2000,
        "text": (
            f"Utterance {i}. The team reviewed the migration plan and agreed the rollout "
            f"order, then assigned the follow-up to the platform group."
        ),
        "is_final": True,
        "confidence": 0.97,
    }


def _group_lag(brokers: str, group: str, topic: str) -> int:
    """Total lag for a consumer group, read from the broker.

    Committed offsets come from the group's own metadata and the high-watermark from the
    broker, so this is the same quantity KEDA's kafka trigger scales on
    (`services/embedding_service/keda-scaledobject.yaml`) rather than a number this test
    computes for itself.
    """
    probe = Consumer({"bootstrap.servers": brokers, "group.id": group, "enable.auto.commit": False})
    try:
        meta = probe.list_topics(topic, timeout=10)
        parts = [TopicPartition(topic, p) for p in meta.topics[topic].partitions]
        committed = probe.committed(parts, timeout=10)
        total = 0
        for tp in committed:
            low, high = probe.get_watermark_offsets(tp, timeout=10, cached=False)
            pos = tp.offset if tp.offset >= 0 else low
            total += max(0, high - pos)
        return total
    finally:
        probe.close()


async def test_a_real_backlog_drains_completely_through_the_real_consumer_loop(
    migrated_db, qdrant_client, redis_client, avro_codec
):
    """The first time this service has been run as a subscribed consumer group under load."""
    brokers = "localhost:9092"
    org_id, meeting_id = uuid.uuid4(), uuid.uuid4()

    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    consumer = EmbeddingConsumer(
        qdrant=qdrant_client,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=Producer({"bootstrap.servers": brokers}),
    )
    # A dedicated group per run. Sharing the service's real group would make the result depend
    # on whatever offsets a previous run or a live worker had already committed.
    consumer.group = f"embedding-load-test.{uuid.uuid4().hex[:12]}"

    # --- produce a real backlog BEFORE the consumer starts, so a backlog genuinely exists ---
    producer = Producer({"bootstrap.servers": brokers})
    chunk_ids = [uuid.uuid4() for _ in range(BACKLOG)]
    for i, cid in enumerate(chunk_ids):
        value = _chunk(org_id=org_id, meeting_id=meeting_id, chunk_id=cid, i=i)
        producer.produce(
            TOPIC,
            key=f"{org_id}:{meeting_id}".encode(),
            value=avro_codec.serialize_for(TOPIC, value),
        )
    producer.flush(30)

    lag_before = _group_lag(brokers, consumer.group, TOPIC)
    assert lag_before >= BACKLOG, (
        f"precondition: a backlog of at least {BACKLOG} must exist before the consumer starts, "
        f"saw {lag_before}"
    )

    # --- run the real loop in a thread and let it drain ---
    stop = threading.Event()
    original_run = consumer.run_forever

    def _run() -> None:
        try:
            original_run(brokers=brokers, poll_timeout=1.0)
        except Exception:  # noqa: BLE001 - the stop path closes the consumer under it
            pass

    thread = threading.Thread(target=_run, daemon=True)
    started = time.monotonic()
    thread.start()

    drained_at = None
    try:
        deadline = started + DRAIN_TIMEOUT_S
        while time.monotonic() < deadline:
            if _group_lag(brokers, consumer.group, TOPIC) == 0:
                drained_at = time.monotonic()
                break
            time.sleep(2.0)
    finally:
        # `run_forever` is `while True:` with no stop path, so the thread is a daemon and dies
        # with the process. Recorded rather than worked around: a consumer that cannot be asked
        # to stop also cannot drain cleanly on SIGTERM, which is a real operability gap.
        stop.set()

    assert drained_at is not None, (
        f"the backlog of {BACKLOG} did not drain within {DRAIN_TIMEOUT_S}s; "
        f"lag still {_group_lag(brokers, consumer.group, TOPIC)}"
    )

    elapsed = drained_at - started
    per_message = elapsed / BACKLOG
    print(
        f"\nSUSTAINED LOAD: {BACKLOG} messages drained in {elapsed:.1f}s "
        f"({per_message:.2f}s/message, {BACKLOG / elapsed:.2f} msg/s) on CPU"
    )
    print(
        f"CONSUMER-LAG SLO: the frozen target is under 5s. At {per_message:.2f}s/message a "
        f"steady arrival rate above {1 / per_message:.2f} msg/s grows the backlog without "
        f"bound on this hardware. GPU packaging is status-overview gap 18."
    )

    # --- every message produced its vector; nothing was silently dropped ---
    collection = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
    count = await qdrant_client.count(collection, exact=True)
    assert count.count == BACKLOG, (
        f"expected {BACKLOG} vectors in {collection}, found {count.count}: the loop committed "
        f"offsets for messages whose effect did not land"
    )

    await qdrant_client.delete_collection(collection)


async def test_the_lag_probe_itself_reports_a_real_backlog(migrated_db, avro_codec):
    """A guard on the measurement, not on the service.

    The drain assertion above is only meaningful if `_group_lag` can distinguish a backlog
    from an empty topic. A probe that always returned 0 would make that test pass without a
    consumer running at all, which is the shape of failure this suite has hit before.
    """
    brokers = "localhost:9092"
    group = f"embedding-lag-probe.{uuid.uuid4().hex[:12]}"

    assert _group_lag(brokers, group, TOPIC) >= 0

    producer = Producer({"bootstrap.servers": brokers})
    org_id, meeting_id = uuid.uuid4(), uuid.uuid4()
    for i in range(3):
        producer.produce(
            TOPIC,
            key=f"{org_id}:{meeting_id}".encode(),
            value=avro_codec.serialize_for(
                TOPIC, _chunk(org_id=org_id, meeting_id=meeting_id, chunk_id=uuid.uuid4(), i=i)
            ),
        )
    producer.flush(30)

    after = _group_lag(brokers, group, TOPIC)
    assert after >= 3, f"the probe must see the three messages just produced, saw {after}"
