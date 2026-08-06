"""EmbeddingConsumer end to end against REAL infrastructure: real BGE-small-en embeddings
(small fixed corpus, per task brief — no random-vector fake), real Qdrant, real Postgres
(`intelligence.outbox`/`inbox` under real RLS), real Schema Registry Avro encode/decode.

Covers the plan's acceptance checks directly:
- A `transcript-chunks` event produces a 384-dim vector in `transcript_chunks__{org_id}`
  carrying `org_id`/`meeting_id`/`chunk_id`/`speaker` payload.
- Redelivering the same `transcript-chunks` event upserts the SAME point (no duplicate),
  proven by stable point id + dedupe (both the Redis-cache path and the durable-inbox path).
- `embedding-events` ack is emitted per processed message (once, not twice on redelivery).
- A record whose `org_id` is missing/unparseable goes to the DLQ, never guessed.
- A deserialize failure (poison bytes) goes to the DLQ, never raises into the caller / blocks.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

import pytest
from confluent_kafka import Consumer, Producer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.qdrant import LOGICAL_TRANSCRIPT_CHUNKS, physical_collection
from services.embedding_service.consumer import EmbeddingConsumer
from tests.conftest import APP_URL

pytestmark = pytest.mark.asyncio


def _transcript_chunk_value(
    *, org_id: uuid.UUID, meeting_id: uuid.UUID, chunk_id: uuid.UUID, text_: str
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
        "speaker_id": "spk-1",
        "start_ms": 0,
        "end_ms": 2000,
        "text": text_,
        "is_final": True,
        "confidence": 0.98,
    }


@pytest.fixture()
def consumer(migrated_db, qdrant_client, redis_client, avro_codec) -> EmbeddingConsumer:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    dlq_producer = Producer({"bootstrap.servers": "localhost:9092"})
    return EmbeddingConsumer(
        qdrant=qdrant_client,
        session_factory=session_factory,
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=dlq_producer,
    )


async def test_transcript_chunk_end_to_end_real_embedding_real_qdrant_real_outbox(
    consumer, qdrant_client, redis_client
):
    org_id = uuid.uuid4()
    meeting_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    value = _transcript_chunk_value(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        text_="We decided to ship the embedding service this sprint.",
    )
    raw = consumer._codec.serialize_for("transcript-chunks", value)

    outcome = await consumer.process_raw(
        topic="transcript-chunks", raw_value=raw, partition=0, offset=1
    )
    assert outcome == "applied"

    name = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
    try:
        count = (await qdrant_client.count(name, exact=True)).count
        assert count >= 1

        points, _ = await qdrant_client.scroll(name, limit=10, with_payload=True, with_vectors=True)
        point = points[0]
        assert len(point.vector) == 384
        assert point.payload["org_id"] == str(org_id)
        assert point.payload["meeting_id"] == str(meeting_id)
        assert point.payload["chunk_id"] == str(chunk_id)
        assert point.payload["speaker"] == "spk-1"
        assert point.payload["is_final"] is True

        # --- Redelivery path 1: Redis dedupe cache still warm ---
        outcome2 = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=2
        )
        assert outcome2 == "duplicate"
        assert (await qdrant_client.count(name, exact=True)).count == count  # no growth

        # --- Redelivery path 2: Redis cache evicted, durable inbox is the authority ---
        await redis_client.delete(f"nexus:{org_id}:dedupe:transcript-chunks:chunk:{chunk_id}")
        outcome3 = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=3
        )
        assert outcome3 == "duplicate"
        assert (await qdrant_client.count(name, exact=True)).count == count  # still no growth

        # --- embedding-events ack emitted exactly once (unique (topic, idempotency_key)) ---
        async with consumer._session_factory() as session:
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            rows = (
                await session.execute(
                    text(
                        "SELECT payload FROM intelligence.outbox "
                        "WHERE topic = 'embedding-events' AND idempotency_key = :k"
                    ),
                    {"k": f"emb:transcript_chunks:{chunk_id}"},
                )
            ).all()
            assert len(rows) == 1
            payload = rows[0].payload
            assert payload["collection"] == name
            assert payload["point_count"] >= 1
    finally:
        await qdrant_client.delete_collection(name)


async def test_missing_org_id_goes_to_dlq_never_guessed(consumer):
    """R6: a record whose org_id is missing or unparseable must go to the DLQ, never guess or
    default a tenant. We stub the codec's decode result (not the infra) to construct the
    otherwise-Avro-illegal case of a malformed org_id without fighting fastavro's logicalType
    validation on the write side."""

    class _BadOrgCodec:
        def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
            return {
                "header": {
                    "event_id": str(uuid.uuid4()),
                    "idempotency_key": "chunk:whatever",
                    "org_id": "not-a-uuid",
                    "schema_version": 1,
                    "producer": "transcription-service",
                    "occurred_at": 0,
                    "correlation_id": None,
                    "causation_id": None,
                },
                "meeting_id": str(uuid.uuid4()),
                "chunk_id": str(uuid.uuid4()),
                "speaker_id": "s",
                "start_ms": 0,
                "end_ms": 1,
                "text": "irrelevant",
                "is_final": True,
                "confidence": 1.0,
            }

    consumer._codec = _BadOrgCodec()
    real_consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "group.id": f"test-dlq-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
        }
    )
    real_consumer.subscribe(["transcript-chunks.dlq"])
    real_consumer.poll(0)  # trigger subscription/assignment

    outcome = await consumer.process_raw(
        topic="transcript-chunks", raw_value=b"irrelevant-bytes", partition=0, offset=99
    )
    assert outcome == "dlq"

    msg = None
    for _ in range(20):
        msg = real_consumer.poll(1.0)
        if msg is not None and not msg.error():
            break
    real_consumer.close()
    assert msg is not None, "expected a message on transcript-chunks.dlq"
    envelope = json.loads(msg.value())
    assert envelope["failure_class"] == "BUSINESS_INVALID"
    assert envelope["org_id"] == "not-a-uuid"
    assert envelope["consumer_group"] == "embedding-service.embed-on-ingest"
    assert base64.b64decode(envelope["raw_value_b64"]) == b"irrelevant-bytes"


async def test_deserialize_failure_goes_to_dlq(consumer):
    """Poison bytes that are not valid Avro against the registered subject -> DLQ, not a raised
    exception (never blocks the partition)."""
    outcome = await consumer.process_raw(
        topic="transcript-chunks", raw_value=b"\x00not-real-avro-bytes", partition=0, offset=100
    )
    assert outcome == "dlq"
