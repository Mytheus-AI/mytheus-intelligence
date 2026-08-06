"""`TranscriptChunkSearchConsumer` end to end against REAL infrastructure: real Schema Registry
Avro encode/decode against the LIVE registered `transcript-chunks-value` schema (version 2, the
one carrying `redacted`), real Postgres (`intelligence.transcript_chunk_search` under real RLS),
real Redis dedupe.

`transcript_chunks` has NO write path anywhere in the product today (confirmed by task 9's own
investigation, recorded in the band's progress ledger) — only `transcription-service`'s own
in-process pipeline. So, like `test_embedding_consumer.py` did for the identical topic, these
tests construct real, schema-valid `TranscriptChunk` Avro messages and push them through the
REAL Schema Registry, rather than fake the event or read a Postgres table nobody can write to in
this environment. This is not a stand-in for genuine content: every message here is validated
against the actual live registered schema before this consumer ever sees it.

Covers the plan's acceptance checks (reinterpreted for this repo's own FTS read model — see
`consumer.py`'s module docstring for why: this consumer builds `intelligence.
transcript_chunk_search`, not a column on `public.transcript_chunks`, which this repo does not
own):
- A `transcript-chunks` event is searchable (GIN-indexed generated `tsv`) immediately after this
  consumer applies it.
- Redelivering the same event never duplicates the row (both the Redis-warm and the
  cache-evicted/durable-inbox paths, plus the table's own `ON CONFLICT DO NOTHING`).
- `redacted=true` chunks are never indexed.
- `is_final=false` chunks are never indexed (defensive; the topic is documented finals-only).
- A record whose `org_id` is missing/unparseable goes to the DLQ, never guessed (R6).
- A deserialize failure (poison bytes) goes to the DLQ, never raises into the caller.
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

from services.search_service.consumer import TranscriptChunkSearchConsumer
from tests.conftest import APP_URL

pytestmark = pytest.mark.asyncio


def _transcript_chunk_value(
    *,
    org_id: uuid.UUID,
    meeting_id: uuid.UUID,
    chunk_id: uuid.UUID,
    text_: str,
    is_final: bool = True,
    redacted: bool = False,
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
        "is_final": is_final,
        "confidence": 0.98,
        "redacted": redacted,
    }


async def _row_for(engine, org_id: uuid.UUID, chunk_id: uuid.UUID):
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        return (
            await conn.execute(
                text("SELECT * FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
                {"c": str(chunk_id)},
            )
        ).first()


async def _cleanup(engine, org_id: uuid.UUID, chunk_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await conn.execute(
            text("DELETE FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"),
            {"c": str(chunk_id)},
        )


@pytest.fixture()
def consumer(migrated_db, redis_client, avro_codec) -> TranscriptChunkSearchConsumer:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    dlq_producer = Producer({"bootstrap.servers": "localhost:9092"})
    return TranscriptChunkSearchConsumer(
        session_factory=session_factory,
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=dlq_producer,
    )


async def test_transcript_chunk_is_indexed_and_searchable_end_to_end(consumer, redis_client):
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    value = _transcript_chunk_value(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        text_="We decided to ship the search service this sprint.",
    )
    raw = consumer._codec.serialize_for("transcript-chunks", value)
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    try:
        outcome = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=1
        )
        assert outcome == "applied"

        row = await _row_for(engine, org_id, chunk_id)
        assert row is not None
        assert row.meeting_id == meeting_id
        assert row.speaker_id == "spk-1"
        assert row.start_ms == 0
        assert row.end_ms == 2000
        assert row.text == "We decided to ship the search service this sprint."

        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            matched = (
                await conn.execute(
                    text(
                        "SELECT ts_rank(tsv, plainto_tsquery('english', 'ship search service')) "
                        "AS rank FROM intelligence.transcript_chunk_search WHERE chunk_id = :c"
                    ),
                    {"c": str(chunk_id)},
                )
            ).scalar_one()
        assert matched > 0

        # --- Redelivery path 1: Redis dedupe cache still warm ---
        outcome2 = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=2
        )
        assert outcome2 == "duplicate"

        # --- Redelivery path 2: Redis cache evicted, durable inbox is the authority ---
        await redis_client.delete(f"nexus:{org_id}:dedupe:transcript-chunks:chunk:{chunk_id}")
        outcome3 = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=3
        )
        assert outcome3 == "duplicate"

        async with engine.begin() as conn:
            await conn.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            count = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM intelligence.transcript_chunk_search "
                        "WHERE chunk_id = :c"
                    ),
                    {"c": str(chunk_id)},
                )
            ).scalar_one()
        assert count == 1  # no growth across three deliveries
    finally:
        await _cleanup(engine, org_id, chunk_id)
        await engine.dispose()


async def test_non_final_chunk_is_not_indexed(consumer):
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    value = _transcript_chunk_value(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        text_="interim guess",
        is_final=False,
    )
    raw = consumer._codec.serialize_for("transcript-chunks", value)
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    try:
        outcome = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=1
        )
        assert outcome == "applied"  # consumed + inbox-recorded; just no index row
        row = await _row_for(engine, org_id, chunk_id)
        assert row is None
    finally:
        await _cleanup(engine, org_id, chunk_id)
        await engine.dispose()


async def test_redacted_chunk_is_not_indexed(consumer):
    org_id, meeting_id, chunk_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    value = _transcript_chunk_value(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        text_="this contains someone's SSN",
        redacted=True,
    )
    raw = consumer._codec.serialize_for("transcript-chunks", value)
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    try:
        outcome = await consumer.process_raw(
            topic="transcript-chunks", raw_value=raw, partition=0, offset=1
        )
        assert outcome == "applied"
        row = await _row_for(engine, org_id, chunk_id)
        assert row is None
    finally:
        await _cleanup(engine, org_id, chunk_id)
        await engine.dispose()


async def test_missing_org_id_goes_to_dlq_never_guessed(consumer):
    """R6: identical proof shape to `test_embedding_consumer.py`'s equivalent test, re-run for
    THIS consumer group — do not trust the base class's own suite by inference.

    A genuine, disclosed finding from an earlier version of this test (whole-band review):
    `BaseConsumer._to_dlq` used to name the DLQ topic purely from the SOURCE topic
    (`f"{topic}.dlq"`), not the consumer group, so `transcript-chunks.dlq` was SHARED between
    every consumer group of `transcript-chunks` (`embedding-service.embed-on-ingest` AND
    `search-service.transcript-index` both landed there, distinguishable only by the in-payload
    `consumer_group` field) — a test reading it from `earliest` with a fresh reader group could
    genuinely pick up ANOTHER consumer's old poison message first, and once did. `_to_dlq` now
    scopes the topic name itself by consumer group (`BaseConsumer.dlq_topic_for`), so this
    consumer's DLQ topic is structurally its own; a unique idempotency_key plus scanning for
    the matching envelope is kept anyway as cheap insurance against a stale message from an
    EARLIER run of this same test/topic combination, not against a different consumer group.
    """
    unique_idem = f"chunk:search-service-test-{uuid.uuid4()}"

    class _BadOrgCodec:
        def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
            return {
                "header": {
                    "event_id": str(uuid.uuid4()),
                    "idempotency_key": unique_idem,
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
                "redacted": False,
            }

    consumer._codec = _BadOrgCodec()
    real_consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "group.id": f"test-dlq-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
        }
    )
    dlq_topic = consumer.dlq_topic_for("transcript-chunks")
    real_consumer.subscribe([dlq_topic])
    real_consumer.poll(0)

    outcome = await consumer.process_raw(
        topic="transcript-chunks", raw_value=b"irrelevant-bytes", partition=0, offset=99
    )
    assert outcome == "dlq"

    envelope = None
    for _ in range(200):  # scan past any older poison messages from earlier runs of this test
        msg = real_consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        candidate = json.loads(msg.value())
        if candidate.get("idempotency_key") == unique_idem:
            envelope = candidate
            break
    real_consumer.close()
    assert envelope is not None, f"expected this test's own message on {dlq_topic}"
    assert envelope["failure_class"] == "BUSINESS_INVALID"
    assert envelope["org_id"] == "not-a-uuid"
    assert envelope["consumer_group"] == "search-service.transcript-index"
    assert base64.b64decode(envelope["raw_value_b64"]) == b"irrelevant-bytes"


async def test_deserialize_failure_goes_to_dlq(consumer):
    outcome = await consumer.process_raw(
        topic="transcript-chunks", raw_value=b"\x00not-real-avro-bytes", partition=0, offset=100
    )
    assert outcome == "dlq"
