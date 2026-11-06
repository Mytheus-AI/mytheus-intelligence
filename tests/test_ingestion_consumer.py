"""`IntegrationEventConsumer` end to end against REAL infrastructure: real Avro-encoded
`integration-events` messages (real Schema Registry), real Postgres (`intelligence.*` under
real RLS), real MinIO (the consumer reads an attachment object exactly as
`integration-service` would have already written it), real Redis.

Covers:
- R8: a `(google_calendar, attachment_to_ingest)` event triggers ingestion end to end (object
  read from `mytheus-attachments`, `document_chunks`/`document_elements` written,
  `document-chunks` outbox rows enqueued, `ingestion_jobs` completed).
- R8: any OTHER `integration-events` message (wrong provider, or right provider but wrong
  `event_type`) is a no-op — consumed and inbox-recorded (BaseConsumer's normal at-least-once
  handling of a shared topic), but no ingestion is triggered.
- The object-key trust boundary: a message whose `object_key` does not start with the message's
  OWN verified `org_id` prefix is rejected before any GET reaches MinIO — proven by pointing the
  payload at a REAL object that belongs to a different org and confirming nothing is written.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import pytest
from confluent_kafka import Producer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.object_storage import BUCKET_ATTACHMENTS, attachment_key, put_object
from services.ingestion_service.consumer import IntegrationEventConsumer
from tests.conftest import APP_URL
from tests.fixtures import sample_documents as docs

pytestmark = pytest.mark.asyncio


def _integration_event_value(
    *, org_id: uuid.UUID, provider: str, event_type: str, payload: dict[str, Any]
) -> dict[str, Any]:
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
        "provider": provider,
        "event_type": event_type,
        "external_id": f"ext-{uuid.uuid4()}",
        "payload_json": json.dumps(payload),
    }


@pytest.fixture()
def consumer(migrated_db, s3_client, redis_client, avro_codec) -> IntegrationEventConsumer:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    dlq_producer = Producer({"bootstrap.servers": "localhost:9092"})
    return IntegrationEventConsumer(
        s3_client=s3_client,
        session_factory=session_factory,
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=dlq_producer,
    )


async def _job_and_chunk_counts(consumer, org_id) -> tuple[int, int]:
    async with consumer._session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        jobs = (
            await session.execute(
                text("SELECT count(*) FROM intelligence.ingestion_jobs WHERE org_id = :o"),
                {"o": str(org_id)},
            )
        ).scalar_one()
        chunks = (
            await session.execute(
                text("SELECT count(*) FROM intelligence.document_chunks WHERE org_id = :o"),
                {"o": str(org_id)},
            )
        ).scalar_one()
    return jobs, chunks


async def test_calendar_attachment_event_triggers_real_ingestion_end_to_end(consumer, s3_client):
    org_id, meeting_id = uuid.uuid4(), uuid.uuid4()
    document_id_for_key = uuid.uuid4()  # a stand-in "attachment id"; integration-service's own
    # naming, opaque to this consumer except as the key it must read exactly.
    key = attachment_key(org_id, meeting_id, document_id_for_key, ".docx")
    put_object(
        s3_client,
        bucket=BUCKET_ATTACHMENTS,
        key=key,
        body=docs.make_docx(),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    value = _integration_event_value(
        org_id=org_id,
        provider="google_calendar",
        event_type="attachment_to_ingest",
        payload={"meeting_id": str(meeting_id), "object_key": key, "filename": "notes.docx"},
    )
    raw = consumer._codec.serialize_for("integration-events", value)

    try:
        outcome = await consumer.process_raw(
            topic="integration-events", raw_value=raw, partition=0, offset=1
        )
        assert outcome == "applied"

        jobs, chunks = await _job_and_chunk_counts(consumer, org_id)
        assert jobs == 1
        assert chunks > 0

        async with consumer._session_factory() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            job = (
                await session.execute(
                    text(
                        "SELECT status, source, bucket, object_key, chunk_count "
                        "FROM intelligence.ingestion_jobs WHERE org_id = :o"
                    ),
                    {"o": str(org_id)},
                )
            ).first()
            outbox_rows = (
                await session.execute(
                    text(
                        "SELECT payload FROM intelligence.outbox "
                        "WHERE org_id = :o AND topic = 'document-chunks'"
                    ),
                    {"o": str(org_id)},
                )
            ).all()
        assert job.status == "completed"
        assert job.source == "calendar_attachment"
        assert job.bucket == BUCKET_ATTACHMENTS
        assert job.object_key == key
        assert job.chunk_count == chunks
        assert len(outbox_rows) == chunks
        for row in outbox_rows:
            assert row.payload["document_id"] is not None
            assert row.payload["element_type"] in {"Title", "NarrativeText", "Table", "ListItem"}
    finally:
        s3_client.delete_object(Bucket=BUCKET_ATTACHMENTS, Key=key)


@pytest.mark.parametrize(
    ("provider", "event_type"),
    [
        ("slack", "message_posted"),
        ("jira", "issue_created"),
        ("linear", "issue_created"),
        ("google_calendar", "event_updated"),  # right provider, wrong event_type
    ],
)
async def test_non_calendar_attachment_events_are_a_no_op(consumer, provider, event_type):
    """R8: this consumer must not widen ingestion triggers beyond Calendar attachments."""
    org_id = uuid.uuid4()
    value = _integration_event_value(
        org_id=org_id,
        provider=provider,
        event_type=event_type,
        payload={"meeting_id": str(uuid.uuid4()), "object_key": f"{org_id}/irrelevant.docx"},
    )
    raw = consumer._codec.serialize_for("integration-events", value)

    outcome = await consumer.process_raw(
        topic="integration-events", raw_value=raw, partition=0, offset=2
    )
    assert outcome == "applied"  # consumed and inbox-recorded; BaseConsumer's normal handling

    jobs, chunks = await _job_and_chunk_counts(consumer, org_id)
    assert jobs == 0  # no ingestion job was ever created
    assert chunks == 0


async def test_cross_tenant_object_key_is_rejected_before_any_fetch(consumer, s3_client):
    """The adversarial case: a message whose OWN envelope org_id is org_a, but whose
    payload_json points at an object under org_b's prefix. Must never fetch it."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    meeting_id, document_id_for_key = uuid.uuid4(), uuid.uuid4()
    # A REAL object that genuinely exists, under org_b's prefix.
    key_under_org_b = attachment_key(org_b, meeting_id, document_id_for_key, ".docx")
    put_object(
        s3_client,
        bucket=BUCKET_ATTACHMENTS,
        key=key_under_org_b,
        body=docs.make_docx(),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    value = _integration_event_value(
        org_id=org_a,  # the envelope's VERIFIED org — org_a
        provider="google_calendar",
        event_type="attachment_to_ingest",
        payload={
            "meeting_id": str(meeting_id),
            "object_key": key_under_org_b,  # but the payload claims org_b's object
            "filename": "notes.docx",
        },
    )
    raw = consumer._codec.serialize_for("integration-events", value)

    try:
        outcome = await consumer.process_raw(
            topic="integration-events", raw_value=raw, partition=0, offset=3
        )
        assert outcome == "applied"  # not a DLQ case — a rejected/no-op business decision

        jobs_a, chunks_a = await _job_and_chunk_counts(consumer, org_a)
        assert jobs_a == 0
        assert chunks_a == 0
        jobs_b, chunks_b = await _job_and_chunk_counts(consumer, org_b)
        assert jobs_b == 0  # org_b's real object was never touched by org_a's forged pointer
        assert chunks_b == 0
    finally:
        s3_client.delete_object(Bucket=BUCKET_ATTACHMENTS, Key=key_under_org_b)


async def test_missing_org_id_on_integration_events_goes_to_dlq_never_guessed(consumer):
    """R6, re-proven for this consumer specifically (BaseConsumer's own machinery is already
    proven generically in test_embedding_consumer.py; this confirms IntegrationEventConsumer
    inherits it rather than accidentally overriding process_raw)."""

    class _BadOrgCodec:
        def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
            return {
                "header": {
                    "event_id": str(uuid.uuid4()),
                    "idempotency_key": "intevt:whatever",
                    "org_id": "not-a-uuid",
                    "schema_version": 1,
                    "producer": "integration-service",
                    "occurred_at": 0,
                    "correlation_id": None,
                    "causation_id": None,
                },
                "provider": "google_calendar",
                "event_type": "attachment_to_ingest",
                "external_id": "ext-1",
                "payload_json": "{}",
            }

    consumer._codec = _BadOrgCodec()
    outcome = await consumer.process_raw(
        topic="integration-events", raw_value=b"irrelevant-bytes", partition=0, offset=99
    )
    assert outcome == "dlq"
