"""`IntegrationEventConsumer` — the `integration-events` consumer, subclassing
`libs.shared.consumer.BaseConsumer` per this repo's established convention (task 7/8: subclass,
never re-implement dedupe/DLQ/offset handling).

--- R8 scope guardrail ---

Risk checkpoint R8 (task brief, verbatim): "`ingestion-service` consumes `integration-events`
ONLY for Calendar attachments. Do not widen ingestion triggers beyond Calendar plus direct
`/ingest` uploads, however tempting." `integration-events` (the real, registered
`IntegrationEvent` schema) also carries Slack and Jira/Linear traffic for OTHER consumers
(`core-api`); this consumer subscribes to the whole topic (Redpanda has no server-side
per-message filtering) but `apply()` immediately no-ops — consumed and inbox-recorded, no
business effect, no `document-chunks` emitted — for anything that is not
`(provider=google_calendar, event_type=attachment_to_ingest)`. This is the correct at-least-once
shape for a shared topic with multiple independent consumer groups (`docs/CLAUDE.md`'s topic
catalog lists both `ingestion-service` and `core-api` as `integration-events` consumers); it is
NOT the same as widening what this consumer ACTS on.

--- The object-key trust boundary (read before touching `_resolve_key`) ---

Per `21-integration-architecture.md`: `integration-service` downloads the calendar attachment
bytes and writes them to `nexus-attachments` BEFORE emitting `integration-events`. This consumer
therefore READS an object another service already wrote; it does not upload one itself (contrast
`routes.py`, which always uploads a fresh object under a server-generated key it fully
controls). `integration-service` does not exist yet in this program (P12, a later plan band) —
this consumer's `payload_json` contract (`meeting_id`, `object_key`, `filename`) is this task's
own reasonable reading of the documented flow, stated plainly as an assumption since there is no
real producer yet to consult; `tests/test_ingestion_consumer.py` constructs synthetic-but-
schema-valid `IntegrationEvent` messages the same way task 7's tests constructed synthetic
`TranscriptChunk`/`DocumentChunk` messages for a producer that also wasn't a live process.

Trusting an `object_key` supplied inside another service's event body is a genuine tenant-
isolation risk if left unchecked: a compromised or buggy `integration-service` (or a forged
message, if the bus were ever reachable by an untrusted producer) could set `object_key` to any
string, including one under ANOTHER org's prefix in the shared `nexus-attachments` bucket. This
consumer is a GET, not a PUT, so the blast radius of an unchecked key is a cross-tenant READ —
still a real leak. `_resolve_key` closes this: the key is used ONLY if it starts with
`f"{org_id}/"` using the SAME verified `org_id` `BaseConsumer.process_raw` already re-asserted
from the event envelope header (never the payload). Anything else is rejected before the GET
ever reaches MinIO (proven in `tests/test_ingestion_consumer.py`).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import boto3
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent
from libs.shared.object_storage import BUCKET_ATTACHMENTS, get_object
from services.ingestion_service.pipeline import complete_job, create_job, fail_job, ingest_document

_CALENDAR_ATTACHMENT = ("google_calendar", "attachment_to_ingest")


class CrossTenantObjectKeyError(ValueError):
    """Raised (and swallowed into a no-op, never a fetch) when an `integration-events` message's
    `object_key` does not start with the verified `org_id` prefix. See the module docstring."""


def _resolve_key(org_id: UUID, object_key: str) -> str:
    prefix = f"{org_id}/"
    if not object_key.startswith(prefix):
        raise CrossTenantObjectKeyError(
            f"object_key {object_key!r} does not start with the verified org prefix {prefix!r}"
        )
    return object_key


class IntegrationEventConsumer(BaseConsumer):
    group = "ingestion-service.attachment-to-ingest"
    topics = ["integration-events"]
    producer_name = "ingestion-service"

    def __init__(self, *, s3_client: boto3.client, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._s3 = s3_client

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        org_id = evt.org_id
        provider = evt.body.get("provider")
        event_type = evt.body.get("event_type")

        if (provider, event_type) != _CALENDAR_ATTACHMENT:
            # R8: not a calendar attachment. Consumed and inbox-recorded by BaseConsumer as
            # normal at-least-once delivery; no ingestion triggered. See module docstring.
            return

        payload = json.loads(evt.body["payload_json"])
        # `meeting_id` is required in the payload_json contract (provenance: which meeting this
        # attachment belongs to) but this consumer has nowhere to persist it today —
        # `ingestion_jobs` has no meeting_id column, and document_chunks/document_elements are
        # keyed by document_id, not meeting_id. Extracting (and requiring) it here still
        # validates the contract's shape even though the value itself is unused past this line.
        _meeting_id = payload["meeting_id"]  # noqa: F841 - see comment above; validates presence
        raw_object_key = payload["object_key"]
        filename = payload.get("filename", "attachment")

        try:
            key = _resolve_key(org_id, raw_object_key)
        except CrossTenantObjectKeyError:
            # Fail closed: never fetch. This is a data-quality/security fault in the producer's
            # event, not a transient error — logged and dropped rather than raised, matching
            # this consumer's own scope (R8 does not extend to acting as integration-service's
            # validator); the message is still recorded as consumed, so it will not be retried
            # forever, and BaseConsumer's normal DLQ path remains reserved for the deserialize /
            # missing-org_id classes it already covers.
            import logging

            logging.getLogger(__name__).error(
                "ingestion-service consumer: rejected cross-tenant object_key for org %s: %r",
                org_id,
                raw_object_key,
            )
            return

        content = get_object(self._s3, bucket=BUCKET_ATTACHMENTS, key=key)
        document_id = uuid4()  # ALWAYS server-generated, even though this event came from
        # another service — document_id identity belongs to ingestion-service's own bounded
        # context, kept uniform with the upload path (routes.py) rather than trusting an
        # identifier from payload_json.

        await create_job(
            session,
            job_id=document_id,
            org_id=org_id,
            source="calendar_attachment",
            bucket=BUCKET_ATTACHMENTS,
            object_key=key,
        )

        async def emit(topic: str, idem: str, event_payload: dict) -> None:  # type: ignore[type-arg]
            await self.emit_outbox(session, evt, topic=topic, idem=idem, payload=event_payload)

        try:
            outcome = await ingest_document(
                session=session,
                org_id=org_id,
                document_id=document_id,
                filename=filename,
                content=content,
                emit=emit,
                redis=self._redis,
            )
        except Exception as exc:  # noqa: BLE001 - record the failure on the job row, then re-raise
            await fail_job(session, job_id=document_id, org_id=org_id, error=str(exc))
            raise

        await complete_job(
            session, job_id=document_id, org_id=org_id, chunk_count=outcome.chunk_count
        )
