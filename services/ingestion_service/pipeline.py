"""The parse pipeline — plan `t9-3.9-ingestion.md` step 2, adapted to this repo's real
conventions (see `task-9-report.md` for the full rationale of each deviation).

`ingest_document` has exactly one job: parse already-fetched bytes, write `document_elements` +
`document_chunks` (idempotently), and call the caller-supplied `emit` once per chunk to enqueue
a `document-chunks` outbox row — all inside the caller's already-open, already-RLS-scoped
transaction. It does NOT touch object storage and does NOT manage the `ingestion_jobs` row's
status lifecycle; both callers (`routes.py`'s `POST /ingest` and `consumer.py`'s
`IntegrationEventConsumer`) own those decisions themselves, because they differ in exactly the
way the task brief's R8 checkpoint cares about: a REST upload WRITES a fresh object under a
server-generated key, while the calendar-attachment consumer READS an object
`integration-service` already wrote (see `consumer.py`'s module docstring for the object-key
trust boundary that applies only to that path).

`document_elements` vs `document_chunks`. The plan's pseudocode shows `write_document_chunk`
and `emit_outbox` per SPLIT piece but is silent on where `document_elements` (also in its own
SQL schema) gets written. This module's reading: `document_elements` holds one row per RAW
unstructured element (`element_id = "el_{pos:04d}"`, `raw` = unstructured's own
`element.to_dict()`) — the parse-time record, pre-split; `document_chunks` holds one row per
600/60-SPLIT piece (`element_id = "el_{pos:04d}_{j}"`) — the embedding-ready, chunked record.
Together they match the service's own one-line description in `CLAUDE.md`: "parsing into
TYPED, CHUNKED elements" — typed (`document_elements`) and chunked (`document_chunks`) are two
different granularities, not two names for the same table's contents.

Splitting: `Title`/`NarrativeText` elements are split 600/60 (`libs/shared/splitter.py`, the
SAME splitter `embedding-service` uses); `Table`/`ListItem` elements keep their typed element
boundary UNSPLIT (one piece each), per the plan's own note ("tables/lists keep the typed element
boundary"). The plan's note additionally suggests `SemanticChunker` for narrative docs; this is
deliberately NOT implemented — see `task-9-report.md` for why (in short: it would require this
service to load its own embedding model in-process, duplicating `embedding-service`'s GPU/model
responsibility inside what should stay a lightweight parsing service, for a refinement no
acceptance check in the plan exercises).

Idempotent re-run. `document_chunks`/`document_elements` both carry a UNIQUE
`(org_id, document_id, element_id)` constraint (migration `b3f7a1c9e402`); every write here uses
`ON CONFLICT ... DO NOTHING`, and the outbox `idempotency_key` (`docchunk:{document_id}:
{element_id}`) is stable across re-parses of the same bytes. Calling `ingest_document` twice with
the same `(org_id, document_id, content)` — a retried job, or the calendar-attachment consumer
processing a redelivered `integration-events` message past its own inbox dedupe — writes zero
additional rows and enqueues zero additional outbox rows the second time (proven in
`tests/test_ingestion_pipeline.py`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.splitter import split_600_60
from services.ingestion_service.parser import ParsedElement, parse_document
from services.ingestion_service.pii import pii_enabled, redact_pii

# One outbox enqueue per chunk: (topic, idempotency_key, payload) -> enqueue.
EmitFn = Callable[[str, str, dict[str, Any]], Awaitable[None]]

_UNSPLIT_TYPES = frozenset({"Table", "ListItem"})

_INSERT_DOCUMENT_ELEMENT = text("""
    INSERT INTO intelligence.document_elements
        (org_id, document_id, element_id, element_type, raw)
    VALUES (:org_id, :document_id, :element_id, :element_type, CAST(:raw AS jsonb))
    ON CONFLICT (org_id, document_id, element_id) DO NOTHING
    """)

_INSERT_DOCUMENT_CHUNK = text("""
    INSERT INTO intelligence.document_chunks
        (org_id, document_id, element_id, element_type, text, "position")
    VALUES (:org_id, :document_id, :element_id, :element_type, :text, :position)
    ON CONFLICT (org_id, document_id, element_id) DO NOTHING
    """)


@dataclass(frozen=True)
class IngestOutcome:
    document_id: UUID
    element_count: int
    chunk_count: int


def _element_pieces(el: ParsedElement, text_: str) -> list[str]:
    if el.element_type in _UNSPLIT_TYPES:
        return [text_]
    return split_600_60(text_) or [text_]


async def ingest_document(
    *,
    session: AsyncSession,
    org_id: UUID,
    document_id: UUID,
    filename: str,
    content: bytes,
    emit: EmitFn,
    redis: Redis,
) -> IngestOutcome:
    import json

    elements = parse_document(filename=filename, content=content)
    pii_on = await pii_enabled(org_id, redis)

    chunk_count = 0
    for el in elements:
        element_id = f"el_{el.position:04d}"
        await session.execute(
            _INSERT_DOCUMENT_ELEMENT,
            {
                "org_id": str(org_id),
                "document_id": str(document_id),
                "element_id": element_id,
                "element_type": el.element_type,
                "raw": json.dumps(el.raw, default=str),
            },
        )

        text_ = redact_pii(el.text) if pii_on else el.text
        for j, piece in enumerate(_element_pieces(el, text_)):
            chunk_element_id = f"el_{el.position:04d}_{j}"
            await session.execute(
                _INSERT_DOCUMENT_CHUNK,
                {
                    "org_id": str(org_id),
                    "document_id": str(document_id),
                    "element_id": chunk_element_id,
                    "element_type": el.element_type,
                    "text": piece,
                    "position": el.position,
                },
            )
            await emit(
                "document-chunks",
                f"docchunk:{document_id}:{chunk_element_id}",
                {
                    "document_id": str(document_id),
                    "element_id": chunk_element_id,
                    "element_type": el.element_type,
                    "text": piece,
                    "position": el.position,
                },
            )
            chunk_count += 1

    return IngestOutcome(
        document_id=document_id, element_count=len(elements), chunk_count=chunk_count
    )


IngestSource = Literal["upload", "calendar_attachment"]


_INSERT_JOB = text("""
    INSERT INTO intelligence.ingestion_jobs
        (id, org_id, status, source, bucket, object_key)
    VALUES (:id, :org_id, 'processing', :source, :bucket, :object_key)
    """)

_COMPLETE_JOB = text("""
    UPDATE intelligence.ingestion_jobs
    SET status = 'completed', chunk_count = :chunk_count, updated_at = now()
    WHERE id = :id AND org_id = :org_id
    """)

_FAIL_JOB = text("""
    UPDATE intelligence.ingestion_jobs
    SET status = 'failed', error = :error, updated_at = now()
    WHERE id = :id AND org_id = :org_id
    """)


async def create_job(
    session: AsyncSession,
    *,
    job_id: UUID,
    org_id: UUID,
    source: IngestSource,
    bucket: str,
    object_key: str,
) -> None:
    await session.execute(
        _INSERT_JOB,
        {
            "id": str(job_id),
            "org_id": str(org_id),
            "source": source,
            "bucket": bucket,
            "object_key": object_key,
        },
    )


async def complete_job(
    session: AsyncSession, *, job_id: UUID, org_id: UUID, chunk_count: int
) -> None:
    await session.execute(
        _COMPLETE_JOB, {"id": str(job_id), "org_id": str(org_id), "chunk_count": chunk_count}
    )


async def fail_job(session: AsyncSession, *, job_id: UUID, org_id: UUID, error: str) -> None:
    await session.execute(_FAIL_JOB, {"id": str(job_id), "org_id": str(org_id), "error": error})
