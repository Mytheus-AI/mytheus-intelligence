"""Transactional outbox base — the only sanctioned emit path for this repo's producers.

Direct port of `mytheus-backend`'s `libs/shared/outbox.py` pattern (`FOR UPDATE SKIP LOCKED`
relay draining, the `(topic, idempotency_key)` unique index, JSONB payload) against this
repo's own `intelligence.outbox` table (see `libs/shared/db.py` for why it is schema-qualified
rather than reusing `mytheus-backend`'s `public.outbox`).

Usage pattern (identical to the backend's):
  async with session.begin():
      ... business effect (e.g. Qdrant upsert; NOT a Postgres write on the agent tier) ...
      await enqueue(session, envelope)           # atomic intent-to-emit

`embedding-service`'s `EmbeddingConsumer.apply()` calls this once per processed message, in
the same DB transaction as the inbox insert (`libs/shared/inbox.record_consumed`), so the
"processed" fact and the "will emit `embedding-events`" intent commit atomically.
"""

from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.events import EventEnvelope


async def enqueue(session: AsyncSession, envelope: EventEnvelope) -> int:
    """Insert an outbox row in the current transaction. Returns the BIGSERIAL row id.

    On a duplicate (topic, idempotency_key) the insert is a no-op and the existing id is
    returned — a second call with the same idempotency key (e.g. consumer redelivery) never
    enqueues a second event.
    """
    result = await session.execute(
        text(
            "INSERT INTO intelligence.outbox "
            "(org_id, topic, partition_key, idempotency_key, schema_subject, payload) "
            "VALUES (:org_id, :topic, :partition_key, :idempotency_key, :schema_subject, "
            "CAST(:payload AS jsonb)) "
            "ON CONFLICT (topic, idempotency_key) DO NOTHING "
            "RETURNING id"
        ),
        {
            "org_id": envelope.org_id,
            "topic": envelope.topic,
            "partition_key": envelope.partition_key,
            "idempotency_key": envelope.idempotency_key,
            "schema_subject": envelope.schema_subject,
            "payload": json.dumps(envelope.to_value()),
        },
    )
    row = result.first()
    if row is not None:
        return int(row[0])
    existing = await session.execute(
        text("SELECT id FROM intelligence.outbox WHERE topic = :topic AND idempotency_key = :ik"),
        {"topic": envelope.topic, "ik": envelope.idempotency_key},
    )
    return int(existing.scalar_one())
