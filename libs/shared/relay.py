"""Transactional-outbox relay: the single sanctioned producer path for this repo.

Direct port of `mytheus-backend`'s `libs/shared/relay.py` `relay_once` against this repo's
`intelligence.outbox` table. Polls unpublished outbox rows (`FOR UPDATE SKIP LOCKED` so
replicas do not double-publish), Avro-serializes each row's value against `{topic}-value`
(via the real Schema Registry — see `libs/shared/avro_codec.py`), produces keyed by `org_id`
(acks=all, idempotent), then marks the row published.

This repo needs its OWN relay loop: `mytheus-backend`'s relay worker (mentioned as already
running against the shared dev stack) drains `public.outbox`, not `intelligence.outbox` — a
relay only ever sees the table it is pointed at, and table names are not otherwise related.
`services/embedding_service/embedding_service/relay_main.py` is this repo's relay entrypoint,
following the same shape as `mytheus-backend/services/relay/main.py`. Established here as the
convention for tasks 8-10: any repo/schema that writes an outbox row needs its own relay
process pointed at `intelligence.outbox` (the schema, not the process, is what's shared).

RLS note. This drains rows across every tenant, which the fail-closed `tenant_isolation`
policy forbids by default (see `mytheus-backend/libs/shared/relay.py`'s identical note). The
fix here is the same one `mytheus-backend` migration `a1c4e7f20b18` applied: a single
permissive `outbox_relay_drain` policy scoped to `intelligence.outbox` and to the
`mytheus_ai_relay` role alone (migration
`0001_intelligence_schema_outbox_inbox_roles`), never a BYPASSRLS role.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Same rationale as mytheus-backend: low on purpose. A row that fails to serialize/produce is
# almost always a deterministic failure (schema mismatch), so retrying many times only delays
# the DLQ while the row holds up its partition key's ordering.
MAX_ATTEMPTS = 5

_SELECT_PENDING = text(
    "SELECT id, topic, partition_key, schema_subject, payload, attempts "
    "FROM intelligence.outbox WHERE published_at IS NULL "
    "ORDER BY created_at "
    "FOR UPDATE SKIP LOCKED "
    "LIMIT :limit"
)

_MARK_PUBLISHED = text("UPDATE intelligence.outbox SET published_at = now() WHERE id = ANY(:ids)")

_BUMP_ATTEMPTS = text("UPDATE intelligence.outbox SET attempts = attempts + 1 WHERE id = ANY(:ids)")

_MARK_DEAD_LETTERED = text(
    "UPDATE intelligence.outbox SET published_at = now(), attempts = attempts + 1 "
    "WHERE id = ANY(:ids)"
)


class RelayProducer(Protocol):
    def serialize(self, subject: str, value: dict[str, Any]) -> bytes: ...
    def produce(self, *, topic: str, key: bytes, value: bytes) -> None: ...
    def flush(self) -> int: ...  # returns messages still in-queue (0 == all delivered)


async def relay_once(session: AsyncSession, producer: RelayProducer, *, limit: int = 500) -> int:
    """Drain one batch of unpublished `intelligence.outbox` rows. Returns the number published.

    Identical semantics to `mytheus-backend`'s `relay_once`: per-row failures are isolated
    (attempts counted, dead-lettered to `{topic}.dlq` after MAX_ATTEMPTS), a broker-side
    non-delivery re-publishes the whole batch (consumers dedupe), and rows are only marked
    published after `producer.flush()` confirms delivery.
    """
    rows = (await session.execute(_SELECT_PENDING, {"limit": limit})).mappings().all()
    if not rows:
        return 0

    produced_ids: list[int] = []
    retry_ids: list[int] = []
    dead_ids: list[int] = []

    for r in rows:
        try:
            value = producer.serialize(r["schema_subject"], r["payload"])
            producer.produce(topic=r["topic"], key=r["partition_key"].encode(), value=value)
        except Exception as exc:  # noqa: BLE001 - any produce-side failure is per-row, not fatal
            if (r["attempts"] or 0) + 1 >= MAX_ATTEMPTS:
                _dead_letter(producer, r, exc)
                dead_ids.append(r["id"])
                logger.error(
                    "outbox relay: row %s dead-lettered to %s.dlq after %d attempts: %s",
                    r["id"],
                    r["topic"],
                    MAX_ATTEMPTS,
                    exc,
                )
            else:
                retry_ids.append(r["id"])
                logger.warning(
                    "outbox relay: row %s failed (attempt %d/%d), will retry: %s",
                    r["id"],
                    (r["attempts"] or 0) + 1,
                    MAX_ATTEMPTS,
                    exc,
                )
            continue
        produced_ids.append(r["id"])

    remaining = producer.flush()
    if remaining:
        raise RuntimeError(
            f"outbox relay: {remaining} message(s) undelivered; not marking published"
        )

    if produced_ids:
        await session.execute(_MARK_PUBLISHED, {"ids": produced_ids})
    if retry_ids:
        await session.execute(_BUMP_ATTEMPTS, {"ids": retry_ids})
    if dead_ids:
        await session.execute(_MARK_DEAD_LETTERED, {"ids": dead_ids})

    return len(produced_ids)


def _dead_letter(producer: RelayProducer, row: Any, exc: Exception) -> None:
    """Publish a poison row's RAW payload to `{topic}.dlq`, same shape as the backend's."""
    envelope = {
        "outbox_id": row["id"],
        "topic": row["topic"],
        "schema_subject": row["schema_subject"],
        "error": f"{type(exc).__name__}: {exc}",
        "payload": row["payload"],
    }
    producer.produce(
        topic=f"{row['topic']}.dlq",
        key=row["partition_key"].encode(),
        value=json.dumps(envelope, default=str).encode(),
    )
