"""Inbox / dedupe for at-least-once event consumers.

Direct port of `mytheus-backend`'s `libs/shared/inbox.py` pattern against this repo's own
`intelligence.inbox` table. Pattern (identical to the backend's):

  1. Check Redis dedupe key `nexus:{org_id}:dedupe:{topic}:{consumer_group}:{idempotency_key}`
     (24h TTL). The key carries the consumer group for the same reason the inbox PK does: two
     groups consume the same topic (`transcript-chunks`: embedding + search; `decisions`:
     knowledge-graph + embedding), and a group-less key let whichever group came second treat
     the other group's fact as its own duplicate and skip its effect (mytheus-docs ADR-0031 D6).
  2. If seen -> drop (return True from is_duplicate).
  3. If new -> run the handler + record_consumed (inbox insert) in the SAME transaction as
     any effect that must be atomic with "this fact was applied" (here: the outbox enqueue
     for the resulting `embedding-events` ack — see `libs/shared/consumer.BaseConsumer`).

The Redis check is fast (avoids a DB round-trip for most redeliveries); the inbox table is the
durable authority when the cache has evicted. The inbox PK is (consumer_group, org_id,
idempotency_key), so each consumer group applies a given fact exactly once, independently —
this matters here because `transcript-chunks` and `document-chunks` are consumed by both
`embedding-service` and `agent-runtime-service` (Plan 5-6 tasks 8-10), and each group's
progress must be independent of the other's.
"""

from __future__ import annotations

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_DEDUPE_TTL = 86_400  # 24 h


def dedupe_key(org_id: str, topic: str, consumer_group: str, idempotency_key: str) -> str:
    return f"nexus:{org_id}:dedupe:{topic}:{consumer_group}:{idempotency_key}"


async def is_duplicate(
    redis: Redis,  # see libs/shared/consumer.py's note on this installed redis-py version's stubs
    org_id: str,
    topic: str,
    idempotency_key: str,
    *,
    consumer_group: str,
) -> bool:
    key = dedupe_key(org_id, topic, consumer_group, idempotency_key)
    result = await redis.set(key, "1", ex=_DEDUPE_TTL, nx=True)
    return result is None  # None -> key already existed -> duplicate


async def record_consumed(
    session: AsyncSession,
    consumer_group: str,
    org_id: str,
    idempotency_key: str,
) -> None:
    """Durably record that this consumer group applied this fact. PK dedupes concurrent replays."""
    await session.execute(
        text("""
            INSERT INTO intelligence.inbox (consumer_group, org_id, idempotency_key)
            VALUES (:consumer_group, :org_id, :idempotency_key)
            ON CONFLICT (consumer_group, org_id, idempotency_key) DO NOTHING
            """),
        {
            "consumer_group": consumer_group,
            "org_id": org_id,
            "idempotency_key": idempotency_key,
        },
    )


async def already_recorded(
    session: AsyncSession,
    consumer_group: str,
    org_id: str,
    idempotency_key: str,
) -> bool:
    """Durable-authority check used when the Redis cache has evicted the dedupe key."""
    result = await session.execute(
        text("""
            SELECT 1 FROM intelligence.inbox
            WHERE consumer_group = :consumer_group AND org_id = :org_id
              AND idempotency_key = :idempotency_key
            """),
        {
            "consumer_group": consumer_group,
            "org_id": org_id,
            "idempotency_key": idempotency_key,
        },
    )
    return result.first() is not None
