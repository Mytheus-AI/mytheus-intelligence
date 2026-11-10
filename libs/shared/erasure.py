"""Finishing an admin data erasure in this tier's stores: the shared half of every consumer.

An admin's erasure (Workspace > Data and privacy) runs in `mytheus-backend`: admin-ops-service
has core-api erase its own tables (one transaction, `services/core_api/routes/erasure.py`),
then emits `admin-events` `erasure_requested` with `target_type` `person` or `meeting`,
`target_id` the `users.id` or meeting id, and `detail_json` `{"erasure_id", "scope"}`. Each
store this repo owns has one consumer that finishes its share:

- `embedding-service.erasure`, store `vectors`: `services/embedding_service/erasure_consumer.py`
- `search-service.erasure`, store `search_index`: `services/search_service/erasure_consumer.py`
- `knowledge-graph-service.erasure`, store `knowledge_graph`:
  `services/knowledge_graph_service/erasure_consumer.py`

`ingestion-service` has no consumer: `document_chunks` / `document_elements` / `ingestion_jobs`
carry neither a meeting nor a person (an uploaded document is the org's, not a meeting's or an
uploader's), so neither erasure scope reaches them.

The rules every store follows, matched to core-api's:
- MEETING: everything derived from the meeting's talk goes; confirmed decisions and actions are
  org records and stay with their content, but the reference to the erased meeting is removed
  (core-api keeps them and hides the meeting with `deleted_at`; this tier cannot check
  `deleted_at` at read time, so it removes the reference instead).
- PERSON: their attribution goes, the words stay. core-api deletes the person's
  `meeting_speakers` rows so their lines "fall back to an unnamed speaker", and keeps the text
  as the meeting's record. Here a transcript line whose speaker is their `users.id` becomes
  `ERASED_SPEAKER`; confirmed records stay with the person removed from them.

`org_id` is ALWAYS the envelope header's (`BaseConsumer` has already DLQ'd a missing or
malformed one); `target_id` must parse as a UUID, or the event is dead-lettered.

Completion. Each consumer emits one `admin-events` `erasure_completed` through
`intelligence.outbox`, in the same transaction as its inbox row: `target_type` / `target_id`
copied from the request, `detail_json` `{"erasure_id", "store", "service", "counts"}`. The
idempotency key `erasure_completed:{erasure_id}:{store}` is stable, so a re-application emits
nothing new. NOT CONNECTED until `admin-events-value` v2 (the `erasure_completed` enum symbol)
is registered: until then the relay cannot encode the row and dead-letters it.

Late arrivals. The indexes are fed by other topics (`transcript-chunks`, `decisions`,
`actions`, `meeting-room-events`) with no ordering against `admin-events`, so a lagging indexer
can store a chunk of an erased meeting after its erasure ran. Every erasure consumer therefore
leaves a TOMBSTONE in Redis, `nexus:{org_id}:erasure:{kind}:{target_id}` = the request's
`occurred_at` (micros), for `TOMBSTONE_TTL_S`; the `ErasureAware*` indexing consumers read it:
a chunk of an erased meeting is not stored, and a person's attribution is removed from a fact
that happened before their erasure (`occurred_at <= tombstone`). A person may keep working in the
org after an erasure, so what they say afterwards is attributed normally.

Dedupe. `BaseConsumer` sets its Redis dedupe key BEFORE `apply` runs and leaves it set if
`apply` raises, so a transient failure (Qdrant or AGE unreachable) followed by a restart would
see the redelivery as a duplicate and the erasure would never finish. `ErasureConsumer`
releases that key when processing fails. (The same gap exists for every `BaseConsumer`; it is
fixed here only for erasure, where losing an event is not acceptable.)
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent, ProcessOutcome
from libs.shared.inbox import dedupe_key
from libs.shared.tenancy import parse_org_id

logger = logging.getLogger(__name__)

TOPIC = "admin-events"
REQUESTED = "erasure_requested"
COMPLETED = "erasure_completed"
ERASED_SPEAKER = "erased"
TOMBSTONE_TTL_S = 30 * 24 * 3600

ErasureKind = Literal["person", "meeting"]


@dataclass(frozen=True)
class ErasureRequest:
    org_id: uuid.UUID
    kind: ErasureKind
    target_id: uuid.UUID
    erasure_id: str
    actor_id: str
    requested_at_us: int

    @property
    def target(self) -> str:
        return str(self.target_id)


class UnusableErasure(ValueError):
    """An `erasure_requested` this store can never apply (a non-UUID target)."""


def occurred_at_us(header: dict[str, Any]) -> int:
    """The header's `occurred_at` in micros. The Avro decoder yields a datetime; a hand-built
    event may carry the raw int. Unknown means now (a fact we cannot date is treated as new)."""
    value = header.get("occurred_at")
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(datetime.now(UTC).timestamp() * 1_000_000)


def parse_request(evt: ConsumedEvent) -> ErasureRequest | None:
    """The erasure an `admin-events` value asks for, or None when it is not one (another event
    type, or a target other than a person or a meeting). Raises `UnusableErasure`."""
    body = evt.body
    if body.get("event_type") != REQUESTED:
        return None
    kind = body.get("target_type")
    if kind not in ("person", "meeting"):
        return None
    try:
        target = uuid.UUID(str(body.get("target_id")))
    except ValueError as exc:
        raise UnusableErasure(
            f"erasure target_id is not a UUID: {body.get('target_id')!r}"
        ) from exc
    try:
        detail = json.loads(body.get("detail_json") or "{}")
    except ValueError:
        detail = {}
    raw = detail.get("erasure_id") if isinstance(detail, dict) else None
    try:
        erasure_id = str(uuid.UUID(str(raw)))
    except ValueError:
        erasure_id = evt.idempotency_key  # an older producer: the request's own key names it
    return ErasureRequest(
        org_id=evt.org_id,
        kind="person" if kind == "person" else "meeting",
        target_id=target,
        erasure_id=erasure_id,
        actor_id=str(body.get("actor_id") or ""),
        requested_at_us=occurred_at_us(evt.header),
    )


def completion_key(erasure_id: str, store: str) -> str:
    return f"{COMPLETED}:{erasure_id}:{store}"


def completion_payload(
    req: ErasureRequest, store: str, service: str, counts: dict[str, int]
) -> dict[str, Any]:
    """The `AdminEvent` fields (no header) of one store's `erasure_completed`."""
    detail = {"erasure_id": req.erasure_id, "store": store, "service": service, "counts": counts}
    return {
        "event_type": COMPLETED,
        "actor_id": req.actor_id,
        "target_type": req.kind,
        "target_id": req.target,
        "detail_json": json.dumps(detail, sort_keys=True),
    }


# --- tombstones ---------------------------------------------------------------------------------


def tombstone_key(org_id: uuid.UUID, kind: ErasureKind, target: str) -> str:
    return f"nexus:{org_id}:erasure:{kind}:{target.lower()}"


async def mark_erased(redis: Any, req: ErasureRequest) -> None:
    await redis.set(
        tombstone_key(req.org_id, req.kind, req.target),
        str(req.requested_at_us),
        ex=TOMBSTONE_TTL_S,
    )


async def erased_at(redis: Any, org_id: uuid.UUID, kind: ErasureKind, target: Any) -> int | None:
    """When `target` was erased (micros), or None. A non-UUID target was never erased."""
    if target is None or target == "":
        return None
    try:
        key = str(uuid.UUID(str(target)))
    except ValueError:
        return None
    raw = await redis.get(tombstone_key(org_id, kind, key))
    if raw is None:
        return None
    try:
        return int(raw.decode() if isinstance(raw, bytes | bytearray) else raw)
    except ValueError:
        return 0  # unreadable: treat as erased, the safe direction


async def meeting_erased(redis: Any, org_id: uuid.UUID, meeting_id: Any) -> bool:
    return await erased_at(redis, org_id, "meeting", meeting_id) is not None


async def person_erased_before(
    redis: Any, org_id: uuid.UUID, user_id: Any, header: dict[str, Any]
) -> bool:
    """True when `user_id` was erased and this fact happened no later than the erasure."""
    at = await erased_at(redis, org_id, "person", user_id)
    return at is not None and occurred_at_us(header) <= at


async def vet_transcript_chunk(redis: Any, evt: ConsumedEvent) -> ConsumedEvent | None:
    """For an indexing consumer of `transcript-chunks`: None when the chunk's meeting is erased
    (store nothing), else the event with the speaker replaced by `ERASED_SPEAKER` when the
    speaker is a person erased after this chunk was spoken, else the event unchanged."""
    if await meeting_erased(redis, evt.org_id, evt.body.get("meeting_id")):
        logger.info("erasure: dropping a late chunk of an erased meeting (org %s)", evt.org_id)
        return None
    if await person_erased_before(redis, evt.org_id, evt.body.get("speaker_id"), evt.header):
        return ConsumedEvent(
            topic=evt.topic,
            org_id=evt.org_id,
            idempotency_key=evt.idempotency_key,
            header=evt.header,
            body={**evt.body, "speaker_id": ERASED_SPEAKER},
            partition=evt.partition,
            offset=evt.offset,
        )
    return evt


# --- the consumer base --------------------------------------------------------------------------


class ErasureConsumer(BaseConsumer):
    """One store's `admin-events` erasure consumer. A subclass sets `group`, `store`,
    `producer_name` and implements `erase`; everything else (filtering, tombstone, completion
    event, dedupe release) is here."""

    topics: ClassVar[list[str]] = [TOPIC]
    store: ClassVar[str]

    @abstractmethod
    async def erase(self, req: ErasureRequest, session: AsyncSession) -> dict[str, int]:
        """Erase this store's share. Idempotent by construction. Returns counts by kind."""

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        try:
            req = parse_request(evt)
        except UnusableErasure as exc:
            logger.error("erasure %s: unusable %s: %s", self.group, evt.idempotency_key, exc)
            self._to_dlq(
                topic=evt.topic,
                partition=evt.partition,
                offset=evt.offset,
                org_id=str(evt.org_id),
                idempotency_key=evt.idempotency_key,
                failure_class="BUSINESS_INVALID",
                error=str(exc),
                raw_value=json.dumps(evt.body, default=str).encode(),
            )
            return
        if req is None:
            return
        await mark_erased(self._redis, req)
        counts = await self.erase(req, session)
        logger.info(
            "erasure %s: %s %s erased from %s for org %s: %s",
            self.group,
            req.kind,
            req.target,
            self.store,
            req.org_id,
            counts,
        )
        await self.emit_outbox(
            session,
            evt,
            topic=TOPIC,
            idem=completion_key(req.erasure_id, self.store),
            payload=completion_payload(req, self.store, self.producer_name, counts),
        )

    async def process_raw(
        self,
        *,
        topic: str,
        raw_value: bytes,
        partition: int,
        offset: int,
        timestamp_ms: int | None = None,
    ) -> ProcessOutcome:
        try:
            return await super().process_raw(
                topic=topic,
                raw_value=raw_value,
                partition=partition,
                offset=offset,
                timestamp_ms=timestamp_ms,
            )
        except Exception:
            await self._release_dedupe(topic, raw_value)
            raise

    async def _release_dedupe(self, topic: str, raw_value: bytes) -> None:
        """Undo `BaseConsumer`'s pre-apply dedupe mark so the redelivery is applied."""
        try:
            header = dict(self._codec.deserialize_for(topic, raw_value)["header"])
            org_id = parse_org_id(header.get("org_id"))
            idem = header.get("idempotency_key")
            if org_id is not None and idem:
                await self._redis.delete(dedupe_key(str(org_id), topic, self.group, str(idem)))
        except Exception:  # noqa: BLE001 - best effort; the original error is what matters
            logger.exception("erasure %s: could not release the dedupe mark", self.group)
