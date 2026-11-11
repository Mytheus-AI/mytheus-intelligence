"""Finishing an admin data erasure in Qdrant (store `vectors`): `libs/shared/erasure.py` has the
shared rules; this module is embedding-service's share, as the only writer of Qdrant.

`VectorErasureConsumer` (group `embedding-service.erasure`, `admin-events`):

- MEETING
  - `transcript_chunks__{org_id}`: every point of the meeting is deleted.
  - `decisions__{org_id}`: the collection only ever holds CONFIRMED decisions (the `decisions`
    topic carries nothing else), and core-api keeps confirmed and superseded decisions. So a
    point that is not `confirmed`/`superseded` (none today; defensive, it is what core-api
    deletes) is deleted, and every kept point loses its `meeting_id` and gains
    `meeting_erased: true`. RAG's decisions arm filters on `meeting_id`, so those points stop
    being reachable through the erased meeting while the decision itself stays an org record.
  - `document_chunks__{org_id}`: documents carry no meeting; nothing to do.
- PERSON
  - `transcript_chunks__{org_id}`: points whose `speaker` is their `users.id` get
    `speaker: "erased"` (the payload holds no text; the vector is the words, which core-api's
    rule keeps).
  - `decisions__{org_id}` holds no person (`confirmed_by` is not in canon's payload).

Every Qdrant call is org-scoped twice, like the rest of the repo: the per-tenant collection and
an `org_id` payload condition. A tenant with no collection yet has nothing to erase.

Late arrivals (`libs/shared/erasure.py` "Late arrivals"): `ErasureAwareEmbeddingConsumer` and
`ErasureAwareDecisionEmbeddingConsumer` are the two indexing consumers with the erasure
tombstone check in front; `worker.py` runs them instead of the bare classes.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import ConsumedEvent
from libs.shared.erasure import (
    ERASED_SPEAKER,
    ErasureConsumer,
    ErasureRequest,
    meeting_erased,
    vet_transcript_chunk,
)
from libs.shared.qdrant import LOGICAL_DECISIONS, LOGICAL_TRANSCRIPT_CHUNKS, physical_collection
from services.embedding_service.consumer import EmbeddingConsumer
from services.embedding_service.decision_consumer import DecisionEmbeddingConsumer

_KEPT_DECISION_STATUSES = ["confirmed", "superseded"]


def _org(org_id: uuid.UUID) -> FieldCondition:
    return FieldCondition(key="org_id", match=MatchValue(value=str(org_id)))


async def _count(client: AsyncQdrantClient, name: str, flt: Filter) -> int:
    return int((await client.count(name, count_filter=flt, exact=True)).count)


async def erase_meeting_vectors(
    client: AsyncQdrantClient, org_id: uuid.UUID, meeting_id: uuid.UUID
) -> dict[str, int]:
    meeting = FieldCondition(key="meeting_id", match=MatchValue(value=str(meeting_id)))
    counts = {
        "transcript_points_deleted": 0,
        "decision_points_deleted": 0,
        "decision_points_anonymized": 0,
    }
    chunks = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
    if await client.collection_exists(chunks):
        flt = Filter(must=[_org(org_id), meeting])
        counts["transcript_points_deleted"] = await _count(client, chunks, flt)
        await client.delete(chunks, points_selector=flt)
    counts.update(await anonymize_meeting_decisions(client, org_id, meeting_id))
    return counts


async def anonymize_meeting_decisions(
    client: AsyncQdrantClient, org_id: uuid.UUID, meeting_id: uuid.UUID
) -> dict[str, int]:
    """`decisions__{org_id}`: delete what core-api deleted, strip the meeting from the rest."""
    name = physical_collection(LOGICAL_DECISIONS, org_id)
    if not await client.collection_exists(name):
        return {"decision_points_deleted": 0, "decision_points_anonymized": 0}
    meeting = FieldCondition(key="meeting_id", match=MatchValue(value=str(meeting_id)))
    unconfirmed = Filter(
        must=[_org(org_id), meeting],
        must_not=[FieldCondition(key="status", match=MatchAny(any=_KEPT_DECISION_STATUSES))],
    )
    deleted = await _count(client, name, unconfirmed)
    if deleted:
        await client.delete(name, points_selector=unconfirmed)
    kept = Filter(must=[_org(org_id), meeting])
    anonymized = await _count(client, name, kept)
    if anonymized:
        await client.set_payload(name, payload={"meeting_erased": True}, points=kept)
        await client.delete_payload(name, keys=["meeting_id"], points=kept)
    return {"decision_points_deleted": deleted, "decision_points_anonymized": anonymized}


async def erase_person_vectors(
    client: AsyncQdrantClient, org_id: uuid.UUID, user_id: uuid.UUID
) -> dict[str, int]:
    name = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
    if not await client.collection_exists(name):
        return {"transcript_points_anonymized": 0}
    speakers = sorted({str(user_id), str(user_id).upper()})
    flt = Filter(must=[_org(org_id), FieldCondition(key="speaker", match=MatchAny(any=speakers))])
    anonymized = await _count(client, name, flt)
    if anonymized:
        await client.set_payload(name, payload={"speaker": ERASED_SPEAKER}, points=flt)
    return {"transcript_points_anonymized": anonymized}


class VectorErasureConsumer(ErasureConsumer):
    group: ClassVar[str] = "embedding-service.erasure"
    store: ClassVar[str] = "vectors"
    producer_name: ClassVar[str] = "embedding-service"

    def __init__(self, *, qdrant: AsyncQdrantClient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._qdrant = qdrant

    async def erase(self, req: ErasureRequest, session: AsyncSession) -> dict[str, int]:
        del session  # Qdrant is not in the relational database
        if req.kind == "meeting":
            return await erase_meeting_vectors(self._qdrant, req.org_id, req.target_id)
        return await erase_person_vectors(self._qdrant, req.org_id, req.target_id)


class ErasureAwareEmbeddingConsumer(EmbeddingConsumer):
    """`EmbeddingConsumer` that never stores a chunk of an erased meeting, and stores a chunk
    spoken by a since-erased person without their attribution."""

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        if evt.topic == "transcript-chunks":
            vetted = await vet_transcript_chunk(self._redis, evt)
            if vetted is None:
                return
            evt = vetted
        await super().apply(evt, session)


class ErasureAwareDecisionEmbeddingConsumer(DecisionEmbeddingConsumer):
    """`DecisionEmbeddingConsumer` that strips the meeting from a decision of an erased meeting
    embedded after the erasure ran (the erasure rule, re-applied to the late point)."""

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        await super().apply(evt, session)
        raw = evt.body.get("meeting_id")
        if await meeting_erased(self._redis, evt.org_id, raw):
            await anonymize_meeting_decisions(self._qdrant, evt.org_id, uuid.UUID(str(raw)))
