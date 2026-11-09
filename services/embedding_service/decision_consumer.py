"""`DecisionEmbeddingConsumer`: embeds confirmed decisions into `decisions__{org_id}` (Plan 7-9
section 3.6 step 5, ADR-0031 D5). `embedding-service` stays the only writer of Qdrant.

Subscribes to `decisions` only (producer `core-api`, on confirm only), so nothing unconfirmed is
ever embedded. One point per decision: the vector is the decision's `content` plus its
`rationale`; the payload is exactly canon's `decisions` collection payload (`org_id`,
`decision_id`, `meeting_id`, `status`, `confidence`), read later by ProactiveAgent's semantic
conflict check and the RAG vector arm.

Idempotency: the point id is `uuid5` of `decision:{decision_id}`, so a redelivery overwrites the
same point. Supersession: when a decision names `supersedes`, the earlier decision's point is
re-labelled `status = superseded` by a payload filter on (`org_id`, `decision_id`), which matches
nothing (and fails nothing) if the earlier decision has not been embedded yet. The ack goes to
`embedding-events` through the outbox, like every other embed.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent
from libs.shared.metrics import observe_embedding_batch
from libs.shared.qdrant import LOGICAL_DECISIONS, ensure_collection, physical_collection, upsert
from services.embedding_service.consumer import _POINT_NAMESPACE
from services.embedding_service.model import embed_batch


def decision_point_id(decision_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"decision:{decision_id}"))


class DecisionEmbeddingConsumer(BaseConsumer):
    group = "embedding-service.embed-decisions"
    topics = ["decisions"]
    producer_name = "embedding-service"

    def __init__(self, *, qdrant: AsyncQdrantClient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._qdrant = qdrant

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        body = evt.body
        if body.get("status") != "confirmed":  # unrepresentable on this topic; fail closed
            return
        org_id = evt.org_id
        decision_id = str(body["decision_id"])
        meeting_id = str(body["meeting_id"])
        text = "\n".join(p for p in (body.get("content"), body.get("rationale")) if p)

        vectors, elapsed = embed_batch([text])
        observe_embedding_batch(elapsed, 1)
        point = PointStruct(
            id=decision_point_id(decision_id),
            vector=vectors[0],
            payload={
                "org_id": str(org_id),
                "decision_id": decision_id,
                "meeting_id": meeting_id,
                "status": "confirmed",
                "confidence": float(body["confidence"]),
            },
        )
        collection = await upsert(self._qdrant, LOGICAL_DECISIONS, org_id, [point])

        supersedes = body.get("supersedes")
        if supersedes and str(supersedes) != decision_id:
            await self.mark_superseded(org_id, str(supersedes))

        await self.emit_outbox(
            session,
            evt,
            topic="embedding-events",
            idem=f"emb:{LOGICAL_DECISIONS}:{decision_id}",
            payload={"collection": collection, "point_count": 1, "meeting_id": meeting_id},
        )

    async def mark_superseded(self, org_id: uuid.UUID, decision_id: str) -> None:
        name = physical_collection(LOGICAL_DECISIONS, org_id)
        await ensure_collection(self._qdrant, name)
        await self._qdrant.set_payload(
            collection_name=name,
            payload={"status": "superseded"},
            points=Filter(
                must=[
                    FieldCondition(key="org_id", match=MatchValue(value=str(org_id))),
                    FieldCondition(key="decision_id", match=MatchValue(value=decision_id)),
                ]
            ),
        )
