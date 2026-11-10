"""`EmbeddingConsumer` — the embed-on-ingest consumer. Plan 3.7 step 4, the band's flagship
at-least-once consumer and the FIRST event-sourced writer in `mytheus-intelligence`.

Subscribes to `transcript-chunks` AND `document-chunks`, splits 600/60 (`splitter.py`), embeds
at batch size = however many pieces one message produced (real BGE-small-en, `model.py`),
upserts into the caller's per-tenant Qdrant collection (`libs/shared/qdrant.py`, ADR-0006), and
emits an `embedding-events` ack through the transactional outbox. Dedupe/inbox/DLQ and the
`org_id` re-assertion are all handled by `BaseConsumer.process_raw`; `apply()` here is ONLY the
business effect, matching the plan's own comment ("dedupe + inbox handled by BaseConsumer").

Idempotency. The Qdrant point id is derived deterministically from the SOURCE message's own
stable key (`chunk:{chunk_id}` / `docchunk:{document_id}:{element_id}`) plus the split-piece
index, via `uuid5` into a fixed namespace — so redelivering the same source event re-computes
the SAME point ids and `client.upsert` overwrites them in place rather than creating
duplicates. This is proven for real against Qdrant in `tests/test_embedding_consumer.py`
(same message applied twice -> same point count, same point ids, no growth).

`DocumentChunk` (per the REAL registered Avro schema, `document-chunks-value`) carries no
`meeting_id` field — the Qdrant payload table in `system.md` §10 lists it as optional
(`meeting_id?`) precisely because document ingests are not always meeting-scoped; this
consumer therefore never reads `evt.body["meeting_id"]` for the document-chunks branch.

Never record and confidential meetings (polish plan, `libs/shared/index_guard.py`). With a
`guard`, every transcript chunk is vetted before it is embedded: a chunk in a never-record window
is acked with zero points and never embedded, and when a chunk is the first to name a
never-record topic the points already stored for its window are deleted
(`qdrant.delete_window`). A chunk of a confidential meeting carries `confidential: true` in its
payload, which the org-wide search path excludes; participant-scoped retrieval still finds it.
Documents are not meeting talk and are not vetted here.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent
from libs.shared.index_guard import IndexGuard
from libs.shared.metrics import observe_embedding_batch
from libs.shared.qdrant import (
    LOGICAL_DOCUMENT_CHUNKS,
    LOGICAL_TRANSCRIPT_CHUNKS,
    delete_window,
    upsert,
)
from services.embedding_service.model import embed_batch
from services.embedding_service.splitter import split_600_60

# Fixed, arbitrary namespace UUID for uuid5-deriving Qdrant point ids from a stable source key.
# Must never change once anything has been upserted — changing it would silently orphan every
# previously-written point (a redelivery would compute a DIFFERENT id and create a duplicate
# instead of overwriting).
_POINT_NAMESPACE = uuid.UUID("6f1f2f2a-8f0a-4b3a-9c8a-3a2f9c8e7d10")


def _point_id(stable_key: str, index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{stable_key}:{index}"))


class EmbeddingConsumer(BaseConsumer):
    group = "embedding-service.embed-on-ingest"
    topics = ["transcript-chunks", "document-chunks"]
    producer_name = "embedding-service"

    def __init__(
        self, *, qdrant: AsyncQdrantClient, guard: IndexGuard | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._qdrant = qdrant
        self._guard = guard

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        org_id = evt.org_id

        if evt.topic == "transcript-chunks":
            logical = LOGICAL_TRANSCRIPT_CHUNKS
            chunk_id = str(evt.body["chunk_id"])
            meeting_id = str(evt.body["meeting_id"])
            stable_key = f"chunk:{chunk_id}"
            idem = f"emb:{logical}:{chunk_id}"
            payload_base: dict[str, Any] = {
                "org_id": str(org_id),
                "meeting_id": meeting_id,
                "chunk_id": chunk_id,
                "speaker": evt.body["speaker_id"],
                "start_ms": evt.body["start_ms"],
                "is_final": evt.body["is_final"],
            }
            text = evt.body["text"]
            if self._guard is not None:
                verdict = await self._guard.check(
                    org_id, uuid.UUID(meeting_id), int(evt.body["start_ms"]), text
                )
                if verdict.purge is not None:
                    await delete_window(
                        self._qdrant, logical, org_id, uuid.UUID(meeting_id), *verdict.purge
                    )
                if verdict.skip:
                    text = ""  # nothing is embedded; acked with zero points below
                else:
                    payload_base["confidential"] = verdict.confidential
        else:  # document-chunks
            logical = LOGICAL_DOCUMENT_CHUNKS
            document_id = str(evt.body["document_id"])
            element_id = str(evt.body["element_id"])
            meeting_id = None
            stable_key = f"docchunk:{document_id}:{element_id}"
            idem = f"emb:{logical}:{document_id}:{element_id}"
            payload_base = {
                "org_id": str(org_id),
                "document_id": document_id,
                "element_id": element_id,
                "element_type": evt.body["element_type"],
            }
            text = evt.body["text"]

        pieces = split_600_60(text)
        if not pieces:
            # Nothing to embed (empty text). Still ack so consumer lag doesn't grow on a
            # legitimately-empty message, but upsert nothing.
            await self.emit_outbox(
                session,
                evt,
                topic="embedding-events",
                idem=idem,
                payload={"collection": "", "point_count": 0, "meeting_id": meeting_id},
            )
            return

        vectors, _elapsed = embed_batch(pieces)
        observe_embedding_batch(_elapsed, len(pieces))
        points = [
            PointStruct(id=_point_id(stable_key, i), vector=vector, payload=payload_base)
            for i, vector in enumerate(vectors)
        ]
        collection_name = await upsert(self._qdrant, logical, org_id, points)

        await self.emit_outbox(
            session,
            evt,
            topic="embedding-events",
            idem=idem,
            payload={
                "collection": collection_name,
                "point_count": len(points),
                "meeting_id": meeting_id,
            },
        )
