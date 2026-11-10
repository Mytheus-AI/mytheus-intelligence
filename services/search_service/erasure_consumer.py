"""Finishing an admin data erasure in the FTS read model (store `search_index`):
`libs/shared/erasure.py` has the shared rules; this is search-service's share, as the owner of
`intelligence.transcript_chunk_search`.

`SearchErasureConsumer` (group `search-service.erasure`, `admin-events`):

- MEETING: every row of the meeting is deleted (its text and its tsvector go with it).
- PERSON: rows whose `speaker_id` is their `users.id` get `speaker_id = 'erased'`; the text stays,
  as core-api's rule keeps the words as the meeting's record. Diarized labels (`1`, `2`, ...)
  carry no identity of their own: core-api deleted the `meeting_speakers` rows that named them.

Each statement runs in `BaseConsumer`'s transaction, under `app.current_org` (RLS) AND with an
explicit `org_id` predicate, so another tenant's rows cannot match even on a connection that
bypasses RLS.

`ErasureAwareSearchConsumer` is the `transcript-chunks` indexer with the erasure tombstone check
in front (`libs/shared/erasure.py` "Late arrivals"); `worker.py` runs it.
"""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import ConsumedEvent
from libs.shared.erasure import (
    ERASED_SPEAKER,
    ErasureConsumer,
    ErasureRequest,
    vet_transcript_chunk,
)
from services.search_service.consumer import TranscriptChunkSearchConsumer

_DELETE_MEETING = text("""
    DELETE FROM intelligence.transcript_chunk_search
    WHERE org_id = :org_id AND meeting_id = :target
    """)
_ANONYMIZE_PERSON = text("""
    UPDATE intelligence.transcript_chunk_search SET speaker_id = :erased
    WHERE org_id = :org_id AND lower(speaker_id) = :target
    """)


class SearchErasureConsumer(ErasureConsumer):
    group: ClassVar[str] = "search-service.erasure"
    store: ClassVar[str] = "search_index"
    producer_name: ClassVar[str] = "search-service"

    async def erase(self, req: ErasureRequest, session: AsyncSession) -> dict[str, int]:
        params = {"org_id": str(req.org_id), "target": req.target}
        if req.kind == "meeting":
            result = await session.execute(_DELETE_MEETING, params)
            return {"deleted": int(result.rowcount or 0)}  # type: ignore[attr-defined]
        result = await session.execute(_ANONYMIZE_PERSON, {**params, "erased": ERASED_SPEAKER})
        return {"anonymized": int(result.rowcount or 0)}  # type: ignore[attr-defined]


class ErasureAwareSearchConsumer(TranscriptChunkSearchConsumer):
    """The FTS indexer that never stores a chunk of an erased meeting, and stores a chunk spoken
    by a since-erased person without their attribution."""

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        vetted = await vet_transcript_chunk(self._redis, evt)
        if vetted is None:
            return
        await super().apply(vetted, session)
