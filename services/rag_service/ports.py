"""What the pipeline needs from the outside world. The real implementations are in
`clients.py`; unit tests and the eval harness pass their own."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from services.rag_service.domain import Caller, Candidate, Scope


class ScopeProvider(Protocol):
    async def scope(self, caller: Caller) -> Scope | None: ...

    async def records(
        self, org_id: UUID, user_id: UUID, decision_ids: Sequence[UUID], action_ids: Sequence[UUID]
    ) -> dict[str, Candidate]:
        """Confirmed decisions/actions by key, already restricted to the asker's meetings."""
        ...


class Embedder(Protocol):
    async def embed(self, org_id: UUID, queries: Sequence[str]) -> list[list[float]]: ...


class VectorSearch(Protocol):
    async def search(
        self,
        org_id: UUID,
        vector: list[float],
        *,
        meeting_ids: Sequence[UUID],
        speaker_ids: Sequence[str],
        kinds: set[str],
        limit: int,
    ) -> list[Candidate]:
        """Ranked hits, best first, across the wanted collections (transcript, document,
        decision). Transcript and decision hits are restricted to `meeting_ids`."""
        ...


class LexicalSearch(Protocol):
    async def search(
        self,
        org_id: UUID,
        query: str,
        *,
        meeting_ids: Sequence[UUID],
        speaker_ids: Sequence[str],
        limit: int,
    ) -> list[Candidate]: ...

    async def passages(
        self, org_id: UUID, candidates: Sequence[Candidate], neighbors: int
    ) -> dict[str, Candidate]:
        """Transcript and document candidates with text (and neighbours), by key."""
        ...


class GraphSearch(Protocol):
    async def decisions(
        self, org_id: UUID, *, person_ids: Sequence[UUID], topics: Sequence[str], limit: int
    ) -> list[Candidate]: ...


class Reranker(Protocol):
    async def rerank(
        self, org_id: UUID, query: str, passages: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        """(index into `passages`, score in 0..1), best first."""
        ...
