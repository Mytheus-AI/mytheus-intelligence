"""`rag_client`: how an agent asks rag-service for grounding (mytheus-docs ADR-0033).

For `QuestionAgent` (and later `ContextGapAgent`). The agent asks AS A PARTICIPANT: retrieval
is participant-scoped (ADR-0033 D1), so `user_id` must be the participant the agent runs for,
and the agent only ever sees that participant's meetings. Never pass a service identity or
another user's id.

Passages only: an agent never asks for the generated answer (`generate` stays false), which
would cost a Sonnet call it does not need.

`HttpRagClient` calls `POST {RAG_URL}/api/rag/query` with the agent's service credentials
(`X-Mytheus-Service`, `X-Mytheus-Service-Token`, `X-Mytheus-Org`) and `X-Mytheus-User`.
rag-service holds the caller secrets in `MYTHEUS_RAG_CALLER_TOKENS`. A failure is an empty
result, never an exception: grounding is optional, and an agent must not die on it.

`FakeRagClient` is for tests: canned passages, and a record of every call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

CitationKind = Literal["transcript", "document", "decision", "action"]


@dataclass(frozen=True)
class RagPassage:
    n: int
    text: str
    score: float
    kind: CitationKind
    meeting_id: UUID | None = None
    chunk_id: UUID | None = None
    document_id: UUID | None = None
    element_id: str | None = None
    decision_id: UUID | None = None
    action_id: UUID | None = None
    meeting_title: str | None = None


class RagClient(Protocol):
    async def retrieve(
        self, *, org_id: UUID, user_id: UUID, query: str, meeting_id: UUID | None = None
    ) -> list[RagPassage]: ...


def _uuid(raw: Any) -> UUID | None:
    return UUID(raw) if raw else None


def parse_passages(body: dict[str, Any]) -> list[RagPassage]:
    out: list[RagPassage] = []
    for p in body.get("answer_passages", []):
        c = p["citation"]
        out.append(
            RagPassage(
                n=p["n"],
                text=p["text"],
                score=p["rerank_score"],
                kind=c["kind"],
                meeting_id=_uuid(c.get("meeting_id")),
                chunk_id=_uuid(c.get("chunk_id")),
                document_id=_uuid(c.get("document_id")),
                element_id=c.get("element_id"),
                decision_id=_uuid(c.get("decision_id")),
                action_id=_uuid(c.get("action_id")),
                meeting_title=p.get("meeting_title"),
            )
        )
    return out


class HttpRagClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        service: str = "agent-runtime-service",
        token: str | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self._url = (base_url or os.environ.get("RAG_URL", "http://localhost:8201")).rstrip("/")
        self._service = service
        self._token = token if token is not None else os.environ.get("RAG_CALLER_TOKEN", "")
        self._timeout = timeout_s

    async def retrieve(
        self,
        *,
        org_id: UUID,
        user_id: UUID,
        query: str,
        meeting_id: UUID | None = None,
        source: str | None = None,
    ) -> list[RagPassage]:
        """`source` ("transcripts" | "documents" | "decisions") forces rag-service's source
        filter; ProactiveAgent's conflict check asks for "decisions" only. Not part of the
        `RagClient` protocol: only callers that hold an `HttpRagClient` use it."""
        if not self._token:
            logger.warning("rag_client: RAG_CALLER_TOKEN unset; no grounding")
            return []
        body: dict[str, Any] = {"query": query[:1000], "generate": False}
        if meeting_id is not None:
            body["meeting_id"] = str(meeting_id)
        if source is not None:
            body["source"] = source
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(
                    f"{self._url}/api/rag/query",
                    json=body,
                    headers={
                        "X-Mytheus-Service": self._service,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                        "X-Mytheus-User": str(user_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("rag_client: rag-service unreachable: %s", exc)
            return []
        if res.status_code != 200:
            logger.warning("rag_client: rag-service answered %s", res.status_code)
            return []
        return parse_passages(res.json())


@dataclass
class FakeRagClient:
    passages: list[RagPassage] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def retrieve(
        self, *, org_id: UUID, user_id: UUID, query: str, meeting_id: UUID | None = None
    ) -> list[RagPassage]:
        self.calls.append(
            {"org_id": org_id, "user_id": user_id, "query": query, "meeting_id": meeting_id}
        )
        return list(self.passages)
