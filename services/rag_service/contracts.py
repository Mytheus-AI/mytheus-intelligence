"""The `POST /api/rag/query` contract (mytheus-docs doc 09 section 3.5, ADR-0033 D3 and D6).

Passages are always returned; `answer` is present only when the caller asked for one
(`generate: true`) and the tenant is not throttled. Every passage carries a citation of one of
four kinds; a passage that cannot bind one never reaches this model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

CitationKind = Literal["transcript", "document", "decision", "action"]


class RagQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    generate: bool = False
    meeting_id: UUID | None = None  # optional: narrow to one meeting (still participant-scoped)


class Citation(BaseModel):
    kind: CitationKind
    meeting_id: UUID | None = None
    chunk_id: UUID | None = None
    speaker: str | None = None
    start_ms: int | None = None
    document_id: UUID | None = None
    element_id: str | None = None
    element_type: str | None = None
    decision_id: UUID | None = None
    action_id: UUID | None = None


class AnswerPassage(BaseModel):
    n: int  # the [n] marker an answer sentence uses
    text: str
    rerank_score: float
    citation: Citation
    context_before: str | None = None
    context_after: str | None = None
    meeting_title: str | None = None
    occurred_at: datetime | None = None


class Answer(BaseModel):
    text: str
    abstained: bool
    cited: list[int] = Field(default_factory=list)


class AppliedFilters(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    person_ids: list[UUID] = Field(default_factory=list)
    source: str | None = None


class RagResponse(BaseModel):
    answer_passages: list[AnswerPassage]
    answer: Answer | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    filters: AppliedFilters = Field(default_factory=AppliedFilters)
    cache: Literal["hit", "miss"] = "miss"
    degraded: bool = False  # budget throttled/blocked: no expansion, no answer
