"""In-process types for the RAG pipeline: the caller, the asker's scope, filters, candidates.

Kept free of I/O so the pipeline, the arms and the eval harness share one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from uuid import UUID

from services.rag_service.contracts import CitationKind


@dataclass(frozen=True)
class Caller:
    """Who is asking. `user_id` is our `users.id` when known (an agent asking for a
    participant); a user's own question arrives with `clerk_user_id` and is resolved by the
    scope call."""

    org_id: UUID
    user_id: UUID | None = None
    clerk_user_id: str | None = None

    def scope_key(self) -> str:
        return str(self.user_id) if self.user_id else f"clerk:{self.clerk_user_id}"


@dataclass(frozen=True)
class MeetingInfo:
    title: str | None
    started_at: datetime


@dataclass(frozen=True)
class Person:
    id: UUID
    email: str


@dataclass(frozen=True)
class Scope:
    """ADR-0033 D1: the meetings the asker is a participant of. Documents are org-wide."""

    user_id: UUID
    meetings: dict[UUID, MeetingInfo]
    people: tuple[Person, ...] = ()


@dataclass(frozen=True)
class Filters:
    """ADR-0033 D2, from the expansion call. Dates are inclusive, on the meeting's start."""

    date_from: date | None = None
    date_to: date | None = None
    person_ids: tuple[UUID, ...] = ()
    source: str | None = None  # transcripts | documents | decisions

    @property
    def narrows(self) -> bool:
        return bool(self.date_from or self.date_to or self.person_ids or self.source)

    def relaxed(self) -> Filters:
        """The corrective retry: drop date and person, keep source."""
        return Filters(source=self.source)

    def wants(self, kind: str) -> bool:
        if self.source is None:
            return True
        return (
            {
                "transcripts": {"transcript"},
                "documents": {"document"},
                "decisions": {"decision", "action"},
            }
            .get(self.source, {kind})
            .__contains__(kind)
        )

    def meeting_ok(self, info: MeetingInfo | None) -> bool:
        if not (self.date_from or self.date_to):
            return True
        if info is None:
            return False
        day = info.started_at.date()
        if self.date_from and day < self.date_from:
            return False
        return not (self.date_to and day > self.date_to)


def allowed_meetings(scope: Scope, filters: Filters, only: UUID | None = None) -> list[UUID]:
    out = [m for m, info in scope.meetings.items() if filters.meeting_ok(info)]
    if only is not None:
        out = [m for m in out if m == only]
    return out


@dataclass
class Candidate:
    """One retrievable unit, before and after hydration. `key` is what RRF fuses on."""

    kind: CitationKind
    key: str
    meeting_id: UUID | None = None
    chunk_id: UUID | None = None
    speaker: str | None = None
    start_ms: int | None = None
    document_id: UUID | None = None
    element_id: str | None = None
    element_type: str | None = None
    decision_id: UUID | None = None
    action_id: UUID | None = None
    text: str = ""
    before: str = ""
    after: str = ""
    score: float = 0.0
    sources: set[str] = field(default_factory=set)

    def with_text(self, text: str, before: str = "", after: str = "") -> Candidate:
        return replace(self, text=text, before=before, after=after)


def transcript_key(chunk_id: UUID | str) -> str:
    return f"transcript:{chunk_id}"


def document_key(document_id: UUID | str, element_id: str) -> str:
    return f"document:{document_id}:{element_id}"


def decision_key(decision_id: UUID | str) -> str:
    return f"decision:{decision_id}"


def action_key(action_id: UUID | str) -> str:
    return f"action:{action_id}"


@dataclass(frozen=True)
class LexicalDecision:
    """One hit of the decision check's lexical arm (core-api FTS over confirmed decisions)."""

    id: UUID
    meeting_id: UUID | None
    content: str
    rationale: str | None = None
    meeting_title: str | None = None
    decided_at: str | None = None  # ISO-8601
