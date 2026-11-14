"""The Ask Myte request and stream payloads (mytheus-docs ADR-0047 D8).

Request limits: `question` 1..2000 characters, `history` at most the last 12 messages of at most
4000 characters each. A body that breaks them is answered in the stream with one
`error {reason: "invalid"}` event, like every other outcome the UI renders.

One additive field beyond D8: `timezone` (optional IANA name). It sets "today" for the model and
is passed to tools that take a time zone (`check_my_capacity`).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION = 2000
MAX_HISTORY = 12
MAX_HISTORY_CONTENT = 4000

Personality = Literal["balanced", "concise", "detailed", "direct", "friendly"]
ErrorReason = Literal["budget", "not_connected", "unavailable", "invalid", "session"]
SourceKind = Literal[
    "transcript", "decision", "action", "promise", "meeting", "document", "person", "notification"
]
CardKind = Literal["meeting", "decision", "tasks", "promises", "capacity"]
ProposalKind = Literal["task", "promise", "attendance", "agenda_item"]


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "myte"]
    content: str = Field(max_length=MAX_HISTORY_CONTENT)


class Scope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["all", "meeting", "person", "as_of"] = "all"
    meeting_id: UUID | None = None
    person_id: UUID | None = None
    as_of: date | None = None
    # The chip's words ("Samar Rihan", a meeting title): names what "he", "this meeting" mean.
    label: str | None = Field(default=None, max_length=160)


class Persona(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(default="Myte", min_length=1, max_length=60)
    personality: Personality = "balanced"
    instructions: str | None = Field(default=None, max_length=2000)


class Page(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["meeting", "decision", "person"] | None = None
    id: UUID | None = None
    title: str | None = Field(default=None, max_length=300)


class MyteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY)
    scope: Scope = Field(default_factory=Scope)
    persona: Persona = Field(default_factory=Persona)
    page: Page = Field(default_factory=Page)
    want_title: bool = False
    timezone: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


def invalid_message(errors: list[dict[str, Any]]) -> str:
    """A plain sentence for the first validation error (shown to the person)."""
    if not errors:
        return "That question could not be read. Try again."
    loc = [str(p) for p in errors[0].get("loc", ())]
    if loc[:1] == ["question"]:
        return f"Questions can be 1 to {MAX_QUESTION} characters."
    if loc[:1] == ["history"]:
        return (
            f"The conversation is too long to send: at most {MAX_HISTORY} earlier messages of "
            f"{MAX_HISTORY_CONTENT} characters each."
        )
    return "That question could not be read. Try again."
