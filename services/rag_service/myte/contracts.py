"""The Ask Myte request and stream payloads (mytheus-docs ADR-0047 D8).

Request limits: `question` 1..2000 characters, `history` at most the last 12 messages of at most
4000 characters each. A body that breaks them is answered in the stream with one
`error {reason: "invalid"}` event, like every other outcome the UI renders.

One additive field beyond D8: `timezone` (optional IANA name). It sets "today" for the model and
is passed to tools that take a time zone (`check_my_capacity`).

ADR-0049 additions: `memories` (<= 30 of <= 300 characters, "What you remember about them"),
`attachments` (<= 3, names <= 120, <= 40,000 characters of text in total, registered as numbered
`document` sources), `mode` ("ask" | "peer_answer", default "ask") and, for `peer_answer`,
`peer {question_id, from_name, question}`: the recipient's Myte drafts a reply to a teammate's
question. In `peer_answer` mode `question` may be empty (the peer question is used); a
non-empty `question` that differs from it is the recipient's instruction for the draft.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTION = 2000
MAX_HISTORY = 12
MAX_HISTORY_CONTENT = 4000
MAX_MEMORIES = 30
MAX_MEMORY_TEXT = 300
MAX_ATTACHMENTS = 3
MAX_ATTACHMENT_NAME = 120
MAX_ATTACHMENT_TOTAL = 40_000
MAX_PEER_QUESTION = 1000

Personality = Literal["balanced", "concise", "detailed", "direct", "friendly"]
ErrorReason = Literal["budget", "not_connected", "unavailable", "invalid", "session"]
SourceKind = Literal[
    "transcript", "decision", "action", "promise", "meeting", "document", "person", "notification"
]
CardKind = Literal["meeting", "decision", "tasks", "promises", "capacity"]
ProposalKind = Literal[
    "task",
    "promise",
    "attendance",
    "agenda_item",
    "schedule_meeting",
    "invite_people",
    "slack_post",
    "export_decisions",
    "share_decision",
    "ask_myte",
    "memory",
]
Mode = Literal["ask", "peer_answer"]


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


class Memory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = Field(default=None, max_length=64)
    text: str = Field(min_length=1, max_length=MAX_MEMORY_TEXT)


class Attachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=MAX_ATTACHMENT_NAME)
    text: str = Field(min_length=1, max_length=MAX_ATTACHMENT_TOTAL)


class Peer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: UUID | None = None
    from_name: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=MAX_PEER_QUESTION)


class MyteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(default="", max_length=MAX_QUESTION)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY)
    scope: Scope = Field(default_factory=Scope)
    persona: Persona = Field(default_factory=Persona)
    page: Page = Field(default_factory=Page)
    want_title: bool = False
    timezone: str | None = Field(default=None, max_length=64)
    memories: list[Memory] = Field(default_factory=list, max_length=MAX_MEMORIES)
    attachments: list[Attachment] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)
    mode: Mode = "ask"
    peer: Peer | None = None

    @field_validator("attachments")
    @classmethod
    def _attachments_fit(cls, v: list[Attachment]) -> list[Attachment]:
        if sum(len(a.text) for a in v) > MAX_ATTACHMENT_TOTAL:
            raise ValueError("attachments are too long")
        return v

    @model_validator(mode="after")
    def _question_or_peer(self) -> MyteRequest:
        if self.mode == "peer_answer":
            if self.peer is None:
                raise ValueError("peer_answer needs peer")
            if not self.question.strip():
                self.question = self.peer.question
        elif not self.question.strip():
            raise ValueError("question must not be blank")
        return self

    @property
    def peer_mode(self) -> bool:
        return self.mode == "peer_answer" and self.peer is not None


def invalid_message(errors: list[dict[str, Any]]) -> str:
    """A plain sentence for the first validation error (shown to the person)."""
    if not errors:
        return "That question could not be read. Try again."
    loc = [str(p) for p in errors[0].get("loc", ())]
    if loc[:1] == ["question"] or (not loc and "question" in str(errors[0].get("msg", ""))):
        return f"Questions can be 1 to {MAX_QUESTION} characters."
    if loc[:1] == ["attachments"]:
        return (
            f"Attach at most {MAX_ATTACHMENTS} files with {MAX_ATTACHMENT_TOTAL:,} characters "
            "of text in total."
        )
    if loc[:1] == ["memories"]:
        return f"At most {MAX_MEMORIES} memories of {MAX_MEMORY_TEXT} characters each can be sent."
    if loc[:1] == ["peer"] or (not loc and "peer" in str(errors[0].get("msg", ""))):
        return "That teammate's question could not be read. Open it again from your Inbox."
    if loc[:1] == ["history"]:
        return (
            f"The conversation is too long to send: at most {MAX_HISTORY} earlier messages of "
            f"{MAX_HISTORY_CONTENT} characters each."
        )
    return "That question could not be read. Try again."
