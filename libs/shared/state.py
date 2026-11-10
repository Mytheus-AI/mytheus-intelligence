"""The shared agent data model: `TranscriptWindow`, `AgentState`, `AgentResult`.

Plan 7-9 section 3.2 step 2 and doc 08 section 3.1. Every agent runs on a `TranscriptWindow`
(30 s of finals merged by speaker turn, `A17`), never on a raw `transcript_chunks` row, and
carries one `AgentState` through its LangGraph graph.

`AgentState` holds exactly the plan's fields plus `result`: the graph's last node writes the
agent's output there. LangGraph nodes return partial state updates, never a different type,
so the output has to live on the state rather than being "returned" by a node.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class Turn:
    """Consecutive finals from one speaker, merged."""

    speaker_id: str  # the LiveKit identity, i.e. `users.id` for a human (transcription-service)
    text: str
    start_ms: int
    end_ms: int
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class TranscriptWindow:
    org_id: UUID
    meeting_id: UUID
    window_id: str  # deterministic: f"{meeting_id}:{start_ms}", so a replay has the same id
    start_ms: int
    end_ms: int
    turns: list[Turn] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c for t in self.turns for c in t.chunk_ids]

    @property
    def speaker_ids(self) -> set[str]:
        return {t.speaker_id for t in self.turns}

    def transcript(self, names: dict[str, str] | None = None) -> str:
        """One line per turn, `speaker: text`. `names` maps speaker ids to display names."""
        names = names or {}
        return "\n".join(f"{names.get(t.speaker_id, t.speaker_id)}: {t.text}" for t in self.turns)


# (seq, text, done) -> an `agent-stream` delta for this window and participant
StreamFn = Callable[[int, str, bool], Awaitable[None]]


@dataclass
class AgentResult:
    kind: str  # AgentResultKind: note | decision_candidate | action_candidate | suggestion | alert
    content: str
    relevance_score: float
    confidence: float
    source_chunk_ids: list[str]
    payload: dict[str, Any] = field(default_factory=dict)
    # Overrides the window's id on `agent-results` (and in the idempotency key). ProactiveAgent
    # sets `proactive:{alert_type}:{subject}` so a replay of any trigger yields the same key.
    window_id: str | None = None


@dataclass
class AgentState:
    meeting_id: UUID
    user_id: UUID
    chunk: TranscriptWindow  # the plan's name for the window
    context: dict[str, Any]  # get_user_context() result
    retrieved_facts: list[dict[str, Any]] = field(default_factory=list)
    org_id: UUID | None = None  # tenant key; BaseAgent fails closed if it is missing
    agent_id: str = ""  # stable identity for agent-to-agent authz and audit
    relevance_score: float | None = None  # PersonalListener pre-gate (0-1)
    confidence: float | None = None  # surfacing gate (0-1)
    prompt_version: str = ""
    token_budget_remaining: int = 0
    correlation_id: str | None = None
    result: AgentResult | None = None
    stream: StreamFn | None = field(default=None, repr=False)  # live deltas; None = no streaming
