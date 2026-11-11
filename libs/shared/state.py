"""The shared agent data model: `TranscriptWindow`, `AgentState`, `AgentResult`.

Plan 7-9 section 3.2 step 2 and doc 08 section 3.1. Every agent runs on a `TranscriptWindow`
(finals merged by speaker turn, `A17`; since the polish program an event-driven 10-15 s window,
`services/agent_runtime_service/windowing.py`), never on a raw `transcript_chunks` row, and
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
class Utterance:
    """One `transcript-chunks` final as it was heard: the unit a card's quote points at. Its
    `chunk_id` is globally unique and stable across replays and window boundaries, which is
    what makes a moment id (`moments.py`) the same for every agent that cites it."""

    chunk_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Turn:
    """Consecutive finals from one speaker, merged."""

    speaker_id: str  # the LiveKit identity, i.e. `users.id` for a human (transcription-service)
    text: str
    start_ms: int
    end_ms: int
    chunk_ids: list[str] = field(default_factory=list)
    # the finals this turn merged, in order (empty for a turn built outside the windower: the
    # turn then counts as one utterance)
    parts: list[Utterance] = field(default_factory=list)

    def utterances(self) -> list[Utterance]:
        if self.parts:
            return list(self.parts)
        cid = self.chunk_ids[0] if self.chunk_ids else f"{self.speaker_id}@{self.start_ms}"
        return [Utterance(cid, self.speaker_id, self.start_ms, self.end_ms, self.text)]


@dataclass
class Trigger:
    """Why a window closed, and the words that closed it (polish: event-driven windows).
    `reason`: `cue` (a phrase like "I'll" or "we decided"), `turn_end` (the speaker changed or
    went quiet after at least the minimum window), `max_span` (the window reached its longest),
    `max_wait` (nothing else closed it in time), `meeting_end`, or `merged` (a worker that fell
    behind merged queued windows)."""

    reason: str
    chunk_id: str = ""
    speaker_id: str = ""
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    cue: str | None = None


@dataclass
class TranscriptWindow:
    org_id: UUID
    meeting_id: UUID
    # f"{meeting_id}:{start_ms}", start_ms the window's first final, so the id carries the
    # moment it points at and a replay of the same finals rebuilds the same id
    window_id: str
    start_ms: int
    end_ms: int
    turns: list[Turn] = field(default_factory=list)
    # speaker id -> "Participant N", assigned by the runtime on a speaker's first appearance in
    # the meeting and kept for the whole meeting, so the model sees the same label for the same
    # person in every window (`services/agent_runtime_service/names.py`). Empty for a window
    # built outside the runtime: the renderers then number speakers in order of first speaking.
    labels: dict[str, str] = field(default_factory=dict)
    trigger: Trigger | None = None
    # wall-clock epoch seconds when the window's last final was produced (its envelope's
    # `occurred_at`, else when the runtime received it): the start of speech-to-card latency
    last_chunk_at: float | None = None
    final: bool = False  # the meeting's last window: agents that accumulate windows flush

    @property
    def utterances(self) -> list[Utterance]:
        return [u for t in self.turns for u in t.utterances()]

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
