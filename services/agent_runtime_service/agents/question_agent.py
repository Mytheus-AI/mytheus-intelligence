"""`QuestionAgent`: one per participant per meeting, proposes questions that participant could ask.

Plan 7-9 section 3.10; doc 08 section 2.1; mytheus-docs ADR-0034. Per closed 30 s window, for
the participant this instance belongs to:

1. **Context-gap heuristic** (no model call): the window needs at least `MIN_TURNS` turns and
   `MIN_WORDS` words, and the agent stays quiet for `QUESTION_COOLDOWN_WINDOWS` windows (default
   1) after it surfaced a suggestion, so the sidebar is not flooded.
2. **Gate** (triage tier, Haiku): `question_gate@vN` scores 0-1 how likely the window leaves a
   gap this participant, given their role, should ask about. Below the threshold
   (`QUESTION_GATE_THRESHOLD`, default 0.5) the graph ends.
3. **Optional grounding** (`QUESTION_RAG_GROUNDING`, off by default; `grounding.py`): passages
   only, retrieved as this participant, so a suggestion is never grounded on a meeting the
   participant did not attend.
4. **Proposal** (extract tier, Sonnet; Haiku when the tenant is throttled): `question_agent@vN`
   answers strict JSON `{"suggestions": [...]}`, at most `MAX_SUGGESTIONS`, each with
   `question`, `priority` 1-3, `rationale`, `sources` (passage numbers) and `confidence`.
5. **Bands per suggestion** (doc 08 section 6.1): `< 0.4` dropped here, `0.4-0.7` kept with
   `surfaced: false`, `> 0.7` kept with `surfaced: true`. `core-api` re-derives the band with
   the tenant's overrides (ADR-0030 D3); the gateway forwards only surfaced ones, to this
   participant only (ADR-0034 D6).
6. **One `AgentResult`** (kind `suggestion`) per window, ids
   `uuid5(CANDIDATE_NS, f"QuestionAgent:{window_id}:{user_id}:{index}")` (ADR-0034 D3).

What the model sees: the participant as "You", everyone else as "Participant N", and a profile of
role / skills / bio; from `question_agent@v2` (ADR-0038 D10) the proposal also gets the same
"Listen especially for" / "Ignore" lines and voice block as PersonalListenerAgent, and the gate
(`question_gate@v2`) the same two lines without the voice. Its threshold, the cooldown and the
bands are unchanged. No ids, names or
email addresses; the router redacts PII on top. Every model call goes through the router, so it
is budget-gated and metered on `usage-events`.

ADR-0039 (`question_agent@v3`, `question_gate@v3`): the proposal's user message starts with the
company block (description, vocabulary, active goals) and the "Never record" line, the gate gets
the line only; a window mainly about a never-record topic scores 0 at the gate and the proposal
never asks about one (C1, C2). With no company brain both render exactly what v2 did. The runtime
never runs this agent for a participant whose attendance is `agent` (C6): they are not there to
ask anything.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Literal
from uuid import UUID, uuid5

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.router import TIER_NAMES, LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow
from services.agent_runtime_service.agents.candidates import (
    CANDIDATE_NS,
    confidence_of,
    gate_threshold,
    parse_candidates,
    parse_score,
    text_of,
)
from services.agent_runtime_service.agents.grounding import Grounder, Passage, ground
from services.agent_runtime_service.agents.personal_listener_agent import (
    band,
    gate_values,
    persona_values,
    render_window,
)

GATE_KEY = "question_gate"
MIN_TURNS = 2
MIN_WORDS = 25
MAX_SUGGESTIONS = 2
DEFAULT_COOLDOWN_WINDOWS = 1
GROUNDING_QUERY_CHARS = 500


def cooldown_windows() -> int:
    try:
        return max(0, int(os.environ.get("QUESTION_COOLDOWN_WINDOWS", DEFAULT_COOLDOWN_WINDOWS)))
    except ValueError:
        return DEFAULT_COOLDOWN_WINDOWS


def has_context_gap_signal(window: TranscriptWindow) -> bool:
    """The cheap pre-filter: enough conversation in the window for a question to be about."""
    words = sum(len(t.text.split()) for t in window.turns)
    return len(window.turns) >= MIN_TURNS and words >= MIN_WORDS


def suggestion_id(window_id: str, user_id: UUID, index: int) -> str:
    return str(uuid5(CANDIDATE_NS, f"QuestionAgent:{window_id}:{user_id}:{index}"))


def parse_priority(value: object) -> int | None:
    """1, 2 or 3 (an int, or a string of one); anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value if isinstance(value, int) and 1 <= value <= 3 else None


def render_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(none)"
    return "\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))


def grounding_query(window: TranscriptWindow) -> str:
    """The window's words, newest last, trimmed from the front: no speaker labels or ids."""
    text = " ".join(t.text for t in window.turns)
    return text[-GROUNDING_QUERY_CHARS:]


class QuestionAgent(BaseAgent):
    name: ClassVar[str] = "QuestionAgent"
    prompt_key: ClassVar[str] = "question_agent"
    scope: ClassVar[Literal["participant", "meeting"]] = "participant"
    kind: ClassVar[str] = "suggestion"

    gate_prompt: Prompt | None = None

    def __init__(
        self, loader: PromptLoader, router: LiteLLMRouter, *, grounder: Grounder | None = None
    ) -> None:
        super().__init__(loader, router)
        self._grounder = grounder
        self._windows_seen = 0
        self._quiet_until = 0  # window count up to which the agent stays quiet

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        await super().spawn(org_id=org_id, meeting_id=meeting_id, user_id=user_id)
        assert self.org_id is not None
        self.gate_prompt = self._loader.load(GATE_KEY, self._loader.resolve(GATE_KEY, self.org_id))

    async def process(self, state: AgentState) -> AgentResult | None:
        self._windows_seen += 1
        if self._windows_seen <= self._quiet_until or not has_context_gap_signal(state.chunk):
            return None  # no model call: nothing to ask about, or just surfaced something
        result = await super().process(state)
        if result is not None and any(s["surfaced"] for s in result.payload["suggestions"]):
            self._quiet_until = self._windows_seen + cooldown_windows()
        return result

    def suggestions(
        self, text: str, window: TranscriptWindow, user_id: UUID, passages: list[Passage]
    ) -> list[dict[str, Any]]:
        """Parse, validate, band and id the model's suggestions; dropped ones are left out."""
        kept: list[dict[str, Any]] = []
        for index, raw in enumerate(parse_candidates(text, key="suggestions")[:MAX_SUGGESTIONS]):
            if not isinstance(raw, dict):
                continue
            question, confidence = text_of(raw, "question"), confidence_of(raw)
            priority = parse_priority(raw.get("priority"))
            if question is None or confidence is None or priority is None:
                continue
            verdict = band(confidence)
            if verdict == "dropped":
                continue
            kept.append(
                {
                    "id": suggestion_id(window.window_id, user_id, index),
                    "question": question,
                    "priority": priority,
                    "confidence": confidence,
                    "surfaced": verdict == "surfaced",
                    "rationale": text_of(raw, "rationale"),
                    "citations": self._citations(raw.get("sources"), passages),
                }
            )
        return kept

    @staticmethod
    def _citations(sources: object, passages: list[Passage]) -> list[dict[str, Any]]:
        """The citations of the passages a suggestion names; numbers outside the list are
        ignored, so a suggestion can never cite something it was not shown."""
        if not isinstance(sources, list):
            return []
        out: list[dict[str, Any]] = []
        for n in sources:
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(passages):
                citation = passages[n - 1].citation
                if citation not in out:
                    out.append(citation)
        return out

    def build_graph(self) -> Any:
        threshold = gate_threshold("QUESTION_GATE_THRESHOLD")

        async def pregate(s: AgentState) -> dict[str, Any]:
            assert self.gate_prompt is not None and s.org_id is not None
            out = await self.router.complete(
                tier="triage",
                system=self.gate_prompt.system,
                user=self.gate_prompt.render_user(
                    **gate_values(s.context), window=render_window(s.chunk, s.user_id)
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=20,
                prompt_ref=self.gate_prompt.ref,
            )
            return {"relevance_score": parse_score(out.text)}

        async def retrieve(s: AgentState) -> dict[str, Any]:
            assert s.org_id is not None
            # As the participant, never as a service identity (ADR-0033 D1, ADR-0034 D9).
            passages = await ground(
                self._grounder,
                org_id=s.org_id,
                user_id=s.user_id,
                query=grounding_query(s.chunk),
            )
            return {"retrieved_facts": [{"text": p.text, "citation": p.citation} for p in passages]}

        async def propose(s: AgentState) -> dict[str, Any]:
            assert self.prompt is not None and s.org_id is not None
            passages = [Passage(text=f["text"], citation=f["citation"]) for f in s.retrieved_facts]
            out = await self.router.complete(
                tier="extract",
                system=self.prompt.system,
                user=self.prompt.render_user(
                    **persona_values(s.context),
                    window=render_window(s.chunk, s.user_id),
                    context=render_passages(passages),
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=500,
                prompt_ref=self.prompt.ref,
            )
            kept = self.suggestions(out.text, s.chunk, s.user_id, passages)
            if not kept:
                return {"confidence": 0.0}
            top = max(kept, key=lambda c: float(c["confidence"]))
            return {
                "confidence": top["confidence"],
                "result": AgentResult(
                    kind=self.kind,
                    content=top["question"],
                    relevance_score=s.relevance_score or 0.0,
                    confidence=top["confidence"],
                    source_chunk_ids=s.chunk.chunk_ids,
                    payload={
                        "suggestions": kept,
                        "prompt": self.prompt.ref,
                        "model_tier": TIER_NAMES.get(out.tier, out.tier),
                        "grounded": bool(passages),
                    },
                ),
            }

        g: Any = StateGraph(AgentState)  # langgraph stubs reject async dict-returning nodes
        g.add_node("pregate", pregate)
        g.add_node("retrieve", retrieve)
        g.add_node("propose", propose)
        g.set_entry_point("pregate")
        g.add_conditional_edges(
            "pregate",
            lambda s: (s.relevance_score or 0.0) >= threshold,
            {True: "retrieve", False: END},
        )
        g.add_edge("retrieve", "propose")
        g.add_edge("propose", END)
        return g.compile()
