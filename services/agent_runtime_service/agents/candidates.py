"""Shared machinery for the meeting-scope candidate agents, `DecisionAgent` and `ActionAgent`.

Plan 7-9 section 3.5; the Phase 8 Part A cross-repo contract, section 1. Per closed window
(event-driven since the polish program, `windowing.py`), once per meeting:

1. **Pre-gate** (triage tier, Haiku): `<agent>_gate@vN` answers `{"score": x}`, how likely the
   window holds a decision / an action item. Below the threshold (env, default 0.5) the graph
   ends and no extraction call is made. An answer that cannot be parsed scores 0.
2. **Extraction** (extract tier, Sonnet; the router downgrades to Haiku only when the tenant is
   throttled): `<agent>@vN` answers strict JSON `{"candidates": [...]}`, each with its own
   `confidence`. Anything that is not that shape is no candidates.
3. **Bands per candidate** (doc 08 section 6.1): `< 0.4` dropped here, `0.4-0.7` kept with
   `surfaced: false`, `> 0.7` kept with `surfaced: true`. No candidate left, no result.
4. **One `AgentResult`** per window carrying every remaining candidate in `payload`, with
   deterministic ids (`uuid5(CANDIDATE_NS, f"{agent}:{window_id}:{index}")`, index in the
   model's order), so a replayed window yields the same ids and the core-api insert is
   idempotent.

What the model sees: the window with every speaker as "Participant N" (numbered in order of
first speaking). No user id, name or email is put in the request; the router redacts PII from
the user message on top. People the model names come back as those labels and are mapped to the
window's speaker ids here; only UUID speaker ids are passed through. These agents never stream.

ADR-0039 C1 / C2: the gate gets `{ground_rules}` (the never-record line, so a window mainly about
one of those topics scores 0 and no extraction is paid for) and the extraction `{company}` and
`{ground_rules}` (`agents/company.py`), from the context the runtime passes (`context["org"]`).
A template that does not name them ignores them, so `DecisionAgent` (whose prompts are not part
of ADR-0039) renders exactly what it did.

Polish, memory inside the meeting (`decision_agent@v3`, `action_agent@v3`): the extraction also
gets `{memory}` (the last two windows and this kind's cards with their status, `memory.py`), and
each answered candidate may carry `"updates": "D2"` (a card label) and `"quote"` (the verbatim
words). `memory.reconcile` turns an update, or a near-duplicate of an open card, into an update
of that card (its id, `updates_candidate_id`), and drops anything repeating a dismissed,
settled or human-edited card. The quote anchors the card's moment (`moments.py`, applied by the
runtime).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from abc import abstractmethod
from typing import Any, ClassVar
from uuid import UUID, uuid5

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.json_answer import json_object
from libs.shared.prompts import Prompt
from libs.shared.router import EXTRACT_MAX_TOKENS, EXTRACTION_EFFORT, TIER_NAMES, Effort
from libs.shared.state import AgentResult, AgentState, TranscriptWindow
from services.agent_runtime_service.agents.company import company_values, gate_ground_rules
from services.agent_runtime_service.agents.personal_listener_agent import band
from services.agent_runtime_service.guidance import guidance_values
from services.agent_runtime_service.memory import (
    KIND_OF_RESULT,
    memory_of,
    reconcile,
    render_memory,
)
from services.agent_runtime_service.names import speaker_labels
from services.agent_runtime_service.policy import thresholds_for

logger = logging.getLogger(__name__)

CANDIDATE_NS = UUID("6f1d3c2e-8a4b-5e7f-9c1d-2b3a4c5d6e7f")
DEFAULT_GATE_THRESHOLD = 0.5
_NUMBER = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_LABEL = re.compile(r"^\s*participant\s*#?\s*(\d+)\s*$", re.IGNORECASE)


def gate_threshold(env_var: str) -> float:
    """The pre-gate threshold from `env_var`; unset or unparseable is the 0.5 default."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return DEFAULT_GATE_THRESHOLD
    try:
        return _clamp(float(raw))
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", env_var, raw, DEFAULT_GATE_THRESHOLD)
        return DEFAULT_GATE_THRESHOLD


def parse_score(text: str) -> float:
    """`{"score": 0.8}` (or a bare number); anything unparseable is 0."""
    try:
        return _clamp(float(json.loads(_strip_fences(text)).get("score")))
    except (ValueError, AttributeError, TypeError):
        pass
    match = _NUMBER.search(text)
    return _clamp(float(match.group(0))) if match else 0.0


def parse_candidates(text: str, key: str = "candidates") -> list[Any]:
    """The `key` list of `{"<key>": [...]}` (`candidates` by default), tolerating code fences,
    prose around the object and a self-corrected second object. Anything else is an empty list
    (never raises)."""
    data = json_object(text, key)  # the model's last object holding `key` (json_answer.py)
    if data is None or not isinstance(data.get(key), list):
        return []
    return list(data[key])


def candidate_id(agent: str, window_id: str, index: int) -> str:
    return str(uuid5(CANDIDATE_NS, f"{agent}:{window_id}:{index}"))


def confidence_of(raw: dict[str, Any]) -> float | None:
    value = raw.get("confidence")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None  # json.loads accepts NaN / Infinity; never let one become "surfaced"
    return _clamp(float(value))


def text_of(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def render_window(window: TranscriptWindow) -> tuple[str, dict[str, str]]:
    """Every speaker as their meeting-wide "Participant N" label (`names.speaker_labels`; in
    order of first speaking for a window built outside the runtime). Returns the transcript and
    the label -> speaker id map (this window's speakers only) used to resolve the answer."""
    by_speaker = speaker_labels(window)
    lines = [f"{by_speaker[t.speaker_id]}: {t.text}" for t in window.turns]
    return "\n".join(lines), {label: speaker for speaker, label in by_speaker.items()}


def resolve_label(value: object, labels: dict[str, str]) -> str | None:
    """A "Participant N" label to the speaker's `users.id`; None unless it is a UUID speaker."""
    if not isinstance(value, str):
        return None
    match = _LABEL.match(value)
    if match is None:
        return None
    speaker = labels.get(f"Participant {int(match.group(1))}")
    try:
        return str(UUID(speaker)) if speaker else None
    except ValueError:
        return None  # an internal identity (audio tap, egress), never a user


def resolve_labels(values: object, labels: dict[str, str]) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        uid = resolve_label(value, labels)
        if uid is not None and uid not in out:
            out.append(uid)
    return out


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def _clamp(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


class CandidateAgent(BaseAgent):
    """A meeting-scope agent that pre-gates a window on Haiku, extracts candidates on Sonnet and
    emits them as one `AgentResult`. Subclasses name the gate prompt, the threshold env var,
    the result kind, and shape one raw model candidate into its contract fields."""

    gate_key: ClassVar[str]
    threshold_env: ClassVar[str]
    kind: ClassVar[str]
    extract_max_tokens: ClassVar[int] = EXTRACT_MAX_TOKENS  # room for thinking (polish)
    effort: ClassVar[Effort] = EXTRACTION_EFFORT  # extraction: low

    gate_prompt: Prompt | None = None

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        await super().spawn(org_id=org_id, meeting_id=meeting_id, user_id=user_id)
        assert self.org_id is not None
        self.gate_prompt = self._loader.load(
            self.gate_key, self._loader.resolve(self.gate_key, self.org_id)
        )

    def user_values(self, context: dict[str, Any] | None = None) -> dict[str, str]:
        """Extra values for the extraction prompt's user template (beyond `window`): the company
        block and the never-record line (ADR-0039), "" for an org without a brain."""
        return company_values(context)

    def memory_values(self, context: dict[str, Any] | None = None) -> dict[str, str]:
        """`{memory}` (polish, decision_agent@v3 / action_agent@v3): the meeting's last two
        windows and the cards of this kind already raised, with their status (`memory.py`).
        "" for a meeting's first window; a v2 template does not name it and ignores it."""
        block, _ = render_memory(memory_of(context), KIND_OF_RESULT[self.kind])
        return {"memory": block}

    def extra(self, raw: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
        """Optional fields of one candidate that need the participant context (ActionAgent's
        `suggested_owner_user_id`). Added only when non-empty."""
        return {}

    @abstractmethod
    def shape(self, raw: dict[str, Any], labels: dict[str, str]) -> dict[str, Any] | None:
        """The agent-specific contract fields of one candidate, or None if it is malformed.
        `content`, `confidence`, `id` and `surfaced` are added by the caller."""

    def candidates(
        self, text: str, window: TranscriptWindow, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Parse, validate, band and id the model's candidates. Dropped ones are left out; the id
        index is the candidate's position in the model's list, so ids never shift."""
        _, labels = render_window(window)
        limits = thresholds_for(context, self.name)
        kept: list[dict[str, Any]] = []
        updates: list[object] = []
        for index, raw in enumerate(parse_candidates(text)):
            if not isinstance(raw, dict):
                continue
            content, confidence = text_of(raw, "content"), confidence_of(raw)
            if content is None or confidence is None:
                continue
            verdict = band(confidence, limits)
            if verdict == "dropped":
                continue
            fields = self.shape(raw, labels)
            if fields is None:
                continue
            kept.append(
                {
                    "id": candidate_id(self.name, window.window_id, index),
                    "content": content,
                    "confidence": confidence,
                    "surfaced": verdict == "surfaced",
                    **fields,
                    **self.extra(raw, context),
                    **({"quote": quote} if (quote := text_of(raw, "quote")) else {}),
                }
            )
            updates.append(raw.get("updates"))
        # polish, memory inside the meeting: an update of an open card keeps that card's id;
        # a repeat of a dismissed or settled card is never emitted
        memory = memory_of(context)
        _, card_labels = render_memory(memory, KIND_OF_RESULT[self.kind])
        return reconcile(kept, KIND_OF_RESULT[self.kind], memory, card_labels, updates)

    def build_graph(self) -> Any:
        threshold = gate_threshold(self.threshold_env)

        async def pregate(s: AgentState) -> dict[str, Any]:
            assert self.gate_prompt is not None and s.org_id is not None
            transcript, _ = render_window(s.chunk)
            out = await self.router.complete(
                tier="triage",
                system=self.gate_prompt.system,
                **self.gate_prompt.user_parts(window=transcript, **gate_ground_rules(s.context)),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=20,
                prompt_ref=self.gate_prompt.ref,
            )
            return {"relevance_score": parse_score(out.text)}

        async def extract(s: AgentState) -> dict[str, Any]:
            assert self.prompt is not None and s.org_id is not None
            transcript, _ = render_window(s.chunk)
            out = await self.router.complete(
                tier="extract",
                system=self.prompt.system,
                **self.prompt.user_parts(
                    window=transcript,
                    **self.user_values(s.context),
                    **self.memory_values(s.context),
                    # polish wave 2: recent kept / corrected / dismissed examples ("" if none)
                    **guidance_values(self.name, s.context),
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=self.extract_max_tokens,
                effort=self.effort,
                prompt_ref=self.prompt.ref,
            )
            kept = self.candidates(out.text, s.chunk, s.context)
            if not kept:
                return {"confidence": 0.0}
            top = max(kept, key=lambda c: float(c["confidence"]))
            return {
                "confidence": top["confidence"],
                "result": AgentResult(
                    kind=self.kind,
                    content=top["content"],
                    relevance_score=s.relevance_score or 0.0,
                    confidence=top["confidence"],
                    source_chunk_ids=s.chunk.chunk_ids,
                    payload={
                        "candidates": kept,
                        "prompt": self.prompt.ref,
                        "model_tier": TIER_NAMES.get(out.tier, out.tier),
                    },
                ),
            }

        g: Any = StateGraph(AgentState)  # langgraph stubs reject async dict-returning nodes
        g.add_node("pregate", pregate)
        g.add_node("extract", extract)
        g.set_entry_point("pregate")
        g.add_conditional_edges(
            "pregate",
            lambda s: (s.relevance_score or 0.0) >= threshold,
            {True: "extract", False: END},
        )
        g.add_edge("extract", END)
        return g.compile()
