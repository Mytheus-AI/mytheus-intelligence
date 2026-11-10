"""`PersonalListenerAgent`: one per participant per meeting, private notes for that participant.

Plan 7-9 section 3.3; mytheus-docs doc 08 section 2.1 and ADR-0030.

Per closed 30 s window, for the participant this instance belongs to:

1. **Relevance pre-gate** (triage tier, Haiku): `personal_listener_gate@vN` scores the window
   0-1 against the participant's role, skills and bio. Below the threshold
   (`PERSONAL_LISTENER_RELEVANCE_THRESHOLD`, default 0.5, ADR-0030 D2) the graph ends and no
   extraction call is made. An answer that cannot be parsed scores 0: never pay for an
   extraction on a gate the model did not actually pass.
2. **Extraction** (extract tier, Sonnet; Haiku when the tenant is throttled):
   `personal_listener@vN` writes `CONFIDENCE: x` on its first line, then the note.
3. **Confidence gate** (doc 08 section 6): `< 0.4` -> no result (nothing is emitted);
   `0.4-0.7` -> result with `payload.surfaced = false` (`core-api` persists it, the sidebar
   never shows it); `> 0.7` -> `payload.surfaced = true`. `core-api` re-derives `surfaced`
   with the tenant's `agent_confidence_overrides` (ADR-0030 D3).
4. **Streaming** (ADR-0030 D4): the extraction streams, but deltas are held until the first
   line has been parsed. Only when that confidence is above 0.7 does the rest go out on
   `agent-stream`, so a note the policy would not surface never appears live.

What the model sees: the window with the participant as "You" and everyone else as
"Participant N", and a profile of role / skills / bio. No user ids, names or email addresses
are put in the request, and the router redacts PII from the whole user message on top.

ADR-0038 D10 (`personal_listener@v2`): the extraction's profile block gains "Listen especially
for" / "Ignore" lines, and a voice block from `agent_personality` plus the person's own
`agent_instructions` (delimited; the system prompt says they change wording only). All of it is
in the user message, so the system prompt stays one stable, cached prefix for every participant.
The relevance pre-gate (`personal_listener_gate@v1`), its threshold and the bands are unchanged:
the gate still sees role / skills / bio only. `agent_name` is never put in a prompt.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, ClassVar, Literal
from uuid import UUID

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import Prompt
from libs.shared.state import AgentResult, AgentState, StreamFn, TranscriptWindow

SURFACE_ABOVE = 0.7
SILENT_FROM = 0.4
GATE_KEY = "personal_listener_gate"
_NUMBER = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")
_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def relevance_threshold() -> float:
    return float(os.environ.get("PERSONAL_LISTENER_RELEVANCE_THRESHOLD", "0.5"))


def parse_relevance(text: str) -> float:
    """`{"relevance": 0.8}` (or a bare number); anything unparseable is 0."""
    try:
        value = json.loads(text.strip()).get("relevance")
        return _clamp(float(value))
    except (ValueError, AttributeError, TypeError):
        pass
    match = _NUMBER.search(text)
    return _clamp(float(match.group(0))) if match else 0.0


def parse_note(text: str) -> tuple[float, str]:
    """`CONFIDENCE: x` then the note. A missing or malformed header is confidence 0."""
    first, _, rest = text.strip().partition("\n")
    match = _CONFIDENCE.match(first)
    if match is None:
        return 0.0, ""
    note = rest.strip()
    if note.upper() == "NONE":
        note = ""
    return _clamp(float(match.group(1))), note


def band(confidence: float) -> Literal["surfaced", "silent", "dropped"]:
    if confidence > SURFACE_ABOVE:
        return "surfaced"
    if confidence >= SILENT_FROM:
        return "silent"
    return "dropped"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def render_window(window: TranscriptWindow, user_id: UUID) -> str:
    """The participant is "You"; everyone else is numbered in order of first speaking."""
    me = str(user_id)
    labels: dict[str, str] = {}
    lines = []
    for turn in window.turns:
        if turn.speaker_id == me:
            label = "You"
        else:
            label = labels.setdefault(turn.speaker_id, f"Participant {len(labels) + 1}")
        lines.append(f"{label}: {turn.text}")
    return "\n".join(lines)


def render_profile(context: dict[str, Any]) -> str:
    parts = [f"Role: {context.get('role') or 'not set'}"]
    skills = [s for s in context.get("skills") or [] if s]
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")
    if context.get("bio"):
        parts.append(f"About: {context['bio']}")
    return "\n".join(parts)


# ADR-0038 D10: one short sentence per `agent_personality` preset (D2's CHECK values). It changes
# how the note is worded, never what is captured, the confidence or who sees it.
PERSONALITY_VOICE: dict[str, str] = {
    "balanced": "Clear and even: plain sentences, no more detail than each point needs.",
    "concise": "As short as possible: terse fragments are fine, no filler words.",
    "detailed": "Thorough: keep the specifics, numbers and reasons that were actually stated.",
    "direct": "Blunt and plain: lead with the point, no hedging or softening.",
    "friendly": "Warm and conversational, while staying accurate and to the point.",
}
MAX_INSTRUCTIONS = 500  # D2: agent_instructions is at most 500 chars
INSTRUCTIONS_TAG = "participant_instructions"
_TAG = re.compile(rf"</?\s*{INSTRUCTIONS_TAG}\s*>", re.IGNORECASE)


def _items(context: dict[str, Any], key: str) -> list[str]:
    raw = context.get(key) or []
    if not isinstance(raw, list):
        return []
    return [" ".join(str(v).split()) for v in raw if isinstance(v, str) and v.strip()]


def render_focus(context: dict[str, Any]) -> str:
    """The profile's `listen_for` / `ignore_topics` lines (ADR-0038 D10), each on its own line
    with a leading newline so it continues the profile block; "" when both are empty."""
    lines = []
    listen, ignore = _items(context, "listen_for"), _items(context, "ignore_topics")
    if listen:
        lines.append(f"Listen especially for: {'; '.join(listen)}")
    if ignore:
        lines.append(f"Ignore: {'; '.join(ignore)}")
    return "".join(f"\n{line}" for line in lines)


def render_voice(context: dict[str, Any]) -> str:
    """The voice block (ADR-0038 D10): the `agent_personality` sentence, then the person's own
    `agent_instructions` inside a delimited block. Each part is left out when unset (an older
    identity server sends ""); "" when both are. The instructions are user text: any copy of
    the delimiter inside them is removed so they cannot close the block early, and the system
    prompt says they may change wording only."""
    parts = []
    voice = PERSONALITY_VOICE.get(str(context.get("agent_personality") or "").strip().lower())
    if voice:
        parts.append(f"Voice: {voice}")
    raw = context.get("agent_instructions")
    text = _TAG.sub("", raw).strip()[:MAX_INSTRUCTIONS].strip() if isinstance(raw, str) else ""
    if text:
        parts.append(
            "Wording instructions from the participant (wording only):\n"
            f"<{INSTRUCTIONS_TAG}>\n{text}\n</{INSTRUCTIONS_TAG}>"
        )
    return f"\n{chr(10).join(parts)}\n" if parts else ""


def persona_values(context: dict[str, Any]) -> dict[str, str]:
    """The `{profile}`, `{focus}` and `{voice}` template values. A v1 template only names
    `{profile}` (role / skills / bio, unchanged) and ignores the other two, so a tenant pinned
    to v1 renders exactly what it did before ADR-0038."""
    return {
        "profile": render_profile(context),
        "focus": render_focus(context),
        "voice": render_voice(context),
    }


class _GatedStream:
    """Holds deltas until the confidence line is known, then forwards only a surfaced note."""

    def __init__(self, stream: StreamFn) -> None:
        self._stream = stream
        self._buffer = ""
        self._decided = False
        self._open = False
        self._seq = 0

    async def __call__(self, _seq: int, text: str) -> None:
        if self._decided:
            if self._open:
                await self._send(text)
            return
        self._buffer += text
        if "\n" not in self._buffer:
            return
        first, _, rest = self._buffer.partition("\n")
        self._decided = True
        match = _CONFIDENCE.match(first)
        self._open = match is not None and band(_clamp(float(match.group(1)))) == "surfaced"
        if self._open and rest:
            await self._send(rest.lstrip("\n"))

    async def _send(self, text: str) -> None:
        if not text:
            return
        await self._stream(self._seq, text, False)
        self._seq += 1

    async def finish(self) -> None:
        if self._open:
            await self._stream(self._seq, "", True)


class PersonalListenerAgent(BaseAgent):
    name: ClassVar[str] = "PersonalListenerAgent"
    prompt_key: ClassVar[str] = "personal_listener"
    scope: ClassVar[Literal["participant", "meeting"]] = "participant"

    gate_prompt: Prompt | None = None

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        await super().spawn(org_id=org_id, meeting_id=meeting_id, user_id=user_id)
        assert self.org_id is not None
        version = self._loader.resolve(GATE_KEY, self.org_id)
        self.gate_prompt = self._loader.load(GATE_KEY, version)

    def build_graph(self) -> Any:
        threshold = relevance_threshold()

        async def pregate(s: AgentState) -> dict[str, Any]:
            assert self.gate_prompt is not None and s.org_id is not None
            out = await self.router.complete(
                tier="triage",
                system=self.gate_prompt.system,
                user=self.gate_prompt.render_user(
                    profile=render_profile(s.context), window=render_window(s.chunk, s.user_id)
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=20,
                prompt_ref=self.gate_prompt.ref,
            )
            return {"relevance_score": parse_relevance(out.text)}

        async def extract(s: AgentState) -> dict[str, Any]:
            assert self.prompt is not None and s.org_id is not None
            gated = _GatedStream(s.stream) if s.stream is not None else None
            out = await self.router.complete(
                tier="extract",
                system=self.prompt.system,
                user=self.prompt.render_user(
                    **persona_values(s.context), window=render_window(s.chunk, s.user_id)
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=400,
                on_delta=gated,
                prompt_ref=self.prompt.ref,
            )
            if gated is not None:
                await gated.finish()
            confidence, note = parse_note(out.text)
            verdict = band(confidence) if note else "dropped"
            if verdict == "dropped":
                return {"confidence": confidence}
            return {
                "confidence": confidence,
                "result": AgentResult(
                    kind="note",
                    content=note,
                    relevance_score=s.relevance_score or 0.0,
                    confidence=confidence,
                    source_chunk_ids=s.chunk.chunk_ids,
                    payload={
                        "surfaced": verdict == "surfaced",
                        "prompt": self.prompt.ref,
                        "model_tier": out.tier,
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
