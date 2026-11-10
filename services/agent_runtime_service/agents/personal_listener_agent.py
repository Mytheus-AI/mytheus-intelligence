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
`personal_listener_gate@v2` sees the same "Listen especially for" / "Ignore" lines (not the
voice block): a window about a listen-for topic is not cut for being outside the role, and an
ignored topic scores low. Its threshold, the gating logic and the bands are unchanged.
`agent_name` is never put in a prompt.

ADR-0039 (`personal_listener@v3`, `personal_listener_gate@v3`):
- C1: the extraction's user message starts with the company block (description, up to 60 glossary
  terms, active goals; `agents/company.py`), from `context["org"]`, which the runtime adds.
- C2: both calls see the "Never record" line; the gate scores a window mainly about one of those
  topics 0, so no note is written (a model instruction, not a guarantee).
- C3: the extraction answers a `COMMITMENTS: [...]` line after the confidence line: the promises
  "You" made, `{content, due_hint}`, at most 5. They ride in the record's
  `payload.commitments` (core-api's consumer writes `commitments` rows, private to the record's
  user) only when there is at least one, so a record without promises is unchanged. The line is
  never streamed.
- C6: a participant whose attendance is `agent` (`context["attendance"] == "agent"`, set by the
  runtime) gets an "Attendance" line in the profile so the note is a catch-up; they never speak,
  so their commitments are always empty, whatever the model says.
With no company brain, a live participant and no promises, v3 renders exactly v2's user messages.
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
from libs.shared.router import EXTRACT_MAX_TOKENS, EXTRACTION_EFFORT
from libs.shared.state import AgentResult, AgentState, StreamFn, TranscriptWindow
from services.agent_runtime_service.agents.company import company_values, gate_ground_rules
from services.agent_runtime_service.guidance import guidance_values
from services.agent_runtime_service.names import speaker_labels
from services.agent_runtime_service.person_memory import memory_focus, memory_values
from services.agent_runtime_service.policy import DEFAULT as DEFAULT_THRESHOLDS
from services.agent_runtime_service.policy import (
    Thresholds,
    thresholds_for,
)

GATE_KEY = "personal_listener_gate"
_NUMBER = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")
_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
COMMITMENTS_PREFIX = "COMMITMENTS:"
MAX_COMMITMENTS = 5  # ADR-0039 C3
MAX_COMMITMENT = 500  # B6: content up to 500
MAX_DUE_HINT = 100  # B6: due_hint up to 100


DEFAULT_NOTE_MIN_SPAN_MS = 30_000


def note_min_span_ms() -> int:
    """Polish (calm): notes run on at least this much meeting time (`NOTE_MIN_SPAN_MS`, default
    30 s), collected from the event-driven windows, or sooner when a window carries a cue (a
    promise is time-sensitive). Fast windows are for the cards; a note per 4 s is noise."""
    try:
        return max(0, int(os.environ.get("NOTE_MIN_SPAN_MS", DEFAULT_NOTE_MIN_SPAN_MS)))
    except ValueError:
        return DEFAULT_NOTE_MIN_SPAN_MS


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
    confidence, note, _ = parse_record(text)
    return confidence, note


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def parse_commitments(line: str) -> list[dict[str, Any]]:
    """The JSON list after `COMMITMENTS:`; malformed entries are dropped one by one, never
    raises. Each kept entry is exactly `{"content": str, "due_hint": str | None}`."""
    try:
        raw = json.loads(line.strip()[len(COMMITMENTS_PREFIX) :].strip() or "[]")
    except ValueError:
        return []
    out: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        due = item.get("due_hint")
        entry = {
            "content": _clip(content, MAX_COMMITMENT),
            "due_hint": _clip(due, MAX_DUE_HINT) if isinstance(due, str) and due.strip() else None,
        }
        if entry not in out:
            out.append(entry)
    return out[:MAX_COMMITMENTS]


def parse_record(text: str) -> tuple[float, str, list[dict[str, Any]]]:
    """`CONFIDENCE: x`, an optional `COMMITMENTS: [...]` line (v3), then the note. A missing or
    malformed confidence header is confidence 0 and nothing else."""
    first, _, rest = text.strip().partition("\n")
    match = _CONFIDENCE.match(first)
    if match is None:
        return 0.0, "", []
    commitments: list[dict[str, Any]] = []
    body = rest.strip()
    second, _, after = body.partition("\n")
    if second.strip().upper().startswith(COMMITMENTS_PREFIX):
        commitments = parse_commitments(second.strip())
        body = after.strip()
    note = "" if body.upper() == "NONE" else body
    return _clamp(float(match.group(1))), note, commitments


def band(
    confidence: float, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> Literal["surfaced", "silent", "dropped"]:
    """The doc 08 band, with the tenant's thresholds when given (`policy.py`, ADR-0030 D3)."""
    return thresholds.band(confidence)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def render_window(window: TranscriptWindow, user_id: UUID) -> str:
    """The participant is "You"; everyone else keeps their meeting-wide "Participant N" label
    (`names.speaker_labels`; numbered in order of first speaking for a window built outside the
    runtime)."""
    me = str(user_id)
    labels = speaker_labels(window, exclude=me)
    return "\n".join(
        f"{'You' if t.speaker_id == me else labels[t.speaker_id]}: {t.text}" for t in window.turns
    )


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


def render_attendance(context: dict[str, Any]) -> str:
    """ADR-0039 C6: the profile's attendance line for a participant who sent their agent; ""
    for everyone else (a live participant renders exactly what v2 did)."""
    if context.get("attendance") != "agent":
        return ""
    return "\nAttendance: not in this meeting; they sent their agent instead"


def gate_values(context: dict[str, Any]) -> dict[str, str]:
    """The pre-gate's `{profile}` and `{focus}` values: the working profile with its "Listen
    especially for" / "Ignore" lines, never the voice block (wording is not relevance). A v1 gate
    template ignores `{focus}`; with both lists empty v2 renders exactly what v1 does. From v3
    (ADR-0039) also `{ground_rules}` (the never-record line) and `{attendance}`, both "" by
    default; older templates ignore them."""
    return {
        "profile": render_profile(context),
        "focus": render_focus(context),
        "attendance": render_attendance(context),
        **gate_ground_rules(context),
        **memory_focus(context),  # polish wave 2 (`personal_listener_gate@v5`); "" if none
    }


def persona_values(context: dict[str, Any]) -> dict[str, str]:
    """The `{profile}`, `{focus}` and `{voice}` template values. A v1 template only names
    `{profile}` (role / skills / bio, unchanged) and ignores the other two, so a tenant pinned
    to v1 renders exactly what it did before ADR-0038. From v3 (ADR-0039) also `{company}`,
    `{ground_rules}` and `{attendance}`, each "" for an org with no brain and a live participant."""
    return {
        "profile": render_profile(context),
        "focus": render_focus(context),
        "voice": render_voice(context),
        "attendance": render_attendance(context),
        **company_values(context),
    }


class _GatedStream:
    """Holds deltas until the confidence line is known, then forwards only a surfaced note. The
    v3 `COMMITMENTS: [...]` line right after it is never forwarded: it is data for core-api, not
    part of the note the participant reads live."""

    def __init__(self, stream: StreamFn, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> None:
        self._stream = stream
        self._thresholds = thresholds
        self._buffer = ""
        self._decided = False
        self._open = False
        self._seq = 0
        self._pending = ""  # after the confidence line, until the commitments line is ruled out
        self._header_done = False

    async def __call__(self, _seq: int, text: str) -> None:
        if self._decided:
            if self._open:
                await self._forward(text)
            return
        self._buffer += text
        if "\n" not in self._buffer:
            return
        first, _, rest = self._buffer.partition("\n")
        self._decided = True
        match = _CONFIDENCE.match(first)
        self._open = (
            match is not None
            and band(_clamp(float(match.group(1))), self._thresholds) == "surfaced"
        )
        if self._open and rest:
            await self._forward(rest)

    async def _forward(self, text: str) -> None:
        if self._header_done:
            await self._send(text)
            return
        self._pending += text
        head = self._pending.lstrip("\n")
        if not head:
            return
        upper = head.upper()
        if len(head) < len(COMMITMENTS_PREFIX) and COMMITMENTS_PREFIX.startswith(upper):
            return  # could still become the commitments line
        if upper.startswith(COMMITMENTS_PREFIX):
            if "\n" not in head:
                return
            head = head.partition("\n")[2]
        self._header_done = True
        self._pending = ""
        await self._send(head.lstrip("\n"))

    async def _send(self, text: str) -> None:
        if not text:
            return
        await self._stream(self._seq, text, False)
        self._seq += 1

    async def finish(self) -> None:
        if not self._open:
            return
        head = self._pending.lstrip("\n")
        if not self._header_done and head and not head.upper().startswith(COMMITMENTS_PREFIX):
            if not COMMITMENTS_PREFIX.startswith(head.upper()):
                await self._send(head)
        await self._stream(self._seq, "", True)


class PersonalListenerAgent(BaseAgent):
    name: ClassVar[str] = "PersonalListenerAgent"
    prompt_key: ClassVar[str] = "personal_listener"
    scope: ClassVar[Literal["participant", "meeting"]] = "participant"
    serves_proxies: ClassVar[bool] = True  # ADR-0039 C6: runs for attendance `agent` too

    @property
    def min_span_ms(self) -> int:
        return note_min_span_ms()

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
                **self.gate_prompt.user_parts(
                    **gate_values(s.context), window=render_window(s.chunk, s.user_id)
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=20,
                prompt_ref=self.gate_prompt.ref,
            )
            return {"relevance_score": parse_relevance(out.text)}

        async def extract(s: AgentState) -> dict[str, Any]:
            assert self.prompt is not None and s.org_id is not None
            limits = thresholds_for(s.context, self.name)
            gated = _GatedStream(s.stream, limits) if s.stream is not None else None
            out = await self.router.complete(
                tier="extract",
                system=self.prompt.system,
                **self.prompt.user_parts(
                    **persona_values(s.context),
                    # polish wave 2: memory across meetings and the person's own note ratings
                    **memory_values(s.context),
                    **guidance_values(self.name, s.context),
                    window=render_window(s.chunk, s.user_id),
                ),
                org_id=s.org_id,
                user_id=s.user_id,
                max_tokens=EXTRACT_MAX_TOKENS,
                effort=EXTRACTION_EFFORT,
                on_delta=gated,
                prompt_ref=self.prompt.ref,
            )
            if gated is not None:
                await gated.finish()
            confidence, note, commitments = parse_record(out.text)
            verdict = band(confidence, limits) if note else "dropped"
            if verdict == "dropped":
                return {"confidence": confidence}
            payload: dict[str, Any] = {
                "surfaced": verdict == "surfaced",
                "prompt": self.prompt.ref,
                "model_tier": out.tier,
            }
            # C3: only promises "You" made; a proxied participant never speaks (C6)
            if commitments and s.context.get("attendance") != "agent":
                payload["commitments"] = commitments
            return {
                "confidence": confidence,
                "result": AgentResult(
                    kind="note",
                    content=note,
                    relevance_score=s.relevance_score or 0.0,
                    confidence=confidence,
                    source_chunk_ids=s.chunk.chunk_ids,
                    payload=payload,
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
