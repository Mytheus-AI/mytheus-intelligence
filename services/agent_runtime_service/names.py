"""Stable speaker labels per meeting, display names after the model answers, and the
never-record filter on everything that leaves the agent tier.

Labels. The models never see a user id, a name or an email (the design since P7): speakers are
"Participant N". Before this module each renderer numbered speakers per WINDOW, so the same
person was "Participant 2" in one window and "Participant 1" in the next, and nothing mapped a
label back to a person: notes read "Participant 2 asked...". Now the runtime keeps one registry
per meeting (`assign_labels`): a speaker gets the next number on first appearance and keeps it
for the whole meeting, and every renderer uses it (`speaker_labels`). A window built outside the
runtime (tests, evals) has no registry and is numbered in order of first speaking, exactly as
before.

Names. The models still see labels only. After a model answers, `Postprocess.result` replaces
every "Participant N" in every user-visible text of the result (note, candidates, suggestions,
fill, alert, agenda update) with the person's display name, from the user context the runtime
already fetched (`display_name`), and leaves the label when there is no name. Streamed deltas
(PersonalListener) go through the same substitution line by line (`Postprocess.stream`).

Never record. `Postprocess` also applies the org's never-record policy
(`libs/shared/never_record.py`) to the same texts: a note line, a candidate, a suggestion, a
fill, an alert or an agenda note that names a never-record topic (or a glossary alias of it)
never leaves the tier, live or durable. The prompts tell the model the rule; this makes the
cases a machine can see a guarantee.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.state import AgentResult, StreamFn, TranscriptWindow

LABEL = "Participant"
_LABEL = re.compile(r"\bParticipant\s*#?\s*(\d+)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(\d+)\s*$")
# payload keys that are data, never shown as text
_DATA_KEYS = frozenset(
    {"id", "prompt", "model_tier", "kind", "citations", "window_id", "status", "type", "hop"}
)


def label(n: int) -> str:
    return f"{LABEL} {n}"


def assign_labels(registry: dict[str, str], window: TranscriptWindow) -> None:
    """Give each speaker of `window` met for the first time in the meeting the next label, then
    hand the window the meeting's labels. `registry` lives as long as the meeting."""
    for turn in window.turns:
        if turn.speaker_id not in registry:
            registry[turn.speaker_id] = label(len(registry) + 1)
    window.labels = dict(registry)


def speaker_labels(window: TranscriptWindow, exclude: str | None = None) -> dict[str, str]:
    """speaker id -> label for the speakers of this window (minus `exclude`), in order of first
    speaking: the meeting's stable label when the runtime assigned one, else the next number
    after every label already in use (so a fallback can never collide with a stable one)."""
    used = [int(m.group(1)) for v in window.labels.values() if (m := _NUMBER.search(v))]
    n = max(used, default=0)
    out: dict[str, str] = {}
    for turn in window.turns:
        sid = turn.speaker_id
        if sid == exclude or sid in out:
            continue
        stable = window.labels.get(sid)
        if stable is None:
            n += 1
            stable = label(n)
        out[sid] = stable
    return out


def display_name(context: Mapping[str, Any] | None) -> str | None:
    """A person's display name from their user context: an explicit name when the identity
    server sends one, else derived from the email's local part ("priya.sharma@acme.com" ->
    "Priya Sharma"). `users` has no name column today, so the email is the source. None when
    neither is there: the label stays."""
    if not context:
        return None
    for key in ("display_name", "name", "full_name"):
        raw = context.get(key)
        if isinstance(raw, str) and raw.strip():
            return " ".join(raw.split())[:80]
    email = context.get("email")
    if not isinstance(email, str) or "@" not in email:
        return None
    local = email.strip().split("@", 1)[0]
    words = [re.sub(r"\d+$", "", w) for w in re.split(r"[._+\-]+", local)]
    words = [w for w in words if w and not w.isdigit()]
    if not words:
        return None
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)[:80]


class Namer:
    """ "Participant N" -> display name, for one meeting at one moment."""

    def __init__(self, names: Mapping[str, str]) -> None:
        self._names = {k.lower(): v for k, v in names.items() if v}

    @classmethod
    def build(
        cls, labels: Mapping[str, str], contexts: Mapping[UUID, Mapping[str, Any] | None]
    ) -> Namer:
        names: dict[str, str] = {}
        for uid, ctx in contexts.items():
            lab = labels.get(str(uid))
            name = display_name(ctx)
            if lab and name:
                names[lab] = name
        return cls(names)

    def text(self, value: str) -> str:
        if not self._names or not value:
            return value

        def swap(m: re.Match[str]) -> str:
            return self._names.get(label(int(m.group(1))).lower(), m.group(0))

        return _LABEL.sub(swap, value)

    def value(self, raw: Any, key: str = "") -> Any:
        """Every text inside `raw`, recursively, except data keys (ids, refs, enums)."""
        if key in _DATA_KEYS or key.endswith("_id") or key.endswith("_ids"):
            return raw
        if isinstance(raw, str):
            return self.text(raw)
        if isinstance(raw, list):
            return [self.value(v) for v in raw]
        if isinstance(raw, dict):
            return {k: self.value(v, str(k)) for k, v in raw.items()}
        return raw


def _texts(raw: Any, key: str = "") -> list[str]:
    if key in _DATA_KEYS or key.endswith("_id") or key.endswith("_ids"):
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [t for v in raw for t in _texts(v)]
    if isinstance(raw, dict):
        return [t for k, v in raw.items() for t in _texts(v, str(k))]
    return []


class LineStream:
    """Forwards a delta stream one complete line at a time, each through `transform` (None
    drops the line). Re-sequences what it sends; `done` flushes the last partial line."""

    def __init__(self, stream: StreamFn, transform: Callable[[str], str | None]) -> None:
        self._stream = stream
        self._transform = transform
        self._buffer = ""
        self._seq = 0

    async def __call__(self, _seq: int, text: str, done: bool) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            await self._emit(line, "\n")
        if done:
            if self._buffer:
                await self._emit(self._buffer, "")
                self._buffer = ""
            await self._stream(self._seq, "", True)
            self._seq += 1

    async def _emit(self, line: str, end: str) -> None:
        out = self._transform(line)
        if out is None:
            return
        await self._stream(self._seq, out + end, False)
        self._seq += 1


class Postprocess:
    """Names and the never-record filter for one window's results."""

    def __init__(self, namer: Namer, policy: NeverRecordPolicy) -> None:
        self.namer = namer
        self.policy = policy

    def line(self, text: str) -> str | None:
        if self.policy.hits(text):
            return None
        return self.namer.text(text)

    def stream(self, stream: StreamFn) -> StreamFn:
        return LineStream(stream, self.line)

    def result(self, result: AgentResult) -> AgentResult | None:
        """The result as it may leave the tier, or None when nothing of it may."""
        payload = copy.deepcopy(result.payload)
        handler = _HANDLERS.get(result.kind, _generic)
        out = handler(self, result, payload)
        if out is None:
            return None
        content, payload, confidence = out
        return dataclasses.replace(
            result,
            content=self.namer.text(content),
            confidence=confidence,
            payload=self.namer.value(payload),
        )


Handled = tuple[str, dict[str, Any], float] | None  # (content, payload, confidence)


def _note(p: Postprocess, r: AgentResult, payload: dict[str, Any]) -> Handled:
    lines = [ln for ln in r.content.split("\n") if not p.policy.hits(ln)]
    content = "\n".join(lines).strip()
    if not content:
        return None
    if p.policy.hits(payload.get("quote"), payload.get("why")):
        # the note's moment stays (an id and a time); words naming the topic never leave
        payload.pop("quote", None)
        payload.pop("why", None)
    if isinstance(payload.get("commitments"), list):
        kept = [c for c in payload["commitments"] if not p.policy.hits(*_texts(c))]
        if kept:
            payload["commitments"] = kept
        else:
            payload.pop("commitments")
    return content, payload, r.confidence


def _items(
    key: str, text_key: str
) -> Callable[[Postprocess, AgentResult, dict[str, Any]], Handled]:
    def handle(p: Postprocess, r: AgentResult, payload: dict[str, Any]) -> Handled:
        items = payload.get(key)
        if not isinstance(items, list):
            return _generic(p, r, payload)
        kept = [i for i in items if isinstance(i, dict) and not p.policy.hits(*_texts(i))]
        if not kept:
            return None
        payload[key] = kept
        if len(kept) == len(items):
            return r.content, payload, r.confidence
        top = max(kept, key=lambda i: float(i.get("confidence") or 0.0))
        return str(top.get(text_key) or ""), payload, float(top.get("confidence") or 0.0)

    return handle


def _fill(p: Postprocess, r: AgentResult, payload: dict[str, Any]) -> Handled:
    gap = payload.get("gap")
    if p.policy.hits(payload.get("query"), gap.get("query") if isinstance(gap, dict) else None):
        return None  # the reference itself is never-record: no fill, no audit text
    fill = payload.get("fill")
    if isinstance(fill, dict) and p.policy.hits(*_texts(fill)):
        payload["fill"] = None
        return "", payload, 0.0
    return r.content, payload, r.confidence


def _meeting_update(p: Postprocess, r: AgentResult, payload: dict[str, Any]) -> Handled:
    for u in payload.get("updates") or []:
        if isinstance(u, dict) and p.policy.hits(u.get("note")):
            u["note"] = ""
    nudge = payload.get("nudge")
    if isinstance(nudge, dict) and p.policy.hits(*_texts(nudge)):
        payload["nudge"] = None
    if not payload.get("updates") and not payload.get("nudge"):
        return None
    content = r.content
    if p.policy.hits(content):
        content = "Agenda updated"
    return content, payload, r.confidence


def _generic(p: Postprocess, r: AgentResult, payload: dict[str, Any]) -> Handled:
    if p.policy.hits(r.content, *_texts(payload)):
        return None
    return r.content, payload, r.confidence


_HANDLERS: dict[str, Callable[[Postprocess, AgentResult, dict[str, Any]], Handled]] = {
    "note": _note,
    "decision_candidate": _items("candidates", "content"),
    "action_candidate": _items("candidates", "content"),
    "suggestion": _items("suggestions", "question"),
    "context_fill": _fill,
    "meeting_update": _meeting_update,
    "alert": _generic,
}
