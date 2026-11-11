"""Stage 1 of the cost pipeline: merge `transcript-chunks` finals into event-driven windows.

Polish program, "Make the meeting calm and fast". The fixed 30 s buckets (`start_ms // 30000`)
were the main source of lateness: a card could not appear before its bucket ended, and the
bucket only closed when a final for the NEXT bucket arrived. A window now closes on what is
said:

- `cue`: a final holding a cue phrase ("I'll", "let's", "we decided", "action item", "can you",
  "by Friday", ...; case-insensitive, `WINDOW_CUES` replaces the list) closes the window right
  after that final, so a commitment or a decision reaches the agents as soon as it is spoken;
- `turn_end`: the speaker changes (a final from someone else arrives), or the speaker goes quiet
  for `turn_gap_s` of wall-clock time, once the window holds at least `min_ms` of speech;
- `max_span`: a final that would stretch the window past `max_ms` closes it first;
- `max_wait`: a window open for `max_wait_s` of wall-clock time closes whatever it holds (a
  short remark followed by silence still reaches the agents);
- `meeting_end`: `close_meeting`.

A meeting never closes windows closer together than `min_gap_ms` of meeting time (`cue` and
`turn_end`; `max_span` and the wall-clock rules are bounded already): a cue arriving too soon
after the last close leaves the window open and it closes on the next rule that fires. Defaults
`min_ms` 10 s, `max_ms` 15 s, `max_wait_s` 15 s, `min_gap_ms` 4 s, `turn_gap_s` 1.5 s
(env `WINDOW_MIN_MS`, `WINDOW_MAX_MS`, `WINDOW_MAX_WAIT_S`, `WINDOW_MIN_GAP_MS`,
`WINDOW_TURN_GAP_S`).

Ids. A window's id is `{meeting_id}:{start_ms}` with `start_ms` the start of its first final,
so the id carries the moment it covers (the UI points at it). Every rule except the two
wall-clock ones depends only on the finals themselves, so a replay after a crash rebuilds the
same windows with the same ids; a window closed by a wall-clock rule may be cut differently on
replay, and the in-meeting ledger (`memory.py`) is what keeps that replay from proposing the
same card twice. The window carries its `trigger` (the reason and the words that closed it) and
`last_chunk_at`, the wall-clock time its last final was produced (speech-to-card latency).

Unchanged: consecutive finals from one speaker merge into one turn (each final is kept as an
`Utterance`, the unit a card's quote points at); a redelivered chunk (same `chunk_id`) is
ignored; a late final for a closed window joins the open one. All chunks of a meeting arrive on
one partition (`{org_id}:{meeting_id}`, ADR-0017), so one runtime process sees a whole meeting
and keeps its open window in memory. Each window remembers the Kafka offsets it holds; the
consumer commits an offset only once every window holding it has been processed.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from uuid import UUID

from libs.shared.state import TranscriptWindow, Trigger, Turn, Utterance

DEFAULT_MIN_MS = 10_000
DEFAULT_MAX_MS = 15_000
DEFAULT_MAX_WAIT_S = 15.0
DEFAULT_MIN_GAP_MS = 4_000
DEFAULT_TURN_GAP_S = 1.5

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
DEFAULT_CUES: tuple[str, ...] = (
    "i'll",
    "i will",
    "i'm going to",
    "we'll",
    "we will",
    "let's",
    "let us",
    "we decided",
    "we've decided",
    "decided",
    "decision is",
    "agreed",
    "we agree",
    "action item",
    "can you",
    "could you",
    "will you",
    "follow up",
    "owner",
    "deadline",
    "by tomorrow",
    "by end of day",
    "by eod",
    "by end of week",
    "by next week",
    *(f"by {d}" for d in _WEEKDAYS),
)


def cue_list(raw: str | None = None) -> tuple[str, ...]:
    """`WINDOW_CUES` (separated by `|` or `,`) or the default list, lowercased."""
    value = os.environ.get("WINDOW_CUES", "") if raw is None else raw
    items = tuple(c.strip().lower() for c in re.split(r"[|,]", value) if c.strip())
    return items or DEFAULT_CUES


def cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive; a straight or curly apostrophe matches either."""
    parts = []
    for cue in sorted(set(cues), key=len, reverse=True):
        escaped = re.escape(cue).replace("'", "['’]")
        parts.append(rf"(?<![\w']){escaped}(?![\w'])")
    return re.compile("|".join(parts), re.IGNORECASE)


@dataclass(frozen=True)
class WindowSettings:
    min_ms: int = DEFAULT_MIN_MS
    max_ms: int = DEFAULT_MAX_MS
    max_wait_s: float = DEFAULT_MAX_WAIT_S
    min_gap_ms: int = DEFAULT_MIN_GAP_MS
    turn_gap_s: float = DEFAULT_TURN_GAP_S
    cues: tuple[str, ...] = DEFAULT_CUES

    @classmethod
    def from_env(cls) -> WindowSettings:
        def num(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except ValueError:
                return default

        return cls(
            min_ms=int(num("WINDOW_MIN_MS", DEFAULT_MIN_MS)),
            max_ms=int(num("WINDOW_MAX_MS", DEFAULT_MAX_MS)),
            max_wait_s=num("WINDOW_MAX_WAIT_S", DEFAULT_MAX_WAIT_S),
            min_gap_ms=int(num("WINDOW_MIN_GAP_MS", DEFAULT_MIN_GAP_MS)),
            turn_gap_s=num("WINDOW_TURN_GAP_S", DEFAULT_TURN_GAP_S),
            cues=cue_list(),
        )


@dataclass
class Chunk:
    org_id: UUID
    meeting_id: UUID
    chunk_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    partition: int = 0
    offset: int = -1
    produced_at: float | None = None  # epoch seconds: the envelope's occurred_at, if known


@dataclass
class _Open:
    window: TranscriptWindow
    seen: set[str] = field(default_factory=set)
    offsets: list[tuple[int, int]] = field(default_factory=list)
    opened_wall: float = 0.0
    last_wall: float = 0.0
    last: Utterance | None = None
    cue: tuple[Utterance, str] | None = None  # the last final that held a cue, and the cue


@dataclass
class ClosedWindow:
    window: TranscriptWindow
    offsets: list[tuple[int, int]]  # (partition, offset) pairs this window held


@dataclass
class _Meeting:
    open: _Open | None = None
    last_close_ms: int | None = None  # meeting time the previous window ended


class Windower:
    def __init__(
        self,
        settings: WindowSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings or WindowSettings()
        self._cue = cue_pattern(self.settings.cues)
        self._meetings: dict[tuple[UUID, UUID], _Meeting] = {}
        self._clock = clock
        self._wall = wall

    def cue_in(self, text: str) -> str | None:
        match = self._cue.search(text)
        return match.group(0).lower().replace("’", "'") if match else None

    def add(self, chunk: Chunk) -> list[ClosedWindow]:
        """Add one final. Returns the windows this closed (zero, one or two)."""
        key = (chunk.org_id, chunk.meeting_id)
        m = self._meetings.setdefault(key, _Meeting())
        closed: list[ClosedWindow] = []
        current = m.open
        if current is not None and chunk.chunk_id in current.seen:
            current.offsets.append((chunk.partition, chunk.offset))
            return closed  # a redelivery: held, so its offset commits with the window
        if current is not None and current.window.turns:
            span_after = max(current.window.end_ms, chunk.end_ms) - current.window.start_ms
            last_speaker = current.window.turns[-1].speaker_id
            if span_after > self.settings.max_ms:
                closed.append(self._close(key, "max_span"))
            elif (
                chunk.speaker_id != last_speaker
                and self._span(current) >= self.settings.min_ms
                and self._gap_ok(m, current.window.end_ms)
            ):
                closed.append(self._close(key, "turn_end"))
        current = self._open_for(key, m, chunk)
        current.offsets.append((chunk.partition, chunk.offset))
        now = self._clock()
        current.last_wall = now
        current.seen.add(chunk.chunk_id)
        utterance = Utterance(
            chunk.chunk_id, chunk.speaker_id, chunk.start_ms, chunk.end_ms, chunk.text
        )
        self._append(current.window, utterance)
        current.last = utterance
        produced = chunk.produced_at if chunk.produced_at else self._wall()
        current.window.last_chunk_at = max(current.window.last_chunk_at or 0.0, produced)
        cue = self.cue_in(chunk.text)
        if cue is not None:
            current.cue = (utterance, cue)
            if self._gap_ok(m, current.window.end_ms):
                closed.append(self._close(key, "cue"))
        return closed

    def expire_idle(self) -> list[ClosedWindow]:
        """The wall-clock rules, called on every tick: a speaker who went quiet ends their turn
        (once the window is long enough, or holds a cue that waited for the minimum gap), and
        a window open for `max_wait_s` closes whatever it holds."""
        now = self._clock()
        out: list[ClosedWindow] = []
        for key, m in list(self._meetings.items()):
            w = m.open
            if w is None or not w.window.turns:
                continue
            quiet = now - w.last_wall >= self.settings.turn_gap_s
            if now - w.opened_wall >= self.settings.max_wait_s:
                out.append(self._close(key, "max_wait"))
            elif quiet and (self._span(w) >= self.settings.min_ms or w.cue is not None):
                out.append(self._close(key, "cue" if w.cue is not None else "turn_end"))
        return out

    def close_meeting(self, org_id: UUID, meeting_id: UUID) -> list[ClosedWindow]:
        key = (org_id, meeting_id)
        m = self._meetings.pop(key, None)
        if m is None or m.open is None or not m.open.window.turns:
            return []
        self._meetings[key] = m
        out = self._close(key, "meeting_end")
        self._meetings.pop(key, None)
        out.window.final = True
        return [out]

    def open_offsets(self) -> list[tuple[int, int]]:
        return [o for m in self._meetings.values() if m.open for o in m.open.offsets]

    # --- internals -----------------------------------------------------------------------

    @staticmethod
    def _span(w: _Open) -> int:
        return w.window.end_ms - w.window.start_ms

    def _gap_ok(self, m: _Meeting, end_ms: int) -> bool:
        return m.last_close_ms is None or end_ms - m.last_close_ms >= self.settings.min_gap_ms

    def _open_for(self, key: tuple[UUID, UUID], m: _Meeting, chunk: Chunk) -> _Open:
        if m.open is None:
            now = self._clock()
            start = chunk.start_ms
            m.open = _Open(
                window=TranscriptWindow(
                    org_id=chunk.org_id,
                    meeting_id=chunk.meeting_id,
                    window_id=f"{chunk.meeting_id}:{start}",
                    start_ms=start,
                    end_ms=chunk.end_ms,
                ),
                opened_wall=now,
                last_wall=now,
            )
        return m.open

    @staticmethod
    def _append(window: TranscriptWindow, u: Utterance) -> None:
        turns = window.turns
        if turns and turns[-1].speaker_id == u.speaker_id:
            last = turns[-1]
            last.text = f"{last.text} {u.text}".strip()
            last.end_ms = max(last.end_ms, u.end_ms)
            last.chunk_ids.append(u.chunk_id)
            last.parts.append(u)
        else:
            turns.append(
                Turn(
                    speaker_id=u.speaker_id,
                    text=u.text,
                    start_ms=u.start_ms,
                    end_ms=u.end_ms,
                    chunk_ids=[u.chunk_id],
                    parts=[u],
                )
            )
        # a late final (earlier than the window's start) widens it back; the id keeps the start
        window.start_ms = min(window.start_ms, u.start_ms)
        window.end_ms = max(window.end_ms, u.end_ms)

    def _close(self, key: tuple[UUID, UUID], reason: str) -> ClosedWindow:
        m = self._meetings[key]
        w = m.open
        assert w is not None
        m.open = None
        m.last_close_ms = w.window.end_ms
        # the triggering quote: the last final that held a cue, else the window's last final
        anchor, cue = w.cue if w.cue is not None else (w.last, None)
        if anchor is not None:
            w.window.trigger = Trigger(
                reason=reason,
                chunk_id=anchor.chunk_id,
                speaker_id=anchor.speaker_id,
                start_ms=anchor.start_ms,
                end_ms=anchor.end_ms,
                text=anchor.text,
                cue=cue,
            )
        else:
            w.window.trigger = Trigger(reason=reason)
        return ClosedWindow(window=w.window, offsets=w.offsets)


def merge_transcripts(windows: list[TranscriptWindow]) -> TranscriptWindow:
    """Several windows of one meeting as one: the first window's id and start, every utterance
    in order (a speaker continuing across a seam stays one turn), the latest labels, the last
    trigger (reason `merged` unless it was the meeting's end), `final` if any was."""
    if len(windows) == 1:
        return windows[0]
    first, last = windows[0], windows[-1]
    merged = TranscriptWindow(
        org_id=first.org_id,
        meeting_id=first.meeting_id,
        window_id=first.window_id,
        start_ms=first.start_ms,
        end_ms=first.end_ms,
        labels=dict(last.labels or first.labels),
    )
    for w in windows:
        for turn in w.turns:
            for u in turn.utterances():
                Windower._append(merged, u)
        merged.last_chunk_at = max(merged.last_chunk_at or 0.0, w.last_chunk_at or 0.0) or None
    merged.final = any(w.final for w in windows)
    trig = last.trigger
    if trig is not None:
        merged.trigger = Trigger(
            reason="meeting_end" if trig.reason == "meeting_end" else "merged",
            chunk_id=trig.chunk_id,
            speaker_id=trig.speaker_id,
            start_ms=trig.start_ms,
            end_ms=trig.end_ms,
            text=trig.text,
            cue=trig.cue,
        )
    return merged


def merge_windows(windows: list[ClosedWindow]) -> ClosedWindow:
    """Queued windows of one meeting as one (a worker that fell behind catches up in one pass
    instead of paying for every window late), holding every offset."""
    if len(windows) == 1:
        return windows[0]
    return ClosedWindow(
        window=merge_transcripts([cw.window for cw in windows]),
        offsets=[o for cw in windows for o in cw.offsets],
    )
