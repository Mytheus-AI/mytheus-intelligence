"""Stage 1 of the cost pipeline: merge `transcript-chunks` finals into 30 s windows (`A17`).

Plan 7-9 section 3.2 step 5, ADR-0004 (window first, then gate). Windows are fixed 30 s
buckets of meeting time, `start_ms // 30000`, so a window's id is deterministic,
`{meeting_id}:{bucket_start_ms}`, and a replay after a crash rebuilds the same window with
the same id (which is what makes `agent-results` idempotent downstream).

A window closes when:
- a final arrives for a later bucket of the same meeting (the normal case);
- the meeting has been quiet for `idle_s` of wall-clock time (the last window of a meeting,
  or a long pause);
- the meeting ends (`close_meeting`).

A late final for an already-closed bucket joins the current open window instead of reopening
the old one. Consecutive finals from the same speaker merge into one turn. A redelivered
chunk (same `chunk_id`) is ignored.

All chunks of a meeting arrive on one partition (`transcript-chunks` is keyed
`{org_id}:{meeting_id}`, ADR-0017), so one runtime process sees a whole meeting and can keep
the open window in memory. Each window remembers the Kafka offsets it holds; the consumer
commits an offset only once every window holding it has been processed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from libs.shared.state import TranscriptWindow, Turn

WINDOW_MS = 30_000


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


@dataclass
class _Open:
    window: TranscriptWindow
    bucket: int
    seen: set[str] = field(default_factory=set)
    offsets: list[tuple[int, int]] = field(default_factory=list)
    last_wall: float = 0.0


@dataclass
class ClosedWindow:
    window: TranscriptWindow
    offsets: list[tuple[int, int]]  # (partition, offset) pairs this window held


class Windower:
    def __init__(self, idle_s: float = 15.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._open: dict[tuple[UUID, UUID], _Open] = {}
        self._idle_s = idle_s
        self._clock = clock

    def add(self, chunk: Chunk) -> list[ClosedWindow]:
        """Add one final. Returns the windows this closed (zero or one)."""
        key = (chunk.org_id, chunk.meeting_id)
        bucket = chunk.start_ms // WINDOW_MS
        closed: list[ClosedWindow] = []
        current = self._open.get(key)
        if current is not None and bucket > current.bucket:
            closed.append(self._close(key))
            current = None
        if current is None:
            start = bucket * WINDOW_MS
            current = _Open(
                window=TranscriptWindow(
                    org_id=chunk.org_id,
                    meeting_id=chunk.meeting_id,
                    window_id=f"{chunk.meeting_id}:{start}",
                    start_ms=start,
                    end_ms=start + WINDOW_MS,
                ),
                bucket=bucket,
            )
            self._open[key] = current
        current.offsets.append((chunk.partition, chunk.offset))
        current.last_wall = self._clock()
        if chunk.chunk_id in current.seen:
            return closed
        current.seen.add(chunk.chunk_id)
        turns = current.window.turns
        if turns and turns[-1].speaker_id == chunk.speaker_id:
            last = turns[-1]
            last.text = f"{last.text} {chunk.text}".strip()
            last.end_ms = max(last.end_ms, chunk.end_ms)
            last.chunk_ids.append(chunk.chunk_id)
        else:
            turns.append(
                Turn(
                    speaker_id=chunk.speaker_id,
                    text=chunk.text,
                    start_ms=chunk.start_ms,
                    end_ms=chunk.end_ms,
                    chunk_ids=[chunk.chunk_id],
                )
            )
        return closed

    def expire_idle(self) -> list[ClosedWindow]:
        now = self._clock()
        stale = [k for k, w in self._open.items() if now - w.last_wall >= self._idle_s]
        return [self._close(k) for k in stale]

    def close_meeting(self, org_id: UUID, meeting_id: UUID) -> list[ClosedWindow]:
        key = (org_id, meeting_id)
        return [self._close(key)] if key in self._open else []

    def open_offsets(self) -> list[tuple[int, int]]:
        return [o for w in self._open.values() for o in w.offsets]

    def _close(self, key: tuple[UUID, UUID]) -> ClosedWindow:
        w = self._open.pop(key)
        return ClosedWindow(window=w.window, offsets=w.offsets)
