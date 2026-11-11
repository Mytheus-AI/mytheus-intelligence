"""The room signals' deterministic half: the source core-api sends after a meeting ends, and the
rules that need no model (a3 "the last word", "fake-consensus detector", "pick up where we
stopped"). `room_signals.py` runs these, asks the model only for what words alone cannot say
(a topic, a stance, an open thread), and emits one `room_signals` record.

Everything here is pure (no I/O, no clock), so every rule is tested on its own.

The meeting clock is the transcript's: milliseconds since the room opened. Room signals arrive
from core-api already on that clock (`at_ms`, `end_ms`).

Presence. A participant row is an invitation, not proof that someone was in the call. A person
counts as present when the source says so (`present`, when core-api sends it) or, failing that,
when they spoke at least once or the room recorded a signal for them. Only present, live (not
"sent their agent") people can be nudged about a listen-for topic or counted for a decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from libs.shared.never_record import NeverRecordPolicy, phrase_pattern

WINDOW_MS = 30_000  # the runtime's never-record window (`runtime.py`, `minutes.py`)
MAX_TOPIC = 80
MAX_RESUME_TOPIC = 120
MAX_QUOTE = 300
MAX_LAST_WORD = 50
MAX_CONSENSUS = 50
MAX_ASK = 12
MAX_POSITIONS = 2

# cut_off
CUT_GAP_MS = 800  # the next speaker comes in within this long after (or over) the fragment
CUT_QUIET_MS = 45_000  # ...and the person says at most CUT_QUIET_WORDS in the next 45 s
CUT_QUIET_WORDS = 8
CUT_MAX_WORDS = 12  # an unpunctuated fragment is short
CUT_MIN_WORDS = 3  # "Yeah" with no full stop is a backchannel, not a cut-off
# spoken_over
OVER_QUIET_MS = 60_000
OVER_QUIET_WORDS = 15
# hand_not_called
HAND_AFTER_MS = 30_000
HAND_SPOKE_WORDS = 15
# topic_silent
TOPIC_MIN_MENTIONS = 2
TOPIC_BEFORE_MS = 10_000
TOPIC_AFTER_MS = 60_000
TOPIC_SPOKE_WORDS = 5
# decision nearby / consensus window
DECISION_NEAR_MS = 180_000
CONSENSUS_BEFORE_MS = 120_000
CONSENSUS_AFTER_MS = 90_000
# resume point
RESUME_TAIL_MS = 300_000  # a cut-off in the last 5 min of transcript
RESUME_TALK_MS = 180_000  # substantive talk in the last 3 min
RESUME_TALK_WORDS = 60
RESUME_TALK_SPEAKERS = 2
RESUME_SCHEDULE_SHARE = 0.7
RESUME_LATE_MS = 60_000
RESUME_CONTEXT_MS = 480_000  # what the model reads: the last 8 minutes
FALLBACK_POSITION_WORDS = 6

KINDS = ("spoken_over", "hand_not_called", "cut_off", "topic_silent")
PRIORITY = {"cut_off": 0, "spoken_over": 1, "hand_not_called": 2, "topic_silent": 3}
HAND_OUTCOMES = ("still_raised_at_end", "lowered_by_host", "gave_up", "left")
NO_TOPIC = "what the room was discussing"

_TRAIL = re.compile(
    r"(?:\b(?:i\s+just\s+want(?:ed)?\s+to(?:\s+(?:say|add|mention|point\s+out|ask))?"
    r"|one\s+(?:more|last)\s+thing|can\s+i\s+(?:just\s+)?(?:add|say|jump\s+in|ask)"
    r"|if\s+i\s+(?:could|may|can)|the\s+thing\s+is|what\s+i\s+mean\s+is"
    r"|but|and|so|because|or|which|that)"
    r"\s*(?:\.\.\.|…|-+|—|,)?|\.\.\.|…|—|--?)\s*$",
    re.IGNORECASE,
)
_BACKCHANNEL = frozenset(
    "yeah yes yep yup ok okay right sure mm mhm hmm uh um huh got it exactly agreed "
    "totally true cool great thanks thank you no nope".split()
)
_WORDS = re.compile(r"[\w']+")


# --- the source -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Participant:
    user_id: UUID
    name: str | None = None
    attendance: str = "live"
    role: str | None = None
    listen_for: tuple[str, ...] = ()
    present: bool | None = None  # None: core-api did not say; inferred (module docstring)


@dataclass(frozen=True)
class Turn:
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    user_id: UUID | None = None


@dataclass(frozen=True)
class SourceDecision:
    id: UUID
    content: str
    status: str
    owner_user_id: UUID | None = None
    quote: str | None = None
    quote_speaker_id: UUID | None = None
    quote_start_ms: int | None = None
    quote_end_ms: int | None = None


@dataclass(frozen=True)
class AgendaItem:
    id: UUID
    title: str
    status: str
    started_ms: int | None = None
    ended_ms: int | None = None


@dataclass(frozen=True)
class RoomSignal:
    id: UUID
    kind: str
    user_id: UUID
    at_ms: int
    end_ms: int
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class RoomSignalsSource:
    meeting_id: UUID
    title: str = "Untitled meeting"
    confidential: bool = False
    host_user_id: UUID | None = None
    previous_meeting_id: UUID | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    room_started_at: datetime | None = None
    ended_at: datetime | None = None
    call_ms: int | None = None
    participants: list[Participant] = field(default_factory=list)
    speakers: dict[str, UUID] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    decisions: list[SourceDecision] = field(default_factory=list)
    agenda: list[AgendaItem] = field(default_factory=list)
    room_signals: list[RoomSignal] = field(default_factory=list)

    def scheduled_ms(self) -> int | None:
        if self.scheduled_start is None or self.scheduled_end is None:
            return None
        ms = int((self.scheduled_end - self.scheduled_start).total_seconds() * 1000)
        return ms if ms > 0 else None

    def scheduled_end_ms(self) -> int | None:
        """The scheduled end on the meeting clock (room opened = 0), when both are known."""
        if self.scheduled_end is None or self.room_started_at is None:
            return None
        return int((self.scheduled_end - self.room_started_at).total_seconds() * 1000)


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _str(raw: object) -> str:
    return " ".join(raw.split()) if isinstance(raw, str) else ""


def _int(raw: object) -> int | None:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _when(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_source(body: object) -> RoomSignalsSource | None:
    """Tolerant: malformed entries are dropped one by one; no meeting id is None."""
    if not isinstance(body, dict):
        return None
    meeting_id = _uuid(body.get("meeting_id"))
    if meeting_id is None:
        return None
    people: list[Participant] = []
    for p in body.get("participants") or []:
        uid = _uuid(p.get("user_id")) if isinstance(p, dict) else None
        if uid is None or any(x.user_id == uid for x in people):
            continue
        assert isinstance(p, dict)
        listen = tuple(
            t for t in (_str(x)[:100] for x in p.get("listen_for") or [] if isinstance(x, str)) if t
        )
        present = p.get("present")
        people.append(
            Participant(
                user_id=uid,
                name=_str(p.get("name"))[:120] or None,
                attendance=_str(p.get("attendance")) or "live",
                role=_str(p.get("role_in_meeting")) or None,
                listen_for=listen[:15],
                present=present if isinstance(present, bool) else None,
            )
        )
    speakers: dict[str, UUID] = {}
    raw_speakers = body.get("speakers")
    for sid, uid in raw_speakers.items() if isinstance(raw_speakers, dict) else []:
        found = _uuid(uid)
        if found is not None:
            speakers[str(sid)] = found
    turns: list[Turn] = []
    for t in body.get("turns") or []:
        if not isinstance(t, dict):
            continue
        text = _str(t.get("text"))
        start, end = _int(t.get("start_ms")), _int(t.get("end_ms"))
        if not text or start is None or end is None:
            continue
        sid = str(t.get("speaker_id") or "")
        turns.append(Turn(sid, start, max(start, end), text, speakers.get(sid) or _uuid(sid)))
    turns.sort(key=lambda x: (x.start_ms, x.end_ms))
    decisions: list[SourceDecision] = []
    for d in body.get("decisions") or []:
        did = _uuid(d.get("id")) if isinstance(d, dict) else None
        if did is None:
            continue
        assert isinstance(d, dict)
        content = _str(d.get("content"))
        if not content:
            continue
        decisions.append(
            SourceDecision(
                id=did,
                content=content[:300],
                status=_str(d.get("status")) or "proposed",
                owner_user_id=_uuid(d.get("owner_user_id")),
                quote=_str(d.get("quote")) or None,
                quote_speaker_id=_uuid(d.get("quote_speaker_id")),
                quote_start_ms=_int(d.get("quote_start_ms")),
                quote_end_ms=_int(d.get("quote_end_ms")),
            )
        )
    agenda: list[AgendaItem] = []
    for a in body.get("agenda") or []:
        aid = _uuid(a.get("id")) if isinstance(a, dict) else None
        if aid is None:
            continue
        assert isinstance(a, dict)
        title = _str(a.get("title"))[:200]
        if title:
            agenda.append(
                AgendaItem(
                    aid,
                    title,
                    _str(a.get("status")) or "pending",
                    _int(a.get("started_ms")),
                    _int(a.get("ended_ms")),
                )
            )
    signals: list[RoomSignal] = []
    for s in body.get("room_signals") or []:
        if not isinstance(s, dict):
            continue
        sid_, uid = _uuid(s.get("id")), _uuid(s.get("user_id"))
        kind = _str(s.get("kind"))
        at, end = _int(s.get("at_ms")), _int(s.get("end_ms"))
        if sid_ is None or uid is None or kind not in ("spoken_over", "hand_not_called"):
            continue
        if at is None:
            continue
        detail = s.get("detail")
        signals.append(
            RoomSignal(
                sid_,
                kind,
                uid,
                at,
                max(at, end if end is not None else at),
                dict(detail) if isinstance(detail, dict) else {},
            )
        )
    return RoomSignalsSource(
        meeting_id=meeting_id,
        title=_str(body.get("title"))[:200] or "Untitled meeting",
        confidential=bool(body.get("confidential")),
        host_user_id=_uuid(body.get("host_user_id")),
        previous_meeting_id=_uuid(body.get("previous_meeting_id")),
        scheduled_start=_when(body.get("scheduled_start")),
        scheduled_end=_when(body.get("scheduled_end")),
        room_started_at=_when(body.get("room_started_at")),
        ended_at=_when(body.get("ended_at")),
        call_ms=_int(body.get("call_ms")),
        participants=people,
        speakers=speakers,
        turns=turns,
        decisions=decisions,
        agenda=agenda,
        room_signals=signals,
    )


# --- small helpers ----------------------------------------------------------------------------


def words(text: str) -> int:
    return len(_WORDS.findall(text))


def norm(text: str) -> str:
    """Lowercase words only: the form quotes are compared in."""
    return " ".join(_WORDS.findall(text.lower()))


def clip(text: str | None, limit: int) -> str | None:
    if not isinstance(text, str):
        return None
    value = " ".join(text.split())
    if not value:
        return None
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def words_by(turns: list[Turn], user_id: UUID, start_ms: int, end_ms: int) -> int:
    """Words `user_id` said in turns that start inside [start_ms, end_ms]."""
    return sum(
        words(t.text) for t in turns if t.user_id == user_id and start_ms <= t.start_ms <= end_ms
    )


def overlaps(t: Turn, start_ms: int, end_ms: int) -> bool:
    return t.start_ms < end_ms and t.end_ms > start_ms


def banned_windows(turns: list[Turn], policy: NeverRecordPolicy) -> set[int]:
    """Every 30 s window that names a never-record topic, and the one after it."""
    if policy.empty:
        return set()
    hit = {t.start_ms // WINDOW_MS for t in turns if policy.hit(t.text)}
    return hit | {w + 1 for w in hit}


def quiet(
    source: RoomSignalsSource, policy: NeverRecordPolicy
) -> tuple[list[Turn], list[RoomSignal]]:
    """The turns and room signals minus the never-record windows (the runtime's rule): no rule
    and no model ever sees them, and nobody is nudged about a moment inside one."""
    banned = banned_windows(source.turns, policy)
    if not banned:
        return list(source.turns), list(source.room_signals)
    turns = [t for t in source.turns if t.start_ms // WINDOW_MS not in banned]
    signals = [s for s in source.room_signals if s.at_ms // WINDOW_MS not in banned]
    return turns, signals


def present_people(source: RoomSignalsSource) -> list[Participant]:
    """Live attendees who were in the call (module docstring)."""
    spoke = {t.user_id for t in source.turns if t.user_id is not None}
    signalled = {s.user_id for s in source.room_signals}
    out = []
    for p in source.participants:
        if p.attendance == "agent":
            continue
        here = (
            p.present if p.present is not None else (p.user_id in spoke or p.user_id in signalled)
        )
        if here:
            out.append(p)
    return out


def assign_labels(source: RoomSignalsSource) -> dict[UUID | str, str]:
    """user id (or an unmapped speaker id) -> "Participant N": speakers in order of first
    speaking, then the participants who never spoke (the minutes' rule)."""
    out: dict[UUID | str, str] = {}
    for t in source.turns:
        who: UUID | str = t.user_id or t.speaker_id
        if who not in out:
            out[who] = f"Participant {len(out) + 1}"
    for p in source.participants:
        if p.user_id not in out:
            out[p.user_id] = f"Participant {len(out) + 1}"
    return out


def clock(ms: int) -> str:
    s = max(0, ms) // 1000
    if s >= 3600:
        return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def render_lines(turns: list[Turn], labels: dict[UUID | str, str]) -> str:
    return "\n".join(
        f"[{clock(t.start_ms)}] {labels.get(t.user_id or t.speaker_id, 'Someone')}: {t.text}"
        for t in turns
    )


# --- the last word: candidates ----------------------------------------------------------------


@dataclass
class Candidate:
    """One "didn't get in" moment, before the model names its topic."""

    user_id: UUID
    kind: str
    at_ms: int | None
    end_ms: int | None
    quote: str | None = None
    topic: str | None = None  # set for topic_silent (the listen-for topic), else by the model
    signal_ids: list[UUID] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)


def trails_off(text: str) -> bool:
    """The words stop mid-thought: a trailing-off ending ("I just want to say", "one more
    thing", "but", "...") or, unpunctuated, a short fragment. A backchannel never does."""
    tokens = _WORDS.findall(text.lower())
    if not tokens or all(w in _BACKCHANNEL for w in tokens):
        return False
    if _TRAIL.search(text.strip()):
        return len(tokens) >= 2
    stripped = text.rstrip().rstrip("\"'”’)")
    if stripped.endswith((".", "?", "!")):
        return False
    return CUT_MIN_WORDS <= len(tokens) <= CUT_MAX_WORDS


def cut_offs(turns: list[Turn]) -> list[Candidate]:
    """Someone's words stopped mid-thought, another speaker came in within 800 ms (or was
    already talking and kept going), and the first person barely spoke in the next 45 s."""
    out: list[Candidate] = []
    for t in turns:
        if t.user_id is None or not trails_off(t.text):
            continue
        taken_over = any(
            o.user_id != t.user_id
            and o.speaker_id != t.speaker_id
            and o.end_ms > t.end_ms
            and o.start_ms <= t.end_ms + CUT_GAP_MS
            and o is not t
            for o in turns
        )
        if not taken_over:
            continue
        if words_by(turns, t.user_id, t.end_ms + 1, t.end_ms + CUT_QUIET_MS) > CUT_QUIET_WORDS:
            continue
        out.append(Candidate(t.user_id, "cut_off", t.start_ms, t.end_ms, quote=t.text))
    return out


def spoken_over(signals: list[RoomSignal], turns: list[Turn]) -> list[Candidate]:
    """A room `spoken_over` signal confirmed by the transcript (another speaker's turn overlaps
    the attempt) or by the room itself (`over_user_id`), when the person did not get in after."""
    out: list[Candidate] = []
    for s in signals:
        if s.kind != "spoken_over":
            continue
        over = _uuid(s.detail.get("over_user_id"))
        confirmed = over is not None or any(
            t.user_id != s.user_id and overlaps(t, s.at_ms, max(s.end_ms, s.at_ms + 1))
            for t in turns
        )
        if not confirmed:
            continue
        if words_by(turns, s.user_id, s.end_ms + 1, s.end_ms + OVER_QUIET_MS) > OVER_QUIET_WORDS:
            continue
        own = [
            t
            for t in turns
            if t.user_id == s.user_id and overlaps(t, s.at_ms - 1000, s.end_ms + 1000)
        ]
        out.append(
            Candidate(
                s.user_id,
                "spoken_over",
                s.at_ms,
                s.end_ms,
                quote=own[0].text if own else None,
                signal_ids=[s.id],
                detail={"over_user_id": str(over) if over else None},
            )
        )
    return out


def hands_not_called(signals: list[RoomSignal], turns: list[Turn]) -> list[Candidate]:
    """A hand that came down without the person being called on, unless they spoke anyway
    (at least 15 words between raising it and 30 s after it came down)."""
    out: list[Candidate] = []
    for s in signals:
        if s.kind != "hand_not_called":
            continue
        if words_by(turns, s.user_id, s.at_ms, s.end_ms + HAND_AFTER_MS) >= HAND_SPOKE_WORDS:
            continue
        outcome = s.detail.get("outcome")
        out.append(
            Candidate(
                s.user_id,
                "hand_not_called",
                s.at_ms,
                s.end_ms,
                signal_ids=[s.id],
                detail={"outcome": outcome if outcome in HAND_OUTCOMES else None},
            )
        )
    return out


def topic_silences(
    people: list[Participant], turns: list[Turn], policy: NeverRecordPolicy
) -> list[Candidate]:
    """A present person's listen-for topic came up in at least two of other people's turns and
    they said at most five words from 10 s before the first mention to 60 s after the last."""
    out: list[Candidate] = []
    for p in people:
        found: list[Candidate] = []
        for topic in p.listen_for:
            if policy.hit(topic):
                continue  # never name a never-record topic, not even back to its listener
            pattern = phrase_pattern(topic)
            if pattern is None:
                continue
            mentions = [t for t in turns if t.user_id != p.user_id and pattern.search(t.text)]
            if len(mentions) < TOPIC_MIN_MENTIONS:
                continue
            first, last = mentions[0], mentions[-1]
            said = words_by(
                turns, p.user_id, first.start_ms - TOPIC_BEFORE_MS, last.end_ms + TOPIC_AFTER_MS
            )
            if said > TOPIC_SPOKE_WORDS:
                continue
            found.append(
                Candidate(
                    p.user_id,
                    "topic_silent",
                    first.start_ms,
                    last.end_ms,
                    topic=clip(topic, MAX_TOPIC),
                )
            )
        if found:
            out.append(min(found, key=lambda c: c.at_ms or 0))
    return out


def pick_one_per_user(cands: list[Candidate]) -> list[Candidate]:
    """One nudge per person: cut_off > spoken_over > hand_not_called > topic_silent; the
    earliest wins a tie. Sorted by moment."""
    best: dict[UUID, Candidate] = {}
    for c in cands:
        cur = best.get(c.user_id)
        key = (PRIORITY[c.kind], c.at_ms if c.at_ms is not None else 1 << 62)
        if cur is None or key < (
            PRIORITY[cur.kind],
            cur.at_ms if cur.at_ms is not None else 1 << 62,
        ):
            best[c.user_id] = c
    return sorted(best.values(), key=lambda c: (c.at_ms or 0, str(c.user_id)))


def nearest_decision(at_ms: int | None, decisions: list[SourceDecision]) -> UUID | None:
    """The decision whose moment is nearest `at_ms`, within 3 minutes."""
    if at_ms is None:
        return None
    near = [
        (abs(d.quote_start_ms - at_ms), d.id)
        for d in decisions
        if d.quote_start_ms is not None and abs(d.quote_start_ms - at_ms) <= DECISION_NEAR_MS
    ]
    return min(near)[1] if near else None


def agenda_at(at_ms: int | None, agenda: list[AgendaItem]) -> AgendaItem | None:
    """The agenda item running at that moment (started, not yet ended), if the clock knows."""
    if at_ms is None:
        return None
    running = [
        a
        for a in agenda
        if a.started_ms is not None
        and a.started_ms <= at_ms
        and (a.ended_ms is None or at_ms < a.ended_ms)
    ]
    return max(running, key=lambda a: a.started_ms or 0) if running else None


def fallback_topic(c: Candidate, agenda: list[AgendaItem], policy: NeverRecordPolicy) -> str:
    """No model answer: the agenda item running at that moment, else a neutral phrase."""
    item = agenda_at(c.at_ms, agenda)
    if item is not None and not policy.hit(item.title):
        return clip(item.title, MAX_TOPIC) or NO_TOPIC
    return NO_TOPIC


# --- consensus ----------------------------------------------------------------------------------


@dataclass
class ConsensusFrame:
    """One decision's moment, read without a model: who was there, who said nothing, who was
    talked over, and the turns the model reads for the rest."""

    decision: SourceDecision
    window_start: int
    window_end: int
    people: list[UUID]
    turns: list[Turn]
    speakers: list[UUID]
    silent: list[UUID]
    talked_over: list[UUID]


def consensus_frames(
    source: RoomSignalsSource,
    turns: list[Turn],
    signals: list[RoomSignal],
    cut: list[Candidate],
    over: list[Candidate],
) -> list[ConsensusFrame]:
    people = [p.user_id for p in present_people(source)]
    frames: list[ConsensusFrame] = []
    for d in source.decisions:
        if d.status not in ("proposed", "confirmed") or d.quote_start_ms is None:
            continue
        start = d.quote_start_ms - CONSENSUS_BEFORE_MS
        end = (d.quote_end_ms or d.quote_start_ms) + CONSENSUS_AFTER_MS
        window = [t for t in turns if overlaps(t, start, end)]
        spoke = {t.user_id for t in window if t.user_id is not None}
        talked_over = {
            c.user_id
            for c in [*cut, *over]
            if c.at_ms is not None and start <= c.at_ms <= end and c.user_id in people
        }
        frames.append(
            ConsensusFrame(
                decision=d,
                window_start=start,
                window_end=end,
                people=people,
                turns=window,
                speakers=[u for u in people if u in spoke],
                silent=[u for u in people if u not in spoke],
                talked_over=sorted(talked_over, key=str),
            )
        )
    return frames[:MAX_CONSENSUS]


STANCES = ("clear_yes", "hedged", "objected", "no_stance")


def strength(n: int, yes: int, objected: int, talked_over: int) -> str:
    if n < 2:
        return "unclear"
    share = yes / n
    if share >= 0.67 and objected == 0:
        return "strong"
    if share < 0.5 or (objected + talked_over >= 1 and share < 0.67):
        return "weak"
    return "mixed"


def tally(frame: ConsensusFrame, stances: dict[UUID, str]) -> dict[str, object]:
    """The counts, the strength and who to ask. Each present person is counted exactly once:
    a clear yes or an objection the model saw wins; a talked-over person with no clear stance is
    `talked_over`; anyone else who spoke without a stance, or said nothing, is `silent`."""
    counts = {"yes": 0, "hedged": 0, "silent": 0, "objected": 0, "talked_over": 0}
    ask: list[dict[str, str]] = []
    over = set(frame.talked_over)
    for uid in frame.people:
        stance = stances.get(uid, "no_stance")
        if stance == "clear_yes":
            counts["yes"] += 1
            continue
        if stance == "objected":
            counts["objected"] += 1
            continue
        if uid in over:
            reason = "talked_over"
        elif stance == "hedged":
            reason = "hedged"
        else:
            reason = "silent"
        counts[reason] += 1
        if uid != frame.decision.owner_user_id:
            ask.append({"user_id": str(uid), "reason": reason})
    n = len(frame.people)
    level = strength(n, counts["yes"], counts["objected"], counts["talked_over"])
    return {
        "decision_id": str(frame.decision.id),
        "strength": level,
        "participants": n,
        **counts,
        "ask": [] if level == "unclear" else ask[:MAX_ASK],
    }


def find_quote(quote: object, turns: list[Turn], user_id: UUID | None) -> Turn | None:
    """The turn of `user_id` (any speaker when None) holding `quote` verbatim (compared as
    lowercase words), or None."""
    if not isinstance(quote, str):
        return None
    wanted = norm(quote)
    if not wanted:
        return None
    for t in turns:
        if (user_id is None or t.user_id == user_id) and wanted in norm(t.text):
            return t
    return None


# --- resume point -----------------------------------------------------------------------------


@dataclass
class OpenThread:
    reason: str  # time_ran_out | agenda_unfinished | cut_off
    agenda_item: AgendaItem | None
    cut_off: Candidate | None
    tail: list[Turn]


def open_thread(
    source: RoomSignalsSource, turns: list[Turn], cut: list[Candidate]
) -> OpenThread | None:
    """Whether the meeting may have ended mid-thread, and why (the model then confirms it).
    A cut-off in the last 5 minutes is the most specific reason; then an agenda item still
    running at the end (or pending ones once 70% of the scheduled time was used); then the call
    ending at or past its scheduled end with real talk in the last 3 minutes."""
    last_end = max((t.end_ms for t in turns), default=None)
    tail = [t for t in turns if last_end is not None and t.end_ms >= last_end - RESUME_CONTEXT_MS]
    late_cut = None
    if last_end is not None:
        recent = [c for c in cut if c.end_ms is not None and c.end_ms >= last_end - RESUME_TAIL_MS]
        late_cut = max(recent, key=lambda c: c.end_ms or 0) if recent else None
    item = next((a for a in source.agenda if a.status == "in_progress"), None)
    if item is None:
        pending = [a for a in source.agenda if a.status == "pending"]
        scheduled = source.scheduled_ms()
        used = source.call_ms if source.call_ms is not None else last_end
        if pending and scheduled and used is not None and used >= RESUME_SCHEDULE_SHARE * scheduled:
            item = pending[0]
    if late_cut is not None:
        return OpenThread("cut_off", item, late_cut, tail)
    if item is not None:
        return OpenThread("agenda_unfinished", item, None, tail)
    if _ran_out(source, turns, last_end):
        return OpenThread("time_ran_out", None, None, tail)
    return None


def _ran_out(source: RoomSignalsSource, turns: list[Turn], last_end: int | None) -> bool:
    if last_end is None:
        return False
    if source.ended_at is not None and source.scheduled_end is not None:
        late = (source.scheduled_end - source.ended_at).total_seconds() * 1000 <= RESUME_LATE_MS
    else:
        end_ms = source.scheduled_end_ms()
        call = source.call_ms if source.call_ms is not None else last_end
        late = end_ms is not None and call >= end_ms - RESUME_LATE_MS
    if not late:
        return False
    recent = [t for t in turns if t.end_ms >= last_end - RESUME_TALK_MS]
    speakers = {t.user_id or t.speaker_id for t in recent}
    return (
        len(speakers) >= RESUME_TALK_SPEAKERS
        and sum(words(t.text) for t in recent) >= RESUME_TALK_WORDS
    )


def fallback_positions(tail: list[Turn]) -> list[Turn]:
    """The last two turns of distinct speakers with something to say (>= 6 words), in order."""
    picked: list[Turn] = []
    seen: set[UUID | str] = set()
    for t in reversed(tail):
        who = t.user_id or t.speaker_id
        if who in seen or words(t.text) < FALLBACK_POSITION_WORDS:
            continue
        seen.add(who)
        picked.append(t)
        if len(picked) == MAX_POSITIONS:
            break
    return list(reversed(picked))
