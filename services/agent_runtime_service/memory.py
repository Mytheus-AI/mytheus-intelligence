"""Memory inside the meeting: the last windows, and a ledger of the cards already raised.

Polish program, "Memory inside the meeting". No agent saw anything before its own window, so a
decision restated two minutes later was proposed again, and a dismissed card came back. Now
every agent call gets, in its context under `"meeting_memory"` (`MeetingMemory`):

- `previous`: the text of the meeting's last two windows (stable "Participant N" labels), kept
  in the runtime's memory (one partition holds a whole meeting, so one process sees it all);
- `entries`: the meeting's decision and action cards (id, kind, text, status), from the ledger;
- `marks`: this meeting's suggestions a participant dismissed or already asked.

The ledger is one Redis hash, `nexus:{org_id}:meeting-ledger:{meeting_id}` (24 h TTL, renewed
on every write), with one writer per field so nobody overwrites anybody:

- `c:{candidate_id}` (THIS runtime, when it emits a candidate or an update):
  `{"kind": "decision"|"action", "text", "moment_id", "seq"}`;
- `s:{candidate_id}` (core-api, when a person confirms, edits or dismisses the card):
  `{"status": "confirmed"|"dismissed"|"edited", "text": <the edited text or null>}`;
- `q:{user_id}:{suggestion_id}` (core-api, when a person dismisses, asks or saves a
  suggestion): `{"status": "dismissed"|"asked"|"saved", "text": <the question>}`.

A card's status is `open` until core-api says otherwise. core-api's state changes are the
source (the gateway's `candidate-resolved` only relays what core-api already stored), and the
runtime's own emits are the only source of what was proposed, so this needs no new topic and no
extra call per window. Redis failing is never fatal: the window runs without memory, as before.

What the memory changes (`reconcile`, for DecisionAgent and ActionAgent): a candidate the
model marks `updates` (a ledger label such as `D2`), or one that is a near-duplicate
(content-word Jaccard >= 0.6, `text_match.similarity`) of an open card of its kind, becomes an
UPDATE of that card: same id, `updates_candidate_id` = that id, the card's original `moment_id`.
Anything that would update or repeat a dismissed, confirmed or human-edited card is dropped:
dismissed is never re-proposed, confirmed is settled, and a person's edit is not overwritten.
QuestionAgent suggestions near-duplicating one the same person dismissed or asked are dropped
the same way (`drop_repeated_suggestions`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from services.agent_runtime_service.text_match import similarity

logger = logging.getLogger(__name__)

LEDGER_TTL_S = 24 * 3600
DUPLICATE_SIMILARITY = 0.6
SAME_SIMILARITY = 0.95
MAX_OPEN = 15
MAX_CONFIRMED = 10
MAX_DISMISSED = 15
MAX_TEXT = 200
CONTEXT_KEY = "meeting_memory"
KIND_OF_RESULT = {"decision_candidate": "decision", "action_candidate": "action"}
_PREFIX = {"decision": "D", "action": "A"}
_CLOSED = frozenset({"dismissed", "confirmed", "edited"})


def ledger_key(org_id: UUID, meeting_id: UUID) -> str:
    return f"nexus:{org_id}:meeting-ledger:{meeting_id}"


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    kind: str  # decision | action
    text: str
    status: str = "open"  # open | confirmed | dismissed | edited
    moment_id: str | None = None
    seq: float = 0.0


@dataclass(frozen=True)
class SuggestionMark:
    user_id: str
    id: str
    text: str
    status: str  # dismissed | asked | saved


@dataclass
class MeetingMemory:
    previous: list[str] = field(default_factory=list)
    entries: list[LedgerEntry] = field(default_factory=list)
    marks: list[SuggestionMark] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.kind == kind]


def memory_of(context: dict[str, Any] | None) -> MeetingMemory:
    found = (context or {}).get(CONTEXT_KEY)
    return found if isinstance(found, MeetingMemory) else MeetingMemory()


def _json(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except (ValueError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_TEXT else flat[: MAX_TEXT - 1].rstrip() + "…"


class MeetingLedger:
    """The Redis side. Every method fails open (logs, returns nothing)."""

    def __init__(self, redis: Any, *, clock: Any = time.time) -> None:
        self._redis = redis
        self._clock = clock

    async def load(
        self, org_id: UUID, meeting_id: UUID
    ) -> tuple[list[LedgerEntry], list[SuggestionMark]]:
        try:
            raw: dict[Any, Any] = await self._redis.hgetall(ledger_key(org_id, meeting_id))
        except Exception:  # noqa: BLE001 - no memory this window, never no window
            logger.warning("meeting ledger unavailable meeting=%s", meeting_id)
            return [], []
        fields = {(k.decode() if isinstance(k, bytes) else str(k)): v for k, v in raw.items()}
        entries: list[LedgerEntry] = []
        marks: list[SuggestionMark] = []
        ids = {k[2:] for k in fields if k.startswith(("c:", "s:"))}
        for cid in ids:
            mine = _json(fields.get(f"c:{cid}", b"{}"))
            state = _json(fields.get(f"s:{cid}", b"{}"))
            kind = str(mine.get("kind") or state.get("kind") or "")
            text = state.get("text") if isinstance(state.get("text"), str) else None
            text = text or (mine.get("text") if isinstance(mine.get("text"), str) else None)
            if kind not in _PREFIX or not text:
                continue
            status = str(state.get("status") or "open")
            entries.append(
                LedgerEntry(
                    id=cid,
                    kind=kind,
                    text=_clip(text),
                    status=status if status in _CLOSED else "open",
                    moment_id=(
                        mine.get("moment_id") if isinstance(mine.get("moment_id"), str) else None
                    ),
                    seq=float(mine.get("seq") or state.get("seq") or 0.0),
                )
            )
        for key, value in fields.items():
            if not key.startswith("q:"):
                continue
            _, user, sid = (key.split(":", 2) + ["", ""])[:3]
            data = _json(value)
            text = data.get("text")
            if (
                isinstance(text, str)
                and text.strip()
                and data.get("status") in ("dismissed", "asked", "saved")
            ):
                marks.append(SuggestionMark(user, sid, _clip(text), str(data["status"])))
        entries.sort(key=lambda e: (e.seq, e.id))
        return entries, marks

    async def record(self, org_id: UUID, meeting_id: UUID, entries: list[LedgerEntry]) -> None:
        if not entries:
            return
        key = ledger_key(org_id, meeting_id)
        mapping = {
            f"c:{e.id}": json.dumps(
                {
                    "kind": e.kind,
                    "text": _clip(e.text),
                    "moment_id": e.moment_id,
                    "seq": e.seq or self._clock(),
                }
            )
            for e in entries
        }
        try:
            await self._redis.hset(key, mapping=mapping)
            await self._redis.expire(key, LEDGER_TTL_S)
        except Exception:  # noqa: BLE001 - the card still went out; only its memory is lost
            logger.warning("meeting ledger write failed meeting=%s", meeting_id)


# --- the prompt block ---------------------------------------------------------------------------


def render_memory(memory: MeetingMemory, kind: str) -> tuple[str, dict[str, str]]:
    """The `{memory}` block for decision_agent@v3 / action_agent@v3 and the label -> card id
    map (`D1`, `A3`) the answer's `updates` is read back with. "" and {} when there is nothing,
    so a meeting's first window renders exactly the plain prompt."""
    parts: list[str] = []
    if memory.previous:
        earlier = "\n".join(memory.previous)
        parts.append(
            "Earlier in this meeting (context only, to understand what is referred to):\n"
            f"<earlier>\n{earlier}\n</earlier>\n"
        )
    mine = memory.of_kind(kind)
    open_ = [e for e in mine if e.status == "open"][-MAX_OPEN:]
    confirmed = [e for e in mine if e.status in ("confirmed", "edited")][-MAX_CONFIRMED:]
    dismissed = [e for e in mine if e.status == "dismissed"][-MAX_DISMISSED:]
    shown = open_ + confirmed + dismissed
    labels: dict[str, str] = {}
    if shown:
        noun = "decisions" if kind == "decision" else "action items"
        lines = []
        for n, e in enumerate(shown, start=1):
            label = f"{_PREFIX[kind]}{n}"
            labels[label] = e.id
            status = "settled" if e.status in ("confirmed", "edited") else e.status
            lines.append(f"{label}. [{status}] {e.text}")
        parts.append(f"Cards already raised in this meeting ({noun}):\n" + "\n".join(lines) + "\n")
    return ("\n".join(parts) + "\n") if parts else "", labels


# --- reconcile the model's answer with the ledger ------------------------------------------------


def _target(raw: object, labels: dict[str, str], ids: dict[str, LedgerEntry]) -> LedgerEntry | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    token = raw.strip().rstrip(".").upper()
    cid = labels.get(token)
    if cid is None and raw.strip().lower() in ids:
        cid = raw.strip().lower()
    return ids.get(cid) if cid else None


def reconcile(
    candidates: list[dict[str, Any]],
    kind: str,
    memory: MeetingMemory,
    labels: dict[str, str],
    updates: list[object],
) -> list[dict[str, Any]]:
    """Each candidate (shaped, banded, id'd) with the model's raw `updates` answer at the same
    index. Returns the candidates to emit: new ones unchanged, updates re-id'd to their card,
    repeats of closed cards removed, at most one update per card."""
    entries = memory.of_kind(kind)
    ids = {e.id: e for e in entries}
    taken: set[str] = set()
    out: list[dict[str, Any]] = []
    for cand, raw in zip(candidates, updates, strict=True):
        target = _target(raw, labels, ids)
        implicit = target is None
        if implicit:
            best = max(entries, key=lambda e: similarity(e.text, cand["content"]), default=None)
            if best is not None and similarity(best.text, cand["content"]) >= DUPLICATE_SIMILARITY:
                target = best
        if target is not None:
            if target.status != "open" or target.id in taken:
                continue  # never re-propose a dismissed / settled / edited card, nor twice
            if implicit and similarity(target.text, cand["content"]) >= SAME_SIMILARITY:
                continue  # the same card said again: nothing new to show
            taken.add(target.id)
            cand = {**cand, "id": target.id, "updates_candidate_id": target.id}
            if target.moment_id:
                cand["moment_id"] = target.moment_id
        out.append(cand)
    return out


def drop_repeated_suggestions(
    suggestions: list[dict[str, Any]], memory: MeetingMemory, user_id: UUID
) -> list[dict[str, Any]]:
    seen = [m.text for m in memory.marks if m.user_id == str(user_id)]
    if not seen:
        return suggestions
    return [
        s
        for s in suggestions
        if not any(
            similarity(t, str(s.get("question") or "")) >= DUPLICATE_SIMILARITY for t in seen
        )
    ]
