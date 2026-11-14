"""`propose_action`: the changes Myte may suggest, validated (ADR-0047 D4, ADR-0049 F1/F3/F4).

A proposal is never a write. It becomes one `proposal` event the UI shows as a card with
Confirm, Edit and Not now; on Confirm the UI calls the existing server action for that write.
This module only decides whether the model's arguments make a proposal the person can trust:

- Every id must be one Myte was given in this answer: a registered source, a tool result, the
  scope or the page (`Known`). An id the model made up is refused with a message telling it to
  look the thing up first; the model can recover in the next round.
- Datetimes are ISO 8601 with a UTC offset (a meeting time without one is ambiguous across time
  zones), normalised to `isoformat()`; `end` after `start`, at most 24 hours, not in the past.
- Text is whitespace-collapsed and clipped to the contract limits.

Every event is flat: `{id, kind, title, note?, ...the kind's fields}`. `title` is always present
(the UI's card heading): the model's own words for task / promise / agenda item / attendance /
new meeting, else derived from the fields. Display names ride beside ids (`attendee_names`,
`user_names`, parallel arrays) so the card can say who without another lookup.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from services.rag_service.myte.tools import PROPOSAL_KINDS

MAX_TITLE = 200
MAX_NOTE = 500
MAX_SLACK = 4000
MAX_MEMORY = 300
MAX_PEER_QUESTION = 1000
MAX_PEOPLE = 20
MAX_DECISIONS = 50
MAX_MEETING = timedelta(hours=24)
PAST_SLACK = timedelta(minutes=5)
DESTINATIONS = ("notion", "confluence", "slack")
_CHANNEL = re.compile(r"^#?([a-z0-9][a-z0-9._-]{0,79})$")
_DEST_LABEL = {"notion": "Notion", "confluence": "Confluence", "slack": "Slack"}


@dataclass
class Known:
    """Ids Myte was given in this answer, with a display name where there is one."""

    meetings: dict[str, str] = field(default_factory=dict)  # id -> title
    decisions: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # id -> (text, mid)
    people: dict[str, str] = field(default_factory=dict)  # id -> name
    asker_id: str | None = None


class Invalid(ValueError):
    """The model's arguments do not make a proposal; the message goes back to the model."""


def _flat(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _uuid(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _meeting(args: dict[str, Any], known: Known, kind: str, *, required: bool) -> str | None:
    raw = args.get("meeting_id")
    if not raw:
        if required:
            raise Invalid(f"{kind} needs the meeting_id of the meeting; look it up first.")
        return None
    meeting = _uuid(raw)
    if meeting is None:
        raise Invalid("meeting_id must be the id of a meeting from your sources.")
    if meeting not in known.meetings:
        raise Invalid("use a meeting_id from your sources; look the meeting up first.")
    return meeting


def _people(raw: Any, known: Known, name: str, *, minimum: int) -> list[str]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise Invalid(f"{name} must be a list of person ids.")
    out: list[str] = []
    for value in raw:
        pid = _uuid(value)
        if pid is None or pid not in known.people:
            raise Invalid(
                f"{name} must be person ids from your sources; look people up with "
                "find_people first."
            )
        if pid not in out:
            out.append(pid)
    if len(out) < minimum:
        raise Invalid(f"{name} needs at least {minimum} person id.")
    if len(out) > MAX_PEOPLE:
        raise Invalid(f"at most {MAX_PEOPLE} people.")
    return out


def _when(value: Any, name: str) -> datetime:
    if not value:
        raise Invalid(f"{name} is required, ISO 8601 with the UTC offset.")
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        raise Invalid(f"{name} must be ISO 8601 like 2026-11-16T10:00:00+05:30.") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Invalid(f"{name} needs the UTC offset, like 2026-11-16T10:00:00+05:30.")
    return parsed


def _names(ids: list[str], known: Known) -> list[str]:
    return [known.people.get(i, "") for i in ids]


def _join(names: list[str]) -> str:
    named = [n for n in names if n]
    if not named:
        return f"{len(names)} people" if len(names) != 1 else "1 person"
    if len(named) <= 2:
        return " and ".join(named)
    return f"{named[0]}, {named[1]} and {len(names) - 2} more"


def build(args: dict[str, Any], known: Known, now: datetime) -> dict[str, Any]:
    """The proposal event (without dedupe or the per-answer cap), or raises `Invalid`."""
    kind = args.get("kind")
    if kind not in PROPOSAL_KINDS:
        raise Invalid("kind must be one of " + ", ".join(PROPOSAL_KINDS) + ".")
    title = _flat(args.get("title"), MAX_TITLE)
    p: dict[str, Any] = {"id": str(uuid.uuid4()), "kind": kind}

    if kind in ("task", "promise", "attendance", "agenda_item"):
        if not title:
            raise Invalid("a title is required.")
        if args.get("due_date"):
            try:
                p["due_date"] = date.fromisoformat(str(args["due_date"])[:10]).isoformat()
            except ValueError:
                raise Invalid("due_date must be a date like 2026-11-20.") from None
        meeting = _meeting(args, known, kind, required=kind in ("attendance", "agenda_item"))
        if meeting:
            p["meeting_id"] = meeting
        if kind == "attendance":
            if args.get("choice") not in ("attend", "agent", "skip"):
                raise Invalid("attendance needs choice: attend, agent or skip.")
            p["choice"] = args["choice"]

    elif kind == "schedule_meeting":
        if not title:
            raise Invalid("schedule_meeting needs a title for the meeting.")
        start, end = _when(args.get("start"), "start"), _when(args.get("end"), "end")
        if end <= start:
            raise Invalid("end must be after start.")
        if end - start > MAX_MEETING:
            raise Invalid("a meeting can be at most 24 hours long.")
        if start < now - PAST_SLACK:
            raise Invalid("start is in the past; check today's date and the time zone.")
        attendees = _people(args.get("attendee_ids"), known, "attendee_ids", minimum=0)
        p.update(
            start=start.isoformat(),
            end=end.isoformat(),
            attendee_ids=attendees,
            attendee_names=_names(attendees, known),
        )

    elif kind == "invite_people":
        meeting = _meeting(args, known, kind, required=True)
        assert meeting is not None
        users = _people(args.get("user_ids"), known, "user_ids", minimum=1)
        names = _names(users, known)
        meeting_title = known.meetings.get(meeting) or "the meeting"
        p.update(meeting_id=meeting, user_ids=users, user_names=names)
        title = title or f"Invite {_join(names)} to {meeting_title}"

    elif kind == "slack_post":
        text = str(args.get("text") or "").strip()[:MAX_SLACK].strip()
        if not text:
            raise Invalid("slack_post needs the text of the message.")
        p["text"] = text
        channel = _flat(args.get("channel"), 81).lower()
        if channel:
            m = _CHANNEL.match(channel)
            if m is None:
                raise Invalid("channel must be a Slack channel name like #product.")
            p["channel"] = f"#{m.group(1)}"
        title = title or ("Post to Slack" + (f" in {p['channel']}" if "channel" in p else ""))

    elif kind == "export_decisions":
        destination = args.get("destination")
        if destination not in DESTINATIONS:
            raise Invalid("destination must be notion, confluence or slack.")
        p["destination"] = destination
        meeting = _meeting(args, known, kind, required=False)
        if meeting:
            p["meeting_id"] = meeting
        raw = args.get("decision_ids")
        if raw is not None:
            if not isinstance(raw, list):
                raise Invalid("decision_ids must be a list of decision ids.")
            ids: list[str] = []
            for value in raw:
                did = _uuid(value)
                if did is None or did not in known.decisions:
                    raise Invalid("decision_ids must be decision ids from your sources.")
                if did not in ids:
                    ids.append(did)
            if len(ids) > MAX_DECISIONS:
                raise Invalid(f"at most {MAX_DECISIONS} decisions.")
            if ids:
                p["decision_ids"] = ids
        if "meeting_id" not in p and "decision_ids" not in p:
            raise Invalid("export_decisions needs a meeting_id or decision_ids from your sources.")
        what = (
            f"decisions from {known.meetings.get(p['meeting_id']) or 'the meeting'}"
            if "meeting_id" in p and "decision_ids" not in p
            else (
                "1 decision"
                if len(p.get("decision_ids", [])) == 1
                else f"{len(p.get('decision_ids', []))} decisions"
            )
        )
        title = title or f"Export {what} to {_DEST_LABEL[destination]}"

    elif kind == "share_decision":
        did = _uuid(args.get("decision_id"))
        if did is None or did not in known.decisions:
            raise Invalid("share_decision needs a decision_id from your sources.")
        text, meeting_id = known.decisions[did]
        p["decision_id"] = did
        if meeting_id:
            p["meeting_id"] = meeting_id
        title = title or _flat(f"Share: {text}" if text else "Share this decision", MAX_TITLE)

    elif kind == "ask_myte":
        pid = _uuid(args.get("person_id"))
        if pid is None or pid not in known.people:
            raise Invalid(
                "ask_myte needs the person_id of a teammate from your sources; find them with "
                "find_people first."
            )
        if known.asker_id and pid == known.asker_id:
            raise Invalid("that is the person you are talking with; they cannot ask themself.")
        question = _flat(args.get("question"), MAX_PEER_QUESTION)
        if not question:
            raise Invalid("ask_myte needs the question to ask.")
        name = _flat(args.get("person_name"), 160) or known.people.get(pid, "")
        if not name:
            raise Invalid("ask_myte needs the teammate's name.")
        p.update(person_id=pid, person_name=name, question=question)
        title = title or f"Ask {name}'s Myte"

    else:  # memory
        text = _flat(args.get("text"), MAX_MEMORY + 1)
        if not text:
            raise Invalid("memory needs the text to remember.")
        if len(text) > MAX_MEMORY:
            raise Invalid(f"memory text must be at most {MAX_MEMORY} characters; shorten it.")
        p["text"] = text
        title = title or _flat(f"Remember: {text}", MAX_TITLE)

    p["title"] = title
    note = _flat(args.get("note"), MAX_NOTE)
    if note:
        p["note"] = note
    return p


def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}
