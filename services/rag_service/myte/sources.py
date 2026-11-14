"""Numbered sources and citations for Ask Myte (mytheus-docs ADR-0047 D5).

Every item a tool returns is registered once as source `n` (1-based, stable within one answer;
the same decision found by two tools keeps its first number) and is given to the model as one
compact line, `[n] kind | title | text | ids`, the ids so it can open the item with another
tool. The model cites `[n]`; `cited()` reads back the numbers that are really in the text and
really registered.

`map_output` turns one tool's structured result (bff's `structuredContent`, the same JSON its
REST routes return) into source drafts, per tool. A tool this module does not know yet, or a
result without structured content, becomes one `document` source of its text, so a new bff tool
is citable the day it ships.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from services.rag_service.myte.contracts import SourceKind

MAX_SOURCES = 80  # per answer
MAX_PER_TOOL = 25  # per tool call
EVENT_TEXT = 1200  # characters of a source's text in its `source` event
LINE_TEXT = 500  # characters of a source's text in the model's line
_CITE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_ID_FIELDS = ("meeting_id", "decision_id", "action_id", "promise_id", "person_id", "id")


@dataclass
class Draft:
    kind: SourceKind
    title: str
    text: str
    meeting_id: str | None = None
    meeting_title: str | None = None
    decision_id: str | None = None
    action_id: str | None = None
    occurred_at: str | None = None
    speaker: str | None = None
    start_ms: int | None = None
    ref_id: str | None = None  # the item's own id (promise, person, notification, ...)
    raw: Any = None
    alias: int | None = None  # the tool's own citation number (ask_meetings)

    def key(self) -> tuple[str, str]:
        ident = self.decision_id or self.action_id or self.ref_id
        if self.kind == "meeting" and not ident:
            ident = self.meeting_id
        if self.kind == "transcript" and self.meeting_id and self.start_ms is not None:
            ident = f"{self.meeting_id}:{self.start_ms}"
        return (self.kind, ident or f"{self.title}\x00{self.text[:200]}")


@dataclass
class Source:
    n: int
    draft: Draft
    detail: Any = None  # the whole structured result it came from (for cards)

    def event(self) -> dict[str, Any]:
        d = self.draft
        out: dict[str, Any] = {
            "n": self.n,
            "kind": d.kind,
            "title": _clip(d.title, 300),
            "text": _clip(d.text, EVENT_TEXT),
        }
        for name in (
            "meeting_id",
            "meeting_title",
            "decision_id",
            "action_id",
            "occurred_at",
            "speaker",
            "start_ms",
        ):
            value = getattr(d, name)
            if value is not None and value != "":
                out[name] = value
        return out

    def line(self) -> str:
        d = self.draft
        ids = [
            f"{name}={value}"
            for name, value in (
                ("meeting_id", d.meeting_id),
                ("decision_id", d.decision_id),
                ("action_id", d.action_id),
                ("id", d.ref_id),
            )
            if value
        ]
        if d.occurred_at:
            ids.append(str(d.occurred_at)[:16])
        parts = [f"[{self.n}] {d.kind}", _flat(d.title, 160), _flat(d.text, LINE_TEXT)]
        if d.meeting_title and d.kind != "meeting":
            parts.append(f"in {_flat(d.meeting_title, 120)}")
        if ids:
            parts.append(", ".join(ids))
        return " | ".join(parts)


@dataclass
class SourceRegistry:
    sources: list[Source] = field(default_factory=list)
    _by_key: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, draft: Draft, detail: Any = None) -> tuple[Source | None, bool]:
        """-> (the source, whether its event must be sent). None when the answer is full.

        The same item found again keeps its number. When the new find says more (a decision
        record after a search hit: the why, the history), the source is upgraded in place and
        its event is sent again with the same `n`, so the person and the model read the same,
        fuller text."""
        key = draft.key()
        if key in self._by_key:
            source = self.sources[self._by_key[key] - 1]
            if len(draft.text) > len(source.draft.text):
                if draft.alias is None:
                    draft.alias = source.draft.alias
                source.draft = draft
                source.detail = detail if detail is not None else source.detail
                return source, True
            return source, False
        if len(self.sources) >= MAX_SOURCES:
            return None, False
        source = Source(n=len(self.sources) + 1, draft=draft, detail=detail)
        self.sources.append(source)
        self._by_key[key] = source.n
        return source, True

    def get(self, n: int) -> Source | None:
        return self.sources[n - 1] if 1 <= n <= len(self.sources) else None

    def cited(self, text: str) -> list[int]:
        found: set[int] = set()
        for m in _CITE.finditer(text):
            for part in m.group(1).split(","):
                n = int(part.strip())
                if self.get(n) is not None:
                    found.add(n)
        return sorted(found)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _flat(text: str, limit: int) -> str:
    return _clip(" ".join(str(text or "").split()).replace("|", "/"), limit)


def _s(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, dict):
        items = body.get("items")
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)][:MAX_PER_TOOL]
    if isinstance(body, list):
        return [i for i in body if isinstance(i, dict)][:MAX_PER_TOOL]
    return []


def _when(*values: Any) -> str | None:
    for v in values:
        if v:
            return str(v)
    return None


# --- per-tool mapping ------------------------------------------------------------------------


def _search(body: Any, _args: dict[str, Any]) -> list[Draft]:
    kinds: dict[str, SourceKind] = {
        "meeting": "meeting",
        "decision": "decision",
        "action": "action",
        "transcript": "transcript",
        "document": "document",
        "promise": "promise",
    }
    out = []
    for h in _items(body):
        kind = kinds.get(str(h.get("kind")), "meeting")
        ident = _s(h.get("id"))
        text = str(h.get("text") or "")
        out.append(
            Draft(
                kind=kind,
                title=(text if kind == "meeting" else (h.get("meeting_title") or text))[:160],
                text=text,
                meeting_id=ident if kind == "meeting" else _s(h.get("meeting_id")),
                meeting_title=_s(h.get("meeting_title")) or (text if kind == "meeting" else None),
                decision_id=ident if kind == "decision" else None,
                action_id=ident if kind == "action" else None,
                ref_id=ident if kind not in ("meeting", "decision", "action") else None,
                occurred_at=_s(h.get("occurred_at")),
                raw=h,
            )
        )
    return out


def _ask(body: Any, _args: dict[str, Any]) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    kinds: dict[str, SourceKind] = {
        "transcript": "transcript",
        "document": "document",
        "decision": "decision",
        "action": "action",
    }
    out = []
    for c in body.get("citations") or []:
        if not isinstance(c, dict):
            continue
        kind = kinds.get(str(c.get("kind")), "transcript")
        title = c.get("meeting_title") or ("Company brain" if kind == "document" else kind)
        out.append(
            Draft(
                kind=kind,
                title=str(title),
                text=str(c.get("text") or ""),
                meeting_id=_s(c.get("meeting_id")),
                meeting_title=_s(c.get("meeting_title")),
                decision_id=_s(c.get("decision_id")),
                action_id=_s(c.get("action_id")),
                ref_id=_s(c.get("document_id")),
                occurred_at=_s(c.get("occurred_at")),
                speaker=_s(c.get("speaker")),
                start_ms=c.get("start_ms") if isinstance(c.get("start_ms"), int) else None,
                raw=c,
                alias=c.get("n") if isinstance(c.get("n"), int) else None,
            )
        )
    return out[:MAX_PER_TOOL]


def _action(a: dict[str, Any], meeting_id: str | None = None, title: str | None = None) -> Draft:
    bits = []
    if a.get("owner_name"):
        bits.append(f"owner {a['owner_name']}")
    if a.get("due_date"):
        bits.append(f"due {a['due_date']}")
    if a.get("status"):
        bits.append(str(a["status"]))
    content = str(a.get("content") or "")
    return Draft(
        kind="action",
        title=content[:160],
        text=content + (f" ({', '.join(bits)})" if bits else ""),
        meeting_id=_s(a.get("meeting_id")) or meeting_id,
        meeting_title=_s(a.get("meeting_title")) or title,
        action_id=_s(a.get("id")),
        occurred_at=_when(a.get("due_date")),
        raw=a,
    )


def _outcome(body: Any, args: dict[str, Any]) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    meeting_id = _s(body.get("id")) or _s(args.get("meeting_id"))
    title = str(body.get("title") or "Meeting")
    meta = [f"status {body['status']}" if body.get("status") else ""]
    if body.get("scheduled_start"):
        meta.append(f"start {body['scheduled_start']}")
    if body.get("host_name"):
        meta.append(f"host {body['host_name']}")
    agenda = [
        f"{i.get('title')} ({i.get('status')})"
        for i in body.get("agenda") or []
        if isinstance(i, dict)
    ]
    if agenda:
        meta.append("agenda: " + "; ".join(agenda))
    out = [
        Draft(
            kind="meeting",
            title=title,
            text=", ".join(m for m in meta if m) or title,
            meeting_id=meeting_id,
            meeting_title=title,
            occurred_at=_s(body.get("scheduled_start")),
            raw=body,
        )
    ]
    for d in body.get("decisions") or []:
        if isinstance(d, dict):
            content = str(d.get("content") or "")
            out.append(
                Draft(
                    kind="decision",
                    title=content[:160],
                    text=content,
                    meeting_id=meeting_id,
                    meeting_title=title,
                    decision_id=_s(d.get("id")),
                    raw=d,
                )
            )
    for a in body.get("actions") or []:
        if isinstance(a, dict):
            out.append(_action(a, meeting_id, title))
    return out[:MAX_PER_TOOL]


def _record(body: Any, args: dict[str, Any]) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    content = str(body.get("content") or "")
    text = [content]
    if body.get("status"):
        text.append(f"Status: {body['status']}.")
    if body.get("rationale"):
        text.append(f"Why: {body['rationale']}")
    history = []
    for s in body.get("lineage") or []:
        if not isinstance(s, dict) or s.get("relation") == "subject":
            continue
        history.append(
            str(s.get("content"))
            if s.get("visible")
            else "a decision from a meeting you were not in"
        )
    if history:
        text.append("History: " + "; ".join(history))
    out = [
        Draft(
            kind="decision",
            title=content[:160],
            text=" ".join(text),
            meeting_id=_s(body.get("meeting_id")),
            meeting_title=_s(body.get("meeting_title")),
            decision_id=_s(body.get("id")) or _s(args.get("decision_id")),
            occurred_at=_s(body.get("decided_at") or body.get("created_at")),
            raw=body,
        )
    ]
    for a in body.get("meeting_actions") or []:
        if isinstance(a, dict):
            out.append(_action(a, _s(body.get("meeting_id")), _s(body.get("meeting_title"))))
    return out[:MAX_PER_TOOL]


def _as_of(body: Any, _args: dict[str, Any]) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    out = []
    for d in (body.get("decisions") or [])[:15]:
        if not isinstance(d, dict):
            continue
        meeting = _dict(d.get("meeting"))
        text = str(d.get("content") or "")
        if d.get("rationale"):
            text += f". Why: {d['rationale']}"
        if d.get("quote"):
            who = f"{d['quote_speaker_name']}: " if d.get("quote_speaker_name") else ""
            text += f'. Said: {who}"{d["quote"]}"'
        out.append(
            Draft(
                kind="decision",
                title=str(d.get("content") or "")[:160],
                text=text,
                meeting_id=_s(meeting.get("id")),
                meeting_title=_s(meeting.get("title")),
                decision_id=_s(d.get("id")),
                occurred_at=_s(d.get("decided_at")),
                raw=d,
            )
        )
    for r in (body.get("replaced") or [])[:5]:
        if not isinstance(r, dict):
            continue
        by = _dict(r.get("replaced_by"))
        what = (
            by.get("content") if by.get("visible") else "a decision from a meeting you were not in"
        )
        out.append(
            Draft(
                kind="decision",
                title=str(r.get("content") or "")[:160],
                text=f"{r.get('content')} (replaced on {str(r.get('replaced_at'))[:10]} by {what})",
                decision_id=_s(r.get("id")),
                occurred_at=_s(r.get("replaced_at")),
                raw=r,
            )
        )
    for p in (body.get("promises") or [])[:5]:
        if isinstance(p, dict):
            due = f", due {p['due']}" if p.get("due") else ""
            out.append(
                Draft(
                    kind="promise",
                    title=str(p.get("content") or "")[:160],
                    text=f"{p.get('content')} ({p.get('owner_name') or 'Unassigned'}{due})",
                    ref_id=_s(p.get("id")),
                    occurred_at=_s(p.get("due")),
                    raw=p,
                )
            )
    return out[:MAX_PER_TOOL]


def _actions(body: Any, _args: dict[str, Any]) -> list[Draft]:
    return [_action(a) for a in _items(body)]


def _promises(body: Any, _args: dict[str, Any]) -> list[Draft]:
    out = []
    for c in _items(body):
        content = str(c.get("content") or "")
        bits = []
        if c.get("recipient_name"):
            bits.append(f"to {c['recipient_name']}")
        if c.get("due_at"):
            bits.append(f"due {c['due_at']}")
        elif c.get("due_hint"):
            bits.append(str(c["due_hint"]))
        source = _dict(c.get("source"))
        if source.get("label"):
            bits.append(f"from {source['label']}")
        if c.get("review_state") == "pending":
            bits.append("found by your Myte, not confirmed yet")
        out.append(
            Draft(
                kind="promise",
                title=content[:160],
                text=content + (f" ({', '.join(bits)})" if bits else ""),
                ref_id=_s(c.get("id")),
                meeting_id=_s(source.get("meeting_id")),
                occurred_at=_s(c.get("due_at")),
                raw=c,
            )
        )
    return out


def _capacity(body: Any, args: dict[str, Any], text: str) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    day = args.get("due_date")
    return [
        Draft(
            kind="promise",
            title=f"Your capacity on {day}" if day else "Your week, day by day",
            text=text,
            ref_id=f"capacity:{day or 'week'}",
            raw=body,
        )
    ]


def _check(body: Any, _args: dict[str, Any]) -> list[Draft]:
    if not isinstance(body, dict):
        return []
    out = []
    for c in (body.get("conflicts") or [])[:MAX_PER_TOOL]:
        if not isinstance(c, dict):
            continue
        verb = "contradicts" if c.get("relation") == "contradicts" else "reopens"
        out.append(
            Draft(
                kind="decision",
                title=str(c.get("decision_content") or "")[:160],
                text=(
                    f"{c.get('decision_content')}. The draft {verb} it: \"{c.get('quote')}\". "
                    f"Why: {c.get('why') or 'it goes against the decision'}"
                ),
                meeting_title=_s(c.get("meeting_title")),
                meeting_id=_s(c.get("meeting_id")),
                decision_id=_s(c.get("decision_id")),
                occurred_at=_s(c.get("decided_at")),
                raw=c,
            )
        )
    return out


def _schedule(body: Any, _args: dict[str, Any]) -> list[Draft]:
    out = []
    for m in _items(body):
        title = str(m.get("title") or "Meeting")
        bits = []
        if m.get("start"):
            bits.append(f"starts {m['start']}")
        if m.get("end"):
            bits.append(f"ends {m['end']}")
        if m.get("live"):
            bits.append("live now")
        if m.get("host_name") or m.get("host"):
            bits.append(f"host {m.get('host_name') or m.get('host')}")
        if m.get("hosting"):
            bits.append("you host")
        if m.get("attendee_count") is not None:
            bits.append(f"{m['attendee_count']} attendees")
        choice = m.get("attendance") or m.get("attendance_choice")
        if choice:
            bits.append(f"your choice: {choice}")
        out.append(
            Draft(
                kind="meeting",
                title=title,
                text=f"{title}: " + ", ".join(bits) if bits else title,
                meeting_id=_s(m.get("id") or m.get("meeting_id")),
                meeting_title=title,
                occurred_at=_s(m.get("start")),
                raw=m,
            )
        )
    return out


def _people(body: Any, _args: dict[str, Any]) -> list[Draft]:
    out = []
    for p in _items(body):
        name = str(p.get("name") or p.get("display_name") or p.get("email") or "Someone")
        bits = [str(p[k]) for k in ("role", "title", "team", "department") if p.get(k)]
        if p.get("email"):
            bits.append(str(p["email"]))
        out.append(
            Draft(
                kind="person",
                title=name,
                text=f"{name}: " + " · ".join(bits) if bits else name,
                ref_id=_s(p.get("id") or p.get("user_id")),
                raw=p,
            )
        )
    return out


def _inbox(body: Any, _args: dict[str, Any]) -> list[Draft]:
    out = []
    for n in _items(body):
        title = str(n.get("title") or n.get("kind") or "Notification")
        text = str(n.get("body") or n.get("text") or title)
        out.append(
            Draft(
                kind="notification",
                title=title[:160],
                text=text,
                meeting_id=_s(n.get("meeting_id")),
                meeting_title=_s(n.get("meeting_title")),
                ref_id=_s(n.get("id")),
                occurred_at=_s(n.get("created_at")),
                raw=n,
            )
        )
    return out


def _my_myte(body: Any, _args: dict[str, Any], text: str) -> list[Draft]:
    if not isinstance(body, dict) and not text:
        return []
    return [
        Draft(
            kind="document",
            title="My settings (Your Myte)",
            text=text or str(body),
            ref_id="myte:settings",
            raw=body,
        )
    ]


_STRUCTURED = {
    "search_meetings": _search,
    "ask_meetings": _ask,
    "get_meeting_outcome": _outcome,
    "get_decision_record": _record,
    "company_as_of": _as_of,
    "list_my_open_actions": _actions,
    "list_my_promises": _promises,
    "check_against_decisions": _check,
    "list_my_schedule": _schedule,
    "find_people": _people,
    "list_my_inbox": _inbox,
}
_WITH_TEXT = {"check_my_capacity": _capacity, "get_my_myte": _my_myte}


def map_output(name: str, args: dict[str, Any], structured: Any, text: str) -> list[Draft]:
    """A successful tool result -> source drafts (possibly none: an empty list is an answer)."""
    if name in _STRUCTURED and structured is not None:
        return _STRUCTURED[name](structured, args)
    if name in _WITH_TEXT:
        return _WITH_TEXT[name](structured, args, text)
    if structured is None and name in _STRUCTURED:
        return []  # a known tool that answered with text only: nothing citable beyond its text
    if text.strip():
        return [Draft(kind="document", title=name.replace("_", " "), text=text, raw=structured)]
    return []


def remap_citations(text: str, mapping: dict[int, int]) -> str:
    """Renumber a tool's own `[k]` markers (ask_meetings' answer) to this answer's `[n]`."""

    def sub(m: re.Match[str]) -> str:
        nums = [mapping.get(int(p.strip())) for p in m.group(1).split(",")]
        kept = [f"[{n}]" for n in nums if n is not None]
        return "".join(kept)

    return _CITE.sub(sub, text)


# --- attachments (ADR-0049 F7) ---------------------------------------------------------------

PASSAGE_CHARS = 1500
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Split text into passages of about `size` characters, at the latest paragraph, line,
    sentence or word break in the second half of each window (a hard cut only when a window has
    none). Whitespace-only passages are dropped."""
    text = text.replace("\r\n", "\n").strip()
    out: list[str] = []
    while text:
        if len(text) <= size:
            out.append(text)
            break
        window = text[:size]
        cut = -1
        for brk in _BREAKS:
            at = window.rfind(brk)
            if at >= size // 2:
                cut = at + len(brk)
                break
        if cut <= 0:
            cut = size
        piece, text = text[:cut].strip(), text[cut:].strip()
        if piece:
            out.append(piece)
    return out


def attachment_drafts(attachments: list[Any]) -> list[Draft]:
    """Each attachment -> numbered `document` drafts, one per passage, the file name as title
    ("name (part i of k)" when it has several). `ref_id` keeps them distinct per file and part."""
    drafts: list[Draft] = []
    for i, att in enumerate(attachments, start=1):
        name = " ".join(str(att.name).split())[:120] or f"Attachment {i}"
        parts = passages(str(att.text))
        for j, part in enumerate(parts, start=1):
            title = name if len(parts) == 1 else f"{name} (part {j} of {len(parts)})"
            drafts.append(
                Draft(
                    kind="document",
                    title=title,
                    text=part,
                    ref_id=f"attachment:{i}:{j}",
                    raw={"attachment": name, "part": j, "parts": len(parts)},
                )
            )
    return drafts
