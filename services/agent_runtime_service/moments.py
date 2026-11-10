"""One card per moment: the moment a result points at, and why the person is seeing it.

Polish program, "One card per moment". One sentence ("I'll send the pricing deck by Friday")
used to produce a note, a promise, a decision card and an action card with nothing linking them.
Now every user-visible item an agent emits carries the same moment fields when it is about the
same words:

- `moment_id`: `uuid5(MOMENT_NS, f"{meeting_id}:{chunk_id}")`, `chunk_id` the final (one
  utterance, `Utterance`) the item quotes. A `transcript-chunks` final is the quote span: it is
  unique, the same for every agent and every window that holds it, and the same on a replay, so
  a note's promise, a decision and an action about one utterance get one id without the agents
  talking to each other. (The window id is not part of the hash on purpose: PersonalListener
  accumulates windows, `runtime.py`, so its window id can differ from the candidate agents' for
  the same words.)
- `quote` (the verbatim words, at most 300 chars), `speaker_id` (a `users.id`, or null for an
  internal identity), `speaker_label` (the meeting-wide "Participant N"; `names.Postprocess`
  swaps it for the display name on the way out), `start_ms` / `end_ms` (meeting time of that
  utterance, the clock of `transcript_chunks.start_ms`): the UI shows the quote, speaker and
  time on the card and opens the live transcript there.
- `why`: one short line. Personal results (a note, a suggestion) say why THIS person sees it
  ("You own billing", "You made a promise here", "Matches your listen-for topic: pricing",
  "Relevant to your role: Designer"). Shared candidates say why the card exists ("Priya took it
  on", "Matches the owned topic: billing", "Agreed by Priya and Raj", "Heard “let's”"),
  "Refines an earlier card" for an update.

Anchoring. A candidate's model-given `quote` (decision_agent@v3 / action_agent@v3) is looked
up in the window's utterances (normalised substring, then word overlap); without one, the
item's own text is matched by content-word overlap. Nothing that matches falls back to the
window's trigger (the utterance that closed it), then its last utterance. A promise is matched
only against the promiser's own utterances.

No model call, no Postgres, no I/O: pure functions over the window and the contexts the runtime
already holds.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID, uuid5

from libs.shared.state import AgentResult, TranscriptWindow, Utterance
from services.agent_runtime_service.agents.company import company_of, owner_topics
from services.agent_runtime_service.names import speaker_labels
from services.agent_runtime_service.text_match import words

MOMENT_NS = UUID("0b6f2a3c-5d4e-5f60-8a71-9c2b3d4e5f61")
MAX_QUOTE = 300
MAX_WHY = 200


def moment_id(meeting_id: UUID, chunk_id: str) -> str:
    return str(uuid5(MOMENT_NS, f"{meeting_id}:{chunk_id}"))


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split())


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def anchor(
    window: TranscriptWindow,
    text: str,
    *,
    quote: str | None = None,
    speaker_id: str | None = None,
) -> tuple[Utterance | None, str | None]:
    """The utterance an item is about, and the verbatim quote when the model gave one that was
    actually said. `speaker_id` restricts the search to that speaker (a promise is the
    promiser's own words) when they spoke in the window."""
    utts = window.utterances
    if speaker_id is not None:
        own = [u for u in utts if u.speaker_id == speaker_id]
        utts = own or utts
    if not utts:
        return None, None
    if quote and quote.strip():
        q = _norm(quote)
        if q:
            for u in utts:
                if q in _norm(u.text):
                    return u, " ".join(quote.split())
    target = words(quote or "") or words(text)
    best: Utterance | None = None
    best_score = 0.0
    for u in utts:
        uw = words(u.text)
        if not uw or not target:
            continue
        score = len(target & uw) / len(target)
        if score > best_score or (score == best_score and score > 0):  # ties: the later one
            best, best_score = u, score
    if best is not None and best_score >= 0.25:
        return best, None
    trig = window.trigger
    if trig is not None and trig.chunk_id:
        for u in utts:
            if u.chunk_id == trig.chunk_id:
                return u, None
    return utts[-1], None


def moment_fields(
    window: TranscriptWindow, u: Utterance | None, quote: str | None
) -> dict[str, Any]:
    if u is None:
        return {}
    label = speaker_labels(window).get(u.speaker_id)
    try:
        speaker: str | None = str(UUID(u.speaker_id))
    except ValueError:
        speaker = None
    return {
        "moment_id": moment_id(window.meeting_id, u.chunk_id),
        "quote": _clip(quote or u.text, MAX_QUOTE),
        "speaker_id": speaker,
        "speaker_label": label,
        "start_ms": int(u.start_ms),
        "end_ms": int(u.end_ms),
    }


# --- why ----------------------------------------------------------------------------------------


def _mentions(blob: str, topic: str) -> bool:
    """A topic is mentioned when the phrase appears, or every content word of it does."""
    if not topic.strip():
        return False
    if _norm(topic) and _norm(topic) in _norm(blob):
        return True
    tw = words(topic)
    return bool(tw) and tw <= words(blob)


def _items(context: dict[str, Any] | None, key: str) -> list[str]:
    raw = (context or {}).get(key) or []
    return [" ".join(v.split()) for v in raw if isinstance(v, str) and v.strip()]


def personal_why(
    blob: str, context: dict[str, Any] | None, user_id: UUID, *, promised: bool = False
) -> str | None:
    """Why this participant sees a note or a suggestion: a topic they own, a promise they made,
    one of their listen-for topics, their role."""
    for owner in owner_topics(company_of(context)):
        if owner.owner_user_id == user_id and _mentions(blob, owner.topic):
            return _clip(f"You own {owner.topic}", MAX_WHY)
    if promised:
        return "You made a promise here"
    for topic in _items(context, "listen_for"):
        if _mentions(blob, topic):
            return _clip(f"Matches your listen-for topic: {topic}", MAX_WHY)
    role = (context or {}).get("role")
    if isinstance(role, str) and role.strip():
        return _clip(f"Relevant to your role: {' '.join(role.split())}", MAX_WHY)
    return None


def _label_of(window: TranscriptWindow, user: object) -> str | None:
    if not isinstance(user, str):
        return None
    return speaker_labels(window).get(user) or window.labels.get(user)


def candidate_why(
    kind: str, cand: dict[str, Any], window: TranscriptWindow, context: dict[str, Any] | None
) -> str | None:
    if cand.get("updates_candidate_id"):
        return "Refines an earlier card"
    blob = f"{cand.get('content') or ''} {cand.get('quote') or ''}"
    if kind == "action_candidate":
        who = _label_of(window, cand.get("assignee_id"))
        if who:
            return f"{who} took it on"
        for owner in owner_topics(company_of(context)):
            if str(owner.owner_user_id) == cand.get("suggested_owner_user_id") and _mentions(
                blob, owner.topic
            ):
                return _clip(f"Matches the owned topic: {owner.topic}", MAX_WHY)
        if cand.get("suggested_owner_user_id"):
            return "Matches a topic with a named owner"
        if cand.get("due_date"):
            return f"Due {cand['due_date']}"
    elif kind == "decision_candidate":
        names = [n for n in (_label_of(window, s) for s in cand.get("stakeholder_ids") or []) if n][
            :3
        ]
        if names:
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
            return _clip(f"Agreed by {joined}", MAX_WHY)
    trig = window.trigger
    if trig is not None and trig.cue:
        return f"Heard “{trig.cue}”"
    return None


# --- annotate one result --------------------------------------------------------------------


def annotate(
    result: AgentResult,
    window: TranscriptWindow,
    user_id: UUID,
    context: dict[str, Any] | None,
) -> AgentResult:
    """The result with moment fields and a why line on every item a person sees. Kinds without
    a quote to point at (context fills, agenda updates, alerts) are returned unchanged."""
    payload = result.payload
    if result.kind == "note":
        commitments = payload.get("commitments")
        first: dict[str, Any] = {}
        if isinstance(commitments, list):
            for c in commitments:
                if not isinstance(c, dict):
                    continue
                u, q = anchor(window, str(c.get("content") or ""), speaker_id=str(user_id))
                fields = moment_fields(window, u, q)
                if fields:
                    c["moment_id"] = fields["moment_id"]
                    first = first or fields
        if not first:
            u, q = anchor(window, result.content)
            first = moment_fields(window, u, q)
        payload.update(first)
        why = personal_why(
            f"{result.content} {first.get('quote', '')}",
            context,
            user_id,
            promised=bool(commitments),
        )
        if why:
            payload["why"] = why
    elif result.kind in ("decision_candidate", "action_candidate"):
        for cand in payload.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            u, q = anchor(window, str(cand.get("content") or ""), quote=cand.get("quote"))
            fields = moment_fields(window, u, q)
            kept = cand.get("moment_id")  # an update keeps its card's original moment
            cand.update(fields)
            if kept:
                cand["moment_id"] = kept
            why = candidate_why(result.kind, cand, window, context)
            if why:
                cand["why"] = why
    elif result.kind == "suggestion":
        for s in payload.get("suggestions") or []:
            if not isinstance(s, dict):
                continue
            blob = f"{s.get('question') or ''} {s.get('rationale') or ''}"
            u, q = anchor(window, blob)
            s.update(moment_fields(window, u, q))
            why = personal_why(f"{blob} {s.get('quote', '')}", context, user_id)
            if why:
                s["why"] = why
    return result
