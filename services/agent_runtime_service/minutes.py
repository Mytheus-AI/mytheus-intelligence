"""Minutes for everyone: after a meeting ends, one model call per participant and one for the
whole meeting (polish program, "Agents that learn and wrap up").

Trigger. `meeting-events` `ended` (core-api's outbox), consumed in its own group
`agent-runtime-minutes` beside the window loop (`run_minutes_loop`). A new group starts at the
end of the topic: replaying every historic meeting on the first deploy would bill a model call
per participant of every old meeting.

Input. core-api's internal `GET /internal/meetings/{id}/minutes-source` (the agent-runtime
caller token, the tenant in `X-Mytheus-Org`): participants by display name, the finals in
spoken order, the decisions and action items recorded during the meeting with their status, and
the agenda. This tier never reads Postgres.

Per meeting end (`MinutesWriter.write`):
1. Never record. The company brain is read strictly (`get_strict`): minutes are durable and
   shareable, so "the brain did not answer" is never treated as "no never-record topics"; the
   run is retried by the loop. Every 30 s window that names a never-record topic, and the
   window after it, is removed from the transcript before any model sees it (the runtime's
   rule, `runtime.py`); the prompt carries the "Never record" line; and every text of the
   answer that names such a topic is dropped before it leaves the tier.
2. Labels. Speakers are "Participant N" in order of first speaking, then the participants who
   never spoke; the model sees no id, name or email (the router also redacts PII). Labels in
   the answer become user ids (owners, askers) and display names after the call.
3. Calls. One shared call ("everyone"; promises always empty) and one call per participant
   (their decisions, open items, unanswered questions and promises), extract tier (Sonnet 5.5)
   through the LiteLLM router, so every call is budget-gated and metered, `minutes@vN` from
   the prompt registry, the meeting record as the cached prefix shared by all N+1 calls.
4. Output. One `agent-results` record per participant (`user_id` = them, scope personal) and
   one shared record (`user_id` = the nil uuid), agent `MeetingAgent`, kind `minutes`,
   `window_id = minutes:{generation}` (one generation per `ended` event, from its idempotency
   key, so a replay republishes the same keys and core-api's inbox dedupes them).
   No transcript: only a shared `unavailable` record (`no_transcript`). Budget blocked or a
   model failure on the shared call: a shared `unavailable` record (`budget` / `model_error`)
   so the follow-up waiting for the minutes is released; a failed personal call is skipped.

Once a generation's records are flushed, `nexus:{org_id}:minutes:{meeting_id}:{generation}` is
set (7 days), so a replayed `ended` (a restart before the commit) does not bill the calls again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import httpx

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import (
    EXTRACT_MAX_TOKENS,
    EXTRACTION_EFFORT,
    BudgetBlocked,
    LiteLLMRouter,
)
from libs.shared.state import AgentResult
from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.agents.company import (
    render_company,
    render_ground_rules,
)

logger = logging.getLogger(__name__)

AGENT = "MeetingAgent"
KIND = "minutes"
PROMPT_KEY = "minutes"
GROUP = "agent-runtime-minutes"
TOPIC = "meeting-events"
NIL = UUID(int=0)
SERVICE_NAME = "agent-runtime-service"
WINDOW_MS = 30_000
MAX_ITEMS = 20
MAX_TEXT = 300
MAX_SUMMARY = 600
MAX_NAME = 120
MAX_DUE = 60
MAX_TRANSCRIPT_CHARS = 160_000  # ~40k tokens: a long meeting; the middle is cut beyond that
CLAIM_TTL_S = 7 * 24 * 3600
_LABEL = re.compile(r"\bParticipant\s*#?\s*(\d+)\b", re.IGNORECASE)
_REF = re.compile(r"^\s*D(\d+)\s*$", re.IGNORECASE)
_DECISION_STATUSES = ("agreed", "tentative", "deferred", "reversed")


# --- the source -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    user_id: UUID
    name: str | None
    attendance: str = "live"
    role: str | None = None


@dataclass(frozen=True)
class Turn:
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Recorded:
    id: UUID
    content: str
    status: str
    assignee_id: UUID | None = None
    due_date: str | None = None


@dataclass
class MinutesSource:
    meeting_id: UUID
    title: str
    confidential: bool
    host_user_id: UUID | None
    participants: list[Person]
    speakers: dict[str, UUID]
    turns: list[Turn]
    decisions: list[Recorded] = field(default_factory=list)
    actions: list[Recorded] = field(default_factory=list)
    agenda: list[tuple[str, str]] = field(default_factory=list)


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _str(raw: object) -> str:
    return " ".join(raw.split()) if isinstance(raw, str) else ""


def parse_source(body: object) -> MinutesSource | None:
    """Tolerant: malformed entries are dropped one by one; no meeting id is None."""
    if not isinstance(body, dict):
        return None
    meeting_id = _uuid(body.get("meeting_id"))
    if meeting_id is None:
        return None
    people: list[Person] = []
    for p in body.get("participants") or []:
        uid = _uuid(p.get("user_id")) if isinstance(p, dict) else None
        if uid is None or any(x.user_id == uid for x in people):
            continue
        assert isinstance(p, dict)
        people.append(
            Person(
                user_id=uid,
                name=_str(p.get("name"))[:MAX_NAME] or None,
                attendance=_str(p.get("attendance")) or "live",
                role=_str(p.get("role_in_meeting")) or None,
            )
        )
    speakers: dict[str, UUID] = {}
    raw_speakers = body.get("speakers")
    for sid, uid in (raw_speakers.items() if isinstance(raw_speakers, dict) else []):
        found = _uuid(uid)
        if found is not None:
            speakers[str(sid)] = found
    turns: list[Turn] = []
    for t in body.get("turns") or []:
        if not isinstance(t, dict):
            continue
        text = _str(t.get("text"))
        start, end = t.get("start_ms"), t.get("end_ms")
        if not text or not isinstance(start, int) or not isinstance(end, int):
            continue
        turns.append(Turn(str(t.get("speaker_id") or ""), start, end, text))

    def recorded(raw: object) -> list[Recorded]:
        out: list[Recorded] = []
        for r in raw if isinstance(raw, list) else []:
            rid = _uuid(r.get("id")) if isinstance(r, dict) else None
            if rid is None:
                continue
            assert isinstance(r, dict)
            content = _str(r.get("content"))
            if content:
                out.append(
                    Recorded(
                        id=rid,
                        content=content[:MAX_TEXT],
                        status=_str(r.get("status")) or "proposed",
                        assignee_id=_uuid(r.get("assignee_id")),
                        due_date=_str(r.get("due_date")) or None,
                    )
                )
        return out

    agenda = [
        (_str(a.get("title"))[:200], _str(a.get("status")) or "pending")
        for a in body.get("agenda") or []
        if isinstance(a, dict) and _str(a.get("title"))
    ]
    return MinutesSource(
        meeting_id=meeting_id,
        title=_str(body.get("title"))[:200] or "Untitled meeting",
        confidential=bool(body.get("confidential")),
        host_user_id=_uuid(body.get("host_user_id")),
        participants=people,
        speakers=speakers,
        turns=turns,
        decisions=recorded(body.get("decisions")),
        actions=recorded(body.get("actions")),
        agenda=agenda,
    )


class SourceUnavailable(Exception):
    """core-api could not answer now (unreachable, 5xx, 409 not ended yet): retry later."""


class MinutesSourceClient(Protocol):
    async def source(self, org_id: UUID, meeting_id: UUID) -> MinutesSource | None: ...


class HttpMinutesSource:
    """`None` for a meeting core-api does not know (404, another tenant); `SourceUnavailable`
    for anything worth retrying."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    async def source(self, org_id: UUID, meeting_id: UUID) -> MinutesSource | None:
        path = f"/internal/meetings/{meeting_id}/minutes-source"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"{path}: {exc}") from None
        if res.status_code == 404:
            return None
        if res.status_code != 200:
            raise SourceUnavailable(f"{path} answered {res.status_code}")
        try:
            return parse_source(res.json())
        except ValueError:
            return None


def source_from_env() -> HttpMinutesSource | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpMinutesSource(url, token)


# --- rendering --------------------------------------------------------------------------------


def label(n: int) -> str:
    return f"Participant {n}"


def assign_labels(source: MinutesSource) -> dict[UUID | str, str]:
    """user id (or an unmapped speaker id) -> "Participant N": speakers in order of first
    speaking, then the participants who never spoke, in the order core-api listed them."""
    out: dict[UUID | str, str] = {}
    for turn in source.turns:
        who: UUID | str = (
            source.speakers.get(turn.speaker_id) or _uuid(turn.speaker_id) or (turn.speaker_id)
        )
        if who not in out:
            out[who] = label(len(out) + 1)
    for p in source.participants:
        if p.user_id not in out:
            out[p.user_id] = label(len(out) + 1)
    return out


def quiet_turns(turns: list[Turn], policy: NeverRecordPolicy) -> list[Turn]:
    """The turns minus every 30 s window that names a never-record topic and the window after
    it (the runtime's rule for live agents)."""
    if policy.empty:
        return list(turns)
    hit_windows = {t.start_ms // WINDOW_MS for t in turns if policy.hit(t.text)}
    banned = hit_windows | {w + 1 for w in hit_windows}
    return [t for t in turns if t.start_ms // WINDOW_MS not in banned]


def _clock(ms: int) -> str:
    s = max(0, ms) // 1000
    if s >= 3600:
        return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def render_transcript(
    source: MinutesSource, turns: list[Turn], labels: dict[UUID | str, str]
) -> str:
    lines: list[str] = []
    for t in turns:
        who: UUID | str = source.speakers.get(t.speaker_id) or _uuid(t.speaker_id) or t.speaker_id
        lines.append(f"[{_clock(t.start_ms)}] {labels.get(who, 'Someone')}: {t.text}")
    text = "\n".join(lines)
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    half = MAX_TRANSCRIPT_CHARS // 2
    return text[:half] + "\n[... the middle of a long meeting is left out ...]\n" + text[-half:]


def render_record(source: MinutesSource, turns: list[Turn], labels: dict[UUID | str, str]) -> str:
    people = []
    for p in source.participants:
        notes = []
        if p.user_id == source.host_user_id:
            notes.append("host")
        if p.attendance == "agent":
            notes.append("sent their agent instead of attending")
        people.append(f"- {labels[p.user_id]}" + (f" ({', '.join(notes)})" if notes else ""))
    decisions = [f"D{n}. {d.content} [{d.status}]" for n, d in enumerate(source.decisions, start=1)]
    actions = []
    for n, a in enumerate(source.actions, start=1):
        extra = [labels[a.assignee_id]] if a.assignee_id in labels else []
        if a.due_date:
            extra.append(f"due {a.due_date}")
        actions.append(
            f"A{n}. {a.content} [{a.status}]" + (f" ({', '.join(extra)})" if extra else "")
        )
    agenda = [f"- {title} [{status}]" for title, status in source.agenda]
    parts = [
        f"Meeting: {source.title}",
        "",
        "Participants:",
        *people,
        "",
        "Recorded decisions:",
        *(decisions or ["(none)"]),
        "",
        "Recorded action items:",
        *(actions or ["(none)"]),
        "",
        "Agenda:",
        *(agenda or ["(none)"]),
        "",
        "Transcript:",
        render_transcript(source, turns, labels),
        "",
    ]
    return "\n".join(parts) + "\n"


def audience(person_label: str | None) -> str:
    if person_label is None:
        return (
            "These minutes are for: everyone in the meeting (the shared minutes). "
            "promises must be []."
        )
    return (
        f"These minutes are for: {person_label} (their personal minutes). promises are only "
        f"what {person_label} committed to do."
    )


# --- parsing the answer -----------------------------------------------------------------------


def _clip(raw: object, limit: int) -> str | None:
    if not isinstance(raw, str):
        return None
    value = " ".join(raw.split())
    if not value:
        return None
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{") :] if "{" in raw else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class Resolver:
    """Labels back to people, and the never-record filter, for one meeting."""

    by_label: dict[str, UUID]
    names: dict[UUID, str]
    policy: NeverRecordPolicy

    @classmethod
    def build(
        cls, source: MinutesSource, labels: dict[UUID | str, str], policy: NeverRecordPolicy
    ) -> Resolver:
        by_label = {lab.lower(): uid for uid, lab in labels.items() if isinstance(uid, UUID)}
        names = {p.user_id: p.name for p in source.participants if p.name}
        return cls(by_label, names, policy)

    def person(self, raw: object) -> UUID | None:
        if not isinstance(raw, str):
            return None
        m = _LABEL.search(raw)
        return self.by_label.get(label(int(m.group(1))).lower()) if m else None

    def text(self, value: str) -> str:
        def swap(m: re.Match[str]) -> str:
            uid = self.by_label.get(label(int(m.group(1))).lower())
            return self.names.get(uid, "someone") if uid else "someone"

        return _LABEL.sub(swap, value)

    def keep(self, value: str | None) -> str | None:
        """The text with names in place of labels, or None when it names a never-record topic."""
        if value is None:
            return None
        named = self.text(value)
        return None if self.policy.hit(named) or self.policy.hit(value) else named


def parse_minutes(
    text: str,
    *,
    resolver: Resolver,
    decisions: list[Recorded],
    personal: bool,
) -> dict[str, Any] | None:
    """The model's answer as the `minutes` payload's content fields, or None when it is not
    the JSON asked for."""
    raw = _json_object(text)
    if raw is None:
        return None

    def entries(key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for e in raw.get(key) or [] if isinstance(raw.get(key), list) else []:
            if not isinstance(e, dict):
                continue
            body = resolver.keep(_clip(e.get("text"), MAX_TEXT))
            if body is None or body.lower() in seen:
                continue
            seen.add(body.lower())
            out.append({**e, "text": body})
            if len(out) == MAX_ITEMS:
                break
        return out

    def decision_id(ref: object) -> str | None:
        m = _REF.match(ref) if isinstance(ref, str) else None
        if not m:
            return None
        n = int(m.group(1))
        return str(decisions[n - 1].id) if 1 <= n <= len(decisions) else None

    def owner(raw_owner: object) -> tuple[str | None, str | None]:
        uid = resolver.person(raw_owner)
        if uid is None:
            return None, None
        return str(uid), resolver.names.get(uid)

    summary = resolver.keep(_clip(raw.get("summary"), MAX_SUMMARY)) or ""
    out: dict[str, Any] = {
        "summary": summary,
        "decisions": [
            {
                "text": d["text"],
                "status": d.get("status") if d.get("status") in _DECISION_STATUSES else "agreed",
                "decision_id": decision_id(d.get("ref")),
            }
            for d in entries("decisions")
        ],
        "open_items": [],
        "unanswered_questions": [],
        "promises": [],
    }
    for o in entries("open_items"):
        uid, name = owner(o.get("owner"))
        out["open_items"].append(
            {
                "text": o["text"],
                "owner_user_id": uid,
                "owner_name": name,
                "due": resolver.keep(_clip(o.get("due"), MAX_DUE)),
            }
        )
    for q in entries("unanswered_questions"):
        uid, name = owner(q.get("asked_by"))
        out["unanswered_questions"].append(
            {"text": q["text"], "asked_by_user_id": uid, "asked_by_name": name}
        )
    if personal:
        for p in entries("promises"):
            to_uid = resolver.person(p.get("to"))
            out["promises"].append(
                {
                    "text": p["text"],
                    "to_name": resolver.names.get(to_uid) if to_uid else None,
                    "due": resolver.keep(_clip(p.get("due"), MAX_DUE)),
                }
            )
    return out


EMPTY_CONTENT: dict[str, Any] = {
    "summary": "",
    "decisions": [],
    "open_items": [],
    "unanswered_questions": [],
    "promises": [],
}


def has_content(content: dict[str, Any]) -> bool:
    return bool(content["summary"]) or any(
        content[k] for k in ("decisions", "open_items", "unanswered_questions", "promises")
    )


# --- the writer -------------------------------------------------------------------------------

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


@dataclass
class Written:
    """One record to emit: `user_id` None is the shared minutes."""

    user_id: UUID | None
    payload: dict[str, Any]

    def result(self, generation: str) -> AgentResult:
        return AgentResult(
            kind=KIND,
            content=str(self.payload.get("summary") or ""),
            relevance_score=1.0,
            confidence=1.0,
            source_chunk_ids=[],
            payload=self.payload,
            window_id=f"minutes:{generation}",
        )


def generation_of(idempotency_key: str, meeting_id: UUID) -> str:
    return hashlib.sha256(f"{meeting_id}:{idempotency_key}".encode()).hexdigest()[:16]


class MinutesWriter:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        source: MinutesSourceClient,
        org_context: OrgContextFn | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.loader = loader
        self.router = router
        self.source = source
        self.org_context = org_context
        self._sem = asyncio.Semaphore(max_concurrency)

    def _payload(
        self, scope: str, prompt_ref: str, generation: str, content: dict[str, Any], **extra: Any
    ) -> dict[str, Any]:
        status = extra.pop("status", None) or ("ready" if has_content(content) else "empty")
        return {
            "scope": scope,
            "prompt_version": prompt_ref,
            "generation": generation,
            "status": status,
            "reason": extra.pop("reason", None),
            **content,
            "promises": content["promises"] if scope == "personal" else [],
        }

    async def write(self, org_id: UUID, meeting_id: UUID, generation: str) -> list[Written]:
        """Everything to emit for one meeting end (possibly nothing). Raises `SourceUnavailable`
        or `OrgContextUnavailable` when the run should be retried."""
        source = await self.source.source(org_id, meeting_id)
        if source is None:
            logger.info("minutes: meeting %s unknown to core-api; nothing written", meeting_id)
            return []
        org = await self.org_context(org_id) if self.org_context is not None else None
        policy = NeverRecordPolicy.from_org(org)
        prompt = self.loader.load(PROMPT_KEY, self.loader.resolve(PROMPT_KEY, org_id))
        turns = quiet_turns(source.turns, policy)
        if not turns:
            reason = "no_transcript"
            return [
                Written(
                    None,
                    self._payload(
                        "shared",
                        prompt.ref,
                        generation,
                        dict(EMPTY_CONTENT),
                        status="unavailable",
                        reason=reason,
                    ),
                )
            ]
        labels = assign_labels(source)
        resolver = Resolver.build(source, labels, policy)
        company = render_company(org) if org is not None else ""
        rules = render_ground_rules(org) if org is not None else ""
        record = render_record(source, turns, labels)

        async def call(person: Person | None) -> dict[str, Any] | None:
            values = {
                "company": company,
                "ground_rules": rules,
                "meeting_record": record,
                "audience": audience(labels[person.user_id] if person else None),
            }
            async with self._sem:
                out = await self.router.complete(
                    tier="extract",
                    system=prompt.system,
                    org_id=org_id,
                    user_id=person.user_id if person else None,
                    max_tokens=EXTRACT_MAX_TOKENS,
                    prompt_ref=prompt.ref,
                    effort=EXTRACTION_EFFORT,
                    **prompt.user_parts(**values),
                )
            return parse_minutes(
                out.text,
                resolver=resolver,
                decisions=source.decisions,
                personal=person is not None,
            )

        everyone: list[Person | None] = [None, *source.participants]
        answers = await asyncio.gather(*(call(p) for p in everyone), return_exceptions=True)
        written: list[Written] = []
        for person, answer in zip(everyone, answers, strict=True):
            if isinstance(answer, BaseException):
                if isinstance(answer, asyncio.CancelledError):
                    raise answer
                reason = "budget" if isinstance(answer, BudgetBlocked) else "model_error"
                if not isinstance(answer, BudgetBlocked):
                    logger.error(
                        "minutes call failed meeting=%s for=%s: %r",
                        meeting_id,
                        person.user_id if person else "shared",
                        answer,
                    )
                if person is None:
                    written.append(
                        Written(
                            None,
                            self._payload(
                                "shared",
                                prompt.ref,
                                generation,
                                dict(EMPTY_CONTENT),
                                status="unavailable",
                                reason=reason,
                            ),
                        )
                    )
                continue
            if answer is None:
                logger.warning("minutes answer unparseable meeting=%s", meeting_id)
                if person is None:
                    written.append(
                        Written(
                            None,
                            self._payload(
                                "shared",
                                prompt.ref,
                                generation,
                                dict(EMPTY_CONTENT),
                                status="unavailable",
                                reason="model_error",
                            ),
                        )
                    )
                continue
            scope = "shared" if person is None else "personal"
            written.append(
                Written(
                    person.user_id if person else None,
                    self._payload(scope, prompt.ref, generation, answer),
                )
            )
        return written


# --- the loop ---------------------------------------------------------------------------------


class ResultSink(Protocol):
    async def agent_result(
        self,
        *,
        org_id: UUID,
        meeting_id: UUID,
        window_id: str,
        user_id: UUID,
        agent: str,
        result: AgentResult,
    ) -> None: ...

    def flush(self, timeout_s: float = 10.0) -> None: ...


def claim_key(org_id: UUID, meeting_id: UUID, generation: str) -> str:
    return f"nexus:{org_id}:minutes:{meeting_id}:{generation}"


@dataclass
class EndedMeeting:
    org_id: UUID
    meeting_id: UUID
    generation: str


def parse_ended(value: dict[str, Any]) -> EndedMeeting | None:
    if str(value.get("event_type")) != "ended":
        return None
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    meeting_id = _uuid(value.get("meeting_id"))
    key = str(header.get("idempotency_key") or "")
    if org_id is None or meeting_id is None or not key:
        return None
    return EndedMeeting(org_id, meeting_id, generation_of(key, meeting_id))


class MinutesLoop:
    """The testable core: one decoded `meeting-events` record in, minutes out, flushed.

    `settle_s`: the transcript's last finals can land a few seconds after the meeting ends, so
    the writer waits that long before reading it. `attempts`: a retryable failure (core-api or
    the company brain unavailable) is retried with backoff; after the last attempt a shared
    `unavailable` record is emitted so nothing waits on these minutes forever."""

    def __init__(
        self,
        writer: MinutesWriter,
        sink: ResultSink,
        *,
        claims: Any = None,  # redis.asyncio: `exists` and `set(..., ex=)`
        settle_s: float = 15.0,
        attempts: int = 3,
        backoff_s: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.writer = writer
        self.sink = sink
        self.claims = claims
        self.settle_s = settle_s
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self._sleep = sleep

    async def _claimed(self, key: str) -> bool:
        if self.claims is None:
            return False
        try:
            return bool(await self.claims.exists(key))
        except Exception:  # noqa: BLE001 - no claim store: write (core-api dedupes the keys)
            return False

    async def handle(self, value: dict[str, Any]) -> int:
        ended = parse_ended(value)
        if ended is None:
            return 0
        key = claim_key(ended.org_id, ended.meeting_id, ended.generation)
        if await self._claimed(key):
            return 0
        if self.settle_s > 0:
            await self._sleep(self.settle_s)
        written: list[Written] | None = None
        for attempt in range(self.attempts):
            try:
                written = await self.writer.write(ended.org_id, ended.meeting_id, ended.generation)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then given up loudly
                logger.warning(
                    "minutes for %s failed (attempt %d): %r", ended.meeting_id, attempt + 1, exc
                )
                if attempt + 1 < self.attempts:
                    await self._sleep(self.backoff_s * (2**attempt))
        if written is None:
            written = [
                Written(
                    None,
                    {
                        "scope": "shared",
                        "prompt_version": f"{PROMPT_KEY}@unknown",
                        "generation": ended.generation,
                        "status": "unavailable",
                        "reason": "model_error",
                        **EMPTY_CONTENT,
                    },
                )
            ]
        for w in written:
            await self.sink.agent_result(
                org_id=ended.org_id,
                meeting_id=ended.meeting_id,
                window_id=f"minutes:{ended.generation}",
                user_id=w.user_id or NIL,
                agent=AGENT,
                result=w.result(ended.generation),
            )
        if written:
            self.sink.flush()  # raises EmitError: the caller must not commit
        if self.claims is not None:
            try:
                await self.claims.set(key, "1", ex=CLAIM_TTL_S)
            except Exception:  # noqa: BLE001 - a replay re-bills at worst; core-api dedupes
                logger.warning("minutes claim not stored for %s", ended.meeting_id)
        return len(written)


async def run_minutes_loop(
    consumer: Any, codec: Any, loop: MinutesLoop, stop: asyncio.Event
) -> None:  # the confluent-kafka harness, like the proactive loop's
    from confluent_kafka import TopicPartition

    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, 1.0)
        if msg is None or msg.error():
            continue
        topic, partition, offset, raw = msg.topic(), msg.partition(), msg.offset(), msg.value()
        if topic is None or partition is None or offset is None:
            continue
        if raw is not None:
            try:
                value = codec.deserialize_for(topic, raw)
            except Exception:  # noqa: BLE001
                logger.exception("undecodable message on %s@%s; skipped", topic, offset)
            else:
                await loop.handle(value)  # EmitError propagates: no commit
        await asyncio.to_thread(
            consumer.commit,
            offsets=[TopicPartition(topic, partition, offset + 1)],
            asynchronous=False,
        )


def consumer_config(brokers: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": GROUP,
        "enable.auto.commit": False,
        # A new group starts at the end: no minutes (and no model calls) for historic meetings.
        "auto.offset.reset": "latest",
        # One meeting's minutes take a minute or two of model time: keep the group alive.
        "max.poll.interval.ms": 900_000,
    }
