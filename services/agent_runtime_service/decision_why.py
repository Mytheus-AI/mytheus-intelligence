"""Decisions that know why, and the assumption watcher (a4 contract sections 1 and 2).

Two jobs, both records of `ProactiveAgent` (kind `alert`, `surfaced: false`, so
notification-service ignores them and core-api persists them and notifies the owner itself):

1. `decision_why`, one per confirmed decision. `ProactiveLoop.handle` calls `on_decision` after
   the conflict check, on the same producer and flush (`proactive/loop.py`). Steps:
   a. core-api `GET /internal/decisions/{id}/why-source`: the decision with its rationale and
      moment quote, the transcript around the moment, up to 40 open actions from other meetings,
      and `already_read` (a `decision_why` was already applied: nothing to do, no model call).
   b. `decision_assumptions@vN` (extract tier): up to 5 stated reasons that must stay true, each
      with a kind, a statement, the verbatim quote, the speaker, a confidence and watch fields.
   c. `decision_ripple@vN` (extract tier): which active company goals (`G1..Gn`) the decision
      advances or risks and which candidate actions (`A1..An`, customer ones marked) it changes,
      blocks or unblocks. Plus the KG `actions_blocked_on_decision` actions (R1 deadline, KG down
      -> none) as `source: "graph"`, `effect: "changes"`, deduped by id.
   d. One record, contract 1a, `user_id` = `confirmed_by` (none without one). Items under
      confidence 0.5 are dropped. Empty lists are still sent: core-api marks the decision read.

2. `assumption_broken`, the watcher's LLM path, `assumption_check@vN` (extract tier): numbered open
   assumptions (`H1..Hn`) against NEW evidence; only clear contradictions come back, each with a
   verbatim evidence quote (checked against the evidence: a quote that is not in it is dropped),
   the speaker, a rationale and a confidence. Breaks of 0.6 or more are sent (core-api applies
   0.7 or more). Two triggers:
   A. the same confirmed-decision hook: evidence = the new decision, its rationale and quote;
      the open assumptions exclude this decision and its meeting. Window
      `assumption-check:decision:{decision_id}`.
   B. `meeting-events` `ended`, own consumer group `agent-runtime-assumptions` (`latest`, like
      minutes), its own loop task (`AssumptionWatchLoop`): evidence = the ended meeting's
      transcript from core-api's minutes source. Window
      `assumption-check:meeting:{meeting_id}:{generation}`; once flushed, the claim
      `nexus:{org_id}:assumption-check:{meeting_id}:{generation}` (7 days) stops a replay from
      billing again.

Gating, before any work: the job's flag (`DECISION_WHY_AGENT`, `ASSUMPTION_WATCH_AGENT`, default
on), the plan feature `proactive_agent` (`PROACTIVE_AGENT_ALL_PLANS=1` overrides it, as for
`ProactiveAgent`), and the tenant budget state `ok`. Every model call goes through
`LiteLLMRouter.complete` (budget gate, metering, PII redaction) with `org_id` set, `user_id`
None (a background check, metered to the tenant) and its `prompt_ref`. `BudgetBlocked` skips
silently.

Privacy and never record:
- The company brain is read strictly: these records are durable, so "the brain did not answer"
  never counts as "no never-record topics". Trigger A then skips; trigger B retries.
- A decision naming a never-record topic (outright, or judged so by the triage check on an
  ambiguous word) gets neither record. Every 30 s window of a transcript that names such a topic,
  and the window after it, is removed before a model sees it (`minutes.quiet_turns`). Every text
  of an answer that names one is dropped. Open assumptions and candidate actions that name one
  are never shown to a model.
- The model sees "Participant N" labels, never a user id, name or email. Labels in answers
  become user ids after the call; a label left in an assumption statement becomes the person's
  display name (the statement is read by the decision's own people), in a break's rationale it
  becomes "someone" (the owner may not have been in the evidence meeting).
- A confidential meeting is never evidence: trigger B skips it, and a decision from one is not
  checked against other decisions' assumptions (trigger A). Its own `decision_why` is still
  written: it stays with the decision.
- URLs are stripped from every text, and every text is clipped to the contract's lengths.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol
from uuid import UUID, uuid5

import httpx

from libs.shared.index_guard import RouterClassifier
from libs.shared.json_answer import json_object
from libs.shared.kg_client import KgClient
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, Goal
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.router import (
    EXTRACT_MAX_TOKENS,
    EXTRACTION_EFFORT,
    SYNTHESIS_EFFORT,
    BudgetBlocked,
    Effort,
    LiteLLMRouter,
)
from libs.shared.state import AgentResult
from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS, confidence_of
from services.agent_runtime_service.agents.company import (
    render_company,
    render_ground_rules,
)
from services.agent_runtime_service.agents.proactive_agent import (
    ConfirmedDecision,
    clip,
    display_name,
    strip_urls,
)
from services.agent_runtime_service.minutes import (
    MinutesSource,
    MinutesSourceClient,
    Person,
    SourceUnavailable,
    Turn,
    generation_of,
    quiet_turns,
)

logger = logging.getLogger(__name__)

AGENT = "ProactiveAgent"
FEATURE = "proactive_agent"
ASSUMPTIONS_KEY = "decision_assumptions"
RIPPLE_KEY = "decision_ripple"
CHECK_KEY = "assumption_check"
GROUP = "agent-runtime-assumptions"
TOPIC = "meeting-events"
SERVICE_NAME = "agent-runtime-service"
MODEL_TIER = "extract"  # the contract names the tier, not the model

MAX_ASSUMPTIONS = 5
MAX_GOALS = 5
MAX_ACTIONS = 12
MAX_GOALS_SHOWN = 30  # A4: at most 30 active goals
MAX_CANDIDATE_ACTIONS = 40
MAX_OPEN = 60
MIN_CONFIDENCE = 0.5  # ripple and assumptions
BREAK_MIN_CONFIDENCE = 0.6  # core-api applies >= 0.7
GRAPH_CONFIDENCE = 0.7  # a still-open action from the decision's own meeting (structural)
MAX_STATEMENT = 300
MAX_QUOTE = 600
MAX_WHY = 200
MAX_RATIONALE = 300
MAX_LABEL = 300
MAX_UNIT = 16
MAX_EXCERPT_CHARS = 24_000  # the stretch around a decision (core-api sends ~3.5 minutes)
MAX_EVIDENCE_CHARS = 60_000  # ~15k tokens of an ended meeting; the middle is cut beyond that
DEFAULT_KG_TIMEOUT_MS = 800
CLAIM_TTL_S = 7 * 24 * 3600

KINDS = ("price", "date", "metric", "dependency", "person", "other")
COMPARATORS = ("lt", "lte", "gt", "gte", "eq")
GOAL_EFFECTS = ("advances", "risks")
ACTION_EFFECTS = ("changes", "blocks", "unblocks")
_LABEL = re.compile(r"\bParticipant\s*#?\s*(\d+)\b", re.IGNORECASE)
_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9]{0,9}-\d{1,7}$")
_REF = re.compile(r"^\s*([A-Za-z])?\s*(\d+)\s*$")
_ELLIPSIS = re.compile(r"\.\.\.|…")
_NON_WORD = re.compile(r"[^a-z0-9]+")


# --- settings and dependencies ----------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class WhySettings:
    why_enabled: bool = True  # DECISION_WHY_AGENT
    watch_enabled: bool = True  # ASSUMPTION_WATCH_AGENT: both watcher triggers
    all_plans: bool = False  # PROACTIVE_AGENT_ALL_PLANS
    kg_timeout_ms: int = DEFAULT_KG_TIMEOUT_MS  # KG_TIMEOUT_MS
    settle_s: float = 15.0  # ASSUMPTION_WATCH_SETTLE_S: the last finals of an ended meeting

    @classmethod
    def from_env(cls) -> WhySettings:
        return cls(
            why_enabled=_env_flag("DECISION_WHY_AGENT", True),
            watch_enabled=_env_flag("ASSUMPTION_WATCH_AGENT", True),
            all_plans=_env_flag("PROACTIVE_AGENT_ALL_PLANS", False),
            kg_timeout_ms=max(1, int(_env_float("KG_TIMEOUT_MS", DEFAULT_KG_TIMEOUT_MS))),
            settle_s=max(0.0, _env_float("ASSUMPTION_WATCH_SETTLE_S", 15.0)),
        )


class PlanFeatures(Protocol):
    async def enabled(self, org_id: UUID, feature: str) -> bool: ...


OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


# --- core-api reads (contract section 2) -------------------------------------------------------


@dataclass(frozen=True)
class CandidateAction:
    id: UUID
    content: str
    due_date: str | None = None
    meeting_title: str | None = None
    customer: bool = False


@dataclass
class WhySource:
    decision_id: UUID
    meeting_id: UUID
    content: str
    rationale: str
    quote: str
    quote_speaker_id: str | None
    confirmed_by: UUID | None
    confidential: bool
    status: str
    meeting_title: str
    participants: list[Person]
    speakers: dict[str, UUID]
    turns: list[Turn]
    candidate_actions: list[CandidateAction] = field(default_factory=list)
    already_read: bool = False


@dataclass(frozen=True)
class OpenAssumption:
    id: UUID
    decision_id: UUID
    decision_content: str
    kind: str
    statement: str
    quote: str | None
    meeting_id: UUID | None


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _str(raw: object) -> str:
    return " ".join(raw.split()) if isinstance(raw, str) else ""


def _people(raw: object) -> list[Person]:
    out: list[Person] = []
    for p in raw if isinstance(raw, list) else []:
        uid = _uuid(p.get("user_id")) if isinstance(p, dict) else None
        if uid is None or any(x.user_id == uid for x in out):
            continue
        assert isinstance(p, dict)
        out.append(Person(user_id=uid, name=_str(p.get("name"))[:120] or None))
    return out


def _speakers(raw: object) -> dict[str, UUID]:
    out: dict[str, UUID] = {}
    for sid, uid in raw.items() if isinstance(raw, dict) else []:
        found = _uuid(uid)
        if found is not None:
            out[str(sid)] = found
    return out


def _turns(raw: object) -> list[Turn]:
    out: list[Turn] = []
    for t in raw if isinstance(raw, list) else []:
        if not isinstance(t, dict):
            continue
        text = _str(t.get("text"))
        start, end = t.get("start_ms"), t.get("end_ms")
        if not text or isinstance(start, bool) or not isinstance(start, int):
            continue
        end = end if isinstance(end, int) and not isinstance(end, bool) else start
        out.append(Turn(str(t.get("speaker_id") or ""), start, end, text))
    return out


def parse_why_source(body: object) -> WhySource | None:
    """Tolerant: malformed entries are dropped one by one; no decision is None."""
    if not isinstance(body, dict) or not isinstance(body.get("decision"), dict):
        return None
    d = body["decision"]
    decision_id, meeting_id = _uuid(d.get("id")), _uuid(d.get("meeting_id"))
    content = _str(d.get("content"))
    if decision_id is None or meeting_id is None or not content:
        return None
    actions: list[CandidateAction] = []
    for a in body.get("candidate_actions") or []:
        aid = _uuid(a.get("id")) if isinstance(a, dict) else None
        if aid is None or any(x.id == aid for x in actions):
            continue
        assert isinstance(a, dict)
        text = _str(a.get("content"))
        if not text:
            continue
        actions.append(
            CandidateAction(
                id=aid,
                content=text[:MAX_LABEL],
                due_date=_str(a.get("due_date"))[:10] or None,
                meeting_title=_str(a.get("meeting_title"))[:200] or None,
                customer=a.get("customer") is True,
            )
        )
    speaker = d.get("quote_speaker_id")
    return WhySource(
        decision_id=decision_id,
        meeting_id=meeting_id,
        content=content,
        rationale=_str(d.get("rationale")),
        quote=_str(d.get("quote")),
        quote_speaker_id=str(speaker) if isinstance(speaker, str) and speaker else None,
        confirmed_by=_uuid(d.get("confirmed_by")),
        # unknown counts as confidential: it is then never used as evidence elsewhere
        confidential=d.get("confidential") is not False,
        status=_str(d.get("status")) or "confirmed",
        meeting_title=_str(body.get("meeting_title"))[:200],
        participants=_people(body.get("participants")),
        speakers=_speakers(body.get("speakers")),
        turns=_turns(body.get("turns")),
        candidate_actions=actions[:MAX_CANDIDATE_ACTIONS],
        already_read=body.get("already_read") is True,
    )


def parse_open(body: object) -> list[OpenAssumption]:
    out: list[OpenAssumption] = []
    items = body.get("items") if isinstance(body, dict) else None
    for a in items if isinstance(items, list) else []:
        if not isinstance(a, dict):
            continue
        aid, did = _uuid(a.get("id")), _uuid(a.get("decision_id"))
        statement = _str(a.get("statement"))
        if aid is None or did is None or not statement or any(x.id == aid for x in out):
            continue
        kind = _str(a.get("kind"))
        out.append(
            OpenAssumption(
                id=aid,
                decision_id=did,
                decision_content=_str(a.get("decision_content")),
                kind=kind if kind in KINDS else "other",
                statement=statement,
                quote=_str(a.get("quote")) or None,
                meeting_id=_uuid(a.get("meeting_id")),
            )
        )
    return out[:MAX_OPEN]


class WhyCore(Protocol):
    async def why_source(self, org_id: UUID, decision_id: UUID) -> WhySource | None: ...

    async def open_assumptions(
        self,
        org_id: UUID,
        *,
        exclude_meeting_id: UUID | None = None,
        exclude_decision_id: UUID | None = None,
    ) -> list[OpenAssumption]: ...


class HttpWhyCore:
    """The two internal reads. `why_source`: None for 404 (unknown, another tenant), 409 (not
    confirmed) or any failure (the caller skips). `open_assumptions`: raises `SourceUnavailable`
    when worth retrying (unreachable, 5xx), else the parsed items."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    async def _get(self, org_id: UUID, path: str, params: dict[str, str]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
            return await c.get(
                f"{self._url}{path}",
                params=params,
                headers={
                    "X-Mytheus-Service": SERVICE_NAME,
                    "X-Mytheus-Service-Token": self._token,
                    "X-Mytheus-Org": str(org_id),
                },
            )

    async def why_source(self, org_id: UUID, decision_id: UUID) -> WhySource | None:
        path = f"/internal/decisions/{decision_id}/why-source"
        try:
            res = await self._get(org_id, path, {})
        except httpx.HTTPError as exc:
            logger.warning("core-api %s unreachable: %s (why skipped)", path, exc)
            return None
        if res.status_code != 200:
            logger.info("core-api %s answered %s (why skipped)", path, res.status_code)
            return None
        try:
            return parse_why_source(res.json())
        except ValueError:
            return None

    async def open_assumptions(
        self,
        org_id: UUID,
        *,
        exclude_meeting_id: UUID | None = None,
        exclude_decision_id: UUID | None = None,
    ) -> list[OpenAssumption]:
        path = "/internal/assumptions/open"
        params = {}
        if exclude_meeting_id is not None:
            params["exclude_meeting_id"] = str(exclude_meeting_id)
        if exclude_decision_id is not None:
            params["exclude_decision_id"] = str(exclude_decision_id)
        try:
            res = await self._get(org_id, path, params)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"{path}: {exc}") from None
        if res.status_code >= 500:
            raise SourceUnavailable(f"{path} answered {res.status_code}")
        if res.status_code != 200:
            return []
        try:
            return parse_open(res.json())
        except ValueError:
            return []


def why_core_from_env() -> HttpWhyCore | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpWhyCore(url, token)


@dataclass
class WhyDeps:
    """No `core`: nothing runs (no source). No `minutes`: trigger B is off. No `kg`: no graph
    actions in the ripple. No `plans`: suspended unless `PROACTIVE_AGENT_ALL_PLANS=1`. No
    `org_context`: no company block and no never-record topics (tests only; production wires
    the strict client)."""

    core: WhyCore | None = None
    minutes: MinutesSourceClient | None = None
    kg: KgClient | None = None
    plans: PlanFeatures | None = None
    org_context: OrgContextFn | None = None


# --- labels and text --------------------------------------------------------------------------


def _label(n: int) -> str:
    return f"Participant {n}"


@dataclass
class Labels:
    """Speakers as "Participant N" in order of first speaking, then the participants who never
    spoke; and the way back."""

    speakers: dict[str, UUID]
    of: dict[UUID | str, str]
    names: dict[UUID, str]

    @classmethod
    def build(
        cls, turns: list[Turn], speakers: dict[str, UUID], participants: list[Person]
    ) -> Labels:
        of: dict[UUID | str, str] = {}
        for t in turns:
            who = cls._who(speakers, t.speaker_id)
            if who not in of:
                of[who] = _label(len(of) + 1)
        for p in participants:
            if p.user_id not in of:
                of[p.user_id] = _label(len(of) + 1)
        names = {p.user_id: display_name(p.name) for p in participants}
        return cls(speakers, of, names)

    @staticmethod
    def _who(speakers: dict[str, UUID], speaker_id: str) -> UUID | str:
        return speakers.get(speaker_id) or _uuid(speaker_id) or speaker_id

    def of_speaker(self, speaker_id: str | None) -> str | None:
        if not speaker_id:
            return None
        return self.of.get(self._who(self.speakers, speaker_id))

    def person(self, raw: object) -> UUID | None:
        if not isinstance(raw, str):
            return None
        m = _LABEL.search(raw)
        if not m:
            return None
        wanted = _label(int(m.group(1)))
        for who, lab in self.of.items():
            if lab == wanted and isinstance(who, UUID):
                return who
        return None

    def swap(self, text: str, *, named: bool) -> str:
        """Labels left in a text become display names (`named`) or "someone"."""

        def one(m: re.Match[str]) -> str:
            uid = self.person(m.group(0))
            if named and uid is not None:
                return self.names.get(uid, "someone")
            return "someone"

        return _LABEL.sub(one, text)


def _clock(ms: int) -> str:
    s = max(0, ms) // 1000
    if s >= 3600:
        return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def render_turns(turns: list[Turn], labels: Labels, limit: int) -> str:
    lines = [
        f"[{_clock(t.start_ms)}] {labels.of_speaker(t.speaker_id) or 'Someone'}: {t.text}"
        for t in turns
    ]
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[... the middle of a long meeting is left out ...]\n" + text[-half:]


def _norm(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def verbatim(quote: str, haystack: str) -> bool:
    """Every part of `quote` (split on an ellipsis) is in `haystack`, ignoring case, punctuation
    and spacing. `haystack` is already `_norm`ed."""
    parts = [_norm(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in haystack for p in parts)


def _text(raw: object, limit: int) -> str | None:
    if not isinstance(raw, str):
        return None
    flat = " ".join(strip_urls(raw).split()).strip().strip('"“”').strip()
    if not flat:
        return None
    return clip(flat, limit)


def _confidence(raw: dict[str, Any]) -> float | None:
    value = confidence_of(raw)
    return None if value is None else round(value, 3)


def resolve_ref(raw: object, prefix: str, n: int) -> int | None:
    """0-based index for "G2" / "g 2" / 2 (labels outside 1..n are None)."""
    if isinstance(raw, bool):
        return None
    number: int | None = None
    if isinstance(raw, int):
        number = raw
    elif isinstance(raw, str):
        m = _REF.match(raw)
        if m and (m.group(1) is None or m.group(1).upper() == prefix):
            number = int(m.group(2))
    if number is None or not 1 <= number <= n:
        return None
    return number - 1


# --- parsing the answers ----------------------------------------------------------------------


def parse_watch(raw: object, sources: str) -> dict[str, Any]:
    """The watch fields, each validated; anything unusable is null. `sources` (normalized) must
    hold an issue key literally."""
    w = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {
        "date": None,
        "value": None,
        "comparator": None,
        "unit": None,
        "issue_key": None,
    }
    d = w.get("date")
    if isinstance(d, str):
        try:
            out["date"] = date.fromisoformat(d.strip()[:10]).isoformat()
        except ValueError:
            pass
    v = w.get("value")
    if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v):
        out["value"] = float(v)
        c = w.get("comparator")
        out["comparator"] = c if c in COMPARATORS else None
    u = w.get("unit")
    if isinstance(u, str) and u.strip():
        out["unit"] = u.strip()[:MAX_UNIT]
    k = w.get("issue_key")
    if isinstance(k, str):
        key = k.strip().upper()
        if _ISSUE_KEY.match(key) and _norm(key) in sources:
            out["issue_key"] = key
    return out


def parse_assumptions(
    text: str, *, labels: Labels, sources: str, policy: NeverRecordPolicy
) -> list[dict[str, Any]] | None:
    """The contract's assumption entries in the model's order, at most 5; None when the answer is
    not the JSON asked for. `sources` is the normalized text the quotes must come from."""
    obj = json_object(text, "assumptions")
    if obj is None or not isinstance(obj.get("assumptions"), list):
        return None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in obj["assumptions"]:
        if not isinstance(raw, dict):
            continue
        confidence = _confidence(raw)
        statement = _text(raw.get("statement"), MAX_STATEMENT)
        if confidence is None or confidence < MIN_CONFIDENCE or statement is None:
            continue
        statement = labels.swap(statement, named=True)
        quote = _text(raw.get("quote"), MAX_QUOTE)
        if quote is not None and not verbatim(quote, sources):
            quote = None  # not the people's words: never shown as a quote
        if policy.hits(statement, quote) or _norm(statement) in seen:
            continue
        seen.add(_norm(statement))
        speaker = labels.person(raw.get("speaker")) if quote is not None else None
        kind = raw.get("kind")
        out.append(
            {
                "kind": kind if kind in KINDS else "other",
                "statement": statement,
                "quote": quote,
                "speaker_user_id": str(speaker) if speaker else None,
                "confidence": confidence,
                "watch": parse_watch(raw.get("watch"), sources),
            }
        )
        if len(out) == MAX_ASSUMPTIONS:
            break
    return out


def parse_ripple(
    text: str,
    *,
    goals: list[Goal],
    actions: list[CandidateAction],
    policy: NeverRecordPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """(goals, actions) entries, strongest first, capped; None when unparseable."""
    obj = json_object(text, "goals") or json_object(text, "actions")
    if obj is None:
        return None

    def entries(key: str) -> list[dict[str, Any]]:
        value = obj.get(key) if obj is not None else None
        return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []

    goal_out: dict[int, dict[str, Any]] = {}
    for raw in entries("goals"):
        i = resolve_ref(raw.get("goal"), "G", len(goals))
        confidence = _confidence(raw)
        effect = raw.get("effect")
        if i is None or confidence is None or confidence < MIN_CONFIDENCE:
            continue
        if effect not in GOAL_EFFECTS or i in goal_out:
            continue
        why = _text(raw.get("why"), MAX_WHY) or ""
        if policy.hits(goals[i].title, why):
            continue
        goal_out[i] = {
            "goal_id": str(goals[i].id),
            "label": clip(goals[i].title, MAX_LABEL),
            "effect": effect,
            "why": why,
            "confidence": confidence,
        }
    action_out: dict[int, dict[str, Any]] = {}
    for raw in entries("actions"):
        i = resolve_ref(raw.get("action"), "A", len(actions))
        confidence = _confidence(raw)
        effect = raw.get("effect")
        if i is None or confidence is None or confidence < MIN_CONFIDENCE:
            continue
        if effect not in ACTION_EFFECTS or i in action_out:
            continue
        why = _text(raw.get("why"), MAX_WHY) or ""
        if policy.hits(actions[i].content, why):
            continue
        action_out[i] = {
            "action_id": str(actions[i].id),
            "label": clip(strip_urls(actions[i].content), MAX_LABEL),
            "effect": effect,
            "why": why,
            "confidence": confidence,
            "source": "agent",
        }

    def ranked(items: dict[int, dict[str, Any]], cap: int) -> list[dict[str, Any]]:
        return sorted(items.values(), key=lambda e: -float(e["confidence"]))[:cap]

    return ranked(goal_out, MAX_GOALS), ranked(action_out, MAX_ACTIONS)


def parse_breaks(
    text: str,
    *,
    open_items: list[OpenAssumption],
    labels: Labels,
    evidence: str,
    policy: NeverRecordPolicy,
) -> list[tuple[OpenAssumption, dict[str, Any]]] | None:
    """(assumption, break fields) for each clear break of 0.6 or more whose evidence quote is in
    the evidence (`evidence` normalized); None when unparseable."""
    obj = json_object(text, "breaks")
    if obj is None or not isinstance(obj.get("breaks"), list):
        return None
    out: dict[int, tuple[OpenAssumption, dict[str, Any]]] = {}
    for raw in obj["breaks"]:
        if not isinstance(raw, dict):
            continue
        i = resolve_ref(raw.get("assumption"), "H", len(open_items))
        confidence = _confidence(raw)
        quote = _text(raw.get("evidence_quote"), MAX_QUOTE)
        if i is None or confidence is None or confidence < BREAK_MIN_CONFIDENCE or i in out:
            continue
        if quote is None or not verbatim(quote, evidence):
            continue  # a break must point at words that were really said or written
        rationale = _text(raw.get("rationale"), MAX_RATIONALE)
        rationale = labels.swap(
            rationale or "New evidence contradicts this assumption.", named=False
        )
        if policy.hits(quote, rationale):
            continue
        speaker = labels.person(raw.get("speaker"))
        out[i] = (
            open_items[i],
            {
                "confidence": confidence,
                "evidence_quote": quote,
                "rationale": rationale,
                "speaker_user_id": str(speaker) if speaker else None,
            },
        )
    return sorted(out.values(), key=lambda b: -float(b[1]["confidence"]))


# --- rendering the inputs ---------------------------------------------------------------------


def render_goals(goals: list[Goal]) -> str:
    if not goals:
        return "(none)"
    lines = []
    for n, g in enumerate(goals, start=1):
        period = f" ({clip(g.period, 20)})" if g.period else ""
        lines.append(f"G{n}. {clip(g.title, 200)}{period}")
    return "\n".join(lines)


def render_actions(actions: list[CandidateAction]) -> str:
    if not actions:
        return "(none)"
    lines = []
    for n, a in enumerate(actions, start=1):
        tags = " [customer]" if a.customer else ""
        extra = []
        if a.due_date:
            extra.append(f"due {a.due_date}")
        if a.meeting_title:
            extra.append(f'from "{clip(a.meeting_title, 80)}"')
        more = f" ({'; '.join(extra)})" if extra else ""
        lines.append(f"A{n}.{tags} {clip(a.content, 240)}{more}")
    return "\n".join(lines)


def render_open(items: list[OpenAssumption]) -> str:
    lines = []
    for n, a in enumerate(items, start=1):
        lines.append(f"H{n}. [{a.kind}] {clip(a.statement, MAX_STATEMENT)}")
        if a.decision_content:
            lines.append(f"    Decision it supports: {clip(a.decision_content, 300)}")
        if a.quote:
            lines.append(f'    Stated as: "{clip(a.quote, 300)}"')
    return "\n".join(lines)


def _date_of(raw: object) -> date | None:
    """An envelope's `occurred_at`: Avro timestamp-micros (int) or a decoded datetime."""
    if isinstance(raw, datetime):
        return (raw if raw.tzinfo else raw.replace(tzinfo=UTC)).astimezone(UTC).date()
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        try:
            return datetime.fromtimestamp(raw / 1_000_000, tz=UTC).date()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def event_date(value: Mapping[str, Any]) -> date:
    """The date of an event (its header's `occurred_at`), else today (UTC)."""
    found = _date_of((value.get("header") or {}).get("occurred_at"))
    return found or datetime.now(UTC).date()


# --- the records ------------------------------------------------------------------------------


@dataclass
class Record:
    """One `agent-results` record to emit (agent `ProactiveAgent`)."""

    org_id: UUID
    meeting_id: UUID
    window_id: str
    user_id: UUID
    result: AgentResult


def why_record(
    *,
    org_id: UUID,
    decision_id: UUID,
    meeting_id: UUID,
    user_id: UUID,
    content: str,
    assumptions: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    prompt_ref: str,
) -> Record:
    confidence = max((float(a["confidence"]) for a in assumptions), default=0.0)
    payload = {
        "alert": {
            "id": str(uuid5(CANDIDATE_NS, f"{AGENT}:decision_why:{decision_id}")),
            "alert_type": "decision_why",
            "priority": 3,
            "surfaced": False,
            "title": "Why this decision was made",
            "body": "",
            "confidence": confidence,
            "rationale": "",
            "subject": {"kind": "decision", "id": str(decision_id)},
            "related": [],
            "link": f"/decisions/{decision_id}",
        },
        "why": {
            "decision_id": str(decision_id),
            "assumptions": assumptions[:MAX_ASSUMPTIONS],
            "ripple": {"goals": goals[:MAX_GOALS], "actions": actions[:MAX_ACTIONS]},
        },
        "prompt": prompt_ref,
        "model_tier": MODEL_TIER,
    }
    window_id = f"why:{decision_id}"
    return Record(
        org_id=org_id,
        meeting_id=meeting_id,
        window_id=window_id,
        user_id=user_id,
        result=AgentResult(
            kind="alert",
            content=f"Why: {clip(strip_urls(content), 90)}",
            relevance_score=confidence,
            confidence=confidence,
            source_chunk_ids=[],
            payload=payload,
            window_id=window_id,
        ),
    )


def broken_record(
    *,
    org_id: UUID,
    meeting_id: UUID,
    user_id: UUID,
    window_id: str,
    subject_kind: str,
    subject_id: UUID,
    breaks: list[tuple[OpenAssumption, dict[str, Any]]],
    source: str,
    source_meeting_id: UUID,
    source_decision_id: UUID | None,
    prompt_ref: str,
) -> Record:
    confidence = max((float(b["confidence"]) for _a, b in breaks), default=0.0)
    entries = [
        {
            "assumption_id": str(a.id),
            "confidence": b["confidence"],
            "evidence_quote": b["evidence_quote"],
            "rationale": b["rationale"],
            "source": source,
            "source_meeting_id": str(source_meeting_id),
            "source_decision_id": str(source_decision_id) if source_decision_id else None,
            "speaker_user_id": b["speaker_user_id"],
        }
        for a, b in breaks
    ]
    title = "An assumption may no longer hold"
    payload = {
        "alert": {
            "id": str(uuid5(CANDIDATE_NS, window_id)),
            "alert_type": "assumption_broken",
            "priority": 2,
            "surfaced": False,
            "title": title,
            "body": "",
            "confidence": confidence,
            "rationale": "",
            "subject": {"kind": subject_kind, "id": str(subject_id)},
            "related": [],
            "link": "",
        },
        "breaks": entries,
        "prompt": prompt_ref,
        "model_tier": MODEL_TIER,
    }
    return Record(
        org_id=org_id,
        meeting_id=meeting_id,
        window_id=window_id,
        user_id=user_id,
        result=AgentResult(
            kind="alert",
            content=title,
            relevance_score=confidence,
            confidence=confidence,
            source_chunk_ids=[],
            payload=payload,
            window_id=window_id,
        ),
    )


# --- the worker -------------------------------------------------------------------------------


@dataclass(frozen=True)
class WhyContext:
    org: CompanyContext
    never: NeverRecordPolicy


class DecisionWhy:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        deps: WhyDeps | None = None,
        settings: WhySettings | None = None,
    ) -> None:
        self.loader = loader
        self.router = router
        self.deps = deps if deps is not None else WhyDeps()
        self.settings = settings if settings is not None else WhySettings.from_env()

    # -- gating --

    async def active(self, org_id: UUID) -> bool:
        """On the tenant's plan (or overridden) and the budget is `ok`. The flags are the
        callers'."""
        if not self.settings.all_plans:
            plans = self.deps.plans
            if plans is None:
                return False
            try:
                if not await plans.enabled(org_id, FEATURE):
                    return False
            except Exception:  # noqa: BLE001 - unknown plan: suspended
                return False
        try:
            return await self.router.budget_state(org_id) == "ok"
        except Exception:  # noqa: BLE001 - unknown budget: suspended (fail closed)
            return False

    async def context(self, org_id: UUID) -> WhyContext:
        """Raises when the company brain cannot be read (strict)."""
        fetch = self.deps.org_context
        org = await fetch(org_id) if fetch is not None else None
        org = org if isinstance(org, CompanyContext) else CompanyContext()
        return WhyContext(org, NeverRecordPolicy.from_org(org))

    async def _never(self, org_id: UUID, never: NeverRecordPolicy, *texts: str | None) -> bool:
        """As `ProactiveAgent._never`: said outright, or an ambiguous word the triage check
        judges (a check that cannot run counts as never-record)."""
        if never.hits(*texts):
            return True
        ambiguous = [t for t in texts if t and never.ambiguous(t)]
        if not ambiguous:
            return False
        try:
            about = await RouterClassifier(self.router).about(org_id, " ".join(ambiguous), never)
        except Exception:  # noqa: BLE001 - unknown: nothing
            about = None
        return about is not False

    def _prompt(self, key: str, org_id: UUID) -> Prompt:
        return self.loader.load(key, self.loader.resolve(key, org_id))

    async def _call(
        self, key: str, org_id: UUID, *, effort: Effort, **values: str
    ) -> tuple[str, str]:
        prompt = self._prompt(key, org_id)
        out = await self.router.complete(
            tier="extract",
            system=prompt.system,
            **prompt.user_parts(**values),
            org_id=org_id,
            user_id=None,  # background check: metered to the tenant
            max_tokens=EXTRACT_MAX_TOKENS,
            effort=effort,
            prompt_ref=prompt.ref,
        )
        return out.text, prompt.ref

    # -- trigger: a confirmed decision --

    async def on_decision(self, d: ConfirmedDecision, meeting_date: date) -> list[Record]:
        """The `decision_why` record and the trigger-A `assumption_broken` record (either may be
        absent). Never raises for an expected failure; the loop also guards it."""
        s = self.settings
        core = self.deps.core
        if not (s.why_enabled or s.watch_enabled) or core is None or d.confirmed_by is None:
            return []
        if not await self.active(d.org_id):
            return []
        try:
            source = await core.why_source(d.org_id, d.decision_id)
        except Exception:  # noqa: BLE001 - no source: skipped
            logger.warning("why-source failed for decision %s; skipped", d.decision_id)
            return []
        if source is None or source.already_read:
            return []
        try:
            ctx = await self.context(d.org_id)
        except Exception:  # noqa: BLE001 - never guess the never-record list for a durable record
            logger.warning("company brain unavailable; decision %s why skipped", d.decision_id)
            return []
        rationale = source.rationale or d.rationale
        texts = (d.content, source.content, rationale, source.quote)
        if await self._never(d.org_id, ctx.never, *texts):
            return []  # ADR-0039 C2: a never-record decision gets nothing
        jobs: list[Awaitable[Record | None]] = []
        if s.why_enabled:
            jobs.append(self._why(d, source, rationale, ctx, meeting_date))
        if s.watch_enabled and not source.confidential:
            jobs.append(self._check_decision(d, source, rationale, ctx, meeting_date))
        out: list[Record] = []
        for done in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(done, Record):
                out.append(done)
            elif isinstance(done, asyncio.CancelledError):
                raise done
            elif isinstance(done, BudgetBlocked):
                continue
            elif isinstance(done, BaseException):
                logger.error("decision %s why/check failed: %r", d.decision_id, done)
        return out

    async def _why(
        self,
        d: ConfirmedDecision,
        source: WhySource,
        rationale: str,
        ctx: WhyContext,
        meeting_date: date,
    ) -> Record | None:
        assert d.confirmed_by is not None
        labels = Labels.build(source.turns, source.speakers, source.participants)
        turns = quiet_turns(source.turns, ctx.never)
        quote = "" if ctx.never.hits(source.quote) else source.quote
        speaker = labels.of_speaker(source.quote_speaker_id)
        moment = f'"{quote}"' + (f" (said by {speaker})" if speaker else "") if quote else "(none)"
        haystack = _norm(" \n ".join([source.content, rationale, quote, *(t.text for t in turns)]))
        company = render_company(ctx.org)
        rules = render_ground_rules(ctx.org)

        async def assumptions() -> tuple[list[dict[str, Any]] | None, str]:
            text, ref = await self._call(
                ASSUMPTIONS_KEY,
                d.org_id,
                effort=EXTRACTION_EFFORT,
                company=company,
                ground_rules=rules,
                meeting_date=meeting_date.isoformat(),
                decision=clip(source.content, 1200),
                rationale=clip(rationale, 1500) if rationale else "(none)",
                moment=moment,
                transcript=render_turns(turns, labels, MAX_EXCERPT_CHARS) or "(none)",
            )
            parsed = parse_assumptions(text, labels=labels, sources=haystack, policy=ctx.never)
            return parsed, ref

        async def ripple_safe() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            try:
                return await self.ripple_for(d.org_id, source, rationale, ctx)
            except asyncio.CancelledError:
                raise
            except BudgetBlocked:
                return [], []
            except Exception as exc:  # noqa: BLE001 - the why still goes out, without a ripple
                logger.error("decision_ripple failed decision=%s: %r", d.decision_id, exc)
                return [], []

        first, rippled, graph = await asyncio.gather(
            assumptions(),
            ripple_safe(),
            self._graph_actions(d.org_id, d.decision_id, ctx),
            return_exceptions=True,
        )
        if isinstance(first, BaseException):
            raise first  # BudgetBlocked or a model failure: no record (the hook logs it)
        if isinstance(rippled, BaseException):
            raise rippled  # only a cancellation gets here
        if isinstance(graph, BaseException):
            raise graph  # only a cancellation gets here
        found, ref = first
        goal_effects, agent_actions = rippled
        if found is None:
            logger.warning("decision_assumptions answer unparseable decision=%s", d.decision_id)
            return None
        seen = {a["action_id"] for a in agent_actions}
        merged = agent_actions + [g for g in graph if g["action_id"] not in seen]
        return why_record(
            org_id=d.org_id,
            decision_id=d.decision_id,
            meeting_id=source.meeting_id,
            user_id=d.confirmed_by,
            content=source.content,
            assumptions=found,
            goals=goal_effects,
            actions=merged[:MAX_ACTIONS],
            prompt_ref=ref,
        )

    async def ripple_for(
        self, org_id: UUID, source: WhySource, rationale: str, ctx: WhyContext
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The `decision_ripple` call: (goals, agent actions). No call without an active goal or
        a candidate action; an unreadable answer is no ripple. Raises `BudgetBlocked` and model
        failures."""
        goals = [g for g in ctx.org.goals[:MAX_GOALS_SHOWN] if not ctx.never.hits(g.title)]
        actions = [a for a in source.candidate_actions if not ctx.never.hits(a.content)]
        if not goals and not actions:
            return [], []
        text, _ref = await self._call(
            RIPPLE_KEY,
            org_id,
            effort=SYNTHESIS_EFFORT,
            # the goals are listed below with their G labels: not twice
            company=render_company(dataclasses.replace(ctx.org, goals=())),
            ground_rules=render_ground_rules(ctx.org),
            decision=clip(source.content + (f" Rationale: {rationale}" if rationale else ""), 1500),
            goals=render_goals(goals),
            actions=render_actions(actions),
        )
        parsed = parse_ripple(text, goals=goals, actions=actions, policy=ctx.never)
        if parsed is None:
            logger.warning("decision_ripple answer unparseable decision=%s", source.decision_id)
            return [], []
        return parsed

    async def _graph_actions(
        self, org_id: UUID, decision_id: UUID, ctx: WhyContext
    ) -> list[dict[str, Any]]:
        """Still-open actions of the decision's own meeting (KG `actions_blocked_on_decision`),
        under the R1 deadline. KG absent, slow or down: none."""
        kg = self.deps.kg
        if kg is None:
            return []
        try:
            nodes = await asyncio.wait_for(
                kg.query(org_id, "actions_blocked_on_decision", {"decision_id": str(decision_id)}),
                self.settings.kg_timeout_ms / 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - R1 (KgUnavailable, a timeout): degrade
            logger.info("ripple: KG unavailable (%s); no graph actions", type(exc).__name__)
            return []
        out: list[dict[str, Any]] = []
        for n in nodes:
            aid = _uuid(n.id)
            text = n.properties.get("content")
            if n.label != "Action" or aid is None or not isinstance(text, str) or not text.strip():
                continue
            if ctx.never.hits(text) or any(o["action_id"] == str(aid) for o in out):
                continue
            out.append(
                {
                    "action_id": str(aid),
                    "label": clip(strip_urls(text), MAX_LABEL),
                    "effect": "changes",
                    "why": "An open task from the meeting that made this decision.",
                    "confidence": GRAPH_CONFIDENCE,
                    "source": "graph",
                }
            )
        return out

    async def _check_decision(
        self,
        d: ConfirmedDecision,
        source: WhySource,
        rationale: str,
        ctx: WhyContext,
        meeting_date: date,
    ) -> Record | None:
        """Trigger A: the new decision as evidence against other decisions' assumptions."""
        assert self.deps.core is not None and d.confirmed_by is not None
        try:
            items = await self.deps.core.open_assumptions(
                d.org_id, exclude_meeting_id=source.meeting_id, exclude_decision_id=d.decision_id
            )
        except Exception:  # noqa: BLE001 - no list: nothing to check
            logger.warning("open assumptions unavailable; decision %s not checked", d.decision_id)
            return None
        labels = Labels.build(source.turns, source.speakers, source.participants)
        speaker = labels.of_speaker(source.quote_speaker_id)
        lines = [f"Decision: {clip(source.content, 1200)}"]
        if rationale:
            lines.append(f"Rationale: {clip(rationale, 1500)}")
        quote = "" if ctx.never.hits(source.quote) else source.quote
        if quote:
            lines.append(f'Said in the meeting{f" by {speaker}" if speaker else ""}: "{quote}"')
        evidence = "\n".join(lines)
        breaks = await self._check(
            d.org_id,
            ctx,
            items,
            labels=labels,
            evidence=evidence,
            haystack=_norm(" \n ".join([source.content, rationale, quote])),
            kind="a newly confirmed decision",
            when=meeting_date,
        )
        if not breaks:
            return None
        found, ref = breaks
        window = f"assumption-check:decision:{d.decision_id}"
        return broken_record(
            org_id=d.org_id,
            meeting_id=source.meeting_id,
            user_id=d.confirmed_by,
            window_id=window,
            subject_kind="decision",
            subject_id=d.decision_id,
            breaks=found,
            source="decision",
            source_meeting_id=source.meeting_id,
            source_decision_id=d.decision_id,
            prompt_ref=ref,
        )

    async def _check(
        self,
        org_id: UUID,
        ctx: WhyContext,
        items: list[OpenAssumption],
        *,
        labels: Labels,
        evidence: str,
        haystack: str,
        kind: str,
        when: date,
    ) -> tuple[list[tuple[OpenAssumption, dict[str, Any]]], str] | None:
        """One `assumption_check` call; None when there is nothing to check or nothing broke."""
        shown = [a for a in items if not ctx.never.hits(a.statement, a.quote, a.decision_content)][
            :MAX_OPEN
        ]
        if not shown or not evidence.strip():
            return None  # no model call
        text, ref = await self._call(
            CHECK_KEY,
            org_id,
            effort=SYNTHESIS_EFFORT,
            company=render_company(ctx.org),
            ground_rules=render_ground_rules(ctx.org),
            assumptions=render_open(shown),
            evidence_kind=kind,
            evidence_date=when.isoformat(),
            evidence=evidence,
        )
        found = parse_breaks(
            text, open_items=shown, labels=labels, evidence=haystack, policy=ctx.never
        )
        if found is None:
            logger.warning("assumption_check answer unparseable")
            return None
        return (found, ref) if found else None

    # -- trigger: a meeting ended --

    async def on_meeting_ended(
        self, org_id: UUID, meeting_id: UUID, generation: str, meeting_date: date
    ) -> list[Record]:
        """Trigger B. Raises `SourceUnavailable` / the brain's error when worth retrying."""
        core, minutes = self.deps.core, self.deps.minutes
        if not self.settings.watch_enabled or core is None or minutes is None:
            return []
        if not await self.active(org_id):
            return []
        source = await minutes.source(org_id, meeting_id)
        if source is None or source.confidential:
            return []  # unknown, or confidential: never evidence for other decisions
        user = recipient(source)
        if user is None:
            return []
        ctx = await self.context(org_id)  # strict: raises -> retried
        turns = quiet_turns(source.turns, ctx.never)
        if not turns:
            return []
        items = await core.open_assumptions(org_id, exclude_meeting_id=meeting_id)
        labels = Labels.build(source.turns, source.speakers, source.participants)
        try:
            found = await self._check(
                org_id,
                ctx,
                items,
                labels=labels,
                evidence=render_turns(turns, labels, MAX_EVIDENCE_CHARS),
                haystack=_norm(" \n ".join(t.text for t in turns)),
                kind="the transcript of a meeting that just ended",
                when=meeting_date,
            )
        except BudgetBlocked:
            return []
        if not found:
            return []
        breaks, ref = found
        return [
            broken_record(
                org_id=org_id,
                meeting_id=meeting_id,
                user_id=user,
                window_id=f"assumption-check:meeting:{meeting_id}:{generation}",
                subject_kind="meeting",
                subject_id=meeting_id,
                breaks=breaks,
                source="meeting",
                source_meeting_id=meeting_id,
                source_decision_id=None,
                prompt_ref=ref,
            )
        ]


def recipient(source: MinutesSource) -> UUID | None:
    """A real user of the meeting: its host, else its first participant."""
    if source.host_user_id is not None:
        return source.host_user_id
    return source.participants[0].user_id if source.participants else None


# --- trigger B: the ended-meeting loop --------------------------------------------------------


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
    return f"nexus:{org_id}:assumption-check:{meeting_id}:{generation}"


@dataclass(frozen=True)
class Ended:
    org_id: UUID
    meeting_id: UUID
    generation: str
    meeting_date: date


def parse_ended(value: Mapping[str, Any]) -> Ended | None:
    """A `meeting-events` `ended` record (the generation is minutes', from its idempotency key)."""
    if str(value.get("event_type")) != "ended":
        return None
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    meeting_id = _uuid(value.get("meeting_id"))
    key = str(header.get("idempotency_key") or "")
    if org_id is None or meeting_id is None or not key:
        return None
    return Ended(org_id, meeting_id, generation_of(key, meeting_id), event_date(value))


async def emit(sink: ResultSink, records: list[Record]) -> int:
    for r in records:
        await sink.agent_result(
            org_id=r.org_id,
            meeting_id=r.meeting_id,
            window_id=r.window_id,
            user_id=r.user_id,
            agent=AGENT,
            result=r.result,
        )
    return len(records)


class AssumptionWatchLoop:
    """The testable core of trigger B: one decoded `meeting-events` record in, at most one
    `assumption_broken` record out, flushed. A retryable failure (core-api or the brain) is
    retried with backoff; after the last attempt the meeting is given up (logged, unclaimed, so
    a replay would try again)."""

    def __init__(
        self,
        why: DecisionWhy,
        sink: ResultSink,
        *,
        claims: Any = None,  # redis.asyncio: `exists` and `set(..., ex=)`
        settle_s: float = 15.0,
        attempts: int = 3,
        backoff_s: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.why = why
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
        except Exception:  # noqa: BLE001 - no claim store: check (core-api dedupes the key)
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
        records: list[Record] | None = None
        for attempt in range(self.attempts):
            try:
                records = await self.why.on_meeting_ended(
                    ended.org_id, ended.meeting_id, ended.generation, ended.meeting_date
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then given up loudly
                logger.warning(
                    "assumption check for %s failed (attempt %d): %r",
                    ended.meeting_id,
                    attempt + 1,
                    exc,
                )
                if attempt + 1 < self.attempts:
                    await self._sleep(self.backoff_s * (2**attempt))
        if records is None:
            logger.error("assumption check for %s given up", ended.meeting_id)
            return 0
        emitted = await emit(self.sink, records)
        if emitted:
            self.sink.flush()  # raises EmitError: the caller must not commit
        if self.claims is not None:
            try:
                await self.claims.set(key, "1", ex=CLAIM_TTL_S)
            except Exception:  # noqa: BLE001 - a replay re-bills at worst; core-api dedupes
                logger.warning("assumption-check claim not stored for %s", ended.meeting_id)
        return emitted


async def run_assumption_watch_loop(
    consumer: Any, codec: Any, loop: AssumptionWatchLoop, stop: asyncio.Event
) -> None:  # the confluent-kafka harness, like the minutes loop's
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
        # A new group starts at the end: no checks (and no model calls) for historic meetings.
        "auto.offset.reset": "latest",
        # One check is one model call over a whole meeting: keep the group alive.
        "max.poll.interval.ms": 900_000,
    }
