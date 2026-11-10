"""`ProactiveAgent`: conflict, workload, dependency and missing-stakeholder alerts (P11).

Plan 10-13 section 3.6; the P11 contract sections 1 and 2. Four checks, two triggers:

- **conflict** (a confirmed decision on `decisions`, `proactive/loop.py`). Candidate prior
  decisions are the AGE `FOLLOWS_FROM` ancestors (`decision_lineage`, depth <= 3, over
  knowledge-graph-service gRPC with an 800 ms deadline) UNION a semantic search of the org's
  confirmed decisions (`decisions__{org_id}`) through rag-service, decision citations only,
  asked AS `confirmed_by`. R1 hot degradation: KG slow or down -> the rag path alone. The
  decision named in `supersedes` (and the decision itself) is never a candidate: replacing a
  decision is not contradicting it. Then ONE extract-tier (Sonnet) call, `proactive_conflict@vN`,
  answers `{conflicts: [{candidate, confidence, rationale}]}`; the best one is the alert.
  Recipients: `confirmed_by`, then the decision meeting's participants (cap 10).
- **workload** (a confirmed action on `actions`). core-api's action context: the assignee's open
  action count (including this one) `>= PROACTIVE_WORKLOAD_THRESHOLD` (6) -> alert, confidence
  0.9, no model call. Recipients: the meeting's participants who are org owners/admins or its
  organizer; if none, the meeting's participants (cap 10).
- **dependency** (same trigger, same context). Any `depends_on` entry whose status is not
  `done` -> alert, confidence 0.85, no model call. Recipients: the assignee, then the meeting's
  participants (cap 10).
- **missing_stakeholder** (every closed window, through the runtime: this agent is meeting-scope
  and `multi_user`). One triage-tier (Haiku) gate, `proactive_stakeholder@vN`, names the
  window's taxonomy topic; KG `related_decisions_for_topic` finds the confirmed decisions on it;
  the people who attended the meetings that produced them (core-api participants route) and are
  absent from this meeting are the stakeholders. At most once per (meeting, person), at most
  `MAX_STAKEHOLDERS` per window. KG unavailable -> skipped silently, before the gate is paid for.

Gating, before any work: `PROACTIVE_AGENT` != 0, the plan feature `proactive_agent` (billing,
cached 5 min per org; `PROACTIVE_AGENT_ALL_PLANS=1` overrides it), and the tenant budget state
`ok`. Throttled or blocked -> suspended: nothing emitted and no model call (plan 10-13 3.6 step
3, doc 18 section 5.4). Model calls go through the router (budget gate, `usage-events`).

Output: one `alert` AgentResult per recipient, exactly the contract's payload, with
`window_id = proactive:{alert_type}:{subject}` and the id
`uuid5(CANDIDATE_NS, "ProactiveAgent:{alert_type}:{subject}:{recipient}")`, so a replayed trigger
yields the same ids. Bands: `< 0.4` no alert, `0.4-0.7` `surfaced: false`, `> 0.7` `surfaced:
true`. No alert writes `decisions` / `actions` or any table: the agent only proposes.

Who sees what (privacy, per recipient). The KG is org-wide and rag ran as `confirmed_by`, so a
prior decision may come from a meeting a recipient never attended. Its text and label go only to
recipients who attended that decision's meeting (core-api participants route, per check); the
others get a redacted conflict alert (generic title, body without the prior text, the related id
kept with an empty label). For missing_stakeholder a recipient is told about an absent person,
with counts and decisions, only from earlier meetings that recipient attended with them; with
none in common there is no alert. A person is marked alerted only by a surfaced (> 0.7) alert,
so a silent window does not block a later strong one. URLs are stripped from alert text.

What a model sees: decision text (conflict) or the window with "Participant N" labels plus the
topic list (stakeholder). Never a user id, a name or an email; the router redacts PII on top.

ADR-0039 C7 adds two more per-window checks, run beside missing_stakeholder with the same gating
(`active`: enabled, plan, budget `ok`), metered through the router, at most once per subject in
Redis (`SET NX`, 7 days; a Redis failure skips the check: once-only cannot be promised):
- **already_decided**: a triage gate (`proactive_already_decided_gate@vN`) says whether the window
  re-opens a settled question and names it as a search query. Then the org's confirmed decisions
  are searched through rag-service (decisions only) ONCE PER PARTICIPANT in the live roster (cap
  10), AS that participant (ADR-0033, ADR-0036 D9): a participant whose own search does not return
  a decision is never told about it. One extract-tier call (`proactive_already_decided@vN`)
  judges the union of the hits; each participant gets the best decision from their OWN hits judged
  at confidence 0.8 or more, at most once per (meeting, decision, participant). Payload (inside
  `alert`, beside the usual keys): `decision_id`, `decision_content`, `decided_at` (the cited
  meeting's date, or null), `meeting_id`, `meeting_title` (THIS meeting: realtime-gateway delivers
  an alert into the room of the record's meeting and drops one whose `meeting_id` differs; the
  decision's own meeting is in `related`). Priority 2.
- **topic_watch**: the org's `topic_watches` (from `GetOrgContext`) whose watcher is not a
  participant (core-api participants route, plus the roster and proxies) are matched against the
  window: exact or glossary-alias match first (word boundary, case-insensitive, confidence 0.9),
  then one triage call (`proactive_topic_watch@vN`) for the rest. Never from a confidential meeting
  (the agenda route's `confidential`; unknown counts as confidential), never from a window that
  names a never-record topic. Each watcher must pass the ADR-0013 PEP for `meeting_context` at the
  WIDEST scope between them and the live participants (the meeting's content comes from all of
  them). Payload: `topic`, `meeting_id`, `meeting_title`, `snippet` (the matching turn's text, up
  to 200 chars, URLs stripped, no speaker). Priority 3. Watchers are absent, so delivery is
  notification-service's `notification_log`. Once per (meeting, watcher, topic), claimed only by a
  surfaced alert (as missing_stakeholder marks people), at most `MAX_RECIPIENTS` per window.

The subject for missing_stakeholder is the meeting, but its id key is `{meeting_id}:{person_id}`:
with the meeting id alone two absent people in one meeting would share an alert id and the second
would be deduped away downstream (a recorded deviation from the contract's literal formula).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal, Protocol
from uuid import UUID, uuid5

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.kg_client import KgClient, KgNode, KgUnavailable, kg_client_from_env
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import EMPTY, CompanyContext
from libs.shared.plan_features import PlanFeaturesClient
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.rag_client import HttpRagClient, RagClient, RagPassage
from libs.shared.router import (
    EXTRACT_MAX_TOKENS,
    SYNTHESIS_EFFORT,
    TIER_NAMES,
    BudgetBlocked,
    LiteLLMRouter,
)
from libs.shared.state import AgentResult, AgentState, TranscriptWindow
from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.a2a.pep import Pep, grantee_role, resolve_scope
from services.agent_runtime_service.agenda import AgendaSource
from services.agent_runtime_service.agents.candidates import (
    CANDIDATE_NS,
    confidence_of,
    parse_candidates,
    text_of,
)
from services.agent_runtime_service.agents.candidates import render_window as labelled_window
from services.agent_runtime_service.agents.company import (
    company_of,
    company_values,
    gate_ground_rules,
)
from services.agent_runtime_service.agents.context_gap_agent import parse_object
from services.agent_runtime_service.agents.personal_listener_agent import band
from services.agent_runtime_service.policy import DEFAULT as DEFAULT_THRESHOLDS
from services.agent_runtime_service.policy import Thresholds, thresholds, thresholds_for
from services.agent_runtime_service.proactive.core import (
    ActionContext,
    CoreContext,
    Participant,
    as_uuid,
    core_context_from_env,
)

logger = logging.getLogger(__name__)

AGENT = "ProactiveAgent"
AlertType = Literal[
    "conflict", "workload", "dependency", "missing_stakeholder", "already_decided", "topic_watch"
]
PRIORITY: dict[str, int] = {
    "conflict": 1,
    "workload": 2,
    "dependency": 2,
    "missing_stakeholder": 3,
    "already_decided": 2,  # ADR-0039 C7
    "topic_watch": 3,  # ADR-0039 C7
}

CONFLICT_KEY = "proactive_conflict"
STAKEHOLDER_KEY = "proactive_stakeholder"
REOPEN_GATE_KEY = "proactive_already_decided_gate"
ALREADY_DECIDED_KEY = "proactive_already_decided"
TOPIC_WATCH_KEY = "proactive_topic_watch"
FEATURE = "proactive_agent"

MAX_RECIPIENTS = 10
MAX_TITLE = 120
MAX_BODY = 400
MAX_LABEL = 80
LINEAGE_DEPTH = 3
MAX_RAG_CANDIDATES = 5
MAX_CANDIDATES = 8
MAX_TOPIC_MEETINGS = 3
MAX_STAKEHOLDERS = 2
WORKLOAD_CONFIDENCE = 0.9
DEPENDENCY_CONFIDENCE = 0.85
DEFAULT_WORKLOAD_THRESHOLD = 6
DEFAULT_GATE_THRESHOLD = 0.5
DEFAULT_KG_TIMEOUT_MS = 800
ALREADY_DECIDED_MIN_CONFIDENCE = 0.8  # ADR-0039 C7
EXACT_MATCH_CONFIDENCE = 0.9  # topic_watch: the watched topic (or an alias) is said outright
MAX_SNIPPET = 200
MAX_WATCH_TOPICS = 40  # distinct watched topics shown to the triage check
MIN_WINDOW_WORDS = 20  # no model call for the C7 checks on a near-empty window
ONCE_TTL_S = 7 * 24 * 3600
SCOPE_WIDTH = {"self": 0, "same_team": 1, "department": 2, "org": 3}


# --- settings and dependencies ----------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class ProactiveSettings:
    enabled: bool = True  # PROACTIVE_AGENT (default 1; 0 disables every check)
    workload_threshold: int = DEFAULT_WORKLOAD_THRESHOLD  # PROACTIVE_WORKLOAD_THRESHOLD
    all_plans: bool = False  # PROACTIVE_AGENT_ALL_PLANS: skip the plan flag (budget still applies)
    gate_threshold: float = DEFAULT_GATE_THRESHOLD
    kg_timeout_ms: int = DEFAULT_KG_TIMEOUT_MS  # KG_TIMEOUT_MS

    @classmethod
    def from_env(cls) -> ProactiveSettings:
        return cls(
            enabled=_env_flag("PROACTIVE_AGENT", True),
            workload_threshold=max(
                1, _env_int("PROACTIVE_WORKLOAD_THRESHOLD", DEFAULT_WORKLOAD_THRESHOLD)
            ),
            all_plans=_env_flag("PROACTIVE_AGENT_ALL_PLANS", False),
            kg_timeout_ms=max(1, _env_int("KG_TIMEOUT_MS", DEFAULT_KG_TIMEOUT_MS)),
        )


class PlanFeatures(Protocol):
    async def enabled(self, org_id: UUID, feature: str) -> bool: ...


class DecisionSearch(Protocol):
    async def search(self, *, org_id: UUID, user_id: UUID, query: str) -> list[RagPassage]: ...


class RagDecisionSearch:
    """rag-service, decisions only. `HttpRagClient` is asked with `source="decisions"` (the
    caller-forced source filter), so transcript hits do not take the top-10 slots; any other
    `RagClient` is filtered here. Either way only `decision` citations with an id come back.
    Participant-scoped: rag-service returns decisions from meetings `user_id` attended."""

    def __init__(self, client: RagClient) -> None:
        self._client = client

    async def search(self, *, org_id: UUID, user_id: UUID, query: str) -> list[RagPassage]:
        if isinstance(self._client, HttpRagClient):
            found = await self._client.retrieve(
                org_id=org_id, user_id=user_id, query=query, source="decisions"
            )
        else:
            found = await self._client.retrieve(org_id=org_id, user_id=user_id, query=query)
        return [p for p in found if p.kind == "decision" and p.decision_id and p.text.strip()]


def _topics() -> list[str]:
    try:
        from services.knowledge_graph_service.taxonomy import distinct_topics

        return distinct_topics()
    except Exception:  # noqa: BLE001 - no taxonomy: the stakeholder check is off
        logger.warning("role-topic taxonomy unavailable; missing_stakeholder check off")
        return []


@dataclass
class ProactiveDeps:
    """No `kg`: lineage and stakeholders are skipped (rag-only conflicts). No `decisions`:
    conflicts come from the graph alone. No `core`: workload, dependency and stakeholders are
    skipped and conflict alerts go to `confirmed_by` only. No `plans`: suspended unless
    `PROACTIVE_AGENT_ALL_PLANS=1`."""

    kg: KgClient | None = None
    decisions: DecisionSearch | None = None
    core: CoreContext | None = None
    plans: PlanFeatures | None = None
    topics: list[str] = field(default_factory=list)
    # ADR-0039 C7. No `redis`: neither new check runs (once-only needs it). No `agenda`, `pep`
    # or `user_context`: topic_watch is off (it cannot tell confidentiality or authorize).
    redis: Any = None
    agenda: AgendaSource | None = None
    pep: Pep | None = None
    user_context: Any = None  # async (org_id, user_id) -> dict | None
    # Polish (never record, one threshold source) for the checks the loop triggers, which have
    # no window contexts: the company brain (`OrgContextClient.get`) and core-api's
    # `agent_confidence_overrides` (`AgentPolicyClient.overrides`). Absent: EMPTY / defaults.
    org_context: Any = None  # async (org_id) -> CompanyContext
    policy: Any = None  # async (org_id) -> {agent: {surface, silent}}


def deps_from_env() -> ProactiveDeps:
    rag: DecisionSearch | None = None
    if os.environ.get("RAG_CALLER_TOKEN", "").strip():
        rag = RagDecisionSearch(HttpRagClient(timeout_s=2.0))
    return ProactiveDeps(
        kg=kg_client_from_env(),
        decisions=rag,
        core=core_context_from_env(),
        plans=PlanFeaturesClient(),
        topics=_topics(),
    )


# --- the triggers ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmedDecision:
    org_id: UUID
    decision_id: UUID
    meeting_id: UUID
    content: str
    rationale: str
    confirmed_by: UUID | None
    supersedes: UUID | None = None


@dataclass(frozen=True)
class ConfirmedAction:
    org_id: UUID
    action_id: UUID
    meeting_id: UUID
    content: str
    assignee_id: UUID | None = None
    due_date: date | None = None


def parse_decision(value: Mapping[str, Any]) -> ConfirmedDecision | None:
    """A decoded `decisions` record (the frozen `Decision`), or None if it is malformed. The org
    is the envelope header's, which core-api derived from the confirming session."""
    org = parse_org_id((value.get("header") or {}).get("org_id"))
    decision, meeting = as_uuid(value.get("decision_id")), as_uuid(value.get("meeting_id"))
    content = value.get("content")
    if org is None or decision is None or meeting is None or not isinstance(content, str):
        return None
    if not content.strip():
        return None
    rationale = value.get("rationale")
    return ConfirmedDecision(
        org_id=org,
        decision_id=decision,
        meeting_id=meeting,
        content=content.strip(),
        rationale=rationale.strip() if isinstance(rationale, str) else "",
        confirmed_by=as_uuid(value.get("confirmed_by")),
        supersedes=as_uuid(value.get("supersedes")),
    )


def parse_action(value: Mapping[str, Any]) -> ConfirmedAction | None:
    org = parse_org_id((value.get("header") or {}).get("org_id"))
    action, meeting = as_uuid(value.get("action_id")), as_uuid(value.get("meeting_id"))
    content = value.get("content")
    if org is None or action is None or meeting is None or not isinstance(content, str):
        return None
    due = value.get("due_date")
    return ConfirmedAction(
        org_id=org,
        action_id=action,
        meeting_id=meeting,
        content=content.strip(),
        assignee_id=as_uuid(value.get("assignee_id")),
        due_date=due if isinstance(due, date) else None,
    )


# --- the alert -------------------------------------------------------------------------------


def alert_id(alert_type: str, subject_key: str, recipient: UUID) -> str:
    return str(uuid5(CANDIDATE_NS, f"{AGENT}:{alert_type}:{subject_key}:{recipient}"))


def clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def display_name(name: str | None) -> str:
    """The context endpoints carry the user's email as `name`. Alert text uses its local part
    ("priya" for priya@acme.com), never the full address; no name is "Someone"."""
    if not name or not name.strip():
        return "Someone"
    local = name.strip().split("@", 1)[0]
    return local or "Someone"


_URL = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+"  # schemes and www.
    r"|\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\.[a-z]{2,}/\S*"  # domain/path
)


def strip_urls(text: str) -> str:
    """Remove URLs before alert text leaves the agent tier: email clients auto-link them, and a
    link lifted from meeting or decision text must not become a clickable one in an alert."""
    return " ".join(_URL.sub("[link removed]", text).split())


def recipients(*groups: list[UUID] | tuple[UUID, ...]) -> list[UUID]:
    out: list[UUID] = []
    for group in groups:
        for uid in group:
            if uid not in out:
                out.append(uid)
    return out[:MAX_RECIPIENTS]


@dataclass(frozen=True)
class View:
    """What one recipient is shown, when it differs from the alert's default text (a recipient
    who may not see a prior decision or another meeting's attendees)."""

    title: str
    body: str
    rationale: str
    related: list[dict[str, str]]


@dataclass
class Alert:
    alert_type: AlertType
    subject_kind: Literal["decision", "action", "meeting"]
    subject_id: UUID
    subject_key: str  # the id/window key; the subject id except for missing_stakeholder
    meeting_id: UUID
    title: str
    body: str
    confidence: float
    rationale: str
    related: list[dict[str, str]]
    link: str | None
    to: list[UUID]
    prompt: str = ""
    model_tier: str = "none"
    source_chunk_ids: list[str] = field(default_factory=list)
    views: dict[UUID, View] = field(default_factory=dict)  # per-recipient overrides
    extra: dict[str, Any] = field(default_factory=dict)  # ADR-0039 C7 keys beside the usual ones
    thresholds: Thresholds = DEFAULT_THRESHOLDS  # the tenant's (ADR-0030 D3)

    @property
    def window_id(self) -> str:
        return f"proactive:{self.alert_type}:{self.subject_key}"

    @property
    def surfaced(self) -> bool:
        return band(self.confidence, self.thresholds) == "surfaced"

    def view(self, recipient: UUID) -> View:
        return self.views.get(recipient) or View(
            self.title, self.body, self.rationale, self.related
        )

    def title_for(self, recipient: UUID) -> str:
        return clip(strip_urls(self.view(recipient).title), MAX_TITLE)

    def payload(self, recipient: UUID) -> dict[str, Any]:
        v = self.view(recipient)
        return {
            "alert": {
                "id": alert_id(self.alert_type, self.subject_key, recipient),
                "alert_type": self.alert_type,
                "priority": PRIORITY[self.alert_type],
                "title": self.title_for(recipient),
                "body": clip(strip_urls(v.body), MAX_BODY),
                "confidence": self.confidence,
                "surfaced": self.surfaced,
                "rationale": strip_urls(v.rationale),
                "subject": {"kind": self.subject_kind, "id": str(self.subject_id)},
                "related": v.related,
                "link": self.link,
                **self.extra,
            },
            "prompt": self.prompt,
            "model_tier": self.model_tier,
        }

    def results(self) -> list[tuple[UUID, AgentResult]]:
        """One `alert` result per recipient. Nothing for a dropped (< 0.4) confidence."""
        if band(self.confidence, self.thresholds) == "dropped":
            return []
        return [
            (
                user,
                AgentResult(
                    kind="alert",
                    content=self.title_for(user),
                    relevance_score=self.confidence,
                    confidence=self.confidence,
                    source_chunk_ids=list(self.source_chunk_ids),
                    payload=self.payload(user),
                    window_id=self.window_id,
                ),
            )
            for user in self.to
        ]


# --- conflict helpers ------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    decision_id: UUID
    text: str
    source: Literal["lineage", "rag"]
    meeting_id: UUID | None = None  # the meeting that confirmed it: who may see its text


def lineage_priors(nodes: list[KgNode], *, offset: int = 0) -> list[Prior]:
    """Ancestors within `LINEAGE_DEPTH` (counting `offset` hops already taken), nearest first.
    Stub nodes (a superseded decision the graph has not seen confirmed) carry no text and are
    left out."""
    out: list[tuple[int, Prior]] = []
    for n in nodes:
        if n.label != "Decision" or n.role != "ancestor":
            continue
        depth = n.depth + offset
        did, text = as_uuid(n.id), n.properties.get("content")
        if did is None or depth > LINEAGE_DEPTH or not isinstance(text, str) or not text.strip():
            continue
        meeting = as_uuid(n.properties.get("meeting_id"))
        out.append((depth, Prior(did, text.strip(), "lineage", meeting)))
    out.sort(key=lambda t: t[0])
    return [p for _d, p in out]


def render_decision(d: ConfirmedDecision) -> str:
    return d.content + (f" Rationale: {d.rationale}" if d.rationale else "")


def render_priors(priors: list[Prior]) -> str:
    return "\n".join(f"[{i + 1}] {clip(p.text, 600)}" for i, p in enumerate(priors))


def parse_conflicts(text: str, n: int) -> list[tuple[int, float, str]]:
    """(0-based candidate index, confidence, rationale) per valid entry, best first. Numbers
    outside 1..n, missing confidences and duplicates are dropped."""
    seen: set[int] = set()
    out: list[tuple[int, float, str]] = []
    for raw in parse_candidates(text, key="conflicts"):
        if not isinstance(raw, dict):
            continue
        index, confidence = raw.get("candidate"), confidence_of(raw)
        if isinstance(index, bool) or not isinstance(index, int) or confidence is None:
            continue
        if not 1 <= index <= n or index in seen:
            continue
        seen.add(index)
        out.append((index - 1, confidence, text_of(raw, "rationale") or ""))
    out.sort(key=lambda t: -t[1])
    return out


def parse_topic(text: str, topics: list[str]) -> tuple[str, float]:
    obj = parse_object(text) or {}
    raw = obj.get("topic")
    topic = raw.strip().lower() if isinstance(raw, str) else ""
    score = confidence_of({"confidence": obj.get("score")}) or 0.0
    return (topic, score) if topic in topics else ("", 0.0)


# --- the agent -------------------------------------------------------------------------------


class ProactiveAgent(BaseAgent):
    name: ClassVar[str] = AGENT
    prompt_key: ClassVar[str] = CONFLICT_KEY
    scope: ClassVar[Literal["participant", "meeting"]] = "meeting"
    kind: ClassVar[str] = "alert"
    multi_user: ClassVar[bool] = True  # the runtime calls process_meeting (missing_stakeholder)

    gate_prompt: Prompt | None = None

    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        deps: ProactiveDeps | None = None,
        settings: ProactiveSettings | None = None,
    ) -> None:
        super().__init__(loader, router)
        self._deps = deps if deps is not None else ProactiveDeps()
        self._settings = settings if settings is not None else ProactiveSettings.from_env()
        self._alerted: set[UUID] = set()  # missing_stakeholder: people already alerted on

    # -- gating --

    async def active(self, org_id: UUID) -> bool:
        """Enabled, on the tenant's plan (or overridden), and the budget is `ok`."""
        if not self._settings.enabled:
            return False
        if not self._settings.all_plans:
            plans = self._deps.plans
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

    async def _company(self, org_id: UUID) -> CompanyContext:
        fetch = self._deps.org_context
        if fetch is None:
            return EMPTY
        try:
            org = await fetch(org_id)
        except Exception:  # noqa: BLE001 - no company context: the prompts render as before
            return EMPTY
        return org if isinstance(org, CompanyContext) else EMPTY

    async def _thresholds(self, org_id: UUID) -> Thresholds:
        fetch = self._deps.policy
        if fetch is None:
            return DEFAULT_THRESHOLDS
        try:
            return thresholds(await fetch(org_id), AGENT)
        except Exception:  # noqa: BLE001 - the defaults, as core-api without overrides
            return DEFAULT_THRESHOLDS

    def _prompt(self, key: str, org_id: UUID) -> Prompt:
        return self._loader.load(key, self._loader.resolve(key, org_id))

    async def _kg(self, org_id: UUID, template: str, params: dict[str, str]) -> list[KgNode]:
        """One KG read under the R1 deadline. Raises `KgUnavailable` on a miss of any kind."""
        kg = self._deps.kg
        if kg is None:
            raise KgUnavailable("not configured")
        try:
            return await asyncio.wait_for(
                kg.query(org_id, template, params), self._settings.kg_timeout_ms / 1000
            )
        except asyncio.CancelledError:
            raise
        except KgUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - timeout or anything else: degrade
            raise KgUnavailable(f"{template}: {type(exc).__name__}") from exc

    async def _participants(self, org_id: UUID, meeting_id: UUID) -> list[Participant]:
        core = self._deps.core
        if core is None:
            return []
        try:
            return await core.participants(org_id, meeting_id) or []
        except Exception:  # noqa: BLE001
            return []

    # -- conflict --

    async def check_decision(self, d: ConfirmedDecision) -> list[Alert]:
        if not await self.active(d.org_id):
            return []
        org = await self._company(d.org_id)
        never = NeverRecordPolicy.from_org(org)
        if never.hits(d.content, d.rationale):
            return []  # ADR-0039 C2: a never-record decision raises nothing anywhere
        priors = [p for p in await self.conflict_candidates(d) if not never.hits(p.text)]
        if not priors:
            return []
        limits = await self._thresholds(d.org_id)
        prompt = self._prompt(CONFLICT_KEY, d.org_id)
        try:
            out = await self.router.complete(
                tier="extract",
                system=prompt.system,
                **prompt.user_parts(
                    decision=clip(render_decision(d), 1200),
                    candidates=render_priors(priors),
                    **company_values({"org": org}),
                ),
                org_id=d.org_id,
                user_id=None,  # background check: metered to the tenant
                max_tokens=EXTRACT_MAX_TOKENS,
                effort=SYNTHESIS_EFFORT,
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            return []
        conflicts = parse_conflicts(out.text, len(priors))
        if not conflicts:
            return []
        index, confidence, rationale = conflicts[0]
        if band(confidence, limits) == "dropped" or never.hits(rationale):
            return []
        prior = priors[index]
        people = await self._participants(d.org_id, d.meeting_id)
        first = [d.confirmed_by] if d.confirmed_by else []
        to = recipients(first, [p.user_id for p in people])
        if not to:
            return []
        # Privacy: the earlier decision may come from a meeting a recipient never attended (the
        # KG is org-wide, rag ran as `confirmed_by`). Only its attendees see its text; everyone
        # else gets the same alert without it (redacted, not dropped: they still learn that the
        # decision they are bound by may clash with something, and can ask the confirmer).
        if prior.meeting_id == d.meeting_id:
            seen: set[UUID] = {p.user_id for p in people}
        elif prior.meeting_id is not None:
            seen = {p.user_id for p in await self._participants(d.org_id, prior.meeting_id)}
        else:
            seen = set()  # unknown source meeting: nobody is shown its text
        redacted = View(
            title="This decision may conflict with an earlier one",
            body=(
                f'The decision "{clip(d.content, 200)}" may contradict an earlier decision from '
                f"a meeting you did not attend. Ask the person who confirmed it for details."
            ),
            rationale="The details are limited to people who attended the earlier meeting.",
            related=[{"kind": "decision", "id": str(prior.decision_id), "label": ""}],
        )
        return [
            Alert(
                alert_type="conflict",
                subject_kind="decision",
                subject_id=d.decision_id,
                subject_key=str(d.decision_id),
                meeting_id=d.meeting_id,
                title=f"Possible conflict: {clip(d.content, 90)}",
                body=(
                    f'The decision "{clip(d.content, 150)}" may contradict the earlier decision '
                    f'"{clip(prior.text, 150)}". {rationale}'
                ),
                confidence=confidence,
                rationale=rationale or "The new decision may contradict an earlier one.",
                related=[
                    {
                        "kind": "decision",
                        "id": str(prior.decision_id),
                        "label": clip(prior.text, MAX_LABEL),
                    }
                ],
                link=f"/decisions/{d.decision_id}",
                to=to,
                prompt=prompt.ref,
                model_tier=TIER_NAMES.get(out.tier, out.tier),
                views={u: redacted for u in to if u not in seen},
                thresholds=limits,
            )
        ]

    async def conflict_candidates(self, d: ConfirmedDecision) -> list[Prior]:
        """Lineage ancestors UNION decision-only rag hits, lineage first, deduped, never the
        decision itself or the one it supersedes. KG down or slow: rag only (R1)."""
        lineage, found = await asyncio.gather(self._lineage(d), self._rag_priors(d))
        excluded = {d.decision_id, d.supersedes}
        out: list[Prior] = []
        for p in [*lineage, *found]:
            if p.decision_id in excluded or any(p.decision_id == q.decision_id for q in out):
                continue
            out.append(p)
        return out[:MAX_CANDIDATES]

    async def _lineage(self, d: ConfirmedDecision) -> list[Prior]:
        try:
            subject = {"decision_id": str(d.decision_id)}
            nodes = await self._kg(d.org_id, "decision_lineage", subject)
            if nodes:
                return lineage_priors(nodes)
            if d.supersedes is None:
                return []
            # The KG worker consumes the same event in parallel, so the new decision may not be
            # in the graph yet: walk from the decision it supersedes (one hop already taken).
            nodes = await self._kg(d.org_id, "decision_lineage", {"decision_id": str(d.supersedes)})
            return lineage_priors(nodes, offset=1)
        except KgUnavailable as exc:
            logger.info("conflict check: KG unavailable (%s); rag path only (R1)", exc)
            return []

    async def _rag_priors(self, d: ConfirmedDecision) -> list[Prior]:
        search = self._deps.decisions
        if search is None or d.confirmed_by is None:
            return []
        try:
            found = await search.search(org_id=d.org_id, user_id=d.confirmed_by, query=d.content)
        except Exception:  # noqa: BLE001 - rag down: lineage only
            logger.warning("conflict check: rag-service failed; lineage only")
            return []
        found = sorted(found, key=lambda p: -p.score)[:MAX_RAG_CANDIDATES]
        return [
            Prior(p.decision_id, p.text.strip(), "rag", p.meeting_id)
            for p in found
            if p.decision_id
        ]

    # -- workload and dependency --

    async def check_action(self, a: ConfirmedAction) -> list[Alert]:
        if not await self.active(a.org_id) or self._deps.core is None:
            return []
        try:
            ctx = await self._deps.core.action_context(a.org_id, a.action_id)
        except Exception:  # noqa: BLE001
            ctx = None
        if ctx is None:
            return []
        never = NeverRecordPolicy.from_org(await self._company(a.org_id))
        if never.hits(a.content) or any(never.hits(dep.content) for dep in ctx.depends_on):
            return []  # ADR-0039 C2
        limits = await self._thresholds(a.org_id)
        alerts = [self._workload(a, ctx), self._dependency(a, ctx)]
        return [dataclasses.replace(x, thresholds=limits) for x in alerts if x is not None]

    def _workload(self, a: ConfirmedAction, ctx: ActionContext) -> Alert | None:
        threshold = self._settings.workload_threshold
        assignee = ctx.assignee_id
        if assignee is None or ctx.open_action_count < threshold:
            return None
        oversight = set(ctx.admins) | ({ctx.organizer_id} if ctx.organizer_id else set())
        watchers = [u for u in ctx.participants if u in oversight]
        to = recipients(watchers or ctx.participants)
        if not to:
            return None
        who, n = display_name(ctx.assignee_name), ctx.open_action_count
        return Alert(
            alert_type="workload",
            subject_kind="action",
            subject_id=a.action_id,
            subject_key=str(a.action_id),
            meeting_id=a.meeting_id,
            title=f"{who} already owns {n} open actions",
            body=(
                f'{who} was just assigned "{clip(a.content, 150)}" and now owns {n} open '
                f"actions. Consider sharing the load before adding more."
            ),
            confidence=WORKLOAD_CONFIDENCE,
            rationale=f"{n} open actions, at or above the workload threshold of {threshold}.",
            related=[
                {"kind": "user", "id": str(assignee), "label": who},
                {"kind": "action", "id": str(a.action_id), "label": clip(a.content, MAX_LABEL)},
            ],
            link=f"/meetings/{a.meeting_id}",
            to=to,
        )

    def _dependency(self, a: ConfirmedAction, ctx: ActionContext) -> Alert | None:
        pending = [dep for dep in ctx.depends_on if dep.status.lower() != "done"]
        if not pending:
            return None
        to = recipients([ctx.assignee_id] if ctx.assignee_id else [], ctx.participants)
        if not to:
            return None
        first = pending[0]
        more = f" and {len(pending) - 1} more" if len(pending) > 1 else ""
        return Alert(
            alert_type="dependency",
            subject_kind="action",
            subject_id=a.action_id,
            subject_key=str(a.action_id),
            meeting_id=a.meeting_id,
            title=f"Blocked by unfinished work: {clip(a.content, 80)}",
            body=(
                f'"{clip(a.content, 150)}" depends on "{clip(first.content, 150)}"{more}, '
                f"which is not done yet."
            ),
            confidence=DEPENDENCY_CONFIDENCE,
            rationale=f"{len(pending)} prior action(s) this one depends on are not done.",
            related=[
                {"kind": "action", "id": str(dep.id), "label": clip(dep.content, MAX_LABEL)}
                for dep in pending
            ],
            link=f"/meetings/{a.meeting_id}",
            to=to,
        )

    # -- missing stakeholder (per closed window, through the runtime) --

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        await super().spawn(org_id=org_id, meeting_id=meeting_id, user_id=user_id)
        assert self.org_id is not None
        self.gate_prompt = self._prompt(STAKEHOLDER_KEY, self.org_id)

    async def process(self, state: AgentState) -> AgentResult | None:
        return None  # one result per recipient: see process_meeting

    def build_graph(self) -> Any:
        """The stakeholder gate as a LangGraph node: the score goes to `relevance_score`, the
        topic to `retrieved_facts[0]`."""

        async def gate(s: AgentState) -> dict[str, Any]:
            assert self.gate_prompt is not None and s.org_id is not None
            out = await self.router.complete(
                tier="triage",
                system=self.gate_prompt.system,
                **self.gate_prompt.user_parts(
                    topics=", ".join(self._deps.topics),
                    window=labelled_window(s.chunk)[0],
                    **gate_ground_rules(s.context),
                ),
                org_id=s.org_id,
                user_id=None,  # shared by every participant: metered to the tenant
                max_tokens=40,
                prompt_ref=self.gate_prompt.ref,
            )
            topic, score = parse_topic(out.text, self._deps.topics)
            return {"relevance_score": score, "retrieved_facts": [{"topic": topic}]}

        g: Any = StateGraph(AgentState)
        g.add_node("gate", gate)
        g.set_entry_point("gate")
        g.add_edge("gate", END)
        return g.compile()

    async def process_meeting(
        self, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
    ) -> list[tuple[UUID, AgentResult]]:
        if self._graph is None or self.org_id is None:
            raise RuntimeError(f"{self.name}.process_meeting before spawn")
        if window.org_id != self.org_id:
            raise ValueError(f"{self.name}: window org_id does not match the spawned agent")
        found = await asyncio.gather(
            self.check_window(window, set(contexts), contexts),
            self.check_already_decided(window, contexts),
            self.check_topic_watch(window, contexts),
        )
        return [r for alerts in found for a in alerts for r in a.results()]

    async def check_window(
        self,
        window: TranscriptWindow,
        roster: set[UUID],
        contexts: dict[UUID, dict[str, Any] | None] | None = None,
    ) -> list[Alert]:
        deps = self._deps
        if deps.kg is None or deps.core is None or not deps.topics or not roster:
            return []  # nothing to find stakeholders with: skip before paying for the gate
        org = _org(contexts or {})
        if _never_window(window, org):
            return []  # ADR-0039 C2: a never-record window raises nothing, and pays nothing
        if not await self.active(window.org_id):
            return []
        limits = _limits(contexts or {})
        topic, score = await self._gate(window, org)
        if not topic or score < self._settings.gate_threshold:
            return []
        try:
            nodes = await self._kg(
                window.org_id, "related_decisions_for_topic", {"topic": topic, "limit": "20"}
            )
        except KgUnavailable as exc:
            logger.info("stakeholder check: KG unavailable (%s); skipped", exc)
            return []
        by_meeting: dict[UUID, list[KgNode]] = {}
        for n in nodes:
            mid = as_uuid(n.properties.get("meeting_id"))
            if n.label != "Decision" or mid is None or mid == window.meeting_id:
                continue
            if mid in by_meeting or len(by_meeting) < MAX_TOPIC_MEETINGS:
                by_meeting.setdefault(mid, []).append(n)
        if not by_meeting:
            return []
        here, *earlier = await asyncio.gather(
            self._participants(window.org_id, window.meeting_id),
            *(self._participants(window.org_id, m) for m in by_meeting),
        )
        present = roster | {p.user_id for p in here}
        attendees = {
            mid: {p.user_id for p in people}
            for mid, people in zip(by_meeting, earlier, strict=True)
        }
        names: dict[UUID, str | None] = {}
        for people in earlier:
            for p in people:
                names.setdefault(p.user_id, p.name)
        counts = {
            u: sum(len(by_meeting[m]) for m, who in attendees.items() if u in who)
            for u in names
            if u not in present
        }
        ranked = sorted(
            (u for u in counts if u not in self._alerted), key=lambda u: (-counts[u], str(u))
        )
        assert self.gate_prompt is not None
        surfaced = band(score, limits) == "surfaced"
        alerts: list[Alert] = []
        for person in ranked:
            if len(alerts) >= MAX_STAKEHOLDERS:
                break
            # Privacy: a recipient learns about this person, and sees decisions, only from the
            # earlier meetings that recipient attended together with them. Someone who attended
            # none of them gets no alert.
            views: dict[UUID, View] = {}
            for r in sorted(present, key=str):
                shared = [m for m, who in attendees.items() if person in who and r in who]
                if shared:
                    views[r] = self._stakeholder_view(person, names, topic, shared, by_meeting)
                if len(views) >= MAX_RECIPIENTS:
                    break
            if not views:
                continue
            if surfaced:  # a silent 0.4-0.7 window must not block a later strong one
                self._alerted.add(person)
            first = next(iter(views.values()))
            alerts.append(
                Alert(
                    alert_type="missing_stakeholder",
                    subject_kind="meeting",
                    subject_id=window.meeting_id,
                    subject_key=f"{window.meeting_id}:{person}",
                    meeting_id=window.meeting_id,
                    title=first.title,
                    body=first.body,
                    confidence=score,
                    rationale=first.rationale,
                    related=first.related,
                    link=f"/meetings/{window.meeting_id}",
                    to=list(views),
                    prompt=self.gate_prompt.ref,
                    model_tier=TIER_NAMES["triage"],
                    source_chunk_ids=window.chunk_ids,
                    views=views,
                    thresholds=limits,
                )
            )
        return alerts

    @staticmethod
    def _stakeholder_view(
        person: UUID,
        names: dict[UUID, str | None],
        topic: str,
        shared: list[UUID],
        by_meeting: dict[UUID, list[KgNode]],
    ) -> View:
        """The alert as one recipient may see it: counts and decisions from `shared` only (the
        earlier meetings both the recipient and the absent person attended)."""
        who = display_name(names.get(person))
        nodes = [n for m in shared for n in by_meeting[m]]
        related = [{"kind": "user", "id": str(person), "label": who}]
        for node in nodes[:3]:
            text = node.properties.get("content")
            if as_uuid(node.id) and isinstance(text, str) and text.strip():
                related.append({"kind": "decision", "id": node.id, "label": clip(text, MAX_LABEL)})
        return View(
            title=f"{who} is not here but has been part of {topic} decisions",
            body=(
                f"This meeting is discussing {topic}. {who} was with you in meetings that "
                f"confirmed {len(nodes)} decision(s) on it and is not in this meeting. "
                f"Consider looping them in."
            ),
            rationale=(
                f"Topic {topic}: {len(nodes)} earlier confirmed decision(s) from meetings you "
                f"and {who} attended."
            ),
            related=related,
        )

    async def check_already_decided(
        self, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
    ) -> list[Alert]:
        """ADR-0039 C7 `already_decided` (module function `check_already_decided`)."""
        return await check_already_decided(self, window, contexts)

    async def check_topic_watch(
        self, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
    ) -> list[Alert]:
        """ADR-0039 C7 `topic_watch` (module function `check_topic_watch`)."""
        return await check_topic_watch(self, window, contexts)

    async def _gate(
        self, window: TranscriptWindow, org: CompanyContext = EMPTY
    ) -> tuple[str, float]:
        assert self._graph is not None and self.prompt is not None
        state = AgentState(
            meeting_id=window.meeting_id,
            user_id=window.meeting_id,
            chunk=window,
            context={"org": org},
            org_id=window.org_id,
            agent_id=self.agent_id,
            prompt_version=self.prompt.version,
        )
        try:
            out = await self._graph.ainvoke(state)
        except BudgetBlocked:
            self.blocked = True
            return "", 0.0
        facts = (out.get("retrieved_facts") if isinstance(out, dict) else out.retrieved_facts) or []
        score = (out.get("relevance_score") if isinstance(out, dict) else out.relevance_score) or 0
        return str((facts[0] if facts else {}).get("topic") or ""), float(score)


# --- ADR-0039 C7: already_decided and topic_watch (per closed window) -------------------------


def once_key(org_id: UUID, check: str, *parts: object) -> str:
    return f"nexus:{org_id}:proactive:{check}:" + ":".join(str(p) for p in parts)


def topic_hash(topic: str) -> str:
    return hashlib.sha256(" ".join(topic.lower().split()).encode()).hexdigest()[:16]


def window_words(window: TranscriptWindow) -> int:
    return sum(len(t.text.split()) for t in window.turns)


def _phrase(text: str) -> re.Pattern[str] | None:
    words = text.lower().split()
    if not words:
        return None
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)", re.I)


def topic_terms(topic: str, org: CompanyContext) -> list[str]:
    """The watched topic plus the aliases of a glossary term that is (or is aliased as) it."""
    wanted = " ".join(topic.lower().split())
    out = [topic]
    for term in org.glossary:
        names = [term.term, *term.aliases]
        if any(" ".join(n.lower().split()) == wanted for n in names):
            out.extend(n for n in names if n not in out)
    return out


def said_turn(window: TranscriptWindow, terms: list[str]) -> str | None:
    """The text of the first turn that says any of `terms` outright, else None."""
    patterns = [p for p in (_phrase(t) for t in terms) if p is not None]
    for turn in window.turns:
        if any(p.search(turn.text) for p in patterns):
            return turn.text
    return None


def snippet_of(text: str) -> str:
    return clip(strip_urls(text), MAX_SNIPPET)


def widest_scope(
    watcher: UUID, watcher_ctx: dict[str, Any], contexts: dict[UUID, dict[str, Any] | None]
) -> str:
    """The widest ADR-0013 scope between the watcher and anyone whose words are in the meeting:
    the meeting's content is everyone's, so the grant must cover the furthest of them. Nobody
    with a known context: `org`."""
    scopes = [
        resolve_scope(watcher, watcher_ctx, uid, ctx)
        for uid, ctx in contexts.items()
        if ctx is not None
    ]
    return max(scopes, key=lambda x: SCOPE_WIDTH[x]) if scopes else "org"


def parse_reopen(text: str) -> tuple[str, float]:
    obj = parse_object(text) or {}
    question = obj.get("question")
    score = confidence_of({"confidence": obj.get("score")}) or 0.0
    if obj.get("reopens") is not True or not isinstance(question, str) or not question.strip():
        return "", 0.0
    return " ".join(question.split())[:300], score


def parse_matches(text: str, n: int) -> dict[int, tuple[float, str]]:
    """0-based candidate index -> (confidence, rationale), the best answer per candidate."""
    out: dict[int, tuple[float, str]] = {}
    for raw in parse_candidates(text, key="matches"):
        if not isinstance(raw, dict):
            continue
        index, confidence = raw.get("candidate"), confidence_of(raw)
        if isinstance(index, bool) or not isinstance(index, int) or confidence is None:
            continue
        if 1 <= index <= n and confidence > out.get(index - 1, (-1.0, ""))[0]:
            out[index - 1] = (confidence, text_of(raw, "rationale") or "")
    return out


def parse_watch_topics(text: str, n: int) -> dict[int, float]:
    """0-based topic index -> score."""
    out: dict[int, float] = {}
    for raw in parse_candidates(text, key="topics"):
        if not isinstance(raw, dict):
            continue
        index = raw.get("topic")
        score = confidence_of({"confidence": raw.get("score")})
        if isinstance(index, bool) or not isinstance(index, int) or score is None:
            continue
        if 1 <= index <= n:
            out[index - 1] = max(score, out.get(index - 1, 0.0))
    return out


def _date(raw: str | None) -> str | None:
    if not raw:
        return None
    return raw.strip()[:10] if len(raw.strip()) >= 10 else raw.strip()


async def _claim(redis: Any, key: str) -> bool:
    """`SET NX` with the once-only TTL. A Redis failure is "not claimed": skip, never repeat."""
    try:
        return bool(await redis.set(key, "1", nx=True, ex=ONCE_TTL_S))
    except Exception:  # noqa: BLE001
        logger.warning("proactive once-only claim failed (%s); skipped", key.split(":")[3])
        return False


def _org(contexts: dict[UUID, dict[str, Any] | None]) -> CompanyContext:
    return next((company_of(c) for c in contexts.values() if c), EMPTY)


def _limits(contexts: dict[UUID, dict[str, Any] | None]) -> Thresholds:
    """The tenant's thresholds for this agent, from any context the runtime filled."""
    return thresholds_for(next((c for c in contexts.values() if c), None), AGENT)


def _never_window(window: TranscriptWindow, org: CompanyContext) -> bool:
    """The window names a never-record topic (or an alias) outright."""
    never = NeverRecordPolicy.from_org(org)
    return any(never.hits(t.text) for t in window.turns)


async def _triage(
    agent: ProactiveAgent, key: str, org_id: UUID, max_tokens: int, **values: str
) -> tuple[str, str]:
    """One triage-tier call metered to the tenant. Returns (text, prompt ref)."""
    prompt = agent._prompt(key, org_id)
    out = await agent.router.complete(
        tier="triage",
        system=prompt.system,
        **prompt.user_parts(**values),
        org_id=org_id,
        user_id=None,  # shared by every participant: metered to the tenant
        max_tokens=max_tokens,
        prompt_ref=prompt.ref,
    )
    return out.text, prompt.ref


async def _meeting_title(agent: ProactiveAgent, org_id: UUID, meeting_id: UUID) -> str:
    source = agent._deps.agenda
    if source is None:
        return ""
    try:
        agenda = await source.agenda(org_id, meeting_id)
    except Exception:  # noqa: BLE001
        return ""
    return agenda.title if agenda is not None else ""


async def check_already_decided(
    agent: ProactiveAgent, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
) -> list[Alert]:
    deps = agent._deps
    people = recipients(sorted((u for u, c in contexts.items() if c is not None), key=str))
    if deps.decisions is None or deps.redis is None or not people:
        return []
    if window_words(window) < MIN_WINDOW_WORDS or not await agent.active(window.org_id):
        return []
    org = _org(contexts)
    if _never_window(window, org):
        return []  # ADR-0039 C2
    never = NeverRecordPolicy.from_org(org)
    limits = _limits(contexts)
    transcript = labelled_window(window)[0]
    try:
        text, _ref = await _triage(
            agent,
            REOPEN_GATE_KEY,
            window.org_id,
            80,
            window=transcript,
            **gate_ground_rules({"org": org}),
        )
    except BudgetBlocked:
        agent.blocked = True
        return []
    question, score = parse_reopen(text)
    if not question or score < agent._settings.gate_threshold or never.hits(question):
        return []

    search = deps.decisions

    async def own(user: UUID) -> list[RagPassage]:
        try:
            return await search.search(org_id=window.org_id, user_id=user, query=question)
        except Exception:  # noqa: BLE001 - this participant's search failed: no alert for them
            return []

    # ADR-0033 / ADR-0036 D9: each participant's search runs AS them, so what they are told about
    # is only ever a decision their own retrieval returned.
    hits = dict(zip(people, await asyncio.gather(*(own(u) for u in people)), strict=True))
    best: dict[UUID, RagPassage] = {}
    for found in hits.values():
        for p in found:
            if p.decision_id is None:
                continue
            if p.decision_id not in best or p.score > best[p.decision_id].score:
                best[p.decision_id] = p
    candidates = sorted(
        (p for p in best.values() if not never.hits(p.text)), key=lambda p: -p.score
    )[:MAX_CANDIDATES]
    if not candidates:
        return []
    prompt = agent._prompt(ALREADY_DECIDED_KEY, window.org_id)
    try:
        out = await agent.router.complete(
            tier="extract",
            system=prompt.system,
            **prompt.user_parts(
                question=question,
                window=transcript,
                candidates="\n".join(
                    f"[{i + 1}] {clip(p.text, 600)}" for i, p in enumerate(candidates)
                ),
                **company_values({"org": org}),
            ),
            org_id=window.org_id,
            user_id=None,
            max_tokens=EXTRACT_MAX_TOKENS,
            effort=SYNTHESIS_EFFORT,
            prompt_ref=prompt.ref,
        )
    except BudgetBlocked:
        agent.blocked = True
        return []
    judged = parse_matches(out.text, len(candidates))
    matched = {
        candidates[i].decision_id: (conf, why)
        for i, (conf, why) in judged.items()
        if conf >= ALREADY_DECIDED_MIN_CONFIDENCE
    }
    if not matched:
        return []
    title = await _meeting_title(agent, window.org_id, window.meeting_id)
    by_decision: dict[UUID, list[UUID]] = {}
    for user in people:
        mine = [p.decision_id for p in hits[user] if p.decision_id in matched]
        if not mine:
            continue  # their own retrieval did not return it: they are never told
        decision = max(mine, key=lambda d: (matched[d][0], str(d)))
        if decision is None:
            continue
        key = once_key(window.org_id, "already_decided", window.meeting_id, decision, user)
        if await _claim(deps.redis, key):
            by_decision.setdefault(decision, []).append(user)
    alerts: list[Alert] = []
    for decision_id, to in by_decision.items():
        passage, (confidence, rationale) = best[decision_id], matched[decision_id]
        content = clip(passage.text.strip(), 500)
        decided_at = _date(passage.occurred_at)
        where = f' in "{clip(passage.meeting_title, 80)}"' if passage.meeting_title else ""
        when = f" on {decided_at}" if decided_at else ""
        related = [{"kind": "decision", "id": str(decision_id), "label": clip(content, MAX_LABEL)}]
        if passage.meeting_id is not None:
            related.append(
                {
                    "kind": "meeting",
                    "id": str(passage.meeting_id),
                    "label": clip(passage.meeting_title or "", MAX_LABEL),
                }
            )
        alerts.append(
            Alert(
                alert_type="already_decided",
                subject_kind="decision",
                subject_id=decision_id,
                subject_key=f"{window.meeting_id}:{decision_id}",
                meeting_id=window.meeting_id,
                title=f"Already decided: {clip(content, 90)}",
                body=(
                    f"This sounds like a question the team already settled{when}{where}: "
                    f'"{clip(content, 200)}".'
                ),
                confidence=confidence,
                rationale=rationale or "An earlier confirmed decision answers this question.",
                related=related,
                link=f"/decisions/{decision_id}",
                to=to,
                prompt=prompt.ref,
                model_tier=TIER_NAMES.get(out.tier, out.tier),
                source_chunk_ids=window.chunk_ids,
                thresholds=limits,
                extra={
                    "decision_id": str(decision_id),
                    "decision_content": content,
                    "decided_at": decided_at,
                    "meeting_id": str(window.meeting_id),
                    "meeting_title": title,
                },
            )
        )
    return alerts


async def check_topic_watch(
    agent: ProactiveAgent, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
) -> list[Alert]:
    deps = agent._deps
    org = _org(contexts)
    if not org.topic_watches or deps.redis is None or deps.agenda is None or deps.core is None:
        return []
    if deps.pep is None or deps.user_context is None or window_words(window) < MIN_WINDOW_WORDS:
        return []
    text = " ".join(t.text for t in window.turns)
    if said_turn(window, list(org.never_record_topics)) is not None or _never_window(window, org):
        return []  # a never-record topic was said: nothing from this window leaves the meeting
    limits = _limits(contexts)
    if not await agent.active(window.org_id):
        return []
    try:
        agenda = await deps.agenda.agenda(window.org_id, window.meeting_id)
    except Exception:  # noqa: BLE001
        agenda = None
    if agenda is None or agenda.confidential is not False:
        return []  # B2: never from a confidential meeting; unknown counts as confidential
    try:
        participants = await deps.core.participants(window.org_id, window.meeting_id)
    except Exception:  # noqa: BLE001
        participants = None
    if participants is None:
        return []  # cannot tell who is in the meeting: fail closed
    present = (
        {p.user_id for p in participants} | set(contexts) | {p.user_id for p in agenda.proxies}
    )
    watches = [w for w in org.topic_watches if w.user_id not in present]
    if not watches or not text.strip():
        return []
    topics: list[str] = []
    for w in watches:
        if all(topic_hash(t) != topic_hash(w.topic) for t in topics):
            topics.append(w.topic)
    topics = topics[:MAX_WATCH_TOPICS]

    matched: dict[str, tuple[float, str]] = {}  # topic hash -> (confidence, snippet text)
    rest: list[str] = []
    for topic in topics:
        turn = said_turn(window, topic_terms(topic, org))
        if turn is not None:
            matched[topic_hash(topic)] = (EXACT_MATCH_CONFIDENCE, turn)
        else:
            rest.append(topic)
    prompt_ref, tier = "", "none"
    if rest:
        try:
            answer, prompt_ref = await _triage(
                agent,
                TOPIC_WATCH_KEY,
                window.org_id,
                120,
                topics="\n".join(f"{n}. {t}" for n, t in enumerate(rest, start=1)),
                window=labelled_window(window)[0],
                **company_values({"org": org}),
            )
            tier = TIER_NAMES["triage"]
        except BudgetBlocked:
            agent.blocked = True
            answer = ""
        for index, score in parse_watch_topics(answer, len(rest)).items():
            if score >= agent._settings.gate_threshold:
                longest = max(window.turns, key=lambda t: len(t.text)).text
                matched[topic_hash(rest[index])] = (score, longest)
    if not matched:
        return []
    title = agenda.title
    alerts: list[Alert] = []
    sent = 0
    contexts_cache: dict[UUID, dict[str, Any] | None] = {}
    for topic in topics:
        h = topic_hash(topic)
        if h not in matched or sent >= MAX_RECIPIENTS:
            continue
        confidence, said = matched[h]
        surfaced = band(confidence, limits) == "surfaced"
        if band(confidence, limits) == "dropped":
            continue
        to: list[UUID] = []
        for watcher in sorted({x.user_id for x in watches if topic_hash(x.topic) == h}, key=str):
            if sent + len(to) >= MAX_RECIPIENTS:
                break
            if watcher not in contexts_cache:
                try:
                    contexts_cache[watcher] = await deps.user_context(window.org_id, watcher)
                except Exception:  # noqa: BLE001
                    contexts_cache[watcher] = None
            ctx = contexts_cache[watcher]
            if ctx is None:
                continue  # unknown watcher: no grant can be checked
            scope = widest_scope(watcher, ctx, contexts)
            if not await deps.pep.authorize(
                window.org_id, grantee_role(ctx), "meeting_context", scope
            ):
                continue  # ADR-0013: deny by default
            key = once_key(window.org_id, "topic_watch", window.meeting_id, watcher, h)
            if surfaced and not await _claim(deps.redis, key):
                continue  # already told about this topic in this meeting
            if not surfaced:
                try:
                    if await deps.redis.exists(key):
                        continue
                except Exception:  # noqa: BLE001
                    continue
            to.append(watcher)
        if not to:
            continue
        sent += len(to)
        snippet = snippet_of(said)
        name = clip(title, 80) if title else "a meeting"
        alerts.append(
            Alert(
                alert_type="topic_watch",
                subject_kind="meeting",
                subject_id=window.meeting_id,
                subject_key=f"{window.meeting_id}:{h}",
                meeting_id=window.meeting_id,
                title=f"{clip(topic, 60)} came up in {name}",
                body=f'A topic you watch, "{clip(topic, 80)}", is being discussed: "{snippet}"',
                confidence=confidence,
                rationale=(
                    "The topic was named in the meeting."
                    if confidence == EXACT_MATCH_CONFIDENCE
                    else "The discussion is about this topic."
                ),
                related=[
                    {"kind": "meeting", "id": str(window.meeting_id), "label": clip(title, 80)}
                ],
                link=f"/meetings/{window.meeting_id}",
                to=to,
                prompt=prompt_ref if confidence != EXACT_MATCH_CONFIDENCE else "",
                model_tier=tier if confidence != EXACT_MATCH_CONFIDENCE else "none",
                source_chunk_ids=window.chunk_ids,
                thresholds=limits,
                extra={
                    "topic": topic,
                    "meeting_id": str(window.meeting_id),
                    "meeting_title": title,
                    "snippet": snippet,
                },
            )
        )
    return alerts
