"""`MeetingAgent`: one per meeting, follows the meeting against its agenda for the host.

mytheus-docs ADR-0038 D7 and D8. Meeting-scope and `multi_user` (the runtime calls
`process_meeting` once per closed 30 s window, and the one result goes to the host, not to the
roster's first user). Per window:

1. **Agenda** from core-api's internal route (`agenda.py`), cached per meeting for
   `MEETING_AGENT_AGENDA_TTL_S` (default 60 s). Any failure, no agenda items, or no host: the
   window is skipped quietly, before any model call.
2. **Budget**: the tenant's budget state; `block` skips the window (no model call, nothing
   emitted). `throttle` changes nothing: the agent already runs on the fast tier.
3. **One fast-tier call** (Haiku 5.5), `meeting_agent@vN`, through the router (budget-gated,
   metered on `usage-events` to the host). It sees the agenda with statuses and how long the
   in-progress item has run, the previous meeting's open actions as plain strings ("C1"...),
   and the window with every speaker as "Participant N". No ids, names or emails; the router
   redacts PII on top. It answers `updates`, `off_agenda` and `raised`.
4. **Updates**: `in_progress` / `covered` only, confidence `> 0.7` (the surfaced band), at most
   one per item, never for an item the host set (`status_source = host`), never for a covered
   or skipped item, never backwards, never a no-op. Notes are clipped to 300 chars.
5. **One nudge at most**, and at most one per meeting per 5 minutes of meeting time, in this
   order: `over_time` (computed, no model: an item in progress longer than its
   `duration_minutes`, once per item), `unaddressed` (a carried-forward open action not raised
   by `MEETING_UNADDRESSED_AT` = 70% of the scheduled time, once per item; scheduled time is
   `scheduled_end - scheduled_start`, else the sum of the items' minutes, else never), then
   `off_agenda` (model confidence `>= 0.75`). Content is clipped to 200 chars.
6. **One `AgentResult`** of kind `meeting_update` when there is at least one update or a nudge,
   `user_id` = the host, payload exactly D7's `{"updates": [...], "nudge": {...} | null,
   "surfaced": true}`. One record per window because the `agent-results` idempotency key is
   `agentresult:{meeting_id}:{window_id}:{user_id}:{agent}` (ADR-0032).

Time is meeting time (the window's `start_ms` / `end_ms`, transcription's meeting timeline), so a
replayed window computes the same durations. State that must outlive a window (when each item
started, statuses this agent already proposed, which over_time / unaddressed nudges went out,
which carried items were raised, when the last nudge went out) lives in one Redis hash,
`nexus:{org_id}:meeting_agent:{meeting_id}`, 24 h TTL. A Redis failure skips the window: without
the rate-limit state the agent cannot promise the host one nudge per 5 minutes.

The agent never writes Postgres: core-api applies `updates` (host wins, never covered ->
in_progress) and never persists the nudge.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from uuid import UUID, uuid5

from langgraph.graph import END, StateGraph

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow
from services.agent_runtime_service.agenda import (
    Agenda,
    AgendaItem,
    AgendaSource,
    agenda_from_env,
)
from services.agent_runtime_service.agents.candidates import (
    CANDIDATE_NS,
    confidence_of,
    text_of,
)
from services.agent_runtime_service.agents.candidates import render_window as labelled_window
from services.agent_runtime_service.agents.context_gap_agent import parse_object
from services.agent_runtime_service.agents.personal_listener_agent import SURFACE_ABOVE

logger = logging.getLogger(__name__)

AGENT = "MeetingAgent"
KIND = "meeting_update"
NudgeType = Literal["off_agenda", "over_time", "unaddressed"]

MAX_NOTE = 300
MAX_NUDGE = 200
OFF_AGENDA_MIN_CONFIDENCE = 0.75
UNADDRESSED_AT = 0.7  # fraction of the scheduled time
NUDGE_INTERVAL_MS = 5 * 60_000
STATE_TTL_S = 24 * 3600
DEFAULT_AGENDA_TTL_S = 60.0
_RANK = {"pending": 0, "in_progress": 1, "covered": 2}


def state_key(org_id: UUID, meeting_id: UUID) -> str:
    return f"nexus:{org_id}:meeting_agent:{meeting_id}"


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def carried_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def nudge_id(meeting_id: UUID, kind: str, subject: str) -> str:
    """Deterministic: a replayed window (or the same over_time item) yields the same id."""
    return str(uuid5(CANDIDATE_NS, f"{AGENT}:{meeting_id}:{kind}:{subject}"))


@dataclass
class MeetingSettings:
    enabled: bool = True
    agenda_ttl_s: float = DEFAULT_AGENDA_TTL_S

    @classmethod
    def from_env(cls) -> MeetingSettings:
        try:
            raw = os.environ.get("MEETING_AGENT_AGENDA_TTL_S", "")
            ttl = max(0.0, float(raw)) if raw.strip() else DEFAULT_AGENDA_TTL_S
        except ValueError:
            ttl = DEFAULT_AGENDA_TTL_S
        return cls(enabled=os.environ.get("MEETING_AGENT", "1") != "0", agenda_ttl_s=ttl)


@dataclass
class MeetingDeps:
    """No `agenda` (no `CORE_URL` / `CORE_CALLER_TOKEN`) or no `redis`: the agent never runs."""

    agenda: AgendaSource | None = None
    redis: Any = None
    clock: Callable[[], float] = time.monotonic


def deps_from_env() -> MeetingDeps:
    agenda = agenda_from_env()
    if agenda is None:
        return MeetingDeps()
    from redis.asyncio import from_url

    return MeetingDeps(
        agenda=agenda, redis=from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
    )


@dataclass
class MeetingState:
    """The Redis hash, decoded. Field names are `{kind}:{subject}`."""

    status: dict[str, str] = field(default_factory=dict)  # item_id -> status this agent proposed
    started: dict[str, int] = field(default_factory=dict)  # item_id -> meeting ms it went live
    over_time: set[str] = field(default_factory=set)  # item_ids nudged over_time
    raised: set[str] = field(default_factory=set)  # carried keys someone raised
    unaddressed: set[str] = field(default_factory=set)  # carried keys nudged unaddressed
    last_nudge_ms: int | None = None
    dirty: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[Any, Any]) -> MeetingState:
        s = cls()
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            value = v.decode() if isinstance(v, bytes) else str(v)
            kind, _, subject = key.partition(":")
            try:
                if kind == "status" and value in _RANK:
                    s.status[subject] = value
                elif kind == "started":
                    s.started[subject] = int(value)
                elif kind == "over_time":
                    s.over_time.add(subject)
                elif kind == "raised":
                    s.raised.add(subject)
                elif kind == "unaddressed":
                    s.unaddressed.add(subject)
                elif key == "last_nudge_ms":
                    s.last_nudge_ms = int(value)
            except ValueError:
                continue
        return s

    def set(self, key: str, value: str | int) -> None:
        self.dirty[key] = str(value)


def effective_status(item: AgendaItem, state: MeetingState) -> str:
    """The route's status, moved forward by what this agent already proposed (core-api may not
    have applied it yet, and the agenda is cached). The host's word and terminal states win."""
    if item.status_source == "host" or item.status in ("covered", "skipped"):
        return item.status
    proposed = state.status.get(str(item.id))
    if proposed and _RANK[proposed] > _RANK.get(item.status, 0):
        return proposed
    return item.status


def render_agenda(items: list[AgendaItem], status: dict[UUID, str], ran: dict[UUID, int]) -> str:
    lines = []
    for n, item in enumerate(items, start=1):
        parts = [status[item.id].replace("_", " ")]
        if item.duration_minutes:
            parts.append(f"{item.duration_minutes} min planned")
        if item.id in ran:
            parts.append(f"running for {ran[item.id]} min")
        lines.append(f"{n}. {item.title} ({', '.join(parts)})")
    return "\n".join(lines)


def render_carried(open_actions: list[str]) -> str:
    return "\n".join(f"C{n}. {t}" for n, t in enumerate(open_actions, start=1)) or "(none)"


@dataclass(frozen=True)
class Assessment:
    updates: list[dict[str, Any]]
    off_agenda: tuple[str, float] | None
    raised: list[int]


def parse_assessment(text: str) -> Assessment:
    """Tolerant: anything malformed is dropped piece by piece, never raises."""
    obj = parse_object(text) or {}
    raw_updates = obj.get("updates")
    updates = [
        u for u in (raw_updates if isinstance(raw_updates, list) else []) if isinstance(u, dict)
    ]
    off = obj.get("off_agenda")
    off_agenda = None
    if isinstance(off, dict):
        content, confidence = text_of(off, "content"), confidence_of(off)
        if content is not None and confidence is not None:
            off_agenda = (content, confidence)
    raw_raised = obj.get("raised")
    raised = [
        n
        for n in (raw_raised if isinstance(raw_raised, list) else [])
        if isinstance(n, int) and not isinstance(n, bool)
    ]
    return Assessment(updates=updates, off_agenda=off_agenda, raised=raised)


class MeetingAgent(BaseAgent):
    name: ClassVar[str] = AGENT
    prompt_key: ClassVar[str] = "meeting_agent"
    scope: ClassVar[Literal["participant", "meeting"]] = "meeting"
    kind: ClassVar[str] = KIND
    multi_user: ClassVar[bool] = True  # the runtime calls process_meeting; the result is the host's

    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        deps: MeetingDeps | None = None,
        settings: MeetingSettings | None = None,
    ) -> None:
        super().__init__(loader, router)
        self._deps = deps if deps is not None else MeetingDeps()
        self._settings = settings if settings is not None else MeetingSettings.from_env()
        self._agenda: Agenda | None = None
        self._agenda_at: float | None = None

    def build_graph(self) -> Any:
        async def assess(s: AgentState) -> dict[str, Any]:
            assert self.prompt is not None and s.org_id is not None
            out = await self.router.complete(
                tier="triage",
                system=self.prompt.system,
                user=self.prompt.render_user(**s.context),
                org_id=s.org_id,
                user_id=s.user_id,  # metered to the host
                max_tokens=400,
                prompt_ref=self.prompt.ref,
            )
            return {"retrieved_facts": [{"text": out.text}]}

        g: Any = StateGraph(AgentState)  # langgraph stubs reject async dict-returning nodes
        g.add_node("assess", assess)
        g.set_entry_point("assess")
        g.add_edge("assess", END)
        return g.compile()

    async def _load_agenda(self, org_id: UUID, meeting_id: UUID) -> Agenda | None:
        source = self._deps.agenda
        if source is None:
            return None
        now = self._deps.clock()
        fresh = self._agenda_at is not None and now - self._agenda_at < self._settings.agenda_ttl_s
        if not fresh:
            try:
                agenda = await source.agenda(org_id, meeting_id)
            except Exception:  # noqa: BLE001 - any failure: skip the window quietly
                logger.warning("agenda read failed meeting=%s; window skipped", meeting_id)
                agenda = None
            if agenda is None:
                return None  # a failure is not cached: the next window asks again
            self._agenda, self._agenda_at = agenda, now
        return self._agenda

    async def _load_state(self, org_id: UUID, meeting_id: UUID) -> MeetingState | None:
        try:
            raw = await self._deps.redis.hgetall(state_key(org_id, meeting_id))
        except Exception:  # noqa: BLE001
            logger.warning("meeting agent state unreadable meeting=%s; window skipped", meeting_id)
            return None
        return MeetingState.parse(raw or {})

    async def _save_state(self, org_id: UUID, meeting_id: UUID, state: MeetingState) -> None:
        if not state.dirty:
            return
        key = state_key(org_id, meeting_id)
        try:
            await self._deps.redis.hset(key, mapping=state.dirty)
            await self._deps.redis.expire(key, STATE_TTL_S)
        except Exception:  # noqa: BLE001 - worst case a nudge repeats once
            logger.warning("meeting agent state not saved meeting=%s", meeting_id)

    async def process_meeting(
        self, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
    ) -> list[tuple[UUID, AgentResult]]:
        if self._graph is None or self.prompt is None or self.org_id is None:
            raise RuntimeError(f"{self.name}.process_meeting before spawn")
        if window.org_id != self.org_id:
            raise ValueError(f"{self.name}: window org_id does not match the spawned agent")
        if not self._settings.enabled or self._deps.redis is None:
            return []
        agenda = await self._load_agenda(window.org_id, window.meeting_id)
        if agenda is None or not agenda.items or agenda.host_user_id is None:
            return []  # D7: only for a meeting with at least one agenda item
        host = agenda.host_user_id
        try:
            if await self.router.budget_state(window.org_id) == "block":
                self.blocked = True
                return []
        except Exception:  # noqa: BLE001 - unknown budget: skipped (fail closed)
            return []
        state = await self._load_state(window.org_id, window.meeting_id)
        if state is None:
            return []
        status = {i.id: effective_status(i, state) for i in agenda.items}
        for item in agenda.items:  # an item seen in progress for the first time starts now
            if status[item.id] == "in_progress" and str(item.id) not in state.started:
                state.started[str(item.id)] = window.start_ms
                state.set(f"started:{item.id}", window.start_ms)

        try:
            assessment = await self._assess(window, agenda, status, state, host)
        except BudgetBlocked:
            self.blocked = True
            return []

        updates, confidences = self._updates(assessment, agenda, status, state, window)
        for n in assessment.raised:
            if 1 <= n <= len(agenda.open_actions):
                k = carried_key(agenda.open_actions[n - 1])
                if k not in state.raised:
                    state.raised.add(k)
                    state.set(f"raised:{k}", 1)
        nudge, nudge_confidence = self._nudge(window, agenda, status, state, assessment)
        await self._save_state(window.org_id, window.meeting_id, state)
        if not updates and nudge is None:
            return []
        if nudge_confidence is not None:
            confidences.append(nudge_confidence)
        titles = {str(i.id): i.title for i in agenda.items}
        content = (
            nudge["content"]
            if nudge is not None
            else "; ".join(
                f"{titles[u['item_id']]}: {u['status'].replace('_', ' ')}" for u in updates
            )
        )
        result = AgentResult(
            kind=KIND,
            content=clip(content, MAX_NOTE),
            relevance_score=0.0,
            confidence=max(confidences),
            source_chunk_ids=window.chunk_ids,
            payload={"updates": updates, "nudge": nudge, "surfaced": True},
        )
        return [(host, result)]

    async def _assess(
        self,
        window: TranscriptWindow,
        agenda: Agenda,
        status: dict[UUID, str],
        state: MeetingState,
        host: UUID,
    ) -> Assessment:
        assert self._graph is not None and self.prompt is not None
        ran = {
            i.id: max(0, (window.end_ms - state.started[str(i.id)]) // 60_000)
            for i in agenda.items
            if status[i.id] == "in_progress" and str(i.id) in state.started
        }
        transcript, _labels = labelled_window(window)
        s = AgentState(
            meeting_id=window.meeting_id,
            user_id=host,
            chunk=window,
            context={
                "agenda": render_agenda(agenda.items, status, ran),
                "carried": render_carried(agenda.open_actions),
                "window": transcript,
            },
            org_id=window.org_id,
            agent_id=self.agent_id,
            prompt_version=self.prompt.version,
        )
        out = await self._graph.ainvoke(s)  # BudgetBlocked propagates to process_meeting
        facts = (out.get("retrieved_facts") if isinstance(out, dict) else out.retrieved_facts) or []
        return parse_assessment(str((facts[0] if facts else {}).get("text") or ""))

    @staticmethod
    def _updates(
        assessment: Assessment,
        agenda: Agenda,
        status: dict[UUID, str],
        state: MeetingState,
        window: TranscriptWindow,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        updates: list[dict[str, Any]] = []
        confidences: list[float] = []
        seen: set[UUID] = set()
        for raw in assessment.updates:
            n, new = raw.get("item"), raw.get("status")
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(agenda.items):
                continue
            item = agenda.items[n - 1]
            confidence = confidence_of(raw)
            if item.id in seen or new not in ("in_progress", "covered") or confidence is None:
                continue
            current = status[item.id]
            if item.status_source == "host" or current in ("covered", "skipped"):
                continue  # the host wins; terminal states stay
            if _RANK[str(new)] <= _RANK.get(current, 0) or confidence <= SURFACE_ABOVE:
                continue  # no-op, backwards, or not in the surfaced band
            seen.add(item.id)
            status[item.id] = str(new)
            state.status[str(item.id)] = str(new)
            state.set(f"status:{item.id}", str(new))
            if new == "in_progress" and str(item.id) not in state.started:
                state.started[str(item.id)] = window.start_ms
                state.set(f"started:{item.id}", window.start_ms)
            note = text_of(raw, "note")
            updates.append(
                {
                    "item_id": str(item.id),
                    "status": str(new),
                    "note": clip(note, MAX_NOTE) if note else "",
                }
            )
            confidences.append(confidence)
        return updates, confidences

    @staticmethod
    def _nudge(
        window: TranscriptWindow,
        agenda: Agenda,
        status: dict[UUID, str],
        state: MeetingState,
        assessment: Assessment,
    ) -> tuple[dict[str, Any] | None, float | None]:
        now = window.end_ms
        if state.last_nudge_ms is not None and now - state.last_nudge_ms < NUDGE_INTERVAL_MS:
            return None, None  # one nudge per meeting per 5 minutes

        def emit(
            kind: NudgeType, subject: str, item_id: str | None, content: str, confidence: float
        ) -> tuple[dict[str, Any], float]:
            state.last_nudge_ms = now
            state.set("last_nudge_ms", now)
            nudge = {
                "id": nudge_id(window.meeting_id, kind, subject),
                "type": kind,
                "item_id": item_id,
                "content": clip(content, MAX_NUDGE),
            }
            return nudge, confidence

        for item in agenda.items:  # over_time: computed, once per item
            sid = str(item.id)
            if status[item.id] != "in_progress" or not item.duration_minutes:
                continue
            if sid in state.over_time or sid not in state.started:
                continue
            ran_ms = now - state.started[sid]
            if ran_ms > item.duration_minutes * 60_000:
                state.over_time.add(sid)
                state.set(f"over_time:{sid}", 1)
                return emit(
                    "over_time",
                    sid,
                    sid,
                    f'"{item.title}" has run {ran_ms // 60_000} min, '
                    f"over its {item.duration_minutes} min slot.",
                    1.0,
                )

        scheduled = agenda.scheduled_minutes
        if scheduled and now >= UNADDRESSED_AT * scheduled * 60_000:
            for text in agenda.open_actions:  # unaddressed: once per carried item
                k = carried_key(text)
                if k in state.raised or k in state.unaddressed:
                    continue
                state.unaddressed.add(k)
                state.set(f"unaddressed:{k}", 1)
                return emit(
                    "unaddressed", k, None, f"Not raised yet from last meeting: {text}", 1.0
                )

        if assessment.off_agenda is not None:
            content, confidence = assessment.off_agenda
            if confidence >= OFF_AGENDA_MIN_CONFIDENCE:
                return emit("off_agenda", window.window_id, None, content, confidence)
        return None, None
