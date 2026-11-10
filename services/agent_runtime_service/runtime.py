"""The spawner and the per-window loop. Plan 7-9 section 3.2 step 12.

Roster. `meeting-room-events` `joined` / `left` carry `participant_id` = the LiveKit identity,
i.e. `users.id` (mytheus-backend `60fa6b7`). The runtime keeps a roster per meeting from them
and spawns one instance of every `participant`-scope agent per user, plus one instance of
every `meeting`-scope agent per meeting. Speakers seen in a window are added too, so a runtime
that restarted mid-meeting (and missed the `joined` events) still serves everyone who talks.
`left` terminates that user's agents; the meeting's last window flushes on `close_meeting`.

`multi_user` agents (P10 `ContextGapAgent`) are meeting-scope but emit one result per
participant: they run once per window in `_run_multi` with every participant's context.

Per closed window, for every (user, agent) pair, concurrently up to `max_concurrency`:
1. `get_user_context(user_id)` over gRPC (skip the user on failure: no context, no spend);
2. `agent.process(AgentState)`; the agent's own graph calls the router, which gates every
   model call on the tenant budget;
3. a result goes to `agent-results`. If an agent hit `BudgetBlocked`, the participant gets one
   in-app "budget exhausted" notification per meeting; captions are untouched.

ADR-0039. With `org_context`, the company brain (`GetOrgContext`, cached in-process) is fetched
once per window and put under `"org"` in every participant's context, for the prompts' company
block and never-record line (C1, C2); without it, or on a failure, nothing is added and every
prompt renders as it did before the brain existed. With `proxies` (core-api's internal agenda
route, cached), each participant whose attendance is `agent` and who is not in the live roster
gets a pass of the agents that serve proxies (`serves_proxies`, PersonalListener only) on every
window, with `"attendance": "agent"` in their context (C6). They are not added to the roster: they
never count as present for meeting-scope agents, ContextGap requesters or the meeting's end.

Polish (stable names, one threshold source, never record). Each meeting keeps one label registry
(`names.assign_labels`): a speaker is "Participant N" from their first appearance to the end of
the meeting, in every agent's prompt. With `policy` (core-api's `agent_confidence_overrides`,
cached), every context carries the tenant's thresholds so each agent bands `surfaced` exactly as
core-api does. A window that names one of the org's never-record topics (or a glossary alias),
and the window after it, run no agent at all. Every result and every streamed delta passes
`names.Postprocess` on its way out: labels become display names (from the contexts fetched
here), and anything naming a never-record topic is dropped.

Roster and agents are in memory. That is safe for the same reason the open windows are: a
meeting's events all arrive on one partition of this consumer group.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from libs.shared import agent_metrics
from libs.shared.base_agent import BaseAgent
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.state import AgentResult, AgentState, StreamFn, TranscriptWindow
from services.agent_runtime_service.memory import (
    CONTEXT_KEY as MEMORY_KEY,
)
from services.agent_runtime_service.memory import (
    KIND_OF_RESULT,
    LedgerEntry,
    MeetingLedger,
    MeetingMemory,
    drop_repeated_suggestions,
)
from services.agent_runtime_service.moments import annotate
from services.agent_runtime_service.names import Namer, Postprocess, assign_labels, speaker_labels
from services.agent_runtime_service.policy import CONTEXT_KEY
from services.agent_runtime_service.windowing import merge_transcripts

RECENT_WINDOWS = 2  # memory inside the meeting: the last two windows' text

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], BaseAgent]
ContextFn = Callable[[UUID, UUID], Awaitable[dict[str, Any] | None]]
OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]
ProxiesFn = Callable[[UUID, UUID], Awaitable[list[UUID]]]
PolicyFn = Callable[[UUID], Awaitable[dict[str, dict[str, float]]]]


class Sink(Protocol):
    async def result(
        self, *, window: TranscriptWindow, user_id: UUID, agent: str, result: AgentResult
    ) -> None: ...

    async def notification(
        self, *, org_id: UUID, user_id: UUID, source_event_id: str, title: str, body: str
    ) -> None: ...

    async def delta(
        self, *, window: TranscriptWindow, user_id: UUID, seq: int, text: str, done: bool = False
    ) -> None: ...


def as_user_id(raw: object) -> UUID | None:
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        return None  # an internal identity (audio tap, egress) or a malformed one


@dataclass
class _Meeting:
    roster: set[UUID] = field(default_factory=set)
    agents: dict[tuple[str, UUID], BaseAgent] = field(default_factory=dict)
    blocked_notified: set[UUID] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)  # speaker id -> "Participant N"
    quiet_until_ms: int = -1  # never record: windows starting before this run no agent
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOWS))
    # (agent name, user) -> the windows an accumulating agent has not run on yet
    pending: dict[tuple[str, UUID], list[TranscriptWindow]] = field(default_factory=dict)


class AgentRuntime:
    def __init__(
        self,
        *,
        sink: Sink,
        user_context: ContextFn,
        agents: list[AgentFactory],
        max_concurrency: int = 8,
        org_context: OrgContextFn | None = None,
        proxies: ProxiesFn | None = None,
        policy: PolicyFn | None = None,
        ledger: MeetingLedger | None = None,
    ) -> None:
        self._sink = sink
        self._ledger = ledger
        self._context = user_context
        self._factories = agents
        self._org_context = org_context
        self._proxies = proxies
        self._policy = policy
        self._meetings: dict[tuple[UUID, UUID], _Meeting] = {}
        self._sem = asyncio.Semaphore(max_concurrency)

    def _meeting(self, org_id: UUID, meeting_id: UUID) -> _Meeting:
        return self._meetings.setdefault((org_id, meeting_id), _Meeting())

    def roster(self, org_id: UUID, meeting_id: UUID) -> set[UUID]:
        return set(self._meeting(org_id, meeting_id).roster)

    async def on_room_event(
        self, org_id: UUID, meeting_id: UUID, event_type: str, participant_id: object
    ) -> None:
        user_id = as_user_id(participant_id)
        if user_id is None:
            return
        m = self._meeting(org_id, meeting_id)
        if event_type == "joined":
            m.roster.add(user_id)
        elif event_type == "left":
            m.roster.discard(user_id)
            for key in [k for k in m.agents if k[1] == user_id]:
                await m.agents.pop(key).terminate()

    async def end_meeting(self, org_id: UUID, meeting_id: UUID) -> None:
        m = self._meetings.pop((org_id, meeting_id), None)
        for agent in (m.agents.values() if m else []):
            await agent.terminate()

    async def _agent(
        self, m: _Meeting, factory: AgentFactory, window: TranscriptWindow, user_id: UUID
    ) -> BaseAgent:
        probe = factory()
        # meeting-scope agents are keyed by the meeting, not by a user
        owner = user_id if probe.scope == "participant" else window.meeting_id
        key = (probe.name, owner)
        if key not in m.agents:
            await probe.spawn(org_id=window.org_id, meeting_id=window.meeting_id, user_id=owner)
            m.agents[key] = probe
        return m.agents[key]

    def _stream_for(self, window: TranscriptWindow, user_id: UUID, post: Postprocess) -> StreamFn:
        async def stream(seq: int, text: str, done: bool) -> None:
            await self._sink.delta(window=window, user_id=user_id, seq=seq, text=text, done=done)

        return post.stream(stream)

    async def process_window(self, window: TranscriptWindow) -> int:
        """Run every agent on one closed window. Returns the number of results emitted."""
        agent_metrics.observe_meeting_window(window)
        m = self._meeting(window.org_id, window.meeting_id)
        assign_labels(m.labels, window)
        m.roster.update(u for u in map(as_user_id, window.speaker_ids) if u is not None)
        users = sorted(m.roster, key=str)
        proxied = [u for u in await self._proxied(window) if u not in m.roster]
        org, overrides, *found = await asyncio.gather(
            self._org(window),
            self._overrides(window),
            *(self._ctx(window, u) for u in [*users, *proxied]),
        )
        never = NeverRecordPolicy.from_org(org)
        memory = await self._memory(m, window)
        if any(never.hits(t.text) for t in window.turns):
            # A never-record topic was named in this window: nothing of it, or of the next
            # window (the talk around a topic rarely names it in every sentence), reaches any
            # agent, so no model is called and nothing is emitted (the indexers skip the same
            # windows, libs/shared/index_guard.py).
            m.quiet_until_ms = window.end_ms + (window.end_ms - window.start_ms)
        if window.start_ms < m.quiet_until_ms:
            logger.info("window %s is never-record; no agent runs", window.window_id)
            agent_metrics.never_record_skip("window", window.org_id)
            m.pending.clear()  # nothing said around it reaches an accumulating agent either
            return 0
        everyone = dict(zip([*users, *proxied], found, strict=True))
        extra: dict[str, Any] = {CONTEXT_KEY: overrides, MEMORY_KEY: memory}
        if org is not None:
            extra["org"] = org
        everyone = {u: (None if c is None else {**c, **extra}) for u, c in everyone.items()}
        post = Postprocess(Namer.build(m.labels, everyone), never)
        contexts = {u: everyone[u] for u in users}
        jobs = [self._run_user(m, window, u, contexts[u], post) for u in users]
        jobs.extend(
            self._run_user(m, window, u, _as_proxy(everyone[u]), post, proxy=True) for u in proxied
        )
        jobs.append(self._run_multi(m, window, contexts, post))
        counts = await asyncio.gather(*jobs)
        labels = speaker_labels(window)
        m.recent.append("\n".join(f"{labels[t.speaker_id]}: {t.text}" for t in window.turns))
        return sum(counts)

    async def _memory(self, m: _Meeting, window: TranscriptWindow) -> MeetingMemory:
        """The last two windows (in process) and the meeting's ledger (Redis, if wired)."""
        memory = MeetingMemory(previous=list(m.recent))
        if self._ledger is not None:
            memory.entries, memory.marks = await self._ledger.load(window.org_id, window.meeting_id)
        return memory

    def _due(
        self, m: _Meeting, agent: BaseAgent, user_id: UUID, window: TranscriptWindow
    ) -> TranscriptWindow | None:
        """The window this agent runs on now, or None to wait. An agent with `min_span_ms`
        (PersonalListener's notes, QuestionAgent's suggestions: fewer, fuller cards) collects
        windows until they span that long, a window carries a cue, or the meeting ends."""
        span = int(getattr(agent, "min_span_ms", 0) or 0)
        if span <= 0:
            return window
        key = (agent.name, user_id)
        held = [*m.pending.pop(key, []), window]
        merged = merge_transcripts(held)
        cue = window.trigger is not None and window.trigger.cue is not None
        if window.final or cue or merged.end_ms - merged.start_ms >= span:
            return merged
        m.pending[key] = held
        return None

    async def _finish(
        self,
        raw: AgentResult,
        window: TranscriptWindow,
        user_id: UUID,
        context: dict[str, Any] | None,
        post: Postprocess,
    ) -> tuple[AgentResult | None, list[LedgerEntry]]:
        """Moments and why lines, the suggestion repeat guard, then names and never-record.
        Returns what may leave the tier and the ledger entries its cards make."""
        memory = (context or {}).get(MEMORY_KEY)
        if raw.kind == "suggestion" and isinstance(memory, MeetingMemory):
            items = raw.payload.get("suggestions")
            if isinstance(items, list):
                kept = drop_repeated_suggestions(items, memory, user_id)
                if not kept:
                    return None, []
                raw.payload["suggestions"] = kept
        annotate(raw, window, user_id, context)
        entries: list[LedgerEntry] = []
        kind = KIND_OF_RESULT.get(raw.kind)
        if kind is not None:
            for c in raw.payload.get("candidates") or []:
                if isinstance(c, dict) and c.get("id") and c.get("content"):
                    entries.append(
                        LedgerEntry(
                            id=str(c["id"]),
                            kind=kind,
                            text=str(c["content"]),
                            moment_id=c.get("moment_id"),
                        )
                    )
        result = post.result(raw)
        if result is None:
            return None, []
        kept_ids = {
            str(c.get("id")) for c in result.payload.get("candidates") or [] if isinstance(c, dict)
        }
        return result, [e for e in entries if e.id in kept_ids]

    async def _remember(self, window: TranscriptWindow, entries: list[LedgerEntry]) -> None:
        if self._ledger is not None and entries:
            await self._ledger.record(window.org_id, window.meeting_id, entries)

    async def _org(self, window: TranscriptWindow) -> CompanyContext | None:
        if self._org_context is None:
            return None
        try:
            return await self._org_context(window.org_id)
        except Exception:  # noqa: BLE001 - no company context: prompts render as before
            logger.warning("org context unavailable org=%s; window runs without it", window.org_id)
            return None

    async def _overrides(self, window: TranscriptWindow) -> dict[str, dict[str, float]]:
        if self._policy is None:
            return {}
        try:
            return await self._policy(window.org_id)
        except Exception:  # noqa: BLE001 - the default thresholds, as core-api without overrides
            logger.warning("agent policy unavailable org=%s; default thresholds", window.org_id)
            return {}

    async def _proxied(self, window: TranscriptWindow) -> list[UUID]:
        if self._proxies is None:
            return []
        try:
            found = await self._proxies(window.org_id, window.meeting_id)
        except Exception:  # noqa: BLE001 - no proxies this window; the live roster still runs
            logger.warning("proxies unavailable meeting=%s", window.meeting_id)
            return []
        return sorted(set(found), key=str)

    async def _ctx(self, window: TranscriptWindow, user_id: UUID) -> dict[str, Any] | None:
        async with self._sem:
            return await self._context(window.org_id, user_id)

    async def _run_multi(
        self,
        m: _Meeting,
        window: TranscriptWindow,
        contexts: dict[UUID, dict[str, Any] | None],
        post: Postprocess,
    ) -> int:
        """Meeting-scope agents that emit one result per participant (`multi_user`, P10's
        `ContextGapAgent`): run once per window with every participant's context. Not under
        the semaphore: they mostly wait on rag-service and agent-responses, not on a model."""
        if not m.roster or all(c is None for c in contexts.values()):
            return 0
        emitted = 0
        for factory in self._factories:
            probe = factory()
            if not getattr(probe, "multi_user", False):
                continue
            agent = await self._agent(m, factory, window, window.meeting_id)
            try:
                results = await agent.process_meeting(window, contexts)  # type: ignore[attr-defined]
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - same rule as _run_user: skip, never crash
                logger.exception(
                    "agent %s failed on window %s; skipped", agent.name, window.window_id
                )
                continue
            for user_id, raw in results:
                result, entries = await self._finish(
                    raw, window, user_id, contexts.get(user_id), post
                )
                if result is None:
                    agent_metrics.never_record_skip("output", window.org_id)
                    continue  # nothing of it may leave the tier (never record)
                try:
                    await self._sink.result(
                        window=window, user_id=user_id, agent=agent.name, result=result
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    # Most likely a registered agent-results schema without this agent's kind
                    # (the AvroCodec serializes against the registry's latest version). Losing
                    # this result beats crash-looping the window and re-billing every call.
                    logger.exception(
                        "could not emit %s result for window %s", agent.name, window.window_id
                    )
                    continue
                await self._remember(window, entries)
                emitted += 1
        return emitted

    async def _run_user(
        self,
        m: _Meeting,
        window: TranscriptWindow,
        user_id: UUID,
        context: dict[str, Any] | None,
        post: Postprocess,
        *,
        proxy: bool = False,
    ) -> int:
        if context is None:
            return 0
        emitted = 0
        for factory in self._factories:
            if proxy and not getattr(factory(), "serves_proxies", False):
                continue  # C6: someone who sent their agent gets notes, nothing else
            agent = await self._agent(m, factory, window, user_id)
            if getattr(agent, "multi_user", False):
                continue  # run once per window by _run_multi
            if agent.scope == "meeting" and user_id != min(m.roster, key=str):
                continue  # a meeting-scope agent runs once per window, not once per user
            due = self._due(m, agent, user_id, window)
            if due is None:
                continue  # an accumulating agent waits for a fuller window
            state = AgentState(
                meeting_id=window.meeting_id,
                user_id=user_id,
                chunk=due,
                context=context,
                org_id=window.org_id,
                stream=self._stream_for(due, user_id, post),
            )
            async with self._sem:
                try:
                    result = await agent.process(state)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - one failed call must not stop the runtime
                    # A provider error (bad key, rate limit, outage) used to escape the window,
                    # kill the consumer before it committed, and replay the same window on every
                    # restart (a crash loop that also blocks the partition, and with a working
                    # key would re-bill). The note for this window is lost; the window completes.
                    logger.exception(
                        "agent %s failed on window %s for user %s; skipped",
                        agent.name,
                        window.window_id,
                        user_id,
                    )
                    result = None
            if agent.blocked and not proxy and user_id not in m.blocked_notified:
                m.blocked_notified.add(user_id)
                await self._sink.notification(
                    org_id=window.org_id,
                    user_id=user_id,
                    source_event_id=f"budget-blocked:{window.meeting_id}",
                    title="AI budget exhausted",
                    body="Live notes are paused for this meeting. Captions continue.",
                )
            entries: list[LedgerEntry] = []
            if result is not None:
                result, entries = await self._finish(result, due, user_id, context, post)
                if result is None:
                    agent_metrics.never_record_skip("output", window.org_id)
            if result is not None:
                await self._sink.result(
                    window=due, user_id=user_id, agent=agent.name, result=result
                )
                await self._remember(due, entries)
                emitted += 1
        return emitted


def _as_proxy(context: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if context is None else {**context, "attendance": "agent"}
