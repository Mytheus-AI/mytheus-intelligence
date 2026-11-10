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

Roster and agents are in memory. That is safe for the same reason the open windows are: a
meeting's events all arrive on one partition of this consumer group.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from libs.shared.base_agent import BaseAgent
from libs.shared.state import AgentResult, AgentState, StreamFn, TranscriptWindow

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], BaseAgent]
ContextFn = Callable[[UUID, UUID], Awaitable[dict[str, Any] | None]]


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


class AgentRuntime:
    def __init__(
        self,
        *,
        sink: Sink,
        user_context: ContextFn,
        agents: list[AgentFactory],
        max_concurrency: int = 8,
    ) -> None:
        self._sink = sink
        self._context = user_context
        self._factories = agents
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

    def _stream_for(self, window: TranscriptWindow, user_id: UUID) -> StreamFn:
        async def stream(seq: int, text: str, done: bool) -> None:
            await self._sink.delta(window=window, user_id=user_id, seq=seq, text=text, done=done)

        return stream

    async def process_window(self, window: TranscriptWindow) -> int:
        """Run every agent on one closed window. Returns the number of results emitted."""
        m = self._meeting(window.org_id, window.meeting_id)
        m.roster.update(u for u in map(as_user_id, window.speaker_ids) if u is not None)
        users = sorted(m.roster, key=str)
        found = await asyncio.gather(*(self._ctx(window, u) for u in users))
        contexts = dict(zip(users, found, strict=True))
        jobs = [self._run_user(m, window, u, contexts[u]) for u in users]
        jobs.append(self._run_multi(m, window, contexts))
        counts = await asyncio.gather(*jobs)
        return sum(counts)

    async def _ctx(self, window: TranscriptWindow, user_id: UUID) -> dict[str, Any] | None:
        async with self._sem:
            return await self._context(window.org_id, user_id)

    async def _run_multi(
        self, m: _Meeting, window: TranscriptWindow, contexts: dict[UUID, dict[str, Any] | None]
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
            for user_id, result in results:
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
                emitted += 1
        return emitted

    async def _run_user(
        self,
        m: _Meeting,
        window: TranscriptWindow,
        user_id: UUID,
        context: dict[str, Any] | None,
    ) -> int:
        if context is None:
            return 0
        emitted = 0
        for factory in self._factories:
            agent = await self._agent(m, factory, window, user_id)
            if getattr(agent, "multi_user", False):
                continue  # run once per window by _run_multi
            if agent.scope == "meeting" and user_id != min(m.roster, key=str):
                continue  # a meeting-scope agent runs once per window, not once per user
            state = AgentState(
                meeting_id=window.meeting_id,
                user_id=user_id,
                chunk=window,
                context=context,
                org_id=window.org_id,
                stream=self._stream_for(window, user_id),
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
            if agent.blocked and user_id not in m.blocked_notified:
                m.blocked_notified.add(user_id)
                await self._sink.notification(
                    org_id=window.org_id,
                    user_id=user_id,
                    source_event_id=f"budget-blocked:{window.meeting_id}",
                    title="AI budget exhausted",
                    body="Live notes are paused for this meeting. Captions continue.",
                )
            if result is not None:
                await self._sink.result(
                    window=window, user_id=user_id, agent=agent.name, result=result
                )
                emitted += 1
        return emitted
