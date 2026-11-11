"""The agent runtime's input loop: `transcript-chunks` + `meeting-room-events` in, agent output
out. Run: `uv run python -m services.agent_runtime_service.consumer`.

Delivery (at-least-once, zero Postgres):
1. Each message's (topic, partition, offset) is tracked as pending.
2. A `transcript-chunks` final joins its meeting's open window and stays pending until that
   window is processed; every other message is done as soon as it is applied.
3. When windows close, they are run through the agents, the emitter is FLUSHED (every
   produced message acknowledged), and only then are their offsets marked done.
4. The committed offset per partition is the lowest still-pending offset, so a crash replays
   from the oldest unprocessed window, never past it.

A malformed message (no tenant, not a UUID meeting, not a final) is logged and skipped: there
is nothing durable to protect here, and the producer's own DLQ handling is where poison is
kept. A transient failure (Redpanda unacknowledged, an unexpected error in an agent) stops the
loop without committing, so the process restarts from the last safe offset.

Polish (calm and fast): windows are event-driven (`windowing.py`: a cue, a turn end, at most
15 s) and each meeting has its own asyncio task (`Loop`), so one meeting's slow model call
never holds up another's cards. Within a window, users run concurrently. Partitions remain the
unit of horizontal scale.

Agent-to-agent (P10, `a2a/bus.py`) runs beside this loop on the same event loop: the responder
on `agent-requests` (group `agent-runtime-a2a`) and this instance's response router on
`agent-responses` (group `agent-runtime-resp-{hostname}`). Every `poll` runs in a worker thread
so a waiting `ContextGapAgent` (which awaits responses inside `process_window`) is never starved.
Without `AGENT_IDENTITY_SECRET` neither starts and ContextGapAgent runs RAG-only.

Polish wave 2: the `agent-feedback` listener (`feedback.py`, group
`agent-runtime-feedback-{hostname}`, `LEARNING_FEEDBACK=0` leaves it unstarted) runs beside it as
well; it only drops the learning caches, so it never blocks or commits anything of the window loop.

ProactiveAgent (P11, `proactive/loop.py`) runs beside it too: `decisions` + `actions` in group
`agent-runtime-proactive`, with its own producer. If that loop dies (an unacknowledged alert),
the process exits without committing, like the window loop, and restarts from the last safe
offset. `PROACTIVE_AGENT=0` leaves it unstarted.

Minutes for everyone (`minutes.py`) runs beside it too: `meeting-events` in group
`agent-runtime-minutes`, its own producer; on `ended` it writes one set of minutes per
participant and one shared set. It needs `CORE_URL` + `CORE_CALLER_TOKEN` (core-api's internal
minutes source); `MINUTES_AGENT=0` leaves it unstarted.

ADR-0039: one `OrgContextClient` (the company brain over `GetOrgContext`, cached in-process) feeds
every window's prompts, and one `CachedAgendaSource` (core-api's internal agenda route) is shared
by the runtime (proxied participants, C6), the MeetingAgent and ProactiveAgent's window checks
(meeting title and confidentiality, C7).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.runtime import AgentRuntime, as_user_id
from services.agent_runtime_service.windowing import (
    Chunk,
    ClosedWindow,
    Windower,
    WindowSettings,
    merge_windows,
)

logger = logging.getLogger(__name__)

GROUP = "agent-runtime-service"
TOPICS = ["transcript-chunks", "meeting-room-events"]
PROMPTS_DIR = Path(__file__).parent / "prompts"


class OffsetTracker:
    """Lowest-pending-offset commit tracking per (topic, partition)."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, int], set[int]] = defaultdict(set)
        self._high: dict[tuple[str, int], int] = {}

    def received(self, topic: str, partition: int, offset: int) -> None:
        tp = (topic, partition)
        self._pending[tp].add(offset)
        self._high[tp] = max(self._high.get(tp, -1), offset)

    def done(self, topic: str, partition: int, offset: int) -> None:
        self._pending[(topic, partition)].discard(offset)

    def committable(self) -> dict[tuple[str, int], int]:
        """Next offset to commit (Kafka semantics: the first offset NOT yet processed)."""
        return {
            tp: (min(pending) if pending else self._high[tp] + 1)
            for tp, pending in self._pending.items()
            if tp in self._high
        }


def parse_chunk(value: dict[str, Any], partition: int, offset: int) -> Chunk | None:
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    try:
        meeting_id = UUID(str(value.get("meeting_id")))
    except ValueError:
        return None
    if org_id is None or not value.get("is_final", True):
        return None
    return Chunk(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=str(value["chunk_id"]),
        speaker_id=str(value["speaker_id"]),
        start_ms=int(value["start_ms"]),
        end_ms=int(value["end_ms"]),
        text=str(value["text"]),
        partition=partition,
        offset=offset,
        produced_at=_produced_at(header.get("occurred_at")),
    )


def _produced_at(raw: object) -> float | None:
    """The envelope's `occurred_at` (timestamp-micros; a datetime once decoded by fastavro)."""
    if isinstance(raw, datetime):
        return raw.timestamp()
    if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
        return float(raw) / 1_000_000
    return None


@dataclass
class _Job:
    windows: list[ClosedWindow] = field(default_factory=list)
    room: tuple[str, object, str, int, int] | None = None  # (event, participant, topic, p, o)


class _Worker:
    """One meeting's consumer task: its windows and room events, in order, one at a time."""

    def __init__(self) -> None:
        self.jobs: deque[_Job] = deque()
        self.wake = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.busy = False

    def put(self, job: _Job) -> None:
        self.jobs.append(job)
        self.wake.set()


class Loop:
    """The testable core: decoded messages in, windows processed, offsets tracked.

    Polish (calm and fast): one asyncio task per meeting (`_Worker`, keyed by org and meeting).
    The poll loop only windows chunks and hands closed windows to their meeting's task, so one
    meeting's slow model call never delays another meeting's cards. Within a meeting, windows
    (and that meeting's room events) are applied in order, one at a time, which is what keeps
    the in-meeting memory and the roster consistent. A task that falls behind merges the
    windows waiting for it into one (`windowing.merge_windows`) and runs the agents once on
    all of them, instead of paying for every window late. Offsets are marked done only after
    the window is processed AND the emitter is flushed; a task that fails (an unacknowledged
    produce) is surfaced by `check()` on the poll loop, which then stops without committing.
    """

    def __init__(self, runtime: AgentRuntime, emitter: Any, windower: Windower) -> None:
        self.runtime = runtime
        self.emitter = emitter
        self.windower = windower
        self.offsets = OffsetTracker()
        self._workers: dict[tuple[UUID, UUID], _Worker] = {}

    async def handle(self, topic: str, partition: int, offset: int, value: dict[str, Any]) -> None:
        self.check()
        self.offsets.received(topic, partition, offset)
        if topic == "transcript-chunks":
            chunk = parse_chunk(value, partition, offset)
            if chunk is None:
                logger.warning("transcript-chunks: skipping malformed message at %s", offset)
                self.offsets.done(topic, partition, offset)
            else:
                self._submit(self.windower.add(chunk))
        elif topic == "meeting-room-events":
            self._room_event(value, topic, partition, offset)
        else:
            self.offsets.done(topic, partition, offset)

    def _room_event(self, value: dict[str, Any], topic: str, partition: int, offset: int) -> None:
        org_id = parse_org_id((value.get("header") or {}).get("org_id"))
        try:
            meeting_id = UUID(str(value.get("meeting_id")))
        except ValueError:
            self.offsets.done(topic, partition, offset)
            return
        if org_id is None:
            self.offsets.done(topic, partition, offset)
            return
        event = str(value.get("event_type"))
        self._worker(org_id, meeting_id).put(
            _Job(room=(event, value.get("participant_id"), topic, partition, offset))
        )

    async def tick(self) -> None:
        """Close windows whose speaker went quiet or that waited long enough."""
        self.check()
        self._submit(self.windower.expire_idle())

    def check(self) -> None:
        """Raise the error of any meeting task that died (nothing it held is committed)."""
        for key, worker in list(self._workers.items()):
            task = worker.task
            if task is not None and task.done():
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
                if not worker.jobs:
                    self._workers.pop(key, None)

    async def drain(self) -> None:
        """Wait until every meeting task has applied everything queued (tests, shutdown)."""
        while True:
            self.check()
            pending = [w for w in self._workers.values() if w.busy or w.jobs]
            if not pending:
                return
            await asyncio.sleep(0.001)

    async def close(self) -> None:
        for worker in self._workers.values():
            if worker.task is not None:
                worker.task.cancel()
        await asyncio.gather(
            *(w.task for w in self._workers.values() if w.task is not None),
            return_exceptions=True,
        )

    def _submit(self, closed: list[ClosedWindow]) -> None:
        for cw in closed:
            w = cw.window
            self._worker(w.org_id, w.meeting_id).put(_Job(windows=[cw]))

    def _worker(self, org_id: UUID, meeting_id: UUID) -> _Worker:
        key = (org_id, meeting_id)
        worker = self._workers.get(key)
        if worker is None or (worker.task is not None and worker.task.done()):
            worker = _Worker()
            self._workers[key] = worker
            worker.task = asyncio.create_task(self._run_worker(key, worker))
        return worker

    async def _run_worker(self, key: tuple[UUID, UUID], worker: _Worker) -> None:
        while True:
            if not worker.jobs:
                worker.wake.clear()
                await worker.wake.wait()
                continue
            job = worker.jobs.popleft()
            worker.busy = True
            try:
                ended = False
                if job.windows:
                    # catch up: every window already waiting behind this one runs with it
                    while worker.jobs and worker.jobs[0].windows:
                        job.windows.extend(worker.jobs.popleft().windows)
                    await self._run_windows(job.windows)
                elif job.room is not None:
                    ended = await self._apply_room(key, job.room)
            finally:
                worker.busy = False
            if ended and not worker.jobs:
                return  # the meeting is over; a later message starts a fresh task

    async def _apply_room(
        self, key: tuple[UUID, UUID], room: tuple[str, object, str, int, int]
    ) -> bool:
        org_id, meeting_id = key
        event, participant, topic, partition, offset = room
        leaver = as_user_id(participant) if event == "left" else None
        if leaver is not None and self.runtime.roster(org_id, meeting_id) == {leaver}:
            # The last participant is leaving: the meeting ends. Its open window runs while they
            # are still in the roster, so their notes get the meeting's last words too (applying
            # `left` first ran that window only for the people who spoke in it), then
            # `end_meeting` flushes whatever the accumulating agents still held.
            await self._run_windows(self.windower.close_meeting(org_id, meeting_id))
            await self.runtime.end_meeting(org_id, meeting_id)
            self.offsets.done(topic, partition, offset)
            return True
        await self.runtime.on_room_event(org_id, meeting_id, event, participant)
        ended = event == "left" and not self.runtime.roster(org_id, meeting_id)
        if ended:
            await self._run_windows(self.windower.close_meeting(org_id, meeting_id))
            await self.runtime.end_meeting(org_id, meeting_id)
        self.offsets.done(topic, partition, offset)
        return ended

    async def _run_windows(self, closed: list[ClosedWindow]) -> None:
        if not closed:
            return
        merged = merge_windows(closed)
        if len(closed) > 1:
            logger.info(
                "meeting %s fell behind: %d windows merged into %s",
                merged.window.meeting_id,
                len(closed),
                merged.window.window_id,
            )
        await self.runtime.process_window(merged.window)
        # every produced message acknowledged before any offset is released; EmitError ends
        # this meeting's task and `check()` stops the poll loop before it commits
        await asyncio.to_thread(self.emitter.flush)
        for partition, offset in merged.offsets:
            self.offsets.done("transcript-chunks", partition, offset)


async def run() -> None:  # pragma: no cover - the confluent-kafka harness
    from confluent_kafka import Consumer, Producer, TopicPartition
    from redis.asyncio import from_url

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.llm_cache import ResponseCache
    from libs.shared.metrics import serve_metrics, set_service
    from libs.shared.org_context import OrgContextClient
    from libs.shared.pii import EgressRedactor
    from libs.shared.prompts import PromptLoader
    from libs.shared.router import LiteLLMRouter
    from libs.shared.user_context import PdpClient, UserContextClient
    from services.agent_runtime_service.a2a.bus import (
        REQUESTS_TOPIC,
        RESPONSES_TOPIC,
        A2AClient,
        Responder,
        ResponseRouter,
        consumer_configs,
        run_request_loop,
        run_response_loop,
    )
    from services.agent_runtime_service.a2a.identity import signer_from_env
    from services.agent_runtime_service.a2a.pep import Pep
    from services.agent_runtime_service.a2a.related import related_from_env
    from services.agent_runtime_service.agenda import CachedAgendaSource, agenda_from_env
    from services.agent_runtime_service.agents import registered_agents
    from services.agent_runtime_service.agents.context_gap_agent import (
        ContextGapDeps,
        ContextGapSettings,
        rag_from_env,
    )
    from services.agent_runtime_service.agents.meeting_agent import MeetingDeps
    from services.agent_runtime_service.agents.proactive_agent import (
        ProactiveAgent,
        ProactiveSettings,
    )
    from services.agent_runtime_service.agents.proactive_agent import (
        deps_from_env as proactive_from_env,
    )
    from services.agent_runtime_service.emitter import Emitter
    from services.agent_runtime_service.feedback import TOPIC as FEEDBACK_TOPIC
    from services.agent_runtime_service.feedback import FeedbackListener, run_feedback_loop
    from services.agent_runtime_service.feedback import (
        consumer_config as feedback_consumer_config,
    )
    from services.agent_runtime_service.learning import LearningClient
    from services.agent_runtime_service.memory import MeetingLedger
    from services.agent_runtime_service.minutes import TOPIC as MINUTES_TOPIC
    from services.agent_runtime_service.minutes import (
        MinutesLoop,
        MinutesWriter,
        run_minutes_loop,
    )
    from services.agent_runtime_service.minutes import (
        consumer_config as minutes_consumer_config,
    )
    from services.agent_runtime_service.minutes import source_from_env as minutes_from_env
    from services.agent_runtime_service.policy import AgentPolicyClient
    from services.agent_runtime_service.proactive.loop import (
        TOPICS as PROACTIVE_TOPICS,
    )
    from services.agent_runtime_service.proactive.loop import (
        ProactiveLoop,
        run_proactive_loop,
    )
    from services.agent_runtime_service.proactive.loop import (
        consumer_config as proactive_consumer_config,
    )
    from services.agent_runtime_service.translation import enabled as translation_enabled
    from services.agent_runtime_service.translation import start_translation

    logging.basicConfig(level=logging.INFO)
    set_service(GROUP)
    serve_metrics()
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    codec = AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    producer = Producer(
        {"bootstrap.servers": brokers, "acks": "all", "enable.idempotence": True, "linger.ms": 5}
    )
    emitter = Emitter(producer, codec)
    redis = from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
    router = LiteLLMRouter(
        BudgetGate(redis, BillingRefresher()),
        emitter,
        redactor=EgressRedactor(redis),
        cache=ResponseCache(redis),
    )
    loader = PromptLoader(PROMPTS_DIR)
    contexts = UserContextClient()
    pdp = PdpClient()
    settings = ContextGapSettings.from_env()
    signer = signer_from_env()
    # A separate producer for the deadline-bound A2A traffic: no batching delay, and its
    # delivery errors never fail a window commit (the window emitter has its own error list).
    a2a_emitter = Emitter(
        Producer(
            {
                "bootstrap.servers": brokers,
                "acks": "all",
                "enable.idempotence": True,
                "linger.ms": 0,
            }
        ),
        codec,
        on_error=lambda err: logger.warning("a2a delivery failed: %s", err),
    )
    responses = ResponseRouter()
    pep = Pep(redis, pdp)
    rag = rag_from_env(settings.deadline_ms / 1000)
    gap = ContextGapDeps(
        rag=rag,
        a2a=A2AClient(a2a_emitter, responses, signer) if signer is not None else None,
        pep=pep,
        related=related_from_env(),
        user_context=contexts.get,
    )
    org_contexts = OrgContextClient()
    agenda_client = agenda_from_env()
    agenda = CachedAgendaSource(agenda_client) if agenda_client is not None else None
    proactive_settings = ProactiveSettings.from_env()
    proactive = proactive_from_env()
    # ADR-0039 C7: once-only claims in Redis, the meeting's title / confidentiality, and the
    # PEP + user contexts to authorize topic watchers
    proactive.redis, proactive.agenda, proactive.pep = redis, agenda, pep
    proactive.user_context = contexts.get
    # Polish: one threshold source (core-api's agent_confidence_overrides, cached) and the
    # company brain for the checks the proactive loop triggers (never record).
    agent_policy = AgentPolicyClient()
    proactive.org_context, proactive.policy = org_contexts.get, agent_policy.overrides
    # Polish wave 2: per-person thresholds, guidance examples and memory across meetings, from
    # core-api (cached; an agent-feedback event drops the caches, see below).
    learning = LearningClient(policy=agent_policy)

    async def proxied(org_id: UUID, meeting_id: UUID) -> list[UUID]:
        if agenda is None:
            return []
        return [p.user_id for p in await agenda.proxies(org_id, meeting_id)]

    runtime = AgentRuntime(
        sink=emitter,
        user_context=contexts.get,
        org_context=org_contexts.get,
        proxies=proxied if agenda is not None else None,
        policy=agent_policy.overrides,
        learning=learning,
        agents=registered_agents(
            loader,
            router,
            context_gap=gap,
            proactive=proactive,
            # ADR-0038 D7: the agenda over core-api's internal route; rate-limit state in the
            # shared Redis under nexus:{org_id}:meeting_agent:{meeting_id}
            meeting=MeetingDeps(agenda=agenda, redis=redis),
        ),
        max_concurrency=int(os.environ.get("AGENT_MAX_CONCURRENCY", "8")),
        # polish: memory inside the meeting, nexus:{org_id}:meeting-ledger:{meeting_id}
        ledger=MeetingLedger(redis),
        redis=redis,  # off the record: no agent runs (libs/shared/off_record.py)
    )
    stop = asyncio.Event()
    background: list[asyncio.Task[None]] = []
    a2a_consumers: list[Any] = []
    if signer is not None:
        configs = consumer_configs(brokers)
        requests_consumer = Consumer(configs["requests"])
        requests_consumer.subscribe([REQUESTS_TOPIC])
        responses_consumer = Consumer(configs["responses"])
        responses_consumer.subscribe([RESPONSES_TOPIC])
        a2a_consumers = [requests_consumer, responses_consumer]
        responder = Responder(
            signer=signer,
            pep=pep,
            rag=rag,
            user_context=contexts.get,
            emitter=a2a_emitter,
            redis=redis,
        )
        background = [
            asyncio.create_task(run_request_loop(requests_consumer, codec, responder, stop)),
            asyncio.create_task(run_response_loop(responses_consumer, codec, responses, stop)),
        ]
    else:
        logger.warning("AGENT_IDENTITY_SECRET unset: agent-to-agent off, ContextGapAgent RAG-only")
    if proactive_settings.enabled:
        proactive_consumer = Consumer(proactive_consumer_config(brokers))
        proactive_consumer.subscribe(PROACTIVE_TOPICS)
        a2a_consumers.append(proactive_consumer)
        alert_emitter = Emitter(
            Producer({"bootstrap.servers": brokers, "acks": "all", "enable.idempotence": True}),
            codec,
        )
        proactive_loop = ProactiveLoop(
            ProactiveAgent(loader, router, deps=proactive, settings=proactive_settings),
            alert_emitter,
        )
        background.append(
            asyncio.create_task(run_proactive_loop(proactive_consumer, codec, proactive_loop, stop))
        )
    if os.environ.get("LEARNING_FEEDBACK", "1") != "0":
        # Every instance hears every feedback event (its own group) and drops what it cached.
        feedback_consumer = Consumer(feedback_consumer_config(brokers))
        feedback_consumer.subscribe([FEEDBACK_TOPIC])
        a2a_consumers.append(feedback_consumer)
        background.append(
            asyncio.create_task(
                run_feedback_loop(feedback_consumer, codec, FeedbackListener(learning), stop)
            )
        )
    minutes_source = minutes_from_env()
    if minutes_source is not None and os.environ.get("MINUTES_AGENT", "1") != "0":
        minutes_consumer = Consumer(minutes_consumer_config(brokers))
        minutes_consumer.subscribe([MINUTES_TOPIC])
        a2a_consumers.append(minutes_consumer)
        minutes_emitter = Emitter(
            Producer({"bootstrap.servers": brokers, "acks": "all", "enable.idempotence": True}),
            codec,
        )
        minutes_loop = MinutesLoop(
            MinutesWriter(
                loader,
                router,
                source=minutes_source,
                # Strict: minutes are durable and shareable, so an unknown never-record list is
                # retried, never treated as "none".
                org_context=org_contexts.get_strict,
            ),
            minutes_emitter,
            claims=redis,
            settle_s=float(os.environ.get("MINUTES_SETTLE_S", "15")),
        )
        background.append(
            asyncio.create_task(run_minutes_loop(minutes_consumer, codec, minutes_loop, stop))
        )
    else:
        logger.warning("minutes off: no CORE_URL / CORE_CALLER_TOKEN, or MINUTES_AGENT=0")
    if translation_enabled():  # live caption translation (translation.py), same event loop
        translation_task, translation_consumer = start_translation(
            brokers=brokers,
            codec=codec,
            redis=redis,
            router=router,
            org_context=org_contexts.get_strict,
            stop=stop,
        )
        background.append(translation_task)
        a2a_consumers.append(translation_consumer)
    loop = Loop(runtime, emitter, Windower(WindowSettings.from_env()))
    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": GROUP,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe(TOPICS)
    committed: dict[tuple[str, int], int] = {}
    try:
        while True:
            for task in background:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    raise RuntimeError("a background consumer loop died") from task.exception()
            # short poll: the wall-clock window rules (a 1.5 s quiet turn end) run on each tick
            msg = await asyncio.to_thread(consumer.poll, 0.25)
            if msg is not None and not msg.error():
                topic, partition, offset = msg.topic(), msg.partition(), msg.offset()
                raw = msg.value()
                if topic is None or partition is None or offset is None or raw is None:
                    continue
                try:
                    value = codec.deserialize_for(topic, raw)
                except Exception:
                    logger.exception("undecodable message on %s@%s; skipped", topic, offset)
                    loop.offsets.received(topic, partition, offset)
                    loop.offsets.done(topic, partition, offset)
                else:
                    await loop.handle(topic, partition, offset, value)
            await loop.tick()
            due = {tp: o for tp, o in loop.offsets.committable().items() if committed.get(tp) != o}
            if due:
                consumer.commit(
                    offsets=[TopicPartition(t, p, o) for (t, p), o in due.items()],
                    asynchronous=False,
                )
                committed.update(due)
    finally:
        await loop.close()
        stop.set()
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        for c in a2a_consumers:
            c.close()
        consumer.close()
        await contexts.close()
        await pdp.close()
        await org_contexts.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run())
