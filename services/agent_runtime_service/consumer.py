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

Windows are processed one at a time on this loop. Within a window, users run concurrently.
That is enough for this project's scale; partitions are the unit of horizontal scale.

Agent-to-agent (P10, `a2a/bus.py`) runs beside this loop on the same event loop: the responder
on `agent-requests` (group `agent-runtime-a2a`) and this instance's response router on
`agent-responses` (group `agent-runtime-resp-{hostname}`). Every `poll` runs in a worker thread
so a waiting `ContextGapAgent` (which awaits responses inside `process_window`) is never starved.
Without `AGENT_IDENTITY_SECRET` neither starts and ContextGapAgent runs RAG-only.

ProactiveAgent (P11, `proactive/loop.py`) runs beside it too: `decisions` + `actions` in group
`agent-runtime-proactive`, with its own producer. If that loop dies (an unacknowledged alert),
the process exits without committing, like the window loop, and restarts from the last safe
offset. `PROACTIVE_AGENT=0` leaves it unstarted.

ADR-0039: one `OrgContextClient` (the company brain over `GetOrgContext`, cached in-process) feeds
every window's prompts, and one `CachedAgendaSource` (core-api's internal agenda route) is shared
by the runtime (proxied participants, C6), the MeetingAgent and ProactiveAgent's window checks
(meeting title and confidentiality, C7).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.windowing import Chunk, ClosedWindow, Windower

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
    )


class Loop:
    """The testable core: decoded messages in, windows processed, offsets tracked."""

    def __init__(self, runtime: AgentRuntime, emitter: Any, windower: Windower) -> None:
        self.runtime = runtime
        self.emitter = emitter
        self.windower = windower
        self.offsets = OffsetTracker()

    async def handle(self, topic: str, partition: int, offset: int, value: dict[str, Any]) -> None:
        self.offsets.received(topic, partition, offset)
        closed: list[ClosedWindow] = []
        if topic == "transcript-chunks":
            chunk = parse_chunk(value, partition, offset)
            if chunk is None:
                logger.warning("transcript-chunks: skipping malformed message at %s", offset)
                self.offsets.done(topic, partition, offset)
            else:
                closed = self.windower.add(chunk)
        elif topic == "meeting-room-events":
            closed = await self._room_event(value)
            self.offsets.done(topic, partition, offset)
        await self.run_closed(closed)

    async def _room_event(self, value: dict[str, Any]) -> list[ClosedWindow]:
        org_id = parse_org_id((value.get("header") or {}).get("org_id"))
        try:
            meeting_id = UUID(str(value.get("meeting_id")))
        except ValueError:
            return []
        if org_id is None:
            return []
        event = str(value.get("event_type"))
        await self.runtime.on_room_event(org_id, meeting_id, event, value.get("participant_id"))
        if event == "left" and not self.runtime.roster(org_id, meeting_id):
            closed = self.windower.close_meeting(org_id, meeting_id)
            await self.run_closed(closed)
            await self.runtime.end_meeting(org_id, meeting_id)
        return []

    async def tick(self) -> None:
        """Close windows of meetings that went quiet."""
        await self.run_closed(self.windower.expire_idle())

    async def run_closed(self, closed: list[ClosedWindow]) -> None:
        for cw in closed:
            await self.runtime.process_window(cw.window)
        if closed:
            self.emitter.flush()  # raises EmitError: nothing below runs, nothing is committed
            for cw in closed:
                for partition, offset in cw.offsets:
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
    from services.agent_runtime_service.delegation.client import delegation_from_env
    from services.agent_runtime_service.emitter import Emitter
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
        delegation=delegation_from_env(),  # polish wave 3: delegated answers
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
    loop = Loop(runtime, emitter, Windower(idle_s=float(os.environ.get("WINDOW_IDLE_S", "15"))))
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
            msg = await asyncio.to_thread(consumer.poll, 1.0)
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
