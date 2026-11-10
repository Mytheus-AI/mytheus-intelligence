"""The `decisions` / `actions` consumer for `ProactiveAgent` (`proactive/loop.py`).

1. Offline: a confirmed decision becomes alerts on `agent-results` through the real `Emitter`,
   flushed before the caller may commit; a malformed record and a failing check are skipped
   (committed, never crash-looped); an unacknowledged alert raises `EmitError` (no commit).
2. The REAL Avro path: `decisions` / `actions` records encoded against the registered
   `decisions-value` / `actions-value` subjects, decoded by the same codec the consumer uses, and
   the alerts encoded against the registered `agent-results-value` (agent `ProactiveAgent`, kind
   `alert`). Skips without a Schema Registry (`MYTHEUS_REQUIRE_REGISTRY=1` refuses to skip).
3. Over a REAL Redpanda: `run_proactive_loop` in group `agent-runtime-proactive`'s config
   consumes a produced decision and the alert lands on `agent-results`. Skips without a broker.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import pytest

from services.agent_runtime_service.emitter import EmitError, Emitter
from services.agent_runtime_service.proactive.loop import (
    GROUP,
    TOPICS,
    ProactiveLoop,
    consumer_config,
    run_proactive_loop,
)
from tests.a2a_helpers import JsonCodec, LoopbackProducer, ScriptedRag, rag_passage
from tests.test_proactive_agent import (
    ACTION,
    ALEX,
    DECISION,
    MEETING,
    ORG,
    PRIOR,
    PRIYA,
    SAM,
    FakeCore,
    ctx,
    make,
)

OTHER_DECISION = uuid.uuid4()


def _header(org: uuid.UUID, idem: str) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "idempotency_key": idem,
        "org_id": str(org),
        "schema_version": 1,
        "producer": "core-api",
        "occurred_at": int(time.time() * 1_000_000),
        "correlation_id": None,
        "causation_id": None,
    }


def decision_value(
    org: uuid.UUID = ORG, decision_id: uuid.UUID = DECISION, supersedes: uuid.UUID | None = None
) -> dict[str, Any]:
    return {
        "header": _header(org, f"decision:{decision_id}:confirmed"),
        "decision_id": str(decision_id),
        "meeting_id": str(MEETING),
        "content": "Launch the analytics beta on 2 December.",
        "rationale": "Marketing wants it before the holidays.",
        "confidence": 0.9,
        "status": "confirmed",
        "confirmed_by": str(PRIYA),
        "supersedes": str(supersedes) if supersedes else None,
    }


def action_value(org: uuid.UUID = ORG) -> dict[str, Any]:
    return {
        "header": _header(org, f"action:{ACTION}:confirmed"),
        "action_id": str(ACTION),
        "meeting_id": str(MEETING),
        "content": "Draft the beta launch checklist",
        "assignee_id": str(PRIYA),
        "due_date": 20_420,  # Avro date: days since 1970-01-01
        "confidence": 0.8,
        "status": "confirmed",
    }


def loop_with(codec: Any, **kw: Any) -> tuple[ProactiveLoop, LoopbackProducer, Any]:
    agent, llm, _ = make(**kw)
    producer = LoopbackProducer()
    return ProactiveLoop(agent, Emitter(producer, codec)), producer, llm


# --- offline -------------------------------------------------------------------------------


async def test_a_decision_becomes_flushed_alerts_and_bad_records_are_skipped():
    loop, producer, _ = loop_with(JsonCodec())
    assert await loop.handle("decisions", decision_value()) == 3  # PRIYA, SAM, ALEX
    assert producer.flushes == 1
    records = producer.on("agent-results")
    assert sorted(r["user_id"] for r in records) == sorted(map(str, (PRIYA, SAM, ALEX)))
    assert all(r["window_id"] == f"proactive:conflict:{DECISION}" for r in records)
    assert all(r["header"]["org_id"] == str(ORG) for r in records)
    assert [p.key for p in producer.produced] == [str(ORG)] * 3  # partition key: org

    assert await loop.handle("decisions", {"header": {}, "decision_id": "x"}) == 0
    assert await loop.handle("transcript-chunks", decision_value()) == 0
    assert producer.flushes == 1  # nothing emitted: nothing to flush


async def test_a_failing_check_is_skipped_and_an_unacknowledged_alert_is_not_committed():
    loop, producer, _ = loop_with(JsonCodec())

    async def boom(_d: Any) -> list[Any]:
        raise RuntimeError("provider outage")

    loop.agent.check_decision = boom  # type: ignore[method-assign]
    assert await loop.handle("decisions", decision_value()) == 0

    loop, producer, _ = loop_with(JsonCodec())
    producer.flush = lambda _t=0: 1  # type: ignore[method-assign]  # one message undelivered
    with pytest.raises(EmitError):
        await loop.handle("decisions", decision_value())


def test_the_consumer_group_topics_and_offset_policy():
    assert GROUP == "agent-runtime-proactive" and TOPICS == ["decisions", "actions"]
    cfg = consumer_config("b:9092")
    assert cfg["group.id"] == GROUP and cfg["enable.auto.commit"] is False
    assert cfg["auto.offset.reset"] == "latest"  # no flood of stale alerts on first deploy


# --- the real Avro path --------------------------------------------------------------------


async def test_real_avro_decisions_and_actions_round_trip(avro_codec):
    rag = ScriptedRag({PRIYA: [rag_passage("Beta launches 15 January.", 0.8, kind="decision")]})
    core = FakeCore(
        contexts={ACTION: ctx(open_action_count=7)},
        participants={MEETING: []},
    )
    loop, producer, llm = loop_with(avro_codec, rag=rag, core=core)

    raw = avro_codec.serialize_for("decisions", decision_value(supersedes=OTHER_DECISION))
    decoded = avro_codec.deserialize_for("decisions", raw)
    assert await loop.handle("decisions", decoded) == 1  # confirmed_by only: no participants

    raw = avro_codec.serialize_for("actions", action_value())
    decoded = avro_codec.deserialize_for("actions", raw)
    assert await loop.handle("actions", decoded) == 2  # the admin and the organizer

    results = [p for p in producer.produced if p.topic == "agent-results"]
    values = [avro_codec.deserialize_for("agent-results", p.value) for p in results]
    assert {v["agent"] for v in values} == {"ProactiveAgent"}
    assert {v["kind"] for v in values} == {"alert"}
    assert all(v["source_chunk_ids"] == [] for v in values)
    conflict = json.loads(values[0]["payload_json"])["alert"]
    assert conflict["alert_type"] == "conflict" and conflict["priority"] == 1
    assert str(values[0]["meeting_id"]) == str(MEETING) and str(values[0]["user_id"]) == str(PRIYA)
    workload = [json.loads(v["payload_json"])["alert"] for v in values[1:]]
    assert {a["alert_type"] for a in workload} == {"workload"}
    assert all(a["priority"] == 2 and a["confidence"] == 0.9 for a in workload)
    # the decoded `supersedes` reached the agent: the superseded id is excluded, PRIOR is used
    assert conflict["related"][0]["id"] == str(PRIOR)


# --- over a real Redpanda ------------------------------------------------------------------


def _brokers() -> str:
    return os.environ.get("REDPANDA_BROKERS", "localhost:9092")


def _broker_or_skip() -> None:
    from confluent_kafka import Producer

    try:
        md = Producer({"bootstrap.servers": _brokers()}).list_topics(timeout=3)
    except Exception:  # noqa: BLE001
        md = None
    missing = {"decisions", "actions", "agent-results"} - set(md.topics) if md else {"broker"}
    if missing:
        if os.environ.get("MYTHEUS_REQUIRE_REGISTRY") == "1":
            raise RuntimeError(f"Redpanda at {_brokers()} lacks {missing}. Refusing to skip.")
        pytest.skip(f"Redpanda at {_brokers()} unreachable or lacks {sorted(missing)}")


async def test_the_loop_over_real_redpanda(avro_codec):
    _broker_or_skip()
    from confluent_kafka import Consumer, Producer

    org = uuid.uuid4()
    loop, _, _ = loop_with(avro_codec)
    loop.sink = Emitter(
        Producer({"bootstrap.servers": _brokers(), "acks": "all", "enable.idempotence": True}),
        avro_codec,
    )
    assigned = asyncio.Event()
    cfg = {**consumer_config(_brokers()), "group.id": f"test-{GROUP}-{uuid.uuid4()}"}
    consumer = Consumer(cfg)
    consumer.subscribe(TOPICS, on_assign=lambda _c, _p: assigned.set())
    stop = asyncio.Event()
    task = asyncio.create_task(run_proactive_loop(consumer, avro_codec, loop, stop))
    scan = Consumer(
        {
            "bootstrap.servers": _brokers(),
            "group.id": f"test-scan-{uuid.uuid4()}",
            "auto.offset.reset": "latest",
        }
    )
    scan_assigned = asyncio.Event()
    scan.subscribe(["agent-results"], on_assign=lambda _c, _p: scan_assigned.set())
    try:
        end = time.monotonic() + 20
        while not (assigned.is_set() and scan_assigned.is_set()) and time.monotonic() < end:
            await asyncio.to_thread(scan.poll, 0.2)
        assert assigned.is_set() and scan_assigned.is_set(), "no partition assignment"
        await asyncio.sleep(1.0)  # let `latest` resolve before producing
        producer = Producer({"bootstrap.servers": _brokers()})
        producer.produce(
            "decisions",
            key=str(org),
            value=avro_codec.serialize_for("decisions", decision_value(org=org)),
        )
        producer.flush(5)
        found: list[dict[str, Any]] = []
        end = time.monotonic() + 20
        while len(found) < 3 and time.monotonic() < end:
            msg = await asyncio.to_thread(scan.poll, 0.5)
            if msg is None or msg.error():
                continue
            value = avro_codec.deserialize_for("agent-results", msg.value())
            if str(value["header"]["org_id"]) == str(org):
                found.append(value)
        assert len(found) == 3, found
        assert {v["agent"] for v in found} == {"ProactiveAgent"}
        assert all(v["window_id"] == f"proactive:conflict:{DECISION}" for v in found)
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)
        consumer.close()
        scan.close()
