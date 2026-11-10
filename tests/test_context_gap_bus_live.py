"""A2A over REAL Redpanda + Schema Registry: the requester, the responder loop and the response
loop exactly as `consumer.run` wires them, with the registered `agent-requests-value`,
`agent-responses-value` and `agent-results-value` Avro subjects. Only rag-service, the PDP and
identity are faked.

1. A request published by `A2AClient` is consumed by `run_request_loop`, answered by the
   responder (retrieving AS the responder), and the response resolves the requester's future
   through `run_response_loop`: `answered`, with the passages.
2. A spoofed envelope org is dead-lettered on `agent-requests.dlq` (JSON, keyed by the derived
   org), and its `org_spoof` entry is a real `context_fill` record on `agent-results`.

Skips without a reachable broker / registry (`MYTHEUS_REQUIRE_REGISTRY=1` refuses to skip), like
the repo's other broker tests. Topics needed: agent-requests, agent-responses,
agent-requests.dlq, agent-results (mytheus-events `scripts/create_topics.sh`).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import fakeredis
import pytest
from confluent_kafka import Consumer, Producer

from services.agent_runtime_service.a2a.bus import (
    DLQ_TOPIC,
    REQUESTS_TOPIC,
    RESPONSES_TOPIC,
    A2AClient,
    Responder,
    ResponseRouter,
    run_request_loop,
    run_response_loop,
)
from services.agent_runtime_service.a2a.identity import IdentitySigner
from services.agent_runtime_service.a2a.pep import Pep
from services.agent_runtime_service.emitter import Emitter
from tests.a2a_helpers import SECRET, FakePdp, ScriptedRag, allow_all, rag_passage

pytestmark = pytest.mark.asyncio

ORG, OTHER = uuid.uuid4(), uuid.uuid4()
ALICE, BOB = uuid.uuid4(), uuid.uuid4()
TEAM = str(uuid.uuid4())


def _brokers() -> str:
    return os.environ.get("REDPANDA_BROKERS", "localhost:9092")


def _broker_or_skip() -> None:
    try:
        md = Producer({"bootstrap.servers": _brokers()}).list_topics(timeout=3)
    except Exception:  # noqa: BLE001
        md = None
    missing = (
        {REQUESTS_TOPIC, RESPONSES_TOPIC, DLQ_TOPIC, "agent-results"} - set(md.topics)
        if md
        else {"broker"}
    )
    if missing:
        if os.environ.get("MYTHEUS_REQUIRE_REGISTRY") == "1":
            raise RuntimeError(f"Redpanda at {_brokers()} lacks {missing}. Refusing to skip.")
        pytest.skip(f"Redpanda at {_brokers()} unreachable or lacks {sorted(missing)}")


def _consumer(group: str) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": _brokers(),
            "group.id": group,
            "enable.auto.commit": True,
            "auto.offset.reset": "earliest",
        }
    )


async def _find(topic: str, predicate: Any, timeout_s: float = 20) -> Any:
    c = _consumer(f"test-a2a-scan-{uuid.uuid4()}")
    c.subscribe([topic])
    try:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            msg = await asyncio.to_thread(c.poll, 0.5)
            if msg is not None and not msg.error() and predicate(msg):
                return msg
        return None
    finally:
        c.close()


async def test_a_round_trip_and_a_spoof_over_real_redpanda(avro_codec):
    _broker_or_skip()
    signer = IdentitySigner(SECRET)
    contexts = {
        ORG: {
            ALICE: {"rbac_tier": "member", "team_ids": [TEAM], "department_id": ""},
            BOB: {"rbac_tier": "member", "team_ids": [TEAM], "department_id": ""},
        }
    }

    async def user_context(org: uuid.UUID, user: uuid.UUID) -> dict[str, Any] | None:
        return contexts.get(org, {}).get(user)

    rag = ScriptedRag({ALICE: [rag_passage("In July the team chose the enterprise tier.", 0.91)]})
    producer = Producer(
        {"bootstrap.servers": _brokers(), "acks": "all", "enable.idempotence": True}
    )
    emitter = Emitter(producer, avro_codec)
    pep = Pep(fakeredis.aioredis.FakeRedis(), FakePdp(allow_all()))
    responder = Responder(
        signer=signer, pep=pep, rag=rag, user_context=user_context, emitter=emitter
    )
    router = ResponseRouter()
    client = A2AClient(emitter, router, signer)

    requests = _consumer(f"test-agent-runtime-a2a-{uuid.uuid4()}")
    requests.subscribe([REQUESTS_TOPIC])
    responses = _consumer(f"test-agent-runtime-resp-{uuid.uuid4()}")
    responses.subscribe([RESPONSES_TOPIC])
    stop = asyncio.Event()
    loops = [
        asyncio.create_task(run_request_loop(requests, avro_codec, responder, stop, poll_s=0.2)),
        asyncio.create_task(run_response_loop(responses, avro_codec, router, stop, poll_s=0.2)),
    ]
    try:
        # 1. a real round trip (generous deadline: first assignment of a new group is slow)
        outcome = await client.ask(
            org_id=ORG,
            requester=BOB,
            responder=ALICE,
            query="Q3 pricing tier decision",
            deadline_s=time.time() + 25,
        )
        assert outcome.result == "answered", outcome
        assert outcome.passages[0].text.startswith("In July")
        assert outcome.passages[0].citation["kind"] == "transcript"
        assert [(c["org_id"], c["user_id"]) for c in rag.calls] == [(ORG, ALICE)]
        assert router.pending == 0

        # 2. a forged envelope org: DLQ + an org_spoof context_fill record under the real org
        cid = str(uuid.uuid4())
        await emitter.agent_request(
            org_id=OTHER,  # header.org_id lies; the bound id says ORG
            requester_agent_id=signer.mint(ORG, BOB),
            responder_target=f"user:{ALICE}",
            query="board compensation",
            deadline_us=int((time.time() + 25) * 1_000_000),
            correlation_id=cid,
        )
        emitter.flush()
        dlq = await _find(DLQ_TOPIC, lambda m: f"areq:{cid}".encode() in m.value())
        assert dlq is not None, "spoofed request never reached agent-requests.dlq"
        body = json.loads(dlq.value())
        assert body["failure_class"] == "ORG_SPOOF" and dlq.key() == str(ORG).encode()

        def is_entry(m: Any) -> bool:
            if m.key() != str(ORG).encode():
                return False
            value = avro_codec.deserialize_for("agent-results", m.value())
            return value["window_id"] == f"a2a:{cid}"

        emitter.flush()
        rec = await _find("agent-results", is_entry)
        assert rec is not None, "org_spoof entry never reached agent-results"
        value = avro_codec.deserialize_for("agent-results", rec.value())
        assert value["kind"] == "context_fill" and value["agent"] == "ContextGapAgent"
        assert str(value["header"]["org_id"]) == str(ORG)  # logicalType uuid decodes to UUID
        assert str(value["user_id"]) == str(BOB)
        entry = json.loads(value["payload_json"])["collaborations"][0]
        assert entry["result"] == "org_spoof" and entry["correlation_id"] == cid
        assert len(rag.calls) == 1  # the spoof was never served
    finally:
        stop.set()
        await asyncio.gather(*loops, return_exceptions=True)
        requests.close()
        responses.close()
