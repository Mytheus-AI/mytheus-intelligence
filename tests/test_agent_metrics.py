"""Agent quality telemetry (libs/shared/agent_metrics.py): what is counted, where, with which
labels. Each test uses a fresh org id, so the global registry's other series do not interfere."""

from __future__ import annotations

import json
import uuid
from typing import Any
from uuid import UUID

import fakeredis
import pytest
from prometheus_client import REGISTRY

from libs.shared import agent_metrics as am
from libs.shared.budget import BudgetGate
from libs.shared.metrics import AGENT_TIME_TO_CARD
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.runtime import AgentRuntime
from tests.a2a_helpers import JsonCodec, LoopbackProducer


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def _unlimited(_org: UUID) -> dict[str, Any]:
    return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9, "enforcement": "enforce"}


class Scripted:
    """Answers each call from a queue (by order), with fixed usage."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)

    async def __call__(self, **kw: Any) -> Any:
        text = self.answers.pop(0) if self.answers else ""
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "cache_read_input_tokens": 400}
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": usage,
        }


class NullSink:
    async def usage(self, **kw: Any) -> None:
        return None


def router(llm: Any) -> LiteLLMRouter:
    return LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), _unlimited), NullSink(), llm)


def window(org: UUID, *lines: str, start: int = 0) -> TranscriptWindow:
    meeting = uuid.uuid4()
    turns = [
        Turn(str(uuid.uuid4()), text, n * 1000, n * 1000 + 900, [f"c{n}"])
        for n, text in enumerate(lines)
    ]
    return TranscriptWindow(org, meeting, f"{meeting}:{start}", start, start + 30_000, turns)


# --- pure helpers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "text", "ok"),
    [
        ("decision_gate@v2", '{"score": 0.8}', True),
        ("decision_gate@v2", '```json\n{"score": 0.8}\n```', True),
        ("decision_gate@v2", "0.8", False),
        ("decision_agent@v2", '{"candidates": []}', True),
        (
            "decision_agent@v2",
            'Here you go: {"candidates": [{"content": "x"}]} hope it helps',
            True,
        ),
        ("decision_agent@v2", '{"candidates": [{"content": "x", "conf', False),
        ("personal_listener@v3", "CONFIDENCE: 0.8\nCOMMITMENTS: []\n- note", True),
        ("personal_listener@v3", "Here is the note", False),
        ("personal_listener_gate@v4", '{"relevance": 0.4}', True),
        ("meeting_agent@v2", '{"updates": [], "off_agenda": null, "raised": []}', True),
        ("context_gap_agent@v2", '{"content": "", "confidence": 0}', True),
        ("minutes@v1", "anything", None),
        (None, "anything", None),
    ],
)
def test_output_formats(ref: str | None, text: str, ok: bool | None):
    assert am.output_ok(ref, text) is ok


def test_agent_names_come_from_the_prompt_key():
    assert am.agent_for_prompt("decision_gate@v2") == "DecisionAgent"
    assert am.agent_for_prompt("personal_listener_gate@v4") == "PersonalListenerAgent"
    assert am.agent_for_prompt("proactive_already_decided_gate@v2") == "ProactiveAgent"
    assert am.agent_for_prompt("meeting_agent@v2") == "MeetingAgent"
    assert am.agent_for_prompt("something_new@v1") == "something_new"
    assert am.agent_for_prompt(None) == "unknown"
    assert am.org_label("not-a-uuid") == "unknown"


def test_the_time_to_card_histogram_is_per_agent_and_workspace():
    org = str(uuid.uuid4())
    AGENT_TIME_TO_CARD.labels("DecisionAgent", org).observe(4.2)
    assert (
        sample("mytheus_agent_time_to_card_seconds_count", agent="DecisionAgent", org_id=org) == 1
    )
    assert (
        sample(
            "mytheus_agent_time_to_card_seconds_bucket", agent="DecisionAgent", org_id=org, le="5.0"
        )
        == 1
    )


# --- router: tokens, spend, parse ----------------------------------------------------------------


async def test_the_router_counts_tokens_spend_and_parse_failures():
    org = uuid.uuid4()
    r = router(Scripted('{"score": 0.9}', "not json at all"))
    for _ in range(2):
        await r.complete(
            tier="triage",
            system="s",
            user="u",
            org_id=org,
            user_id=None,
            max_tokens=20,
            prompt_ref="decision_gate@v2",
        )
    labels = {"agent": "DecisionAgent", "org_id": str(org)}
    assert sample("mytheus_agent_parsed_total", **labels) == 2
    assert sample("mytheus_agent_parse_failures_total", **labels) == 1
    tok = {**labels, "tier": "triage"}
    assert sample("mytheus_agent_tokens_total", **tok, kind="input") == 2 * 600  # uncached
    assert sample("mytheus_agent_tokens_total", **tok, kind="cache_read") == 2 * 400
    assert sample("mytheus_agent_tokens_total", **tok, kind="output") == 2 * 200
    # Haiku 5.5: 600 x $0.10/M + 400 x $0.01/M + 200 x $0.50/M per call
    per_call = (600 * 0.10 + 400 * 0.01 + 200 * 0.50) / 1_000_000
    assert sample("mytheus_agent_spend_usd_total", **tok) == pytest.approx(2 * per_call)


async def test_a_metrics_failure_never_breaks_a_call(monkeypatch):
    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("registry gone")

    monkeypatch.setattr(am, "usd", boom)
    out = await router(Scripted('{"score": 1}')).complete(
        tier="triage",
        system="s",
        user="u",
        org_id=uuid.uuid4(),
        user_id=None,
        max_tokens=20,
        prompt_ref="decision_gate@v2",
    )
    assert out.text == '{"score": 1}'


# --- BaseAgent: relevance gate and dropped -------------------------------------------------------


async def _decide(org: UUID, *answers: str) -> AgentResult | None:
    agent = DecisionAgent(PromptLoader(PROMPTS_DIR), router(Scripted(*answers)))
    meeting = uuid.uuid4()
    await agent.spawn(org_id=org, meeting_id=meeting, user_id=meeting)
    w = window(org, "Let's go with Postgres.", "Agreed.")
    state = AgentState(meeting_id=meeting, user_id=meeting, chunk=w, context={}, org_id=org)
    return await agent.process(state)


async def test_the_pre_gate_and_the_dropped_band_are_counted_per_agent_and_workspace():
    org = uuid.uuid4()
    labels = {"agent": "DecisionAgent", "org_id": str(org)}
    assert await _decide(org, '{"score": 0.1}') is None  # gate closed
    assert await _decide(org, '{"score": 0.9}', '{"candidates": []}') is None  # nothing kept
    low = json.dumps({"candidates": [{"content": "Use Postgres", "confidence": 0.2}]})
    assert await _decide(org, '{"score": 0.9}', low) is None  # below 0.4: dropped
    kept = json.dumps({"candidates": [{"content": "Use Postgres", "confidence": 0.9}]})
    assert await _decide(org, '{"score": 0.9}', kept) is not None
    assert sample("mytheus_agent_gate_total", **labels, gate="relevance", outcome="fail") == 1
    assert sample("mytheus_agent_gate_total", **labels, gate="relevance", outcome="pass") == 3
    assert sample("mytheus_agent_gate_total", **labels, gate="confidence", outcome="dropped") == 2


# --- emitter: surfaced and silent ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "confidence", "outcome"),
    [
        ({"surfaced": True}, 0.9, "surfaced"),
        ({"surfaced": False}, 0.6, "silent"),
        ({"candidates": [{"surfaced": False}, {"surfaced": True}]}, 0.9, "surfaced"),
        ({"candidates": [{"surfaced": False}]}, 0.6, "silent"),
        ({"alert": {"surfaced": False}}, 0.9, "silent"),
        ({}, 0.75, "surfaced"),
        ({}, 0.5, "silent"),
    ],
)
async def test_every_emitted_result_is_counted_in_its_band(
    payload: dict[str, Any], confidence: float, outcome: str
):
    org = uuid.uuid4()
    emitter = Emitter(LoopbackProducer(), JsonCodec())
    result = AgentResult(
        kind="x",
        content="c",
        relevance_score=0,
        confidence=confidence,
        source_chunk_ids=[],
        payload=payload,
    )
    await emitter.result(
        window=window(org, "hi"), user_id=uuid.uuid4(), agent="QuestionAgent", result=result
    )
    assert (
        sample(
            "mytheus_agent_gate_total",
            agent="QuestionAgent",
            org_id=str(org),
            gate="confidence",
            outcome=outcome,
        )
        == 1
    )


# --- runtime: meeting time and never-record skips ------------------------------------------------


class _Sink:
    def __init__(self) -> None:
        self.results: list[AgentResult] = []

    async def result(self, **kw: Any) -> None:
        self.results.append(kw["result"])

    async def notification(self, **kw: Any) -> None:
        return None

    async def delta(self, **kw: Any) -> None:
        return None


class _Echo:
    """A participant agent that emits the window's text as a note, without a model."""

    name = "EchoAgent"
    scope = "participant"
    blocked = False

    async def spawn(self, **kw: Any) -> None:
        return None

    async def process(self, state: AgentState) -> AgentResult:
        text = " ".join(t.text for t in state.chunk.turns)
        return AgentResult(
            kind="note",
            content=text,
            relevance_score=1,
            confidence=0.9,
            source_chunk_ids=[],
            payload={"surfaced": True},
        )

    async def terminate(self) -> None:
        return None


async def test_the_runtime_counts_meeting_time_and_never_record_skips():
    org = uuid.uuid4()
    company = CompanyContext(never_record_topics=("layoffs",))

    async def ctx(_org: UUID, _user: UUID) -> dict[str, Any]:
        return {"role": "x"}

    async def org_ctx(_org: UUID) -> CompanyContext:
        return company

    sink = _Sink()
    runtime = AgentRuntime(
        sink=sink,  # type: ignore[arg-type]
        user_context=ctx,
        agents=[_Echo],  # type: ignore[list-item]
        org_context=org_ctx,
    )
    w1 = window(org, "Shipping on Friday.", start=0)
    w2 = window(org, "The layoffs list is final.", start=30_000)
    for w in (w1, w2):
        await runtime.process_window(w)
    o = str(org)
    assert sample("mytheus_agent_meeting_seconds_total", org_id=o) == 60
    assert sample("mytheus_agent_never_record_skips_total", stage="window", org_id=o) == 1
    assert len(sink.results) == 1


class _Leaky(_Echo):
    """Writes about a never-record topic the window never named (the prompt did not hold)."""

    name = "LeakyAgent"

    async def process(self, state: AgentState) -> AgentResult:
        return AgentResult(
            kind="note",
            content="About the layoffs plan",
            relevance_score=1,
            confidence=0.9,
            source_chunk_ids=[],
            payload={"surfaced": True},
        )


async def test_a_result_the_never_record_filter_removes_is_counted_as_an_output_skip():
    org = uuid.uuid4()

    async def ctx(_org: UUID, _user: UUID) -> dict[str, Any]:
        return {"role": "x"}

    async def org_ctx(_org: UUID) -> CompanyContext:
        return CompanyContext(never_record_topics=("layoffs",))

    sink = _Sink()
    runtime = AgentRuntime(sink=sink, user_context=ctx, agents=[_Leaky], org_context=org_ctx)  # type: ignore[arg-type,list-item]
    await runtime.process_window(window(org, "Shipping on Friday."))
    assert sink.results == []
    assert sample("mytheus_agent_never_record_skips_total", stage="output", org_id=str(org)) == 1
