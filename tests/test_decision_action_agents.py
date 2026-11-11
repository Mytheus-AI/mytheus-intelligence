"""`DecisionAgent` / `ActionAgent`: Plan 7-9 section 3.5 and the Phase 8 Part A contract
section 1, every model call faked.

1. Routing: the pre-gate goes to Haiku, extraction to Sonnet (models and usage `model_tier`).
2. Below the gate threshold no extraction call is made.
3. Bands per candidate: `< 0.4` dropped, `0.4-0.7` silent, `> 0.7` surfaced; none left -> None.
4. Candidate ids are deterministic (`uuid5` over agent, window id and the model's index).
5. "Participant N" labels map back to UUID speaker ids; no id, name or email reaches the model.
6. Malformed model output is no candidates.
7. One instance per meeting through `AgentRuntime`, one result per window per agent.
8. The shipped prompts load, render and are registered.
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.action_agent import ActionAgent, parse_due_date
from services.agent_runtime_service.agents.candidates import (
    CANDIDATE_NS,
    CandidateAgent,
    gate_threshold,
    parse_candidates,
    parse_score,
    render_window,
)
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.runtime import AgentRuntime

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
ALICE, BOB, CARA = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
HAIKU, SONNET = "anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"
MONDAY = date(2026, 11, 9)


class Sink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.deltas: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        self.notifications.append(kw)

    async def delta(self, **kw: Any) -> None:
        self.deltas.append(kw)


def cands(*items: dict[str, Any]) -> str:
    return json.dumps({"candidates": list(items)})


class ScriptedLLM:
    """Haiku answers every gate with `score`; Sonnet answers by which extractor is asking.
    The PersonalListener and QuestionAgent gates (only in the runtime test) always score 0, so
    those participant-scoped agents stay quiet."""

    def __init__(self, score: float, decision: str = "", action: str = "") -> None:
        self.score = score
        self.decision = decision
        self.action = action
        self.calls: list[dict[str, Any]] = []

    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]

    def _answer(self, model: str, system: str) -> str:
        if "haiku" in model:
            if "personal meeting note-taker" in system:
                return '{"relevance": 0}'
            if "question coach" in system:
                return '{"score": 0}'
            return json.dumps({"score": self.score})
        return self.decision if "extract decisions" in system else self.action

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system = kw["messages"][0]["content"][0]["text"]
        assert not kw.get("stream"), "candidate agents never stream"
        return {
            "choices": [{"message": {"content": self._answer(kw["model"], system)}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }


def gate() -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": 0,
            "token_quota": 10**6,
            "hard_cap": 10**6,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


def window(window_id: str = f"{MEETING}:0", extra: list[Turn] | None = None) -> TranscriptWindow:
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=window_id,
        start_ms=0,
        end_ms=30_000,
        turns=[
            Turn(
                speaker_id=str(ALICE),
                text="QA signed off, so let's ship the beta on the 2nd.",
                start_ms=0,
                end_ms=5_000,
                chunk_ids=["c1"],
            ),
            Turn(
                speaker_id=str(BOB),
                text="Agreed. I'll write the release notes by Friday.",
                start_ms=5_000,
                end_ms=9_000,
                chunk_ids=["c2"],
            ),
            *(extra or []),
        ],
    )


def make(cls: type[CandidateAgent], router: LiteLLMRouter) -> CandidateAgent:
    loader = PromptLoader(PROMPTS_DIR)
    if cls is ActionAgent:
        return ActionAgent(loader, router, today=lambda: MONDAY)
    return cls(loader, router)


async def run(
    cls: type[CandidateAgent],
    llm: ScriptedLLM,
    sink: Sink | None = None,
    w: TranscriptWindow | None = None,
    context: dict[str, Any] | None = None,
) -> Any:
    router = LiteLLMRouter(gate(), sink or Sink(), llm)
    agent = make(cls, router)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    state = AgentState(
        meeting_id=MEETING, user_id=ALICE, chunk=w or window(), context=context or {}, org_id=ORG
    )
    return await agent.process(state)


DECISION = cands(
    {
        "content": "Ship the beta on 2 December",
        "rationale": "QA signed off",
        "stakeholders": ["Participant 1", "Participant 2"],
        "confidence": 0.86,
    }
)
ACTION = cands(
    {
        "content": "Write the release notes",
        "assignee": "Participant 2",
        "due_date": "2026-11-13",
        "depends_on": ["beta date"],
        "confidence": 0.9,
    }
)


# --- 1. routing ------------------------------------------------------------------------------


@pytest.mark.parametrize("cls,script", [(DecisionAgent, DECISION), (ActionAgent, ACTION)])
async def test_the_gate_routes_haiku_and_extraction_routes_sonnet(cls, script):
    sink, llm = Sink(), ScriptedLLM(0.9, decision=script, action=script)
    result = await run(cls, llm, sink)
    assert llm.models() == [HAIKU, SONNET]
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku", "sonnet"]
    assert result.payload["model_tier"] == "sonnet"
    # ADR-0039 advanced the action prompts, the polish plan the decision ones (never record),
    # polish wave 2 both: v3 in-meeting memory, v5 = v3 plus the {guidance} feedback examples
    assert result.payload["prompt"] == f"{cls.prompt_key}@v5"
    # extraction: adaptive thinking at effort low, with room for it (polish: model settings)
    extract = llm.calls[1]
    assert extract["thinking"] == {"type": "adaptive"}
    assert extract["output_config"] == {"effort": "low"} and extract["max_tokens"] == 8192
    assert result.kind == cls.kind
    assert result.relevance_score == 0.9
    assert result.source_chunk_ids == ["c1", "c2"]


async def test_the_contract_payload_shapes():
    d = await run(DecisionAgent, ScriptedLLM(0.9, decision=DECISION))
    (c,) = d.payload["candidates"]
    assert set(c) == {"id", "content", "confidence", "surfaced", "rationale", "stakeholder_ids"}
    assert c["rationale"] == "QA signed off"
    assert c["stakeholder_ids"] == [str(ALICE), str(BOB)]
    assert d.content == "Ship the beta on 2 December" and d.confidence == 0.86
    json.dumps(d.payload)  # serializable as payload_json

    a = await run(ActionAgent, ScriptedLLM(0.9, action=ACTION))
    (c,) = a.payload["candidates"]
    assert set(c) == {
        "id",
        "content",
        "confidence",
        "surfaced",
        "assignee_id",
        "due_date",
        "depends_on",
    }
    assert c["assignee_id"] == str(BOB)
    assert c["due_date"] == "2026-11-13" and c["depends_on"] == ["beta date"]


# --- 2. pre-gate -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,env", [(DecisionAgent, "DECISION_GATE_THRESHOLD"), (ActionAgent, "ACTION_GATE_THRESHOLD")]
)
async def test_below_the_gate_threshold_no_extraction_call(cls, env, monkeypatch):
    monkeypatch.setenv(env, "0.5")
    sink, llm = Sink(), ScriptedLLM(0.3, decision=DECISION, action=ACTION)
    assert await run(cls, llm, sink) is None
    assert llm.models() == [HAIKU]
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku"]

    llm = ScriptedLLM(0.5, decision=DECISION, action=ACTION)
    assert await run(cls, llm) is not None
    assert llm.models() == [HAIKU, SONNET]


def test_gate_parsing_and_threshold(monkeypatch):
    assert parse_score('{"score": 0.82}') == 0.82
    assert parse_score('```json\n{"score": 0.6}\n```') == 0.6
    assert parse_score("score 0.6") == 0.6
    assert parse_score('{"score": 7}') == 1.0
    assert parse_score("no idea") == 0.0
    assert parse_score('{"score": NaN}') == 0.0
    monkeypatch.delenv("DECISION_GATE_THRESHOLD", raising=False)
    assert gate_threshold("DECISION_GATE_THRESHOLD") == 0.5
    monkeypatch.setenv("DECISION_GATE_THRESHOLD", "0.65")
    assert gate_threshold("DECISION_GATE_THRESHOLD") == 0.65
    monkeypatch.setenv("DECISION_GATE_THRESHOLD", "high")
    assert gate_threshold("DECISION_GATE_THRESHOLD") == 0.5


# --- 3. bands --------------------------------------------------------------------------------


async def test_bands_apply_per_candidate():
    script = cands(
        *(
            {"content": f"decision {c}", "confidence": c, "stakeholders": []}
            for c in (0.39, 0.4, 0.55, 0.71, 0.9)
        )
    )
    result = await run(DecisionAgent, ScriptedLLM(0.8, decision=script))
    got = [(c["confidence"], c["surfaced"]) for c in result.payload["candidates"]]
    assert got == [(0.4, False), (0.55, False), (0.71, True), (0.9, True)]
    assert result.content == "decision 0.9" and result.confidence == 0.9
    # ids keep the model's index: the dropped 0.39 at index 0 does not shift the others
    ids = [c["id"] for c in result.payload["candidates"]]
    assert ids == [
        str(uuid.uuid5(CANDIDATE_NS, f"DecisionAgent:{MEETING}:0:{i}")) for i in (1, 2, 3, 4)
    ]


@pytest.mark.parametrize("cls", [DecisionAgent, ActionAgent])
async def test_a_window_with_every_candidate_dropped_emits_nothing(cls):
    script = cands({"content": "maybe", "confidence": 0.39}, {"content": "eh", "confidence": 0.1})
    assert await run(cls, ScriptedLLM(0.9, decision=script, action=script)) is None
    empty = cands()
    assert await run(cls, ScriptedLLM(0.9, decision=empty, action=empty)) is None


# --- 4. deterministic ids --------------------------------------------------------------------


async def test_candidate_ids_are_deterministic():
    first = await run(ActionAgent, ScriptedLLM(0.9, action=ACTION))
    again = await run(ActionAgent, ScriptedLLM(0.9, action=ACTION))
    assert first.payload["candidates"][0]["id"] == again.payload["candidates"][0]["id"]
    assert first.payload["candidates"][0]["id"] == str(
        uuid.uuid5(uuid.UUID("6f1d3c2e-8a4b-5e7f-9c1d-2b3a4c5d6e7f"), f"ActionAgent:{MEETING}:0:0")
    )
    other_window = await run(
        ActionAgent, ScriptedLLM(0.9, action=ACTION), w=window(f"{MEETING}:30000")
    )
    decision = await run(DecisionAgent, ScriptedLLM(0.9, decision=ACTION))
    ids = {
        first.payload["candidates"][0]["id"],
        other_window.payload["candidates"][0]["id"],
        decision.payload["candidates"][0]["id"],
    }
    assert len(ids) == 3


# --- 5. labels and what reaches the model ----------------------------------------------------


async def test_labels_map_back_to_uuid_speakers_only():
    w = window(
        extra=[Turn(speaker_id="egress-recorder", text="(beep)", start_ms=9_000, end_ms=9_100)]
    )
    decision = cands(
        {
            "content": "Use Postgres",
            "confidence": 0.8,
            "stakeholders": [
                "participant 1",
                "Participant 3",
                "Participant 9",
                "Bob",
                4,
                "Participant 1",
            ],
        }
    )
    d = await run(DecisionAgent, ScriptedLLM(0.9, decision=decision), w=w)
    assert d.payload["candidates"][0]["stakeholder_ids"] == [str(ALICE)]

    actions = cands(
        {"content": "a", "assignee": "Participant 3", "due_date": "Friday", "confidence": 0.8},
        {"content": "b", "assignee": "Bob", "due_date": None, "confidence": 0.8},
        {"content": "c", "assignee": "Participant 2", "due_date": "2026-11-13", "confidence": 0.8},
    )
    a = await run(ActionAgent, ScriptedLLM(0.9, action=actions), w=w)
    got = [(c["assignee_id"], c["due_date"]) for c in a.payload["candidates"]]
    assert got == [(None, None), (None, None), (str(BOB), "2026-11-13")]


@pytest.mark.parametrize("cls", [DecisionAgent, ActionAgent])
async def test_no_id_name_or_email_reaches_the_model(cls):
    llm = ScriptedLLM(0.9, decision=DECISION, action=ACTION)
    w = window(
        extra=[
            Turn(
                speaker_id=str(CARA),
                text="Send it to jane@acme.com when it's out.",
                start_ms=9_000,
                end_ms=12_000,
            )
        ]
    )
    context = {"name": "Alice Zebedee", "email": "alice.zebedee@acme.com", "role": "PM"}
    await run(cls, llm, w=w, context=context)
    assert len(llm.calls) == 2
    for call in llm.calls:
        sent = json.dumps(call["messages"])
        for uid in (ALICE, BOB, CARA, MEETING, ORG):
            assert str(uid) not in sent
        assert "Zebedee" not in sent and "@acme.com" not in sent
        assert "[REDACTED_EMAIL]" in sent
        assert "Participant 1: QA signed off" in sent and "Participant 3:" in sent


async def test_the_action_prompt_carries_the_reference_date():
    llm = ScriptedLLM(0.9, action=ACTION)
    await run(ActionAgent, llm)
    assert "Today is Monday 2026-11-09 (UTC)." in llm.calls[1]["messages"][1]["content"]


def test_render_window_numbers_speakers_in_order_of_first_speaking():
    w = window()
    w.turns.append(Turn(speaker_id=str(CARA), text="+1", start_ms=9_000, end_ms=9_500))
    w.turns.append(Turn(speaker_id=str(ALICE), text="Done.", start_ms=9_600, end_ms=9_900))
    text, labels = render_window(w)
    assert [ln.split(":")[0] for ln in text.splitlines()] == [
        "Participant 1",
        "Participant 2",
        "Participant 3",
        "Participant 1",
    ]
    assert labels == {
        "Participant 1": str(ALICE),
        "Participant 2": str(BOB),
        "Participant 3": str(CARA),
    }


# --- 6. malformed output ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "There is a decision here.",
        "{not json",
        '{"candidates": "none"}',
        '["a"]',
        cands({"content": "", "confidence": 0.9}),
        cands({"content": "no confidence"}),
        cands({"content": "bool confidence", "confidence": True}),
        cands({"content": "string confidence", "confidence": "0.9"}),
        '{"candidates": [{"content": "nan", "confidence": NaN}]}',
        '{"candidates": ["just a string", 3]}',
    ],
)
async def test_malformed_output_is_no_result(text: str):
    assert await run(DecisionAgent, ScriptedLLM(0.9, decision=text)) is None


def test_candidates_are_parsed_through_fences_and_prose():
    inner = cands({"content": "x", "confidence": 0.8})
    assert len(parse_candidates(f"```json\n{inner}\n```")) == 1
    assert len(parse_candidates(f"Here you go:\n{inner}\nThanks")) == 1
    assert parse_candidates("```\nnope\n```") == []


def test_due_date_parsing():
    assert parse_due_date("2026-12-02") == "2026-12-02"
    assert parse_due_date("2026-12-02T17:00:00Z") == "2026-12-02"
    assert parse_due_date("Friday") is None and parse_due_date("2026-13-40") is None
    assert parse_due_date(None) is None and parse_due_date(20261202) is None


# --- 7. one instance per meeting through the runtime -----------------------------------------


async def test_one_instance_per_meeting_and_one_result_per_window():
    sink = Sink()
    llm = ScriptedLLM(0.9, decision=DECISION, action=ACTION)
    router = LiteLLMRouter(gate(), sink, llm)

    async def ctx(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
        return {"role": "Engineer"}

    runtime = AgentRuntime(
        sink=sink, user_context=ctx, agents=registered_agents(PromptLoader(PROMPTS_DIR), router)
    )
    for u in (ALICE, BOB, CARA):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    await runtime.process_window(window())
    await runtime.process_window(window(f"{MEETING}:30000"))

    m = runtime._meetings[(ORG, MEETING)]
    per_participant = ("PersonalListenerAgent", "QuestionAgent")
    meeting_scoped = sorted(k for k in m.agents if k[0] not in per_participant)
    assert meeting_scoped == [
        ("ActionAgent", MEETING),
        ("ContextGapAgent", MEETING),
        ("DecisionAgent", MEETING),
        ("MeetingAgent", MEETING),
        ("ProactiveAgent", MEETING),
    ]
    assert len([k for k in m.agents if k[0] == "PersonalListenerAgent"]) == 3
    assert len([k for k in m.agents if k[0] == "QuestionAgent"]) == 3

    by_agent: dict[str, list[dict[str, Any]]] = {}
    for r in sink.results:
        by_agent.setdefault(r["agent"], []).append(r)
    assert set(by_agent) == {"DecisionAgent", "ActionAgent"}  # the PL and question gates scored 0
    lowest = min((ALICE, BOB, CARA), key=str)
    for agent, results in by_agent.items():
        assert len(results) == 2, agent  # once per window, not once per participant
        assert [r["window"].window_id for r in results] == [f"{MEETING}:0", f"{MEETING}:30000"]
        assert {r["user_id"] for r in results} == {lowest}
    # 3 PL gates + 2 x (gate + extract) per window
    assert llm.models().count(SONNET) == 4


# --- 8. shipped prompts ----------------------------------------------------------------------


def test_the_shipped_prompts_are_registered_and_render():
    loader = PromptLoader(PROMPTS_DIR)
    for key in ("decision_gate", "decision_agent", "action_gate", "action_agent"):
        prompt = loader.load(key, loader.resolve(key, ORG))
        rendered = prompt.render_user(
            window="Participant 1: hi",
            today="Monday 2026-11-09",
            company="",
            ground_rules="",
            owners="",
            memory="",
            guidance="",
        )
        assert "Participant 1: hi" in rendered
        assert "{" not in rendered  # every placeholder filled, no stray literal braces
        assert prompt.system and "Participant 1" in prompt.system
    assert "Monday 2026-11-09" in loader.load("action_agent", "v1").render_user(
        window="x", today="Monday 2026-11-09"
    )
    registry = json.loads((Path(PROMPTS_DIR) / "registry.json").read_text())["defaults"]
    assert registry["decision_gate"] == "v2"  # polish: never record everywhere
    # polish wave 2: v3 memory inside the meeting, v5 = v3 plus the {guidance} feedback
    # examples (v4 = v2 plus {guidance}, kept for pins only)
    assert registry["decision_agent"] == "v5"
    assert registry["action_agent"] == "v5"
    assert registry["action_gate"] == "v3"  # polish: past the 512-token cache minimum


async def test_spawn_fails_closed_without_org_id():
    router = LiteLLMRouter(gate(), Sink(), ScriptedLLM(0.9))
    for cls in (DecisionAgent, ActionAgent):
        with pytest.raises(ValueError):
            await make(cls, router).spawn(org_id=None, meeting_id=MEETING, user_id=MEETING)
