"""`QuestionAgent`: Plan 7-9 section 3.10 and mytheus-docs ADR-0034, every model call faked.

1. Routing: gate on Haiku, proposal on Sonnet, both metered (`usage-events` model tiers).
2. The context-gap heuristic and the cooldown make no model call at all.
3. Below the gate threshold no proposal call is made.
4. Bands per suggestion: `< 0.4` dropped, `0.4-0.7` silent, `> 0.7` surfaced; priority in {1,2,3}.
5. Ids are deterministic and include the participant.
6. The budget gate is honoured: a blocked tenant makes no call and gets no result.
7. Role-tuned: the participant's profile is in the request; no id reaches the model.
8. Grounding: off by default; on, it runs as the participant (never a service identity),
   requests passages only, never grounds user A on a meeting only B attended, and a grounder
   failure does not fail the window. Citations are only of passages the model was shown.
9. One instance per participant through `AgentRuntime`; each result's `user_id` is its owner.
10. The prompts load, render and are registered.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS
from services.agent_runtime_service.agents.grounding import (
    Passage,
    ground,
    grounder_from_env,
)
from services.agent_runtime_service.agents.question_agent import (
    QuestionAgent,
    has_context_gap_signal,
    parse_priority,
    suggestion_id,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.runtime import AgentRuntime

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
ALICE, BOB = uuid.uuid4(), uuid.uuid4()
HAIKU, SONNET = "anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"
PROFILE = {"role": "Finance lead", "skills": ["budgeting"], "bio": "Owns vendor spend."}


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


def sugg(*items: dict[str, Any]) -> str:
    return json.dumps({"suggestions": list(items)})


ONE = sugg(
    {
        "question": "What is the budget ceiling for the new vendor?",
        "priority": 1,
        "rationale": "You own vendor spend.",
        "sources": [],
        "confidence": 0.86,
    }
)


class ScriptedLLM:
    """Haiku answers the question gate with `score` (every other gate 0, so the other agents
    stay quiet in the runtime test); Sonnet answers the question prompt with `answer`."""

    def __init__(self, score: float, answer: str = ONE) -> None:
        self.score = score
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]

    def users(self) -> list[str]:
        return [c["messages"][-1]["content"] for c in self.calls]

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system = kw["messages"][0]["content"][0]["text"]
        if "haiku" in kw["model"]:
            if "question coach" in system:
                text = json.dumps({"score": self.score})
            elif "personal meeting note-taker" in system:
                text = '{"relevance": 0}'
            else:
                text = '{"score": 0}'
        else:
            text = self.answer
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }


def budget(consumed: int = 0, cap: int = 10**6) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": cap,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


def window(window_id: str = f"{MEETING}:0", turns: list[Turn] | None = None) -> TranscriptWindow:
    default = [
        Turn(
            speaker_id=str(BOB),
            text="We are going with the new analytics vendor from next quarter, starting with "
            "the enterprise tier for every team.",
            start_ms=0,
            end_ms=6_000,
            chunk_ids=["c1"],
        ),
        Turn(
            speaker_id=str(ALICE),
            text="Okay, and who signs the contract on our side?",
            start_ms=6_000,
            end_ms=9_000,
            chunk_ids=["c2"],
        ),
    ]
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=window_id,
        start_ms=0,
        end_ms=30_000,
        turns=default if turns is None else turns,
    )


async def make(llm: ScriptedLLM, sink: Sink | None = None, **kw: Any) -> QuestionAgent:
    router = LiteLLMRouter(kw.pop("gate", None) or budget(), sink or Sink(), llm)
    agent = QuestionAgent(PromptLoader(PROMPTS_DIR), router, **kw)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=ALICE)
    return agent


def state(w: TranscriptWindow | None = None, user: uuid.UUID = ALICE) -> AgentState:
    return AgentState(
        meeting_id=MEETING, user_id=user, chunk=w or window(), context=PROFILE, org_id=ORG
    )


# --- 1. routing ------------------------------------------------------------------------------


async def test_the_gate_routes_haiku_and_the_proposal_routes_sonnet():
    sink, llm = Sink(), ScriptedLLM(0.9)
    result = await (await make(llm, sink)).process(state())
    assert llm.models() == [HAIKU, SONNET]
    # a gate's few tokens must not be spent on thinking (it returned empty text: score 0)
    assert llm.calls[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in llm.calls[1]
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku", "sonnet"]
    assert all(u["user_id"] == ALICE for u in sink.usage_calls)
    assert result is not None and result.kind == "suggestion"
    assert result.payload["prompt"] == "question_agent@v2"
    assert result.payload["model_tier"] == "sonnet"
    assert result.payload["grounded"] is False
    assert result.source_chunk_ids == ["c1", "c2"]
    (s,) = result.payload["suggestions"]
    assert set(s) == {
        "id",
        "question",
        "priority",
        "confidence",
        "surfaced",
        "rationale",
        "citations",
    }
    assert s["priority"] == 1 and s["surfaced"] is True and s["citations"] == []
    assert result.content == s["question"] and result.confidence == 0.86
    json.dumps(result.payload)


# --- 2. heuristic and cooldown ---------------------------------------------------------------


async def test_a_thin_window_makes_no_model_call():
    llm = ScriptedLLM(0.9)
    agent = await make(llm)
    short = window(turns=[Turn(str(BOB), "Can everyone hear me?", 0, 2_000, ["c1"])])
    assert not has_context_gap_signal(short)
    assert await agent.process(state(short)) is None
    two_short = window(
        turns=[Turn(str(BOB), "Hi all.", 0, 1_000, ["c1"]), Turn(str(ALICE), "Hi!", 1_000, 2_000)]
    )
    assert await agent.process(state(two_short)) is None
    assert llm.calls == []


async def test_after_a_surfaced_suggestion_the_next_window_is_quiet(monkeypatch):
    monkeypatch.delenv("QUESTION_COOLDOWN_WINDOWS", raising=False)
    llm = ScriptedLLM(0.9)
    agent = await make(llm)
    assert await agent.process(state(window(f"{MEETING}:0"))) is not None
    assert await agent.process(state(window(f"{MEETING}:30000"))) is None  # cooldown, no call
    assert len(llm.calls) == 2
    assert await agent.process(state(window(f"{MEETING}:60000"))) is not None
    assert len(llm.calls) == 4


async def test_a_silent_suggestion_does_not_start_the_cooldown():
    quiet = sugg({"question": "Is legal involved?", "priority": 3, "confidence": 0.5})
    llm = ScriptedLLM(0.9, quiet)
    agent = await make(llm)
    assert await agent.process(state(window(f"{MEETING}:0"))) is not None
    assert await agent.process(state(window(f"{MEETING}:30000"))) is not None


# --- 3. gate ---------------------------------------------------------------------------------


async def test_below_the_gate_threshold_no_proposal_call(monkeypatch):
    monkeypatch.setenv("QUESTION_GATE_THRESHOLD", "0.5")
    sink, llm = Sink(), ScriptedLLM(0.3)
    assert await (await make(llm, sink)).process(state()) is None
    assert llm.models() == [HAIKU]
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku"]


# --- 4. bands and priority -------------------------------------------------------------------


async def test_bands_per_suggestion_and_malformed_ones_skipped():
    answer = sugg(
        {"question": "Surfaced?", "priority": 2, "confidence": 0.9},
        {"question": "Silent?", "priority": "3", "confidence": 0.55},
    )
    result = await (await make(ScriptedLLM(0.9, answer))).process(state())
    assert [
        (s["question"], s["surfaced"], s["priority"]) for s in result.payload["suggestions"]
    ] == [
        ("Surfaced?", True, 2),
        ("Silent?", False, 3),
    ]

    dropped = sugg({"question": "Generic?", "priority": 1, "confidence": 0.39})
    assert await (await make(ScriptedLLM(0.9, dropped))).process(state()) is None

    bad = sugg(
        {"question": "No priority?", "confidence": 0.9},
        {"question": "Priority 4?", "priority": 4, "confidence": 0.9},
    )
    assert await (await make(ScriptedLLM(0.9, bad))).process(state()) is None
    for text in ("not json", '{"suggestions": "x"}', "{}", '{"suggestions": [{"q": 1}]}'):
        assert await (await make(ScriptedLLM(0.9, text))).process(state()) is None


async def test_at_most_two_suggestions_per_window():
    many = sugg(*({"question": f"Q{i}?", "priority": 2, "confidence": 0.8} for i in range(4)))
    result = await (await make(ScriptedLLM(0.9, many))).process(state())
    assert [s["question"] for s in result.payload["suggestions"]] == ["Q0?", "Q1?"]


def test_priority_parsing():
    assert [parse_priority(v) for v in (1, 2, 3, "2", 3.0)] == [1, 2, 3, 2, 3]
    assert [parse_priority(v) for v in (0, 4, "high", None, True, 1.5)] == [None] * 6


# --- 5. ids ----------------------------------------------------------------------------------


async def test_ids_are_deterministic_and_per_participant():
    first = await (await make(ScriptedLLM(0.9))).process(state())
    again = await (await make(ScriptedLLM(0.9))).process(state())
    (a,), (b,) = first.payload["suggestions"], again.payload["suggestions"]
    assert (
        a["id"] == b["id"] == str(uuid.uuid5(CANDIDATE_NS, f"QuestionAgent:{MEETING}:0:{ALICE}:0"))
    )
    assert suggestion_id(f"{MEETING}:0", BOB, 0) != a["id"]


# --- 6. budget -------------------------------------------------------------------------------


async def test_a_blocked_tenant_makes_no_call_and_gets_no_result():
    sink, llm = Sink(), ScriptedLLM(0.9)
    agent = await make(llm, sink, gate=budget(consumed=10**6, cap=10**6))
    assert await agent.process(state()) is None
    assert agent.blocked is True
    assert llm.calls == [] and sink.usage_calls == []


# --- 7. role-tuned, no ids -------------------------------------------------------------------


async def test_the_profile_is_sent_and_no_id_reaches_the_model():
    llm = ScriptedLLM(0.9)
    await (await make(llm)).process(state())
    for user in llm.users():
        assert "Role: Finance lead" in user and "Skills: budgeting" in user
        assert "You: Okay, and who signs the contract" in user
        assert "Participant 1: We are going with" in user
        for uid in (ALICE, BOB, MEETING, ORG):
            assert str(uid) not in user
    assert "Context from earlier meetings and documents:\n(none)" in llm.users()[1]


# --- 8. grounding ----------------------------------------------------------------------------

M_SHARED, M_ONLY_BOB = uuid.uuid4(), uuid.uuid4()


class FakeRag:
    """Participant-scoped like rag-service (ADR-0033 D1): a passage of a meeting is returned
    only to its attendees. Records every call."""

    def __init__(self) -> None:
        self.attended = {ALICE: {M_SHARED}, BOB: {M_SHARED, M_ONLY_BOB}}
        self.corpus = [
            (M_SHARED, "Q3 vendor budget was capped at 40k.", "c-shared"),
            (M_ONLY_BOB, "Board approved 90k for analytics.", "c-bob"),
        ]
        self.calls: list[dict[str, Any]] = []

    async def passages(
        self, *, org_id: uuid.UUID, user_id: uuid.UUID, query: str, limit: int
    ) -> list[Passage]:
        self.calls.append({"org_id": org_id, "user_id": user_id, "query": query, "limit": limit})
        return [
            Passage(text, {"kind": "transcript", "chunk_id": cid, "meeting_id": str(m)})
            for m, text, cid in self.corpus
            if m in self.attended.get(user_id, set())
        ][:limit]


GROUNDED = sugg(
    {
        "question": "Does the 40k cap from Q3 still hold?",
        "priority": 1,
        "rationale": "You own vendor spend.",
        "sources": [1, 7, "2", True],
        "confidence": 0.88,
    }
)


async def test_grounding_runs_as_the_participant_and_never_on_a_meeting_they_missed():
    rag, llm = FakeRag(), ScriptedLLM(0.9, GROUNDED)
    result = await (await make(llm, grounder=rag)).process(state(user=ALICE))
    (call,) = rag.calls
    assert call["user_id"] == ALICE and call["org_id"] == ORG
    assert "analytics vendor" in call["query"] and str(BOB) not in call["query"]
    proposal = llm.users()[1]
    assert "[1] Q3 vendor budget was capped at 40k." in proposal
    assert "90k" not in proposal  # Bob's meeting never reaches Alice's suggestion
    (s,) = result.payload["suggestions"]
    assert s["citations"] == [
        {"kind": "transcript", "chunk_id": "c-shared", "meeting_id": str(M_SHARED)}
    ]
    assert result.payload["grounded"] is True
    assert all(str(M_ONLY_BOB) not in json.dumps(c) for c in s["citations"])


async def test_bob_sees_his_own_meeting_through_his_own_agent():
    rag, llm = FakeRag(), ScriptedLLM(0.9, GROUNDED)
    agent = QuestionAgent(
        PromptLoader(PROMPTS_DIR), LiteLLMRouter(budget(), Sink(), llm), grounder=rag
    )
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=BOB)
    await agent.process(state(user=BOB))
    assert rag.calls[0]["user_id"] == BOB
    assert "90k" in llm.users()[1]


async def test_no_grounding_call_when_the_gate_fails():
    rag = FakeRag()
    assert await (await make(ScriptedLLM(0.1), grounder=rag)).process(state()) is None
    assert rag.calls == []


async def test_a_grounder_failure_or_timeout_does_not_fail_the_window():
    class Broken:
        async def passages(self, **_kw: Any) -> list[Passage]:
            raise RuntimeError("rag-service down")

    class Slow:
        async def passages(self, **_kw: Any) -> list[Passage]:
            await asyncio.sleep(5)
            return []

    result = await (await make(ScriptedLLM(0.9), grounder=Broken())).process(state())
    assert result is not None and result.payload["grounded"] is False
    assert await ground(Slow(), org_id=ORG, user_id=ALICE, query="x", timeout_s=0.01) == []


async def test_grounding_drops_passages_without_a_known_citation_kind():
    class Mixed:
        async def passages(self, **_kw: Any) -> list[Passage]:
            return [
                Passage("ok", {"kind": "decision", "decision_id": "d1", "meeting_id": "m"}),
                Passage("bad kind", {"kind": "web"}),
                Passage("no citation", {}),
                Passage("   ", {"kind": "document", "document_id": "x"}),
            ]

    out = await ground(Mixed(), org_id=ORG, user_id=ALICE, query="x")
    assert [p.text for p in out] == ["ok"]


def test_grounding_is_off_by_default(monkeypatch):
    monkeypatch.delenv("QUESTION_RAG_GROUNDING", raising=False)
    assert grounder_from_env() is None


async def test_the_rag_adapter_asks_as_the_participant_and_maps_every_citation_kind():
    from libs.shared.rag_client import FakeRagClient, RagPassage
    from services.agent_runtime_service.agents.grounding import RagClientGrounder

    m1, c1, d1, a1, doc = (uuid.uuid4() for _ in range(5))
    client = FakeRagClient(
        passages=[
            RagPassage(
                1,
                "Said in standup.",
                0.9,
                "transcript",
                meeting_id=m1,
                chunk_id=c1,
                meeting_title="Standup",
            ),
            RagPassage(2, "From the Q3 plan.", 0.8, "document", document_id=doc),
            RagPassage(3, "We chose vendor A.", 0.7, "decision", meeting_id=m1, decision_id=d1),
            RagPassage(4, "Bob drafts the contract.", 0.6, "action", action_id=a1),
            RagPassage(5, "Over the limit.", 0.5, "transcript", meeting_id=m1, chunk_id=c1),
        ]
    )
    out = await RagClientGrounder(client).passages(org_id=ORG, user_id=ALICE, query="q", limit=4)
    assert client.calls == [{"org_id": ORG, "user_id": ALICE, "query": "q", "meeting_id": None}]
    assert out == [
        Passage(
            "Said in standup.",
            {"kind": "transcript", "meeting_id": str(m1), "chunk_id": str(c1), "title": "Standup"},
        ),
        Passage("From the Q3 plan.", {"kind": "document", "document_id": str(doc)}),
        Passage(
            "We chose vendor A.",
            {"kind": "decision", "meeting_id": str(m1), "decision_id": str(d1)},
        ),
        Passage("Bob drafts the contract.", {"kind": "action", "action_id": str(a1)}),
    ]


def test_grounding_on_uses_the_http_rag_client(monkeypatch):
    from libs.shared.rag_client import HttpRagClient
    from services.agent_runtime_service.agents.grounding import RagClientGrounder

    monkeypatch.setenv("QUESTION_RAG_GROUNDING", "1")
    grounder = grounder_from_env()
    assert isinstance(grounder, RagClientGrounder)
    assert isinstance(grounder._client, HttpRagClient)


async def test_http_rag_client_sends_the_participant_and_never_asks_for_an_answer(monkeypatch):
    import httpx

    from libs.shared import rag_client
    from services.agent_runtime_service.agents.grounding import RagClientGrounder

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answer_passages": []})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        rag_client.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    client = rag_client.HttpRagClient("http://rag", token="t")
    assert (
        await RagClientGrounder(client).passages(org_id=ORG, user_id=ALICE, query="q", limit=4)
        == []
    )
    assert seen["body"] == {"query": "q", "generate": False}
    assert seen["headers"]["x-mytheus-user"] == str(ALICE)
    assert seen["headers"]["x-mytheus-org"] == str(ORG)


# --- 9. through the runtime ------------------------------------------------------------------


async def test_one_instance_per_participant_and_each_result_is_its_owners():
    sink, llm = Sink(), ScriptedLLM(0.9)
    router = LiteLLMRouter(budget(), sink, llm)

    async def context(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
        return PROFILE

    runtime = AgentRuntime(
        sink=sink, user_context=context, agents=registered_agents(PromptLoader(PROMPTS_DIR), router)
    )
    await runtime.on_room_event(ORG, MEETING, "joined", str(ALICE))
    await runtime.on_room_event(ORG, MEETING, "joined", str(BOB))
    await runtime.process_window(window())
    questions = [r for r in sink.results if r["agent"] == "QuestionAgent"]
    assert sorted(r["user_id"] for r in questions) == sorted([ALICE, BOB])
    ids = {r["user_id"]: r["result"].payload["suggestions"][0]["id"] for r in questions}
    assert ids[ALICE] != ids[BOB]
    keys = [k for k in runtime._meeting(ORG, MEETING).agents if k[0] == "QuestionAgent"]
    assert sorted(k[1] for k in keys) == sorted([ALICE, BOB])


# --- 10. prompts -----------------------------------------------------------------------------


def test_the_prompts_load_render_and_are_registered():
    loader = PromptLoader(PROMPTS_DIR)
    registry = json.loads((Path(PROMPTS_DIR) / "registry.json").read_text("utf-8"))
    assert registry["defaults"]["question_gate"] == "v2"  # ADR-0038 D10
    assert registry["defaults"]["question_agent"] == "v2"
    for key, version in (
        ("question_gate", "v1"),
        ("question_gate", "v2"),
        ("question_agent", "v1"),
        ("question_agent", "v2"),
    ):
        prompt = loader.load(key, version)
        values = {"profile": "Role: x", "focus": "", "voice": "", "window": "You: hi"}
        rendered = prompt.render_user(**values, context="(none)")
        assert "Role: x" in rendered and "You: hi" in rendered
    for version in ("v1", "v2"):
        assert '{"suggestions": []}' in loader.load("question_agent", version).system
