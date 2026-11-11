"""Delegated answers (polish wave 3): an absent person's agent answers only from what they
pre-approved, on the ContextGapAgent peer path. No paid model: a scripted LLM.

1. The detector: a question that names the delegate, not spoken by them; mentions are ignored.
2. A covered question -> the pre-approved answer, one surfaced `peer` fill per live
   participant, and one `delegated` block (same id on every copy) for core-api's log.
3. Not covered (or a statement number out of range, or garbage) -> the fixed refusal, written
   by code, never by the model.
4. `addressed: false` -> nothing. The same question twice in a meeting -> answered once.
5. Budget blocked -> no answer, agent marked blocked. No delegates -> the gap pass runs as
   before. The client parses core-api's body and fails closed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.agents.context_gap_agent import (
    ContextGapAgent,
    ContextGapDeps,
    ContextGapSettings,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.delegation import answerer as answerer_mod
from services.agent_runtime_service.delegation.client import (
    Delegate,
    Statement,
    delegation_from_env,
    parse,
)
from tests.a2a_helpers import budget

ORG, MEETING = uuid.uuid4(), uuid.uuid4()
MAYA, RAJ, SARAH = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
VENDOR = Statement(uuid.uuid4(), "Either vendor under $40k is fine")
SLIP = Statement(uuid.uuid4(), "The launch can slip one week, not more")
DELEGATE = Delegate(SARAH, "Sarah Chen", (VENDOR, SLIP))
MARKER = "sent their AI agent instead"


class Sink:
    async def usage(self, **kw: Any) -> None:
        pass

    async def result(self, **kw: Any) -> None:
        pass

    async def notification(self, **kw: Any) -> None:
        pass

    async def delta(self, **kw: Any) -> None:
        pass


class LLM:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        system = kw["messages"][0]["content"][0]["text"]
        if MARKER in system:
            self.calls.append(kw)
            text = self.verdict
        else:
            text = '{"is_gap": false, "query": "", "score": 0}'
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 30},
        }


class Source:
    def __init__(self, delegates: list[Delegate]) -> None:
        self.delegates_ = delegates

    async def delegates(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> list[Delegate]:
        assert org_id == ORG and meeting_id == MEETING
        return self.delegates_


def window(text: str, speaker: uuid.UUID = MAYA, window_id: str | None = None) -> TranscriptWindow:
    turns = [
        Turn(str(RAJ), "We have two quotes on the table now.", 0, 3_000, [str(uuid.uuid4())]),
        Turn(str(speaker), text, 3_000, 8_000, [str(uuid.uuid4())]),
    ]
    return TranscriptWindow(ORG, MEETING, window_id or f"{MEETING}:0", 0, 30_000, turns)


async def make(verdict: str, *, gate: Any = None, delegates: list[Delegate] | None = None):
    llm = LLM(verdict)
    router = LiteLLMRouter(gate or budget(), Sink(), llm)
    agent = ContextGapAgent(
        PromptLoader(PROMPTS_DIR),
        router,
        deps=ContextGapDeps(
            rag=None, delegation=Source([DELEGATE] if delegates is None else delegates)
        ),
        settings=ContextGapSettings(a2a=False),
    )
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    return agent, llm


CONTEXTS: dict[uuid.UUID, dict[str, Any] | None] = {MAYA: {"user_id": str(MAYA)}, RAJ: {}}


def covered(n: int = 1, answer: str = "Sarah pre-approved either vendor under $40k.") -> str:
    return json.dumps({"addressed": True, "statement": n, "answer": answer})


# --- 1. detector ----------------------------------------------------------------------------


def test_the_detector_wants_a_question_for_the_delegate():
    find = answerer_mod.find_question
    assert find(window("Is Sarah OK with vendor B at $35k?"), DELEGATE) is not None
    assert find(window("what does sarah think about the launch"), DELEGATE) is not None
    assert find(window("Sarah sent the deck this morning."), DELEGATE) is None
    assert find(window("Is Raj OK with vendor B?"), DELEGATE) is None
    # Sarah's own words are never a question to her agent.
    assert find(window("Is Sarah OK with this?", speaker=SARAH), DELEGATE) is None
    assert answerer_mod.names_delegate("Sarah's view on it?", DELEGATE)
    assert not answerer_mod.names_delegate("Sarahs", DELEGATE)


# --- 2. covered ------------------------------------------------------------------------------


async def test_a_covered_question_is_answered_from_the_statement_to_every_live_person():
    agent, llm = await make(covered())
    results = await agent.process_meeting(window("Is Sarah OK with vendor B at $35k?"), CONTEXTS)
    assert len(llm.calls) == 1
    user_text = json.dumps(llm.calls[0]["messages"][1:])
    assert "[1] Either vendor under $40k is fine" in user_text and "Sarah" in user_text
    assert [u for u, _ in results] == [MAYA, RAJ]
    blocks = [r.payload["delegated"] for _, r in results]
    assert blocks[0] == blocks[1]
    block = blocks[0]
    assert block["covered"] is True and block["statement_id"] == str(VENDOR.id)
    assert block["statement_text"] == VENDOR.text and block["asked_by"] == str(MAYA)
    assert block["delegator_user_id"] == str(SARAH)
    _, first = results[0]
    fill = first.payload["fill"]
    assert fill["surfaced"] is True and fill["hop"] == "peer"
    assert fill["responder"] == f"user:{SARAH}"
    assert fill["content"] == "Sarah's agent: Sarah pre-approved either vendor under $40k."
    entry = first.payload["collaborations"][0]
    assert entry["id"] == fill["id"] and entry["requester"] == str(MAYA)
    assert entry["result"] == "answered" and entry["hop"] == "peer"
    assert first.kind == "context_fill" and first.confidence > 0.7


# --- 3. refusals -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        json.dumps({"addressed": True, "statement": None, "answer": ""}),
        json.dumps({"addressed": True, "statement": 7, "answer": "Sure, $55k is fine."}),
        json.dumps({"addressed": True, "statement": 1, "answer": ""}),
        "I think she would be fine with it",
    ],
)
async def test_anything_not_covered_gets_the_fixed_refusal(verdict: str):
    agent, _ = await make(verdict)
    results = await agent.process_meeting(window("Can Sarah approve vendor B at $55k?"), CONTEXTS)
    block = results[0][1].payload["delegated"]
    assert block["covered"] is False and block["statement_id"] is None
    assert block["answer"] == "Sarah didn't pre-approve an answer to that."
    assert results[0][1].payload["collaborations"][0]["result"] == "no_answer"


# --- 4. not addressed / once -----------------------------------------------------------------


async def test_not_addressed_says_nothing_and_a_repeat_is_answered_once():
    agent, _ = await make(json.dumps({"addressed": False, "statement": None, "answer": ""}))
    assert await agent.process_meeting(window("Did Sarah send the deck?"), CONTEXTS) == []
    agent, llm = await make(covered())
    q = "Is Sarah OK with vendor B at $35k?"
    assert await agent.process_meeting(window(q), CONTEXTS)
    again = await agent.process_meeting(window(q + "  ", window_id=f"{MEETING}:30000"), CONTEXTS)
    assert again == [] and len(llm.calls) == 1


# --- 5. budget, no delegates, client ---------------------------------------------------------


async def test_a_blocked_budget_answers_nothing():
    agent, llm = await make(covered(), gate=budget(consumed=10**8, quota=10, cap=10))
    assert await agent.process_meeting(window("Is Sarah OK with vendor B?"), CONTEXTS) == []
    assert agent.blocked is True and llm.calls == []


async def test_without_delegates_nothing_is_asked():
    agent, llm = await make(covered(), delegates=[])
    assert await agent.process_meeting(window("Is Sarah OK with vendor B?"), CONTEXTS) == []
    assert llm.calls == []


def test_the_client_parses_and_fails_closed(monkeypatch: pytest.MonkeyPatch):
    sid = uuid.uuid4()
    body = {
        "delegates": [
            {
                "user_id": str(SARAH),
                "name": "Sarah Chen",
                "statements": [{"id": str(sid), "text": " a  b "}],
            },
            {"user_id": "not-a-uuid", "name": "x", "statements": []},
            {"user_id": str(RAJ), "name": "Raj", "statements": []},
        ]
    }
    got = parse(body)
    assert got == [Delegate(SARAH, "Sarah Chen", (Statement(sid, "a b"),))]
    assert got[0].first_name == "Sarah"
    assert parse("nope") == [] and parse({"delegates": "x"}) == []
    monkeypatch.delenv("CORE_URL", raising=False)
    assert delegation_from_env() is None
    monkeypatch.setenv("CORE_URL", "http://core")
    monkeypatch.setenv("CORE_CALLER_TOKEN", "t" * 40)
    assert delegation_from_env() is not None
    monkeypatch.setenv("DELEGATED_ANSWERS", "0")
    assert delegation_from_env() is None


def test_the_prompt_is_registered_and_cacheable():
    loader = PromptLoader(PROMPTS_DIR)
    prompt = loader.load("delegated_answer", loader.resolve("delegated_answer", ORG))
    assert prompt.ref == "delegated_answer@v1"
    assert "{" not in prompt.system.replace('{"addressed"', "")
    assert len(prompt.system) >= 2400
    rendered = prompt.render_user(name="Sarah", statements="[1] x", question="Is Sarah OK?")
    assert "Is Sarah OK?" in rendered
