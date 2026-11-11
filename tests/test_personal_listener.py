"""`PersonalListenerAgent`: Plan 7-9 section 3.3 acceptance checks, every model call faked.

1. One instance per participant per meeting (spawn count against the roster).
2. A window below the relevance threshold makes no extraction call.
3. Confidence bands: `> 0.7` surfaced, `0.4-0.7` silent, `< 0.4` and `NONE` produce nothing.
4. Streaming: only a surfaced note streams, and the confidence line never does.
5. A window containing `jane@acme.com` leaves redacted; no user id reaches the model.
6. The shipped prompt files load, render and are registered.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.personal_listener_agent import (
    PersonalListenerAgent,
    band,
    parse_note,
    parse_relevance,
    render_window,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.runtime import AgentRuntime

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
ALICE, BOB, CARA = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
CONTEXT = {"role": "Backend engineer", "skills": ["python", "postgres"], "bio": ""}


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


class ScriptedLLM:
    """Haiku answers the gate, Sonnet answers the note (streamed in small pieces)."""

    def __init__(self, relevance: float, note: str) -> None:
        self.gate = json.dumps({"relevance": relevance})
        self.note = note
        self.calls: list[dict[str, Any]] = []

    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        text = self.gate if "haiku" in kw["model"] else self.note
        if kw.get("stream"):
            return self._stream(text)
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 10},
        }

    async def _stream(self, text: str) -> Any:
        for i in range(0, len(text), 7):
            yield {"choices": [{"delta": {"content": text[i : i + 7]}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 60}}


def gate() -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": 0,
            "token_quota": 10**6,
            "hard_cap": 10**6,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


def window(
    text: str = "We ship the billing fix Friday, Alice owns the migration.",
) -> TranscriptWindow:
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:0",
        start_ms=0,
        end_ms=30_000,
        turns=[
            Turn(speaker_id=str(BOB), text=text, start_ms=0, end_ms=5_000, chunk_ids=["c1"]),
            Turn(
                speaker_id=str(ALICE),
                text="Okay, I can do it.",
                start_ms=5_000,
                end_ms=8_000,
                chunk_ids=["c2"],
            ),
        ],
    )


async def run_agent(
    llm: ScriptedLLM, sink: Sink | None = None, w: TranscriptWindow | None = None
) -> Any:
    sink = sink or Sink()
    router = LiteLLMRouter(gate(), sink, llm)
    agent = PersonalListenerAgent(PromptLoader(PROMPTS_DIR), router)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=ALICE)
    w = w or window()

    async def stream(seq: int, text: str, done: bool) -> None:
        sink.deltas.append({"seq": seq, "text": text, "done": done})

    state = AgentState(
        meeting_id=MEETING, user_id=ALICE, chunk=w, context=CONTEXT, org_id=ORG, stream=stream
    )
    return await agent.process(state)


# --- 1. spawn count ------------------------------------------------------------------------


async def test_one_personal_listener_per_participant_per_meeting():
    sink, llm = Sink(), ScriptedLLM(0.9, "CONFIDENCE: 0.9\n- You own the migration.")
    router = LiteLLMRouter(gate(), sink, llm)

    async def ctx(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
        return CONTEXT

    runtime = AgentRuntime(
        sink=sink, user_context=ctx, agents=registered_agents(PromptLoader(PROMPTS_DIR), router)
    )
    for u in (ALICE, BOB, CARA):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    await runtime.process_window(window())
    second = window()
    second.window_id = f"{MEETING}:30000"
    await runtime.process_window(second)

    m = runtime._meetings[(ORG, MEETING)]
    listeners = [k for k in m.agents if k[0] == "PersonalListenerAgent"]
    assert len(listeners) == 3
    assert {k[1] for k in listeners} == {ALICE, BOB, CARA}
    assert len(sink.results) == 6  # 3 participants x 2 windows, instances reused
    assert {r["agent"] for r in sink.results} == {"PersonalListenerAgent"}


# --- 2. pre-gate ---------------------------------------------------------------------------


async def test_a_window_below_the_relevance_threshold_makes_no_extraction_call(monkeypatch):
    monkeypatch.setenv("PERSONAL_LISTENER_RELEVANCE_THRESHOLD", "0.5")
    llm, sink = ScriptedLLM(0.3, "CONFIDENCE: 0.9\n- never"), Sink()
    assert await run_agent(llm, sink) is None
    assert llm.models() == ["anthropic/claude-haiku-5-5"]
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku"]

    llm = ScriptedLLM(0.5, "CONFIDENCE: 0.9\n- You own the migration.")
    assert await run_agent(llm) is not None
    assert llm.models() == ["anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"]


def test_relevance_parsing_fails_to_zero():
    assert parse_relevance('{"relevance": 0.82}') == 0.82
    assert parse_relevance("relevance 0.6") == 0.6
    assert parse_relevance('{"relevance": 7}') == 1.0
    assert parse_relevance("no idea") == 0.0


# --- 3. confidence bands -------------------------------------------------------------------


@pytest.mark.parametrize(
    "note,surfaced",
    [
        ("CONFIDENCE: 0.91\n- You own the migration, due Friday.", True),
        ("CONFIDENCE: 0.55\n- Something about a migration.", False),
        ("CONFIDENCE: 0.4\n- Borderline.", False),
    ],
)
async def test_emitted_notes_carry_their_band(note: str, surfaced: bool):
    result = await run_agent(ScriptedLLM(0.9, note))
    assert result is not None and result.kind == "note"
    assert result.payload["surfaced"] is surfaced
    assert result.source_chunk_ids == ["c1", "c2"]
    assert result.relevance_score == 0.9
    assert not result.content.startswith("CONFIDENCE")


@pytest.mark.parametrize(
    "note", ["CONFIDENCE: 0.39\n- weak", "CONFIDENCE: 0\nNONE", "no header at all", ""]
)
async def test_low_confidence_or_empty_notes_produce_nothing(note: str):
    assert await run_agent(ScriptedLLM(0.9, note)) is None


def test_bands_and_note_parsing():
    assert band(0.71) == "surfaced" and band(0.7) == "silent" and band(0.4) == "silent"
    assert band(0.39) == "dropped"
    assert parse_note("CONFIDENCE: 0.8\n- a\n- b") == (0.8, "- a\n- b")
    assert parse_note("confidence:0.5\nNONE") == (0.5, "")


# --- 4. streaming --------------------------------------------------------------------------


async def test_only_a_surfaced_note_streams_and_never_its_confidence_line():
    sink = Sink()
    note = "CONFIDENCE: 0.9\n- You own the migration, due Friday."
    await run_agent(ScriptedLLM(0.9, note), sink)
    streamed = "".join(d["text"] for d in sink.deltas)
    assert streamed == "- You own the migration, due Friday."
    assert sink.deltas[-1]["done"] is True
    assert [d["seq"] for d in sink.deltas] == list(range(len(sink.deltas)))

    silent = Sink()
    await run_agent(ScriptedLLM(0.9, "CONFIDENCE: 0.6\n- maybe something"), silent)
    assert silent.deltas == []


# --- 5. what reaches the model -------------------------------------------------------------


async def test_pii_is_redacted_and_no_user_id_reaches_the_model():
    llm = ScriptedLLM(0.9, "CONFIDENCE: 0.9\n- Reply to [REDACTED_EMAIL].")
    await run_agent(llm, w=window("Send the invoice to jane@acme.com today."))
    for call in llm.calls:
        sent = json.dumps(call["messages"])
        assert "jane@acme.com" not in sent and "[REDACTED_EMAIL]" in sent
        for uid in (ALICE, BOB):
            assert str(uid) not in sent
        assert "You: Okay, I can do it." in sent and "Participant 1:" in sent


def test_the_participant_is_you_and_others_are_numbered():
    w = window()
    w.turns.append(Turn(speaker_id=str(CARA), text="+1", start_ms=9_000, end_ms=9_500))
    w.turns.append(Turn(speaker_id=str(BOB), text="Thanks", start_ms=9_600, end_ms=9_900))
    lines = render_window(w, ALICE).splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        "Participant 1",
        "You",
        "Participant 2",
        "Participant 1",
    ]


# --- 6. shipped prompts ----------------------------------------------------------------------


def test_the_shipped_prompts_are_registered_and_render():
    loader = PromptLoader(PROMPTS_DIR)
    for key in ("personal_listener", "personal_listener_gate"):
        prompt = loader.load(key, loader.resolve(key, ORG))
        rendered = prompt.render_user(
            profile="Role: x",
            focus="",
            voice="",
            window="You: hi",
            company="",
            ground_rules="",
            attendance="",
            person_memory="",
            guidance="",
            memory_focus="",
        )
        assert "Role: x" in rendered and "You: hi" in rendered

    assert json.loads((Path(PROMPTS_DIR) / "registry.json").read_text())["defaults"] == {
        "personal_listener": "v5",
        "personal_listener_gate": "v6",
        "decision_gate": "v2",
        "decision_agent": "v5",
        "action_gate": "v3",
        "action_agent": "v5",
        "question_gate": "v4",
        "question_agent": "v4",
        "context_gap_gate": "v2",
        "context_gap_agent": "v2",
        "proactive_conflict": "v2",
        "proactive_stakeholder": "v3",
        "meeting_agent": "v3",
        "proactive_already_decided_gate": "v2",
        "proactive_already_decided": "v2",
        "proactive_topic_watch": "v2",
        "minutes": "v1",  # minutes for everyone
        "delegated_answer": "v1",
        "promise_extract": "v1",  # promise ledger: the person's own messages
    }


def test_a_crlf_checkout_still_splits_system_from_user(tmp_path: Path):
    (tmp_path / "registry.json").write_text('{"defaults": {"k": "v1"}}')
    (tmp_path / "k@v1.md").write_bytes(b"System line.\r\n---user---\r\nWindow: {window}\r\n")
    prompt = PromptLoader(tmp_path).load("k", "v1")
    assert prompt.system == "System line." and prompt.render_user(window="x") == "Window: x"
