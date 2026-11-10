"""Polish plan, Phase 0: names, not "Participant 2", and one threshold source.

1. Labels are stable per meeting: a speaker keeps the label of their first appearance in every
   window and every agent's prompt; a window built outside the runtime numbers as before.
2. The model never sees a name or an email; after it answers, every "Participant N" in every
   user-visible text (note, streamed deltas, candidates, suggestions, fill, alert, agenda note)
   becomes the person's display name, and stays the label when there is no name.
3. The tenant's `agent_confidence_overrides` decide `surfaced` in the agent (the same
   comparisons core-api makes), fetched through a cached client, so the gateway's live
   forward and core-api's reload agree.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import fakeredis
import httpx
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents.candidates import render_window as labelled
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.personal_listener_agent import (
    PersonalListenerAgent,
    render_window,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.names import (
    LineStream,
    Namer,
    Postprocess,
    assign_labels,
    display_name,
    speaker_labels,
)
from services.agent_runtime_service.policy import (
    CONTEXT_KEY,
    AgentPolicyClient,
    Thresholds,
    parse_overrides,
    thresholds,
    thresholds_for,
)
from services.agent_runtime_service.runtime import AgentRuntime

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
ALICE, BOB, CARA = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
EMAILS = {ALICE: "alice.wong@acme.com", BOB: "bob_okafor@acme.com", CARA: "cara@acme.com"}
PROFILE = {"role": "Backend engineer", "skills": ["python"], "bio": ""}


def win(start: int, *turns: tuple[uuid.UUID | str, str]) -> TranscriptWindow:
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:{start}",
        start_ms=start,
        end_ms=start + 30_000,
        turns=[
            Turn(speaker_id=str(s), text=t, start_ms=start + i, end_ms=start + i + 1, chunk_ids=[])
            for i, (s, t) in enumerate(turns)
        ],
    )


# --- 1. stable labels ------------------------------------------------------------------------


def test_a_speaker_keeps_the_label_of_their_first_appearance_for_the_whole_meeting():
    registry: dict[str, str] = {}
    first = win(0, (BOB, "Hi."), (CARA, "Hello."))
    second = win(30_000, (CARA, "So, the release."), (ALICE, "Friday."), (BOB, "Agreed."))
    assign_labels(registry, first)
    assign_labels(registry, second)
    assert registry == {
        str(BOB): "Participant 1",
        str(CARA): "Participant 2",
        str(ALICE): "Participant 3",
    }
    # renumbered per window before: Cara was "Participant 1" in the second window
    assert labelled(second)[0].splitlines() == [
        "Participant 2: So, the release.",
        "Participant 3: Friday.",
        "Participant 1: Agreed.",
    ]
    assert labelled(second)[1] == {
        "Participant 2": str(CARA),
        "Participant 3": str(ALICE),
        "Participant 1": str(BOB),
    }
    # PersonalListener: the participant is "You", everyone else keeps their meeting-wide label
    assert render_window(second, ALICE).splitlines() == [
        "Participant 2: So, the release.",
        "You: Friday.",
        "Participant 1: Agreed.",
    ]


def test_a_window_built_outside_the_runtime_numbers_as_before():
    w = win(0, (CARA, "a"), (BOB, "b"), (CARA, "c"))
    assert speaker_labels(w) == {str(CARA): "Participant 1", str(BOB): "Participant 2"}
    assert render_window(w, CARA) == "You: a\nParticipant 1: b\nYou: c"


def test_a_speaker_missing_from_the_registry_never_takes_a_used_number():
    w = win(0, (ALICE, "a"), (BOB, "b"))
    w.labels = {str(ALICE): "Participant 4"}
    assert speaker_labels(w) == {str(ALICE): "Participant 4", str(BOB): "Participant 5"}


# --- 2. display names --------------------------------------------------------------------------


def test_display_names_come_from_the_context_and_fall_back_to_nothing():
    assert display_name({"email": "priya.sharma@acme.com"}) == "Priya Sharma"
    assert display_name({"email": "bob_okafor2@acme.com"}) == "Bob Okafor"
    assert display_name({"display_name": "Dr. Lee", "email": "x@y.z"}) == "Dr. Lee"
    assert display_name({"email": "1234@acme.com"}) is None
    assert display_name({"role": "x"}) is None and display_name(None) is None


def test_every_label_in_every_text_becomes_a_name_and_an_unknown_label_stays():
    labels = {str(ALICE): "Participant 1", str(BOB): "Participant 2", "egress": "Participant 3"}
    namer = Namer.build(labels, {ALICE: {"email": EMAILS[ALICE]}, BOB: {"email": EMAILS[BOB]}})
    assert namer.text(
        "Participant 2 asked Participant 1; participant 3 and Participant 9 nodded"
    ) == ("Bob Okafor asked Alice Wong; participant 3 and Participant 9 nodded")
    post = Postprocess(namer, NeverRecordPolicy())
    cand = AgentResult(
        kind="decision_candidate",
        content="Participant 2 ships Friday",
        relevance_score=0.9,
        confidence=0.9,
        source_chunk_ids=[],
        payload={
            "candidates": [
                {
                    "id": "Participant 2",  # a data key is never rewritten
                    "content": "Participant 2 ships Friday",
                    "rationale": "Participant 1 agreed",
                    "stakeholder_ids": [str(BOB)],
                    "confidence": 0.9,
                }
            ],
            "prompt": "decision_agent@v2",
        },
    )
    out = post.result(cand)
    assert out is not None and out.content == "Bob Okafor ships Friday"
    [c] = out.payload["candidates"]
    assert (c["content"], c["rationale"], c["id"]) == (
        "Bob Okafor ships Friday",
        "Alice Wong agreed",
        "Participant 2",
    )
    assert cand.payload["candidates"][0]["content"] == "Participant 2 ships Friday"  # untouched


async def test_streamed_deltas_are_named_line_by_line_even_when_a_label_is_split():
    sent: list[tuple[int, str, bool]] = []

    async def sink(seq: int, text: str, done: bool) -> None:
        sent.append((seq, text, done))

    namer = Namer({"Participant 2": "Bob Okafor"})
    stream = LineStream(sink, Postprocess(namer, NeverRecordPolicy()).line)
    for i, piece in enumerate(["- Partic", "ipant 2 asked", " for the logs\n- You ", "agreed"]):
        await stream(i, piece, False)
    await stream(9, "", True)
    assert "".join(t for _s, t, _d in sent) == "- Bob Okafor asked for the logs\n- You agreed"
    assert [s for s, _t, _d in sent] == list(range(len(sent))) and sent[-1][2] is True


# --- 2b. through the runtime: the model sees labels, the participant reads names --------------


class Sink:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.deltas: list[dict[str, Any]] = []
        self.usage_calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        pass

    async def delta(self, **kw: Any) -> None:
        self.deltas.append(kw)


class NoteLLM:
    """The gate passes; the note names whoever the transcript calls "Participant 1"."""

    def __init__(self, confidence: float = 0.9) -> None:
        self.calls: list[dict[str, Any]] = []
        self.confidence = confidence

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if "haiku" in kw["model"]:
            text = json.dumps({"relevance": 0.9})
        else:
            text = (
                f"CONFIDENCE: {self.confidence}\nCOMMITMENTS: []\n"
                "- Participant 1 asked You for the logs"
            )
        if kw.get("stream"):
            return self._stream(text)
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 9}}

    async def _stream(self, text: str) -> Any:
        for i in range(0, len(text), 5):
            yield {"choices": [{"delta": {"content": text[i : i + 5]}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 9}}


def gate() -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9}

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


async def contexts(_org: uuid.UUID, user: uuid.UUID) -> dict[str, Any]:
    return {**PROFILE, "email": EMAILS[user]}


async def test_the_model_sees_labels_and_the_participant_reads_names_live_and_stored():
    sink, llm = Sink(), NoteLLM()
    router = LiteLLMRouter(gate(), sink, llm)
    runtime = AgentRuntime(
        sink=sink,
        user_context=contexts,
        agents=[lambda: PersonalListenerAgent(PromptLoader(PROMPTS_DIR), router)],
    )
    await runtime.on_room_event(ORG, MEETING, "joined", str(ALICE))
    await runtime.process_window(win(0, (BOB, "Hi all."), (ALICE, "Hi.")))
    await runtime.process_window(win(30_000, (ALICE, "Can you send the logs?"), (BOB, "Sure.")))
    sent = " ".join(str(c["messages"]) for c in llm.calls)
    assert "Participant 1: Sure." in sent  # Bob kept "Participant 1" in the second window
    assert "Bob" not in sent and "acme.com" not in sent  # never a name or an email
    alice = [r["result"] for r in sink.results if r["user_id"] == ALICE]
    assert alice and all(r.content == "- Bob Okafor asked You for the logs" for r in alice)
    live = "".join(d["text"] for d in sink.deltas if d["user_id"] == ALICE)
    assert "Bob Okafor asked You for the logs" in live and "Participant" not in live


# --- 3. one threshold source -----------------------------------------------------------------


def test_thresholds_mirror_core_api_and_reject_a_bad_entry():
    t = thresholds(
        {"PersonalListenerAgent": {"surface": 0.9, "silent": 0.5}}, "PersonalListenerAgent"
    )
    assert t == Thresholds(0.9, 0.5)
    assert [t.band(x) for x in (0.95, 0.9, 0.5, 0.49)] == [
        "surfaced",
        "silent",
        "silent",
        "dropped",
    ]
    assert thresholds({"X": {"surface": "high"}}, "X") == Thresholds()
    assert thresholds({"X": {"surface": 0.3, "silent": 0.6}}, "X") == Thresholds()
    assert thresholds({"X": {"surface": 0.6}}, "X") == Thresholds(0.6, 0.4)
    assert thresholds_for({CONTEXT_KEY: {"X": {"surface": 0.5, "silent": 0.1}}}, "X").silent == 0.1
    assert parse_overrides(
        {"agent_confidence_overrides": {"A": {"surface": 0.8, "silent": 0.2}}}
    ) == {"A": {"surface": 0.8, "silent": 0.2}}


async def test_the_policy_client_caches_per_org_and_a_failure_is_the_defaults():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["X-Mytheus-Org"] == str(ALICE):
            return httpx.Response(503)
        body = {"agent_confidence_overrides": {"DecisionAgent": {"surface": 0.8, "silent": 0.5}}}
        return httpx.Response(200, json=body)

    now = [0.0]
    client = AgentPolicyClient(
        "http://core",
        token="t" * 32,
        ttl_s=60,
        clock=lambda: now[0],
        transport=httpx.MockTransport(handler),
    )
    assert await client.overrides(ORG) == {"DecisionAgent": {"surface": 0.8, "silent": 0.5}}
    assert await client.overrides(ORG) == {"DecisionAgent": {"surface": 0.8, "silent": 0.5}}
    assert len(calls) == 1 and calls[0].url.path == "/internal/agent-policy"
    assert calls[0].headers["X-Mytheus-Service"] == "agent-runtime-service"
    assert await client.overrides(ALICE) == {}  # failure: the defaults ...
    now[0] = 16.0
    assert await client.overrides(ALICE) == {} and len(calls) == 3  # ... cached 15 s only
    assert await AgentPolicyClient("", token="").overrides(ORG) == {}  # unconfigured: defaults


class CandidateLLM:
    def __init__(self, confidence: float) -> None:
        self.answer = json.dumps(
            {
                "candidates": [
                    {"content": "Ship Friday", "stakeholders": [], "confidence": confidence}
                ]
            }
        )

    async def __call__(self, **kw: Any) -> Any:
        text = json.dumps({"score": 0.9}) if "haiku" in kw["model"] else self.answer
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 9}}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),  # 0.75 > 0.7: surfaced with the defaults
        ({"DecisionAgent": {"surface": 0.8, "silent": 0.4}}, False),  # silent for this tenant
        ({"DecisionAgent": {"surface": 0.9, "silent": 0.8}}, None),  # below silent: dropped
    ],
)
async def test_the_agent_bands_with_the_tenants_overrides(overrides: dict[str, Any], expected: Any):
    agent = DecisionAgent(
        PromptLoader(PROMPTS_DIR), LiteLLMRouter(gate(), Sink(), CandidateLLM(0.75))
    )
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    state = AgentState(
        meeting_id=MEETING,
        user_id=ALICE,
        chunk=win(0, (ALICE, "Let's ship Friday."), (BOB, "Agreed.")),
        context={**PROFILE, CONTEXT_KEY: overrides},
        org_id=ORG,
    )
    result = await agent.process(state)
    if expected is None:
        assert result is None
    else:
        assert result is not None and result.payload["candidates"][0]["surfaced"] is expected


async def test_the_runtime_hands_every_agent_the_tenants_overrides_and_the_stream_follows_them():
    """A 0.75 note is surfaced (and streamed live) by default, but with this tenant's surface
    0.8 it is silent: stored, never streamed, exactly what core-api will reload."""
    seen: list[uuid.UUID] = []

    async def policy(org: uuid.UUID) -> dict[str, Any]:
        seen.append(org)
        return {"PersonalListenerAgent": {"surface": 0.8, "silent": 0.4}}

    sink, llm = Sink(), NoteLLM(confidence=0.75)
    router = LiteLLMRouter(gate(), sink, llm)
    runtime = AgentRuntime(
        sink=sink,
        user_context=contexts,
        agents=[lambda: PersonalListenerAgent(PromptLoader(PROMPTS_DIR), router)],
        policy=policy,
    )
    await runtime.on_room_event(ORG, MEETING, "joined", str(ALICE))
    await runtime.process_window(win(0, (BOB, "Send the logs?"), (ALICE, "Sure.")))
    [stored] = [r["result"] for r in sink.results if r["user_id"] == ALICE]
    assert stored.payload["surfaced"] is False and seen == [ORG]
    assert all(not d["text"] for d in sink.deltas)  # nothing streamed (only the done marker)


async def test_never_record_text_is_dropped_by_the_same_postprocess():
    org = CompanyContext(never_record_topics=("salaries",))
    post = Postprocess(Namer({}), NeverRecordPolicy.from_org(org))
    note = AgentResult(
        kind="note",
        content="- Salary bands go up 4%\n- You own the migration",
        relevance_score=1,
        confidence=0.9,
        source_chunk_ids=[],
    )
    out = post.result(note)
    assert out is not None and out.content == "- You own the migration"
