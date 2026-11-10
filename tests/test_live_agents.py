"""Polish program, "Make the meeting calm and fast": the agent side, every model call faked.

1. Memory inside the meeting: the ledger in Redis (`nexus:{org_id}:meeting-ledger:{meeting_id}`,
   one writer per field), the `{memory}` prompt block, and `reconcile` (an update keeps its
   card's id, a near-duplicate of an open card becomes an update, a repeat of a dismissed /
   settled / edited card is never emitted).
2. The candidate agents end to end: decision_agent@v3 / action_agent@v3 get the memory and
   the model's `updates` / `quote` come back as `updates_candidate_id` and an anchored quote.
3. One card per moment: a promise in a note and the action quoting the same words share one
   `moment_id`; why lines; never-record still strips the words.
4. The runtime: memory reaches every agent, emitted cards land in the ledger, a dismissed
   suggestion does not return, notes accumulate windows (fewer, fuller notes).
5. The emitter: the top-level moment fields and `mytheus_agent_time_to_card_seconds`.
6. One task per meeting: a slow meeting never holds up another; a task that fell behind merges
   its queued windows.
7. The v3 prompts with no memory render the v2 message apart from the stretch's heading.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import fakeredis
import pytest

from libs.shared.metrics import AGENT_TIME_TO_CARD
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, TopicOwner
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow, Trigger, Turn, Utterance
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR, Loop
from services.agent_runtime_service.emitter import Emitter, top_item
from services.agent_runtime_service.memory import (
    CONTEXT_KEY,
    LedgerEntry,
    MeetingLedger,
    MeetingMemory,
    SuggestionMark,
    drop_repeated_suggestions,
    ledger_key,
    reconcile,
    render_memory,
)
from services.agent_runtime_service.moments import anchor, annotate, moment_id
from services.agent_runtime_service.names import Namer, Postprocess
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.windowing import Chunk, Windower
from tests.test_decision_action_agents import (
    ALICE,
    BOB,
    MEETING,
    ORG,
    ScriptedLLM,
    Sink,
    cands,
    gate,
    make,
)

D1, D2, D3 = (str(uuid.uuid4()) for _ in range(3))


def memory(*entries: LedgerEntry, previous: list[str] | None = None) -> MeetingMemory:
    return MeetingMemory(previous=previous or [], entries=list(entries))


def utt(cid: str, speaker: uuid.UUID, start: int, text: str) -> Utterance:
    return Utterance(cid, str(speaker), start, start + 3_000, text)


def moment_window() -> TranscriptWindow:
    a = utt("c1", ALICE, 0, "QA signed off on the build.")
    b = utt("c2", BOB, 3_000, "Okay. I'll send the pricing deck to the client by Friday.")
    turns = [
        Turn(str(u.speaker_id), u.text, u.start_ms, u.end_ms, [u.chunk_id], [u]) for u in (a, b)
    ]
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:0",
        start_ms=0,
        end_ms=6_000,
        turns=turns,
        labels={str(ALICE): "Participant 1", str(BOB): "Participant 2"},
        trigger=Trigger("cue", "c2", str(BOB), 3_000, 6_000, b.text, "i'll"),
    )


# --- 1. the ledger and reconcile ------------------------------------------------------------


async def test_the_ledger_merges_the_runtime_and_core_api_fields():
    redis = fakeredis.aioredis.FakeRedis()
    ledger = MeetingLedger(redis, clock=lambda: 100.0)
    await ledger.record(
        ORG,
        MEETING,
        [
            LedgerEntry(D1, "decision", "Ship the beta on 2 December", moment_id="m1"),
            LedgerEntry(D2, "action", "Write the release notes"),
            LedgerEntry(D3, "decision", "Use Mongo for the ledger"),
        ],
    )
    key = ledger_key(ORG, MEETING)
    assert key == f"nexus:{ORG}:meeting-ledger:{MEETING}"
    assert 0 < await redis.ttl(key) <= 24 * 3600
    # core-api's fields: a dismissal, an edit (new text), a suggestion someone dismissed
    await redis.hset(key, f"s:{D3}", json.dumps({"status": "dismissed", "text": None}))
    await redis.hset(key, f"s:{D2}", json.dumps({"status": "edited", "text": "Write notes v2"}))
    await redis.hset(
        key, f"q:{ALICE}:{D1}", json.dumps({"status": "dismissed", "text": "Who owns QA?"})
    )
    entries, marks = await ledger.load(ORG, MEETING)
    by_id = {e.id: e for e in entries}
    assert by_id[D1].status == "open" and by_id[D1].moment_id == "m1"
    assert by_id[D2].status == "edited" and by_id[D2].text == "Write notes v2"
    assert by_id[D3].status == "dismissed"
    assert marks == [SuggestionMark(str(ALICE), D1, "Who owns QA?", "dismissed")]


async def test_a_broken_redis_means_no_memory_never_no_window():
    class Broken:
        async def hgetall(self, _key: str) -> Any:
            raise ConnectionError("down")

        async def hset(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError("down")

    ledger = MeetingLedger(Broken())
    assert await ledger.load(ORG, MEETING) == ([], [])
    await ledger.record(ORG, MEETING, [LedgerEntry(D1, "decision", "x")])  # no raise


def test_the_memory_block_lists_this_kinds_cards_with_labels():
    mem = memory(
        LedgerEntry(D1, "decision", "Ship the beta on 2 December", seq=1),
        LedgerEntry(D2, "action", "Write the release notes", seq=2),
        LedgerEntry(D3, "decision", "Use Mongo", status="dismissed", seq=3),
        previous=["Participant 1: we were talking about the beta"],
    )
    block, labels = render_memory(mem, "decision")
    assert "<earlier>\nParticipant 1: we were talking about the beta\n</earlier>" in block
    assert "D1. [open] Ship the beta on 2 December" in block
    assert "D2. [dismissed] Use Mongo" in block
    assert "release notes" not in block
    assert labels == {"D1": D1, "D2": D3}
    assert render_memory(MeetingMemory(), "action") == ("", {})


def _cand(content: str, cid: str = "new-id") -> dict[str, Any]:
    return {"id": cid, "content": content, "confidence": 0.9, "surfaced": True}


def test_reconcile_updates_merges_and_never_reproposes():
    mem = memory(
        LedgerEntry(D1, "decision", "Ship the beta on 2 December", moment_id="m-old"),
        LedgerEntry(D2, "decision", "Use Mongo for the ledger service", status="dismissed"),
        LedgerEntry(D3, "decision", "Price the pro plan at 49 dollars", status="confirmed"),
    )
    _, labels = render_memory(mem, "decision")
    out = reconcile(
        [
            _cand("Ship the beta on 9 December", "a"),  # explicit update of D1
            _cand("Use Mongo for the ledger service", "b"),  # a dismissed card again
            _cand("Price the pro plan at 49 dollars a month", "c"),  # a settled card again
            _cand("Hire a second designer", "d"),  # new
        ],
        "decision",
        mem,
        labels,
        ["D1", None, None, None],
    )
    assert [c["id"] for c in out] == [D1, "d"]
    assert out[0]["updates_candidate_id"] == D1 and out[0]["moment_id"] == "m-old"
    assert "updates_candidate_id" not in out[1]


def test_a_near_duplicate_of_an_open_card_is_an_update_and_the_same_card_is_silence():
    mem = memory(LedgerEntry(D1, "action", "Send the pricing deck to the client"))
    out = reconcile(
        [_cand("Send the pricing deck to the client by Friday")], "action", mem, {}, [None]
    )
    assert out[0]["id"] == D1 and out[0]["updates_candidate_id"] == D1
    same = reconcile([_cand("Send the pricing deck to the client")], "action", mem, {}, [None])
    assert same == []
    # an update of a dismissed card, named by label, is dropped
    gone = memory(LedgerEntry(D1, "action", "Send the deck", status="dismissed"))
    _, labels = render_memory(gone, "action")
    assert reconcile([_cand("Send the deck today")], "action", gone, labels, ["A1"]) == []


def test_a_dismissed_or_asked_suggestion_does_not_return_for_that_person():
    mem = MeetingMemory(
        marks=[SuggestionMark(str(ALICE), "s1", "Who owns the QA sign-off?", "dismissed")]
    )
    items = [{"question": "Who owns the QA sign off?"}, {"question": "What is the budget?"}]
    assert drop_repeated_suggestions(items, mem, ALICE) == [items[1]]
    assert drop_repeated_suggestions(items, mem, BOB) == items


# --- 2. the candidate agents with memory ---------------------------------------------------


async def test_the_decision_agent_sees_the_memory_and_answers_an_update():
    llm = ScriptedLLM(
        0.9,
        decision=cands(
            {
                "content": "Ship the beta on 9 December",
                "rationale": "QA needs a week",
                "stakeholders": ["Participant 1"],
                "quote": "let's ship the beta on the 2nd",
                "updates": "D1",
                "confidence": 0.9,
            }
        ),
    )
    router = LiteLLMRouter(gate(), Sink(), llm)
    agent = make(DecisionAgent, router)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    from tests.test_decision_action_agents import window

    mem = memory(
        LedgerEntry(D1, "decision", "Ship the beta on 2 December", moment_id="m1"),
        previous=["Participant 2: the beta date is still open"],
    )
    state = AgentState(
        meeting_id=MEETING,
        user_id=ALICE,
        chunk=window(),
        context={CONTEXT_KEY: mem},
        org_id=ORG,
    )
    result = await agent.process(state)
    assert result is not None
    sent = llm.calls[-1]["messages"][1]["content"]
    text = " ".join(part["text"] for part in sent) if isinstance(sent, list) else str(sent)
    assert "D1. [open] Ship the beta on 2 December" in text
    assert "the beta date is still open" in text and "New transcript stretch:" in text
    (cand,) = result.payload["candidates"]
    assert cand["id"] == D1 and cand["updates_candidate_id"] == D1
    assert cand["quote"] == "let's ship the beta on the 2nd" and cand["moment_id"] == "m1"
    assert result.payload["prompt"] == "decision_agent@v3"


# --- 3. moments -------------------------------------------------------------------------------


def test_a_promise_and_the_action_quoting_it_share_one_moment():
    w = moment_window()
    note = AgentResult(
        kind="note",
        content="Participant 2 will send the pricing deck by Friday.",
        relevance_score=0.9,
        confidence=0.9,
        source_chunk_ids=["c1", "c2"],
        payload={"commitments": [{"content": "Send the pricing deck", "due_hint": "Friday"}]},
    )
    annotate(note, w, BOB, {"role": "Sales"})
    action = AgentResult(
        kind="action_candidate",
        content="Send the pricing deck to the client",
        relevance_score=0.9,
        confidence=0.9,
        source_chunk_ids=["c1", "c2"],
        payload={
            "candidates": [
                {
                    "id": "x",
                    "content": "Send the pricing deck to the client",
                    "assignee_id": str(BOB),
                    "quote": "I'll send the pricing deck to the client by Friday",
                    "confidence": 0.9,
                }
            ]
        },
    )
    annotate(action, w, MEETING, {})
    expected = moment_id(MEETING, "c2")
    cand = action.payload["candidates"][0]
    assert note.payload["moment_id"] == expected
    assert note.payload["commitments"][0]["moment_id"] == expected
    assert cand["moment_id"] == expected
    assert cand["start_ms"] == 3_000 and cand["speaker_id"] == str(BOB)
    assert cand["speaker_label"] == "Participant 2"
    assert cand["why"] == "Participant 2 took it on"
    assert note.payload["why"] == "You made a promise here"


def test_why_lines_name_ownership_listen_for_and_role():
    w = moment_window()
    owners = CompanyContext(topic_owners=(TopicOwner("pricing deck", BOB),))

    def why(user: uuid.UUID, context: dict[str, Any]) -> str | None:
        r = AgentResult("note", "The client gets the pricing deck.", 0.9, 0.9, [], {})
        return annotate(r, w, user, context).payload.get("why")

    assert why(BOB, {"org": owners}) == "You own pricing deck"
    assert why(ALICE, {"listen_for": ["client"]}) == "Matches your listen-for topic: client"
    assert why(ALICE, {"role": "Designer"}) == "Relevant to your role: Designer"


def test_a_quote_the_model_invented_falls_back_to_the_closest_words():
    w = moment_window()
    u, quote = anchor(w, "Send the pricing deck", quote="I will email everyone the slides")
    assert u is not None and u.chunk_id == "c2" and quote is None
    u, _ = anchor(w, "nothing in common here")
    assert u is not None and u.chunk_id == "c2"  # the trigger


def test_names_and_never_record_apply_to_the_moment_fields():
    w = moment_window()
    note = AgentResult("note", "Participant 2 shares pricing.", 0.9, 0.9, [], {"commitments": []})
    annotate(note, w, ALICE, {"role": "PM"})
    post = Postprocess(
        Namer({"Participant 2": "Bob Jones"}),
        NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("pricing deck",))),
    )
    out = post.result(note)
    assert out is not None and "quote" not in out.payload and out.payload["moment_id"]
    cand = AgentResult(
        "action_candidate",
        "x",
        0.9,
        0.9,
        [],
        {"candidates": [{"id": "1", "content": "Book the venue", "confidence": 0.9}]},
    )
    annotate(cand, w, ALICE, {})
    named = Postprocess(Namer({"Participant 2": "Bob Jones"}), NeverRecordPolicy.from_org(None))
    got = named.result(cand)
    assert got is not None and got.payload["candidates"][0]["speaker_label"] in (
        "Bob Jones",
        "Participant 1",
    )


# --- 4. the runtime -------------------------------------------------------------------------


class _Probe:
    """A scripted agent: records its contexts, answers a fixed result."""

    name = "DecisionAgent"
    scope = "meeting"
    blocked = False
    seen: list[dict[str, Any]] = []
    answer: AgentResult | None = None

    async def spawn(self, **_kw: Any) -> None:
        return None

    async def terminate(self) -> None:
        return None

    async def process(self, state: AgentState) -> AgentResult | None:
        type(self).seen.append({"context": state.context, "window": state.chunk})
        return type(self).answer


def _probe(name: str, scope: str, answer: AgentResult | None, span: int = 0) -> type[_Probe]:
    return type(
        f"Probe{name}",
        (_Probe,),
        {"name": name, "scope": scope, "answer": answer, "seen": [], "min_span_ms": span},
    )


def _win(start: int, text: str, speaker: uuid.UUID = ALICE, cue: str | None = None) -> Any:
    u = utt(f"c{start}", speaker, start, text)
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:{start}",
        start_ms=start,
        end_ms=start + 3_000,
        turns=[Turn(u.speaker_id, u.text, u.start_ms, u.end_ms, [u.chunk_id], [u])],
        trigger=Trigger(
            "cue" if cue else "turn_end", u.chunk_id, u.speaker_id, start, 0, text, cue
        ),
    )


async def test_memory_reaches_the_agents_and_emitted_cards_land_in_the_ledger():
    redis = fakeredis.aioredis.FakeRedis()
    decision = AgentResult(
        "decision_candidate",
        "Ship on Friday",
        0.9,
        0.9,
        [],
        {"candidates": [{"id": D1, "content": "Ship on Friday", "confidence": 0.9}]},
    )
    probe = _probe("DecisionAgent", "meeting", decision)
    sink = Sink()

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {"role": "PM"}

    rt = AgentRuntime(
        sink=sink, user_context=ctx, agents=[probe], ledger=MeetingLedger(redis)  # type: ignore[list-item]
    )
    await rt.on_room_event(ORG, MEETING, "joined", str(ALICE))
    await rt.process_window(_win(0, "let's ship on Friday", cue="let's"))
    await rt.process_window(_win(5_000, "sounds good"))
    first, second = probe.seen
    assert first["context"][CONTEXT_KEY].previous == []
    mem = second["context"][CONTEXT_KEY]
    assert mem.previous == ["Participant 1: let's ship on Friday"]
    assert [(e.id, e.text, e.status) for e in mem.entries] == [(D1, "Ship on Friday", "open")]
    cand = sink.results[0]["result"].payload["candidates"][0]
    assert cand["moment_id"] == moment_id(MEETING, "c0") and cand["quote"]


async def test_a_dismissed_suggestion_never_reaches_the_sink():
    redis = fakeredis.aioredis.FakeRedis()
    await redis.hset(
        ledger_key(ORG, MEETING),
        f"q:{ALICE}:s1",
        json.dumps({"status": "dismissed", "text": "Who signs off QA?"}),
    )
    suggestion = AgentResult(
        "suggestion",
        "Who signs off QA?",
        0.9,
        0.9,
        [],
        {"suggestions": [{"id": "s2", "question": "Who signs off on QA?", "confidence": 0.9}]},
    )
    probe = _probe("QuestionAgent", "participant", suggestion)
    sink = Sink()

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {"role": "PM"}

    rt = AgentRuntime(
        sink=sink, user_context=ctx, agents=[probe], ledger=MeetingLedger(redis)  # type: ignore[list-item]
    )
    await rt.on_room_event(ORG, MEETING, "joined", str(ALICE))
    assert await rt.process_window(_win(0, "QA is the open item")) == 0
    assert sink.results == []


async def test_notes_accumulate_short_windows_until_a_span_or_a_cue():
    note = AgentResult("note", "A note.", 0.9, 0.9, [], {})
    probe = _probe("PersonalListenerAgent", "participant", note, span=8_000)
    sink = Sink()

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {"role": "PM"}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[probe])  # type: ignore[list-item]
    await rt.on_room_event(ORG, MEETING, "joined", str(ALICE))
    assert await rt.process_window(_win(0, "one")) == 0
    assert await rt.process_window(_win(3_000, "two")) == 0
    assert await rt.process_window(_win(9_000, "three")) == 1  # 12 s collected
    assert [u.text for u in probe.seen[0]["window"].utterances] == ["one", "two", "three"]
    assert await rt.process_window(_win(12_000, "I'll do it", cue="i'll")) == 1  # a cue: now
    last = _win(20_000, "bye")
    last.final = True
    assert await rt.process_window(last) == 1  # the meeting's end flushes


# --- 5. the emitter ---------------------------------------------------------------------------


class _Codec:
    def __init__(self) -> None:
        self.values: list[dict[str, Any]] = []

    def serialize_for(self, _topic: str, value: dict[str, Any]) -> bytes:
        self.values.append(value)
        return b"x"


class _Producer:
    def produce(self, *_a: Any, **_k: Any) -> None:
        return None

    def poll(self, _t: float) -> int:
        return 0

    def flush(self, _t: float) -> int:
        return 0


async def test_the_record_carries_the_moment_and_the_latency_is_observed():
    codec = _Codec()
    emitter = Emitter(_Producer(), codec)
    w = moment_window()
    w.last_chunk_at = time.time() - 2.0
    result = AgentResult(
        "action_candidate",
        "Send the deck",
        0.9,
        0.9,
        ["c2"],
        {
            "candidates": [
                {
                    "id": D1,
                    "content": "Send the deck",
                    "confidence": 0.9,
                    "moment_id": "m",
                    "why": "Bob took it on",
                    "updates_candidate_id": D1,
                }
            ]
        },
    )
    before = AGENT_TIME_TO_CARD.labels("ActionAgent", str(ORG))._sum.get()  # type: ignore[attr-defined]
    await emitter.result(window=w, user_id=MEETING, agent="ActionAgent", result=result)
    (value,) = codec.values
    assert value["moment_id"] == "m" and value["why"] == "Bob took it on"
    assert value["updates_candidate_id"] == D1
    assert (value["window_start_ms"], value["window_end_ms"]) == (0, 6_000)
    after = AGENT_TIME_TO_CARD.labels("ActionAgent", str(ORG))._sum.get()  # type: ignore[attr-defined]
    assert 1.5 < after - before < 10
    assert top_item(AgentResult("alert", "x", 0, 0, [], {})) == {}


# --- 6. one task per meeting ----------------------------------------------------------------


class _SlowRuntime:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.done: list[tuple[uuid.UUID, str, int]] = []

    async def process_window(self, window: TranscriptWindow) -> int:
        if window.meeting_id == MEETING:
            await self.release.wait()
        self.done.append((window.meeting_id, window.window_id, len(window.utterances)))
        return 0

    async def on_room_event(self, *_a: Any) -> None:
        return None

    def roster(self, *_a: Any) -> set[uuid.UUID]:
        return {ALICE}

    async def end_meeting(self, *_a: Any) -> None:
        return None


class _Flush:
    def flush(self) -> None:
        return None


def _value(meeting: uuid.UUID, start: int, text: str) -> dict[str, Any]:
    return {
        "header": {"org_id": str(ORG)},
        "meeting_id": str(meeting),
        "chunk_id": str(uuid.uuid4()),
        "speaker_id": str(ALICE),
        "start_ms": start,
        "end_ms": start + 1_000,
        "text": text,
        "is_final": True,
    }


async def test_a_slow_meeting_never_holds_up_another_and_catches_up_in_one_pass():
    rt = _SlowRuntime()
    loop = Loop(rt, _Flush(), Windower())  # type: ignore[arg-type]
    other = uuid.uuid4()
    await loop.handle("transcript-chunks", 0, 1, _value(MEETING, 0, "let's start"))
    await asyncio.sleep(0.01)  # MEETING's task is now stuck on its model call
    await loop.handle("transcript-chunks", 0, 2, _value(MEETING, 5_000, "I'll write it"))
    await loop.handle("transcript-chunks", 0, 3, _value(MEETING, 10_000, "we decided"))
    await loop.handle("transcript-chunks", 1, 1, _value(other, 0, "let's go"))
    for _ in range(50):
        await asyncio.sleep(0.001)
    assert [d[0] for d in rt.done] == [other]  # the other meeting did not wait
    assert loop.offsets.committable()[("transcript-chunks", 0)] == 1  # nothing past MEETING
    rt.release.set()
    await loop.drain()
    mine = [d for d in rt.done if d[0] == MEETING]
    assert mine == [(MEETING, f"{MEETING}:0", 1), (MEETING, f"{MEETING}:5000", 2)]
    assert loop.offsets.committable()[("transcript-chunks", 0)] == 4
    await loop.close()


def test_a_chunk_carries_its_production_time():
    from services.agent_runtime_service.consumer import parse_chunk

    value = _value(MEETING, 0, "hi")
    value["header"]["occurred_at"] = 1_700_000_000_000_000
    chunk = parse_chunk(value, 0, 0)
    assert isinstance(chunk, Chunk) and chunk.produced_at == pytest.approx(1_700_000_000.0)


# --- 7. the v3 prompts ------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["decision_agent", "action_agent"])
def test_v3_with_no_memory_renders_the_v2_message_with_the_new_heading(key: str):
    loader = PromptLoader(PROMPTS_DIR)
    values = {
        "window": "Participant 1: hi",
        "company": "",
        "ground_rules": "",
        "owners": "",
        "today": "Monday 2026-11-09",
        "memory": "",
    }
    v2 = loader.load(key, "v2").render_user(**values)
    v3 = loader.load(key, "v3").render_user(**values)
    assert v3 == v2.replace("Transcript (30 seconds):", "New transcript stretch:")
    system = loader.load(key, "v3").system
    assert '"updates"' in system and '"quote"' in system and "DISMISSED" in system
    assert "{" not in loader.load(key, "v3").render_user(
        **{**values, "memory": render_memory(memory(LedgerEntry(D1, key[0:6], "x")), "action")[0]}
    )


async def test_the_action_agent_carries_quote_and_update_through():
    llm = ScriptedLLM(
        0.9,
        action=cands(
            {
                "content": "Send the pricing deck to the client by Friday",
                "assignee": "Participant 2",
                "due_date": "2026-11-13",
                "depends_on": [],
                "quote": "I'll write the release notes by Friday",
                "updates": None,
                "confidence": 0.9,
            }
        ),
    )
    router = LiteLLMRouter(gate(), Sink(), llm)
    agent = make(ActionAgent, router)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    from tests.test_decision_action_agents import window

    mem = memory(LedgerEntry(D2, "action", "Send the pricing deck to the client"))
    result = await agent.process(
        AgentState(
            meeting_id=MEETING,
            user_id=ALICE,
            chunk=window(),
            context={CONTEXT_KEY: mem},
            org_id=ORG,
        )
    )
    assert result is not None
    (cand,) = result.payload["candidates"]
    assert cand["id"] == D2 and cand["updates_candidate_id"] == D2  # merged, not duplicated
    assert cand["quote"] == "I'll write the release notes by Friday"
