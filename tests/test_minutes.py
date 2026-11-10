"""Minutes for everyone (`services/agent_runtime_service/minutes.py`): every model call and the
core-api route faked, no paid call.

1. One call per participant plus one shared call, extract tier, `minutes@v1`, budget-gated and
   metered; the meeting record is the cached prefix shared by every call.
2. The model sees labels only; the answer's labels become user ids (owners, askers) and display
   names; refs become decision ids; the shared minutes never carry promises.
3. Never record: the window naming a topic and the window after it never reach the model, and
   an answer naming one is dropped before it leaves the tier.
4. No transcript, budget blocked, a model failure: a shared `unavailable` record only.
5. The loop: `ended` only, one generation per event, records on `agent-results` with the
   canonical key shape, flushed, claimed in Redis so a replay bills nothing; retries, then an
   `unavailable` record so nothing waits forever.
6. The source client speaks the caller-token scheme; the MeetingAgent carries the minutes'
   open items forward; the record encodes against the `agent-results` schema with `minutes`.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import fastavro
import httpx
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.org_context import CompanyContext, OrgContextUnavailable
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agenda import parse_agenda
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.minutes import (
    NIL,
    HttpMinutesSource,
    MinutesLoop,
    MinutesSource,
    MinutesWriter,
    SourceUnavailable,
    claim_key,
    generation_of,
    parse_source,
    quiet_turns,
)

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
HOST, ALICE, BOB = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
D1 = uuid.uuid4()

SHARED = {
    "summary": "Participant 1 and Participant 2 agreed the price rise.",
    "decisions": [{"text": "Raise the price by 8%", "status": "agreed", "ref": "D1"}],
    "open_items": [{"text": "Confirm the vendor split", "owner": "Participant 2", "due": "Friday"}],
    "unanswered_questions": [{"text": "Does legal review this?", "asked_by": "Participant 1"}],
    "promises": [{"text": "Should never appear", "to": None, "due": None}],
}
PERSONAL = {
    **SHARED,
    "promises": [{"text": "Send the comparison to Participant 1", "to": "Participant 1"}],
}


def source_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "meeting_id": str(MEETING),
        "title": "Pricing review",
        "status": "ended",
        "confidential": False,
        "host_user_id": str(HOST),
        "participants": [
            {
                "user_id": str(HOST),
                "name": "Hana Host",
                "attendance": "live",
                "role_in_meeting": "host",
            },
            {
                "user_id": str(ALICE),
                "name": "Alice Doe",
                "attendance": "live",
                "role_in_meeting": "attendee",
            },
            {
                "user_id": str(BOB),
                "name": "Bob Roe",
                "attendance": "agent",
                "role_in_meeting": "attendee",
            },
        ],
        "speakers": {},
        "turns": [
            {
                "speaker_id": str(HOST),
                "start_ms": 0,
                "end_ms": 4000,
                "text": "We should raise the price by 8% hana@acme.com",
            },
            {
                "speaker_id": str(ALICE),
                "start_ms": 5000,
                "end_ms": 9000,
                "text": "Agreed. I will confirm the vendor split by Friday.",
            },
        ],
        "turns_truncated": False,
        "decisions": [{"id": str(D1), "content": "Raise the price by 8%", "status": "proposed"}],
        "actions": [],
        "agenda": [{"title": "Pricing", "status": "covered"}],
    }
    body.update(over)
    return body


class FakeSource:
    def __init__(self, value: MinutesSource | None | Exception) -> None:
        self.value = value
        self.calls = 0

    async def source(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> MinutesSource | None:
        assert org_id == ORG and meeting_id == MEETING
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class LLM:
    """Answers the shared call (whose audience says "everyone") and personal calls apart."""

    def __init__(self, shared: Any = SHARED, personal: Any = PERSONAL) -> None:
        self.shared, self.personal = shared, personal
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        user = self.text(kw)
        answer = self.shared if "everyone in the meeting" in user else self.personal
        if isinstance(answer, Exception):
            raise answer
        return {
            "choices": [
                {"message": {"content": answer if isinstance(answer, str) else json.dumps(answer)}}
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120},
        }

    @staticmethod
    def text(kw: dict[str, Any]) -> str:
        content = kw["messages"][1]["content"]
        if isinstance(content, list):
            return "".join(part["text"] for part in content)
        return str(content)


class Usage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.calls.append(kw)


def budget(*, cap: int = 10**7, consumed: int = 0) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": 10**7,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


def writer(
    llm: LLM,
    source: Any = None,
    *,
    org: CompanyContext | None = None,
    gate: BudgetGate | None = None,
    usage: Usage | None = None,
) -> MinutesWriter:
    async def org_context(_org: uuid.UUID) -> CompanyContext:
        return org or CompanyContext()

    router = LiteLLMRouter(gate or budget(), usage or Usage(), completion_fn=llm)
    return MinutesWriter(
        PromptLoader(PROMPTS_DIR),
        router,
        source=source if source is not None else FakeSource(parse_source(source_body())),
        org_context=org_context,
    )


def by_user(written: list[Any]) -> dict[Any, dict[str, Any]]:
    return {w.user_id: w.payload for w in written}


# --- 1, 2: calls and content ------------------------------------------------------------------


async def test_one_call_per_participant_plus_the_shared_call():
    llm, usage = LLM(), Usage()
    written = await writer(llm, usage=usage).write(ORG, MEETING, "gen1")
    assert len(llm.calls) == 4  # shared + 3 participants (Bob sent his agent: still minutes)
    assert {c["model"] for c in llm.calls} == {"anthropic/claude-sonnet-5-5"}
    assert all(c["output_config"] == {"effort": "low"} for c in llm.calls)
    assert {u["model_tier"] for u in usage.calls} == {"sonnet"}
    assert {u["user_id"] for u in usage.calls} == {None, HOST, ALICE, BOB}
    # The meeting record is the cached prefix of every call, identical across them.
    prefixes = {c["messages"][1]["content"][0]["text"] for c in llm.calls}
    assert len(prefixes) == 1
    prefix = prefixes.pop()
    assert "Transcript:" in prefix and "D1. Raise the price by 8% [proposed]" in prefix
    assert "Hana" not in prefix and str(HOST) not in prefix  # labels only, never names or ids
    assert "Participant 3 (sent their agent instead of attending)" in prefix
    assert "[REDACTED_EMAIL]" in prefix  # the router redacts PII before it leaves

    out = by_user(written)
    assert set(out) == {None, HOST, ALICE, BOB}
    shared, mine = out[None], out[ALICE]
    assert shared["scope"] == "shared" and shared["promises"] == []
    assert shared["prompt_version"] == "minutes@v1" and shared["generation"] == "gen1"
    assert shared["status"] == "ready" and shared["reason"] is None
    assert shared["summary"] == "Hana Host and Alice Doe agreed the price rise."
    assert shared["decisions"] == [
        {"text": "Raise the price by 8%", "status": "agreed", "decision_id": str(D1)}
    ]
    assert shared["open_items"] == [
        {
            "text": "Confirm the vendor split",
            "owner_user_id": str(ALICE),
            "owner_name": "Alice Doe",
            "due": "Friday",
        }
    ]
    assert shared["unanswered_questions"] == [
        {
            "text": "Does legal review this?",
            "asked_by_user_id": str(HOST),
            "asked_by_name": "Hana Host",
        }
    ]
    assert mine["scope"] == "personal"
    assert mine["promises"] == [
        {"text": "Send the comparison to Hana Host", "to_name": "Hana Host", "due": None}
    ]


async def test_the_audience_names_the_participant_by_label():
    llm = LLM()
    await writer(llm).write(ORG, MEETING, "g")
    tails = sorted(c["messages"][1]["content"][1]["text"] for c in llm.calls)
    assert any("for: everyone in the meeting" in t for t in tails)
    assert any("for: Participant 2 (their personal minutes)" in t for t in tails)


async def test_an_empty_answer_is_empty_and_a_bad_one_unavailable():
    nothing = {
        "summary": "",
        "decisions": [],
        "open_items": [],
        "unanswered_questions": [],
        "promises": [],
    }
    out = by_user(await writer(LLM(nothing, nothing)).write(ORG, MEETING, "g"))
    assert out[None]["status"] == "empty" and out[ALICE]["status"] == "empty"
    out = by_user(await writer(LLM("not json", "not json")).write(ORG, MEETING, "g"))
    assert list(out) == [None]
    assert (out[None]["status"], out[None]["reason"]) == ("unavailable", "model_error")


# --- 3: never record --------------------------------------------------------------------------


async def test_never_record_windows_never_reach_the_model_and_answers_are_filtered():
    org = CompanyContext(never_record_topics=("layoffs",))
    body = source_body(
        turns=[
            {"speaker_id": str(HOST), "start_ms": 0, "end_ms": 4000, "text": "Price first."},
            {
                "speaker_id": str(ALICE),
                "start_ms": 31_000,
                "end_ms": 35_000,
                "text": "About the layoffs next quarter",
            },
            {
                "speaker_id": str(HOST),
                "start_ms": 61_000,
                "end_ms": 65_000,
                "text": "Those numbers are grim",
            },
            {
                "speaker_id": str(ALICE),
                "start_ms": 95_000,
                "end_ms": 99_000,
                "text": "Back to vendors.",
            },
        ]
    )
    leaky = {**SHARED, "open_items": [{"text": "Plan the layoff list", "owner": None}]}
    llm = LLM(leaky, leaky)
    out = by_user(
        await writer(llm, FakeSource(parse_source(body)), org=org).write(ORG, MEETING, "g")
    )
    prompt = LLM.text(llm.calls[0])
    assert "layoffs next quarter" not in prompt and "grim" not in prompt
    assert "Price first." in prompt and "Back to vendors." in prompt
    assert "Never record (company ground rule): layoffs" in prompt
    assert out[None]["open_items"] == []


def test_quiet_turns_without_topics_keeps_everything():
    from libs.shared.never_record import NeverRecordPolicy

    src = parse_source(source_body())
    assert src is not None
    assert quiet_turns(src.turns, NeverRecordPolicy()) == src.turns


# --- 4: no transcript, budget, failure --------------------------------------------------------


async def test_no_transcript_is_one_unavailable_shared_record_and_no_call():
    llm = LLM()
    written = await writer(llm, FakeSource(parse_source(source_body(turns=[])))).write(
        ORG, MEETING, "g"
    )
    assert llm.calls == []
    assert [(w.user_id, w.payload["status"], w.payload["reason"]) for w in written] == [
        (None, "unavailable", "no_transcript")
    ]


async def test_budget_blocked_is_unavailable_and_calls_nothing():
    llm = LLM()
    written = await writer(llm, gate=budget(cap=10, consumed=11)).write(ORG, MEETING, "g")
    assert llm.calls == []
    assert [(w.user_id, w.payload["reason"]) for w in written] == [(None, "budget")]


async def test_a_failed_personal_call_is_skipped():
    llm = LLM(SHARED, RuntimeError("provider down"))
    out = by_user(await writer(llm).write(ORG, MEETING, "g"))
    assert list(out) == [None] and out[None]["status"] == "ready"


# --- 5: the loop ------------------------------------------------------------------------------


class Sink:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.flushes = 0

    async def agent_result(self, **kw: Any) -> None:
        self.results.append(kw)

    def flush(self, timeout_s: float = 10.0) -> None:
        self.flushes += 1


def ended(event_type: str = "ended", key: str = "meeting:x:ended:t1") -> dict[str, Any]:
    return {
        "header": {"org_id": str(ORG), "idempotency_key": key},
        "meeting_id": str(MEETING),
        "event_type": event_type,
        "livekit_room": None,
    }


async def _no_sleep(_s: float) -> None:
    return None


async def test_loop_emits_flushes_claims_and_skips_a_replay():
    llm, sink, redis = LLM(), Sink(), fakeredis.aioredis.FakeRedis()
    loop = MinutesLoop(writer(llm), sink, claims=redis, sleep=_no_sleep)
    assert await loop.handle(ended("updated")) == 0
    assert await loop.handle(ended()) == 4
    gen = generation_of("meeting:x:ended:t1", MEETING)
    assert sink.flushes == 1
    assert {r["window_id"] for r in sink.results} == {f"minutes:{gen}"}
    assert {(r["agent"], r["result"].kind) for r in sink.results} == {("MeetingAgent", "minutes")}
    assert {r["user_id"] for r in sink.results} == {NIL, HOST, ALICE, BOB}
    assert await redis.exists(claim_key(ORG, MEETING, gen))
    calls = len(llm.calls)
    assert await loop.handle(ended()) == 0 and len(llm.calls) == calls
    # The meeting ending again is a new generation.
    assert await loop.handle(ended(key="meeting:x:ended:t2")) == 4


async def test_loop_retries_then_releases_waiters_with_unavailable():
    sink = Sink()
    flaky = FakeSource(SourceUnavailable("core-api down"))
    loop = MinutesLoop(writer(LLM(), flaky), sink, sleep=_no_sleep, attempts=3)
    assert await loop.handle(ended()) == 1
    assert flaky.calls == 3
    [only] = sink.results
    assert only["user_id"] == NIL
    assert only["result"].payload["status"] == "unavailable"

    async def no_brain(_org: uuid.UUID) -> CompanyContext:
        raise OrgContextUnavailable(str(ORG))

    w = writer(LLM())
    w.org_context = no_brain
    sink2 = Sink()
    assert await MinutesLoop(w, sink2, sleep=_no_sleep, attempts=2).handle(ended()) == 1
    assert sink2.results[0]["result"].payload["reason"] == "model_error"


async def test_an_unknown_meeting_writes_nothing():
    sink = Sink()
    loop = MinutesLoop(writer(LLM(), FakeSource(None)), sink, sleep=_no_sleep)
    assert await loop.handle(ended()) == 0 and sink.flushes == 0


# --- 6: wiring --------------------------------------------------------------------------------


async def test_source_client_speaks_the_caller_token_scheme():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith(f"{MEETING}/minutes-source"):
            return httpx.Response(200, json=source_body())
        return httpx.Response(404)

    client = HttpMinutesSource("http://core", "tok", transport=httpx.MockTransport(handler))
    got = await client.source(ORG, MEETING)
    assert got is not None and got.title == "Pricing review" and len(got.turns) == 2
    h = seen[0].headers
    assert (h["X-Mytheus-Service"], h["X-Mytheus-Service-Token"], h["X-Mytheus-Org"]) == (
        "agent-runtime-service",
        "tok",
        str(ORG),
    )
    assert await client.source(ORG, uuid.uuid4()) is None

    def busy(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(409)

    with pytest.raises(SourceUnavailable):
        await HttpMinutesSource("http://core", "t", transport=httpx.MockTransport(busy)).source(
            ORG, MEETING
        )


def test_meeting_agent_carries_minutes_open_items_forward():
    agenda = parse_agenda(
        {
            "host_user_id": str(HOST),
            "items": [],
            "carry_forward": {
                "decisions": [],
                "open_actions": ["Send the contract"],
                "open_items": ["send the contract", "Confirm the vendor split (owner: Alice)"],
                "questions": ["Does legal review this?"],
            },
        }
    )
    assert agenda is not None
    assert agenda.open_actions == ["Send the contract", "Confirm the vendor split (owner: Alice)"]
    assert agenda.questions == ["Does legal review this?"]


async def test_record_encodes_against_agent_results_with_the_minutes_kind():
    events = Path(__file__).resolve().parents[2] / "p2-minutes-events"
    schema_path = events / "schemas" / "agent-results" / "v1.avsc"
    if not schema_path.exists():
        pytest.skip("mytheus-events worktree not beside this one")
    named: dict[str, Any] = {}
    fastavro.parse_schema(
        json.loads((events / "schemas" / "_envelope" / "EventEnvelopeHeader.avsc").read_text()),
        named_schemas=named,
    )
    schema = fastavro.parse_schema(json.loads(schema_path.read_text()), named_schemas=named)

    class Codec:
        def serialize_for(self, _topic: str, value: dict[str, Any]) -> bytes:
            buf = io.BytesIO()
            fastavro.schemaless_writer(buf, schema, value)
            return buf.getvalue()

    class Producer:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def produce(self, _topic: str, key: str, value: bytes, on_delivery: Any) -> None:
            self.sent.append(value)

        def poll(self, _t: float) -> None:
            return None

        def flush(self, _t: float) -> int:
            return 0

    producer = Producer()
    loop = MinutesLoop(writer(LLM()), Emitter(producer, Codec()), sleep=_no_sleep)
    assert await loop.handle(ended()) == 4
    back = fastavro.schemaless_reader(io.BytesIO(producer.sent[0]), schema, schema)
    assert (back["agent"], back["kind"]) == ("MeetingAgent", "minutes")
    assert back["header"]["idempotency_key"] == (
        f"agentresult:{MEETING}:{back['window_id']}:{back['user_id']}:MeetingAgent"
    )
