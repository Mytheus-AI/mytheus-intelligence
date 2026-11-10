"""`MeetingAgent` (mytheus-docs ADR-0038 D7, D8): every model call and core-api route faked.

1. Skips without an agenda, without agenda items, without a host, and when the internal agenda
   route fails in any way (quietly, before any model call).
2. The agenda route speaks the caller-token scheme of `a2a/related.py`; the agenda is cached per
   meeting for a short TTL and a failure is not cached.
3. `over_time` is computed (no model), once per item, after `duration_minutes`.
4. At most one nudge per meeting per 5 minutes; `off_agenda` needs confidence >= 0.75;
   `unaddressed` names a carried-forward item nobody raised, after 70% of the scheduled time.
5. Updates: surfaced band only, at most one per item, the host's status wins, never backwards.
6. One record per window: kind `meeting_update`, user_id = the host, payload exactly D7's.
7. Budget: blocked -> no model call, nothing emitted; every call metered to the host on Haiku.
8. Runtime wiring, and Avro: the record encodes against an `agent-results` schema carrying D8's
   symbols, and a registry without them costs the record, never the window.
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Any

import fakeredis
import fastavro
import httpx
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentResult, TranscriptWindow, Turn
from services.agent_runtime_service.agenda import (
    Agenda,
    AgendaItem,
    HttpAgendaClient,
    agenda_from_env,
    parse_agenda,
)
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.meeting_agent import (
    MeetingAgent,
    MeetingDeps,
    MeetingSettings,
    carried_key,
    nudge_id,
    state_key,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.runtime import AgentRuntime

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
HOST, ALICE, BOB = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
ITEM_A, ITEM_B, ITEM_C = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
MIN = 60_000

NOTHING = json.dumps({"updates": [], "off_agenda": None, "raised": []})


def agenda(**overrides: Any) -> Agenda:
    base: dict[str, Any] = {
        "host_user_id": HOST,
        "items": [
            AgendaItem(ITEM_A, 0, "Budget review", 10, "pending"),
            AgendaItem(ITEM_B, 1, "Hiring plan", 1, "in_progress", "agent"),
            AgendaItem(ITEM_C, 2, "Launch date", None, "pending"),
        ],
        "open_actions": ["Send the vendor contract", "Fix the login bug"],
        "scheduled_minutes": 30.0,
    }
    base.update(overrides)
    return Agenda(**base)


class FakeAgenda:
    def __init__(self, value: Agenda | None | Exception) -> None:
        self.value = value
        self.calls = 0

    async def agenda(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> Agenda | None:
        assert org_id == ORG and meeting_id == MEETING
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class ScriptedLLM:
    """Answers each call with the next scripted text (the last one repeats)."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or [NOTHING]
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40},
        }

    def user(self, i: int = -1) -> str:
        return str(self.calls[i]["messages"][1]["content"])


class Sink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        self.notifications.append(kw)

    async def delta(self, **kw: Any) -> None:
        return None


def budget(*, consumed: int = 0, cap: int = 10**6) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": 10**6,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def window(
    start_min: float = 0, text: str = "Let us look at the budget numbers now."
) -> TranscriptWindow:
    start = int(start_min * MIN)
    c1, c2 = str(uuid.uuid4()), str(uuid.uuid4())
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:{start}",
        start_ms=start,
        end_ms=start + 30_000,
        turns=[
            Turn(str(ALICE), text, start, start + 5_000, [c1]),
            Turn(str(BOB), "Sure, go ahead.", start + 5_000, start + 8_000, [c2]),
        ],
    )


async def make(
    llm: ScriptedLLM,
    source: Any = None,
    *,
    sink: Sink | None = None,
    gate: BudgetGate | None = None,
    redis: Any = None,
    clock: Clock | None = None,
    ttl: float = 0.0,
) -> tuple[MeetingAgent, Sink, Any]:
    sink = sink or Sink()
    redis = redis if redis is not None else fakeredis.aioredis.FakeRedis()
    router = LiteLLMRouter(gate or budget(), sink, llm)
    deps = MeetingDeps(
        agenda=source if source is not None else FakeAgenda(agenda()),
        redis=redis,
        clock=clock or Clock(),
    )
    agent = MeetingAgent(
        PromptLoader(PROMPTS_DIR), router, deps=deps, settings=MeetingSettings(agenda_ttl_s=ttl)
    )
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    return agent, sink, redis


async def run(agent: MeetingAgent, w: TranscriptWindow) -> list[tuple[uuid.UUID, AgentResult]]:
    return await agent.process_meeting(w, {ALICE: {"role": "x"}, BOB: {"role": "y"}})


def answer(
    updates: list[dict[str, Any]] | None = None,
    off: dict[str, Any] | None = None,
    raised: list[int] | None = None,
) -> str:
    return json.dumps({"updates": updates or [], "off_agenda": off, "raised": raised or []})


# --- 1. skips ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,  # the route failed (unreachable / 401 / 404 / bad body)
        RuntimeError("boom"),  # the client itself raised
        agenda(items=[]),  # D7: only for a meeting with at least one agenda item
        agenda(host_user_id=None),  # nobody to tell
    ],
)
async def test_skips_quietly_without_a_usable_agenda(value: Any):
    llm = ScriptedLLM(answer(off={"content": "x", "confidence": 0.99}))
    agent, sink, _ = await make(llm, FakeAgenda(value))
    assert await run(agent, window()) == []
    assert llm.calls == [] and sink.usage_calls == []  # nothing paid for


async def test_without_a_configured_agenda_route_it_never_runs(monkeypatch):
    monkeypatch.delenv("CORE_URL", raising=False)
    monkeypatch.delenv("CORE_CALLER_TOKEN", raising=False)
    assert agenda_from_env() is None
    llm = ScriptedLLM()
    router = LiteLLMRouter(budget(), Sink(), llm)
    agent = MeetingAgent(PromptLoader(PROMPTS_DIR), router, deps=MeetingDeps())
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    assert await run(agent, window()) == [] and llm.calls == []


async def test_a_redis_failure_skips_the_window():
    class DownRedis:
        async def hgetall(self, key: str) -> Any:
            raise ConnectionError("redis down")

    llm = ScriptedLLM(answer(off={"content": "x", "confidence": 0.99}))
    agent, _, _ = await make(llm, redis=DownRedis())
    assert await run(agent, window()) == [] and llm.calls == []


# --- 2. the agenda route ----------------------------------------------------------------------


BODY = {
    "host_user_id": str(HOST),
    "previous_meeting_id": str(uuid.uuid4()),
    "scheduled_start": "2026-11-10T10:00:00Z",
    "scheduled_end": "2026-11-10T10:45:00Z",
    "items": [
        {
            "id": str(ITEM_B),
            "position": 1,
            "title": "Hiring plan",
            "duration_minutes": 5,
            "status": "in_progress",
            "status_source": "agent",
        },
        {"id": str(ITEM_A), "position": 0, "title": "Budget", "duration_minutes": None},
        {"id": "not-a-uuid", "position": 2, "title": "dropped"},
        {"id": str(uuid.uuid4()), "position": 3, "title": "  "},
    ],
    "carry_forward": {"decisions": ["Use vendor X"], "open_actions": ["Send contract", ""]},
    "extra": "ignored",
}


async def test_the_route_uses_the_caller_token_scheme_and_parses_the_d6_shape():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BODY)

    client = HttpAgendaClient("http://core/", "secret", transport=httpx.MockTransport(handler))
    got = await client.agenda(ORG, MEETING)
    (req,) = seen
    assert req.url.path == f"/internal/meetings/{MEETING}/agenda" and not req.url.query
    assert req.headers["X-Mytheus-Service"] == "agent-runtime-service"
    assert req.headers["X-Mytheus-Service-Token"] == "secret"
    assert req.headers["X-Mytheus-Org"] == str(ORG)
    assert got is not None and got.host_user_id == HOST
    assert [i.id for i in got.items] == [ITEM_A, ITEM_B]  # by position, malformed dropped
    assert got.items[0].status == "pending" and got.items[0].status_source is None
    assert got.items[1].status_source == "agent" and got.items[1].duration_minutes == 5
    assert got.open_actions == ["Send contract"] and got.decisions == ["Use vendor X"]
    assert got.scheduled_minutes == 45.0


def test_scheduled_time_falls_back_to_the_items_minutes_then_to_none():
    body = {**BODY, "scheduled_start": None}
    body.pop("scheduled_end")
    parsed = parse_agenda(body)
    assert parsed is not None and parsed.scheduled_minutes == 5.0
    body["items"] = [{"id": str(ITEM_A), "position": 0, "title": "Budget"}]
    parsed = parse_agenda(body)
    assert parsed is not None and parsed.scheduled_minutes is None
    assert parse_agenda({"items": "nope"}) is None and parse_agenda([]) is None


@pytest.mark.parametrize("status", [401, 404, 500])
async def test_a_non_200_is_none(status: int):
    client = HttpAgendaClient(
        "http://core", "t", transport=httpx.MockTransport(lambda r: httpx.Response(status))
    )
    assert await client.agenda(ORG, MEETING) is None


async def test_unreachable_and_bad_bodies_are_none():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = HttpAgendaClient("http://core", "t", transport=httpx.MockTransport(down))
    assert await client.agenda(ORG, MEETING) is None
    junk = HttpAgendaClient(
        "http://core", "t", transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<"))
    )
    assert await junk.agenda(ORG, MEETING) is None


async def test_a_failing_route_skips_the_agent_quietly_end_to_end():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = ScriptedLLM()
    source = HttpAgendaClient("http://core", "t", transport=httpx.MockTransport(down))
    agent, sink, _ = await make(llm, source)
    assert await run(agent, window()) == [] and llm.calls == [] and sink.results == []


async def test_the_agenda_is_cached_per_meeting_for_the_ttl_and_failures_are_not():
    clock = Clock()
    source = FakeAgenda(agenda())
    agent, _, _ = await make(ScriptedLLM(), source, clock=clock, ttl=60)
    await run(agent, window(0))
    clock.now += 30
    await run(agent, window(0.5))
    assert source.calls == 1
    clock.now += 31
    await run(agent, window(1))
    assert source.calls == 2
    source.value = None  # route down after the TTL: skip, and ask again next window
    clock.now += 61
    assert await run(agent, window(2)) == []
    assert await run(agent, window(2.5)) == []
    assert source.calls == 4


# --- 3. over_time ----------------------------------------------------------------------------


async def test_over_time_is_computed_once_per_item():
    llm = ScriptedLLM(NOTHING)
    agent, _, redis = await make(llm)
    # ITEM_B (1 min slot) is in progress when first seen at 0:00
    assert await run(agent, window(0)) == []
    (host, result), *_ = await run(agent, window(1))  # ends at 1:30 > 1:00 slot
    assert host == HOST
    nudge = result.payload["nudge"]
    assert nudge == {
        "id": nudge_id(MEETING, "over_time", str(ITEM_B)),
        "type": "over_time",
        "item_id": str(ITEM_B),
        "content": '"Hiring plan" has run 1 min, over its 1 min slot.',
    }
    assert result.payload["updates"] == [] and result.confidence == 1.0
    # long after the rate limit: never again for this item
    assert await run(agent, window(20)) == []
    raw = await redis.hgetall(state_key(ORG, MEETING))
    assert raw[f"over_time:{ITEM_B}".encode()] == b"1"
    assert raw[f"started:{ITEM_B}".encode()] == b"0"


async def test_the_model_sees_how_long_the_item_has_run_and_the_carried_items():
    llm = ScriptedLLM(NOTHING)
    agent, _, _ = await make(llm)
    await run(agent, window(0))
    await run(agent, window(3))
    user = llm.user()
    assert "1. Budget review (pending, 10 min planned)" in user
    assert "2. Hiring plan (in progress, 1 min planned, running for 3 min)" in user
    assert "3. Launch date (pending)" in user
    assert "C1. Send the vendor contract" in user and "C2. Fix the login bug" in user
    assert "Participant 1: Let us look" in user
    assert str(ALICE) not in user and str(HOST) not in user and str(ITEM_A) not in user


async def test_an_item_the_agent_moved_to_in_progress_starts_its_clock_then():
    llm = ScriptedLLM(
        answer([{"item": 1, "status": "in_progress", "note": "Started.", "confidence": 0.9}]),
        NOTHING,
    )
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", 2, "pending")])
    agent, _, _ = await make(llm, FakeAgenda(a))
    (_, first), *_ = await run(agent, window(5))  # starts at 5:00
    assert first.payload["updates"][0]["status"] == "in_progress"
    assert await run(agent, window(6)) == []  # 6:30: 1.5 min of a 2 min slot
    (_, late), *_ = await run(agent, window(7))  # 7:30: 2.5 min > 2 min
    assert late.payload["nudge"]["type"] == "over_time"
    assert late.payload["nudge"]["item_id"] == str(ITEM_A)


# --- 4. nudges --------------------------------------------------------------------------------


async def test_one_nudge_per_meeting_per_five_minutes():
    off = answer(off={"content": "The group moved to the office move.", "confidence": 0.8})
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], open_actions=[])
    agent, _, _ = await make(ScriptedLLM(off), FakeAgenda(a))
    (_, first), *_ = await run(agent, window(0))
    assert first.payload["nudge"]["type"] == "off_agenda"
    assert first.payload["nudge"]["item_id"] is None
    assert first.payload["nudge"]["id"] == nudge_id(MEETING, "off_agenda", f"{MEETING}:0")
    assert await run(agent, window(1)) == []  # same drift, inside 5 minutes: nothing to emit
    assert await run(agent, window(4.5)) == []  # 5:00 end is still < 5 min after 0:30
    (_, again), *_ = await run(agent, window(5))  # 5:30 - 0:30 = 5 min
    assert again.payload["nudge"]["type"] == "off_agenda"


async def test_a_rate_limited_over_time_nudge_waits_and_still_fires_once():
    off = answer(off={"content": "Drifted to lunch plans.", "confidence": 0.9})
    llm = ScriptedLLM(off, NOTHING)
    agent, _, _ = await make(llm)
    (_, first), *_ = await run(agent, window(0))  # off_agenda takes the 0:30 slot
    assert first.payload["nudge"]["type"] == "off_agenda"
    assert await run(agent, window(2)) == []  # ITEM_B is over time but rate-limited
    (_, later), *_ = await run(agent, window(5))
    assert later.payload["nudge"]["type"] == "over_time"
    assert await run(agent, window(11)) == []


async def test_off_agenda_needs_confidence_of_at_least_0_75():
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], open_actions=[])
    weak = answer(off={"content": "Maybe drifting.", "confidence": 0.74})
    agent, _, _ = await make(ScriptedLLM(weak), FakeAgenda(a))
    assert await run(agent, window(0)) == []
    edge = answer(off={"content": "Drifted.", "confidence": 0.75})
    agent, _, _ = await make(ScriptedLLM(edge), FakeAgenda(a))
    (_, result), *_ = await run(agent, window(0))
    assert result.payload["nudge"]["type"] == "off_agenda" and result.confidence == 0.75


async def test_unaddressed_names_an_unraised_carried_item_after_70_percent():
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")])
    # C1 is raised early; scheduled 30 min -> the check opens at 21:00
    llm = ScriptedLLM(answer(raised=[1, 9]), NOTHING)
    agent, _, redis = await make(llm, FakeAgenda(a))
    assert await run(agent, window(1)) == []
    assert await run(agent, window(20)) == []  # 20:30 < 21:00
    (_, result), *_ = await run(agent, window(21))
    nudge = result.payload["nudge"]
    key = carried_key("Fix the login bug")
    assert nudge == {
        "id": nudge_id(MEETING, "unaddressed", key),
        "type": "unaddressed",
        "item_id": None,
        "content": "Not raised yet from last meeting: Fix the login bug",
    }
    assert await run(agent, window(30)) == []  # once per carried item; C1 was raised
    raw = await redis.hgetall(state_key(ORG, MEETING))
    assert raw[f"raised:{carried_key('Send the vendor contract')}".encode()] == b"1"


async def test_unaddressed_never_fires_without_a_scheduled_time():
    a = agenda(
        items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], scheduled_minutes=None
    )
    agent, _, _ = await make(ScriptedLLM(NOTHING), FakeAgenda(a))
    assert await run(agent, window(500)) == []


# --- 5. updates -------------------------------------------------------------------------------


async def test_update_rules():
    host_set = AgendaItem(ITEM_A, 0, "Budget review", 10, "in_progress", "host")
    covered = AgendaItem(ITEM_B, 1, "Hiring plan", 10, "covered", "agent")
    pending = AgendaItem(ITEM_C, 2, "Launch date", None, "pending")
    a = agenda(items=[host_set, covered, pending], open_actions=[])
    llm = ScriptedLLM(
        answer(
            [
                {"item": 1, "status": "covered", "note": "Host owns this.", "confidence": 0.95},
                {"item": 2, "status": "in_progress", "note": "Backwards.", "confidence": 0.95},
                {"item": 3, "status": "in_progress", "note": "Weak.", "confidence": 0.7},
                {
                    "item": 3,
                    "status": "covered",
                    "note": "Done: ship on " + "x" * 400,
                    "confidence": 0.9,
                },
                {"item": 3, "status": "covered", "note": "Duplicate.", "confidence": 0.99},
                {"item": 4, "status": "covered", "note": "No such item.", "confidence": 0.99},
                {"item": True, "status": "covered", "confidence": 0.99},
                {"item": 3, "status": "skipped", "confidence": 0.99},
            ]
        )
    )
    agent, _, _ = await make(llm, FakeAgenda(a))
    (_, result), *_ = await run(agent, window(0))
    (update,) = result.payload["updates"]
    assert update["item_id"] == str(ITEM_C) and update["status"] == "covered"
    assert update["note"].startswith("Done: ship on") and len(update["note"]) <= 300
    assert result.payload["nudge"] is None and result.confidence == 0.9


async def test_an_update_already_proposed_is_not_repeated_while_the_agenda_lags():
    """The agenda is cached and core-api applies asynchronously: the agent's own proposal
    counts as the current status, so the same in_progress is not re-sent every window."""
    start = answer([{"item": 1, "status": "in_progress", "note": "Started.", "confidence": 0.9}])
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], open_actions=[])
    agent, _, _ = await make(ScriptedLLM(start), FakeAgenda(a))
    assert len(await run(agent, window(0))) == 1
    assert await run(agent, window(1)) == []


async def test_a_host_reset_wins_over_the_agents_own_proposal():
    start = answer([{"item": 1, "status": "covered", "note": "Done.", "confidence": 0.9}])
    source = FakeAgenda(
        agenda(items=[AgendaItem(ITEM_A, 0, "Budget", None, "in_progress")], open_actions=[])
    )
    llm = ScriptedLLM(start)
    agent, _, _ = await make(llm, source)
    assert len(await run(agent, window(0))) == 1
    source.value = agenda(
        items=[AgendaItem(ITEM_A, 0, "Budget", None, "pending", "host")], open_actions=[]
    )
    assert await run(agent, window(1)) == []  # host-set: not proposed again
    assert "1. Budget (pending)" in llm.user()


# --- 6. the record ----------------------------------------------------------------------------


async def test_one_record_per_window_for_the_host_with_the_d7_payload():
    llm = ScriptedLLM(
        NOTHING,
        answer(
            [{"item": 1, "status": "in_progress", "note": "Numbers first.", "confidence": 0.9}],
            off={"content": "ignored: over_time wins the slot", "confidence": 0.99},
        ),
    )
    agent, _, _ = await make(llm)
    assert await run(agent, window(0)) == []
    results = await run(agent, window(1))  # an update and the over_time nudge, one record
    assert len(results) == 1
    host, r = results[0]
    assert host == HOST and r.kind == "meeting_update" and r.window_id is None
    assert set(r.payload) == {"updates", "nudge", "surfaced"} and r.payload["surfaced"] is True
    for u in r.payload["updates"]:
        assert set(u) == {"item_id", "status", "note"}
        assert uuid.UUID(u["item_id"]) and u["status"] in ("in_progress", "covered")
    n = r.payload["nudge"]
    assert set(n) == {"id", "type", "item_id", "content"} and uuid.UUID(n["id"])
    assert n["type"] == "over_time" and len(n["content"]) <= 200
    assert r.content == n["content"]
    assert r.payload["updates"] == [
        {"item_id": str(ITEM_A), "status": "in_progress", "note": "Numbers first."}
    ]
    assert all(uuid.UUID(c) for c in r.source_chunk_ids)
    json.dumps(r.payload)  # payload_json


async def test_nothing_to_say_emits_nothing_but_the_call_is_metered_to_the_host():
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], open_actions=[])
    llm = ScriptedLLM(NOTHING)
    agent, sink, _ = await make(llm, FakeAgenda(a))
    assert await run(agent, window(0)) == []
    (call,) = llm.calls
    assert "haiku" in call["model"] and call["thinking"] == {"type": "disabled"}
    (usage,) = sink.usage_calls
    assert usage["user_id"] == HOST and usage["model_tier"] == "haiku" and usage["org_id"] == ORG
    assert call["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


async def test_a_malformed_model_answer_emits_only_what_is_computed():
    agent, _, _ = await make(ScriptedLLM("not json at all {"))
    assert await run(agent, window(0)) == []
    (_, r), *_ = await run(agent, window(1))
    assert r.payload["nudge"]["type"] == "over_time" and r.payload["updates"] == []


# --- 7. budget --------------------------------------------------------------------------------


async def test_a_blocked_budget_skips_the_agent_without_a_model_call():
    llm = ScriptedLLM(answer(off={"content": "x", "confidence": 0.99}))
    agent, sink, redis = await make(llm, gate=budget(consumed=10**6, cap=10**6))
    assert await run(agent, window(0)) == []
    assert await run(agent, window(5)) == []  # not even the computed over_time
    assert agent.blocked and llm.calls == [] and sink.usage_calls == []
    assert await redis.hgetall(state_key(ORG, MEETING)) == {}


async def test_a_call_that_would_cross_the_cap_is_blocked_too():
    llm = ScriptedLLM(answer(off={"content": "x", "confidence": 0.99}))
    agent, _, _ = await make(llm, gate=budget(consumed=10**6 - 100, cap=10**6))
    assert await run(agent, window(0)) == []
    assert agent.blocked and llm.calls == []


# --- 8. runtime wiring and the Avro contract --------------------------------------------------


async def test_registered_once_per_meeting_and_the_result_goes_to_the_host():
    meeting_answer = answer(
        [{"item": 1, "status": "in_progress", "note": "Started.", "confidence": 0.9}]
    )

    class RoutingLLM(ScriptedLLM):
        """The MeetingAgent gets its answer; every other agent's gate stays shut."""

        async def __call__(self, **kw: Any) -> Any:
            self.calls.append(kw)
            system = kw["messages"][0]["content"][0]["text"]
            mine = system.startswith("You follow a live meeting against its agenda")
            text = meeting_answer if mine else json.dumps({"relevance": 0, "score": 0})
            return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1}}

    sink = Sink()
    router = LiteLLMRouter(budget(), sink, RoutingLLM())
    a = agenda(items=[AgendaItem(ITEM_A, 0, "Budget review", None, "pending")], open_actions=[])
    deps = MeetingDeps(agenda=FakeAgenda(a), redis=fakeredis.aioredis.FakeRedis())

    async def ctx(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
        return {"role": "Engineer"}

    runtime = AgentRuntime(
        sink=sink,
        user_context=ctx,
        agents=registered_agents(PromptLoader(PROMPTS_DIR), router, meeting=deps),
    )
    for u in (ALICE, BOB):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    await runtime.process_window(window(0))
    mine = [r for r in sink.results if r["agent"] == "MeetingAgent"]
    assert len(mine) == 1 and mine[0]["user_id"] == HOST  # the host, not in the roster
    assert mine[0]["result"].kind == "meeting_update"
    keys = [k for k in runtime._meeting(ORG, MEETING).agents if k[0] == "MeetingAgent"]
    assert keys == [("MeetingAgent", MEETING)]


def _schema(*, with_d8: bool) -> Any:
    header = {
        "type": "record",
        "name": "EventEnvelopeHeader",
        "namespace": "io.nexus.events",
        "fields": [
            {"name": "event_id", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "idempotency_key", "type": "string"},
            {"name": "org_id", "type": {"type": "string", "logicalType": "uuid"}},
            {"name": "schema_version", "type": "int", "default": 1},
            {"name": "producer", "type": "string"},
            {"name": "occurred_at", "type": {"type": "long", "logicalType": "timestamp-micros"}},
            {"name": "correlation_id", "type": ["null", "string"], "default": None},
            {"name": "causation_id", "type": ["null", "string"], "default": None},
        ],
    }
    agents = [
        "PersonalListenerAgent",
        "DecisionAgent",
        "ActionAgent",
        "QuestionAgent",
        "ContextGapAgent",
        "ProactiveAgent",
    ]
    kinds = ["note", "decision_candidate", "action_candidate", "suggestion", "alert"]
    kinds.append("context_fill")
    if with_d8:  # ADR-0038 D8, appended (BACKWARD)
        agents.append("MeetingAgent")
        kinds.append("meeting_update")
    uuid_t = {"type": "string", "logicalType": "uuid"}
    return fastavro.parse_schema(
        {
            "type": "record",
            "name": "AgentResult",
            "namespace": "io.nexus.events",
            "fields": [
                {"name": "header", "type": header},
                {"name": "meeting_id", "type": uuid_t},
                {"name": "window_id", "type": "string"},
                {"name": "user_id", "type": uuid_t},
                {
                    "name": "agent",
                    "type": {"type": "enum", "name": "AgentClass", "symbols": agents},
                },
                {
                    "name": "kind",
                    "type": {"type": "enum", "name": "AgentResultKind", "symbols": kinds},
                    "default": "note",
                },
                {"name": "relevance_score", "type": "float"},
                {"name": "confidence", "type": "float"},
                {"name": "content", "type": "string"},
                {"name": "payload_json", "type": "string"},
                {"name": "source_chunk_ids", "type": {"type": "array", "items": uuid_t}},
            ],
        }
    )


class FastavroCodec:
    """Stands in for `AvroCodec`, which serializes against the registry's latest version."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema

    def serialize_for(self, topic: str, value: dict[str, Any]) -> bytes:
        assert topic == "agent-results"
        out = io.BytesIO()
        fastavro.schemaless_writer(out, self.schema, value)
        return out.getvalue()


class Producer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any, bytes]] = []

    def produce(self, topic: str, key: Any, value: bytes, on_delivery: Any) -> None:
        self.sent.append((topic, key, value))

    def poll(self, _t: float) -> int:
        return 0


async def _meeting_record() -> tuple[uuid.UUID, AgentResult, TranscriptWindow]:
    agent, _, _ = await make(ScriptedLLM(NOTHING))
    await run(agent, window(0))
    w = window(1)
    ((host, result),) = await run(agent, w)
    return host, result, w


async def test_the_record_encodes_against_an_agent_results_schema_with_the_d8_symbols():
    host, result, w = await _meeting_record()
    schema = _schema(with_d8=True)
    producer = Producer()
    emitter = Emitter(producer, FastavroCodec(schema))
    await emitter.result(window=w, user_id=host, agent="MeetingAgent", result=result)
    ((topic, key, value),) = producer.sent
    assert topic == "agent-results" and key == str(ORG)
    decoded = fastavro.schemaless_reader(io.BytesIO(value), schema, schema)
    assert decoded["agent"] == "MeetingAgent" and decoded["kind"] == "meeting_update"
    assert str(decoded["user_id"]) == str(HOST) and decoded["window_id"] == w.window_id
    assert decoded["header"]["idempotency_key"] == (
        f"agentresult:{MEETING}:{w.window_id}:{HOST}:MeetingAgent"
    )
    assert json.loads(decoded["payload_json"]) == result.payload


async def test_a_registry_without_the_d8_symbols_costs_the_record_not_the_window():
    host, result, w = await _meeting_record()
    emitter = Emitter(Producer(), FastavroCodec(_schema(with_d8=False)))
    with pytest.raises(ValueError):
        await emitter.result(window=w, user_id=host, agent="MeetingAgent", result=result)

    class OldRegistrySink(Sink):
        async def result(self, **kw: Any) -> None:
            self.results.append(kw)  # the attempt
            await emitter.result(**kw)  # raises for MeetingAgent, as AvroCodec would

    sink = OldRegistrySink()
    router = LiteLLMRouter(budget(), sink, ScriptedLLM(NOTHING))
    deps = MeetingDeps(agenda=FakeAgenda(agenda()), redis=fakeredis.aioredis.FakeRedis())
    runtime = AgentRuntime(
        sink=sink,
        user_context=lambda o, u: _ctx(),
        agents=[lambda: MeetingAgent(PromptLoader(PROMPTS_DIR), router, deps=deps)],
    )
    await runtime.on_room_event(ORG, MEETING, "joined", str(ALICE))
    assert await runtime.process_window(window(0)) == 0
    assert await runtime.process_window(window(1)) == 0  # over_time produced, dropped, no crash
    assert [r["agent"] for r in sink.results] == ["MeetingAgent"]


async def _ctx() -> dict[str, Any]:
    return {"role": "x"}
