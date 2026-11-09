"""agent-runtime-service: Plan 7-9 section 3.2 acceptance checks, with every model call faked.

No test here reaches a real LLM (CLAUDE.md: never hit a paid model in the suite), Redpanda,
Postgres, or identity-tenant-service.

1. Windowing: 360 finals over 45 min become 90 windows; same-speaker finals merge; a
   redelivered chunk is ignored; ids are deterministic.
2. Budget gate: ok / throttle / block from the mirror; `alert_only` never degrades; a mirror
   miss reads through billing; every failure blocks.
3. Router: `block` calls no model; `throttle` forces Haiku; extraction never routes to Opus;
   exactly one `usage-events` record per call with its tier; the system prompt is cached.
4. BaseAgent: fails closed without `org_id`; a blocked budget returns nothing.
5. Spawner: one agent per participant; `left` terminates; one budget notification.
6. Offsets: nothing is committed past an unprocessed window.
7. Zero Postgres: the runtime's modules import no database driver.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from langgraph.graph import END, StateGraph

from libs.shared import user_context
from libs.shared.base_agent import BaseAgent
from libs.shared.budget import BudgetGate, BudgetView, mirror_key
from libs.shared.llm_cache import ResponseCache
from libs.shared.pii import EgressRedactor
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import AgentResult, AgentState
from services.agent_runtime_service.consumer import Loop, OffsetTracker
from services.agent_runtime_service.emitter import EmitError, Emitter
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.windowing import WINDOW_MS, Chunk, Windower

ROOT = Path(__file__).resolve().parents[1]
ORG = uuid.uuid4()
MEETING = uuid.uuid4()


# --- fakes ---------------------------------------------------------------------------------


class Sink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.flushes = 0

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        self.notifications.append(kw)

    def flush(self) -> None:
        self.flushes += 1


class FakeLLM:
    """Records every call; answers with fixed text and usage."""

    def __init__(self, text: str = "0.9") -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = text

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if kw.get("stream"):
            return self._stream()
        return {
            "choices": [{"message": {"content": self.text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    async def _stream(self) -> Any:
        for piece in ("Ship ", "it."):
            yield {"choices": [{"delta": {"content": piece}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}


def budget_gate(
    redis: Any, *, consumed: int = 0, quota: int = 1000, cap: int = 2000, mode: str = "enforce"
) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any] | None:
        return {
            "tokens_consumed": consumed,
            "token_quota": quota,
            "hard_cap": cap,
            "enforcement": mode,
        }

    return BudgetGate(redis, refresh)


@pytest.fixture
def prompts(tmp_path: Path) -> PromptLoader:
    (tmp_path / "registry.json").write_text(
        json.dumps({"defaults": {"two_step": "v1"}, "pins": {str(ORG): {"two_step": "v2"}}})
    )
    for v in ("v1", "v2"):
        (tmp_path / f"two_step@{v}.md").write_text(
            f"You are a test agent {v}.\n---user---\nWindow:\n{{window}}"
        )
    return PromptLoader(tmp_path)


class TwoStepAgent(BaseAgent):
    """The PersonalListener shape: a Haiku pre-gate, then a Sonnet extraction if it passes."""

    name = "PersonalListenerAgent"
    prompt_key = "two_step"
    scope = "participant"
    spawned = 0
    terminated = 0

    async def spawn(self, **kw: Any) -> None:
        await super().spawn(**kw)
        TwoStepAgent.spawned += 1

    async def terminate(self) -> None:
        TwoStepAgent.terminated += 1
        await super().terminate()

    def build_graph(self) -> Any:
        assert self.prompt is not None
        prompt = self.prompt

        async def pregate(s: AgentState) -> dict[str, Any]:
            out = await self.router.complete(
                tier="triage",
                system=prompt.system,
                user=prompt.render_user(window=s.chunk.transcript()),
                org_id=s.org_id,  # type: ignore[arg-type]
                user_id=s.user_id,
                max_tokens=5,
            )
            return {"relevance_score": float(out.text)}

        async def extract(s: AgentState) -> dict[str, Any]:
            out = await self.router.complete(
                tier="extract",
                system=prompt.system,
                user=s.chunk.transcript(),
                org_id=s.org_id,  # type: ignore[arg-type]
                user_id=s.user_id,
                max_tokens=50,
            )
            return {
                "result": AgentResult(
                    kind="note",
                    content=out.text,
                    relevance_score=s.relevance_score or 0.0,
                    confidence=0.8,
                    source_chunk_ids=s.chunk.chunk_ids,
                )
            }

        g = StateGraph(AgentState)
        g.add_node("pregate", pregate)
        g.add_node("extract", extract)
        g.set_entry_point("pregate")
        g.add_conditional_edges(
            "pregate", lambda s: (s.relevance_score or 0) > 0.35, {True: "extract", False: END}
        )
        g.add_edge("extract", END)
        return g.compile()


def chunk(start_ms: int, speaker: str = "a", cid: str | None = None, offset: int = 0) -> Chunk:
    return Chunk(
        org_id=ORG,
        meeting_id=MEETING,
        chunk_id=cid or str(uuid.uuid4()),
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=start_ms + 5_000,
        text=f"t{start_ms}",
        offset=offset,
    )


# --- 1. windowing --------------------------------------------------------------------------


def test_360_finals_over_45_minutes_make_90_windows():
    w = Windower()
    closed = []
    for i in range(360):  # one final every 7.5 s
        closed += w.add(chunk(i * 7_500, speaker="ab"[i % 2]))
    closed += w.close_meeting(ORG, MEETING)
    assert len(closed) == 90
    assert closed[0].window.window_id == f"{MEETING}:0"
    assert closed[1].window.window_id == f"{MEETING}:{WINDOW_MS}"


def test_same_speaker_finals_merge_and_a_redelivered_chunk_is_ignored():
    w = Windower()
    w.add(chunk(0, "a", cid="c1"))
    w.add(chunk(5_000, "a", cid="c2"))
    w.add(chunk(5_000, "a", cid="c2"))  # redelivery
    w.add(chunk(10_000, "b", cid="c3"))
    (cw,) = w.close_meeting(ORG, MEETING)
    assert [(t.speaker_id, t.chunk_ids) for t in cw.window.turns] == [
        ("a", ["c1", "c2"]),
        ("b", ["c3"]),
    ]
    assert cw.window.transcript() == "a: t0 t5000\nb: t10000"


def test_a_quiet_meeting_closes_on_idle():
    now = [0.0]
    w = Windower(idle_s=15, clock=lambda: now[0])
    w.add(chunk(0))
    assert w.expire_idle() == []
    now[0] = 16
    assert len(w.expire_idle()) == 1


# --- 2. budget gate ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("consumed", "est", "expected"),
    [(0, 10, "ok"), (1000, 10, "throttle"), (1995, 10, "block"), (2000, 0, "block")],
)
async def test_gate_decides_from_the_mirror(consumed, est, expected):
    redis = fakeredis.aioredis.FakeRedis()
    await redis.hset(
        mirror_key(ORG),
        mapping={
            "tokens_consumed": consumed,
            "token_quota": 1000,
            "hard_cap": 2000,
            "enforcement": "enforce",
        },
    )

    async def never(_o: uuid.UUID) -> None:
        raise AssertionError("a mirror hit must not read through")

    assert await BudgetGate(redis, never).check(ORG, est) == expected


async def test_alert_only_never_degrades():
    gate = budget_gate(fakeredis.aioredis.FakeRedis(), consumed=5000, mode="alert_only")
    assert await gate.check(ORG, 10) == "ok"


async def test_a_miss_reads_through_and_every_failure_blocks():
    redis = fakeredis.aioredis.FakeRedis()
    assert await budget_gate(redis).check(ORG, 10) == "ok"  # miss -> billing answered

    async def down(_o: uuid.UUID) -> None:
        return None

    assert await BudgetGate(redis, down).check(ORG, 10) == "block"

    async def garbage(_o: uuid.UUID) -> dict[str, Any]:
        return {"tokens_consumed": "lots"}

    assert await BudgetGate(redis, garbage).check(ORG, 10) == "block"

    class BrokenRedis:
        async def hgetall(self, _k: str) -> dict[str, Any]:
            raise ConnectionError("redis down")

    assert await budget_gate(BrokenRedis()).check(ORG, 10) == "ok"  # read through billing
    assert await BudgetGate(BrokenRedis(), down).check(ORG, 10) == "block"


def test_budget_view_parses_bytes():
    v = BudgetView.parse({b"tokens_consumed": b"5", b"token_quota": b"10", b"hard_cap": b"20"})
    assert (v.tokens_consumed, v.enforcement) == (5, "alert_only")


# --- 3. router -----------------------------------------------------------------------------


async def test_block_calls_no_model_and_meters_nothing():
    llm, sink = FakeLLM(), Sink()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis(), consumed=2000), sink, llm)
    with pytest.raises(BudgetBlocked):
        await router.complete(
            tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
        )
    assert llm.calls == [] and sink.usage_calls == []


async def test_throttle_forces_haiku_and_ok_routes_extract_to_sonnet():
    llm, sink = FakeLLM(), Sink()
    redis = fakeredis.aioredis.FakeRedis()
    throttled = LiteLLMRouter(budget_gate(redis, consumed=1000), sink, llm)
    out = await throttled.complete(
        tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
    )
    assert out.tier == "triage" and llm.calls[-1]["model"] == "anthropic/claude-haiku-5-5"

    ok = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)
    out = await ok.complete(
        tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
    )
    assert llm.calls[-1]["model"] == "anthropic/claude-sonnet-5-5"
    assert all("opus" not in c["model"] for c in llm.calls)
    assert [u["model_tier"] for u in sink.usage_calls] == ["haiku", "sonnet"]
    assert sink.usage_calls[-1]["tokens"] == 120
    system = llm.calls[-1]["messages"][0]["content"][0]
    assert system["cache_control"] == {"type": "ephemeral"}


async def test_a_workspace_id_is_sent_as_the_anthropic_header(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), Sink(), llm)
    await router.complete(
        tier="triage", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
    )
    assert llm.calls[-1]["extra_headers"] == {"anthropic-workspace-id": "wrkspc_test"}
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    plain = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), Sink(), llm)
    await plain.complete(
        tier="triage", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
    )
    assert "extra_headers" not in llm.calls[-1]


async def test_high_stakes_is_refused_while_unmapped():
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), Sink(), FakeLLM())
    with pytest.raises(ValueError):
        await router.complete(
            tier="high_stakes", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
        )


async def test_streaming_hands_each_delta_on_and_meters_once():
    llm, sink = FakeLLM(), Sink()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)
    deltas: list[tuple[int, str]] = []

    async def on_delta(seq: int, text: str) -> None:
        deltas.append((seq, text))

    out = await router.complete(
        tier="extract",
        system="s",
        user="u",
        org_id=ORG,
        user_id=None,
        max_tokens=5,
        on_delta=on_delta,
    )
    assert out.text == "Ship it." and deltas == [(0, "Ship "), (1, "it.")]
    assert len(sink.usage_calls) == 1 and sink.usage_calls[0]["tokens"] == 55


async def test_pii_is_redacted_before_egress_unless_the_tenant_turned_it_off():
    """Plan 7-9 section 3.3 acceptance: a window containing `jane@acme.com` is redacted before
    egress when `pii_redaction_enabled=true` (the canon default, and the answer when the
    setting is absent or Redis is down)."""
    window = "Jane: mail me at jane@acme.com or call 415-555-0134"
    redis = fakeredis.aioredis.FakeRedis()
    llm = FakeLLM()
    router = LiteLLMRouter(budget_gate(redis), Sink(), llm, redactor=EgressRedactor(redis))

    async def sent() -> str:
        await router.complete(
            tier="triage", system="s", user=window, org_id=ORG, user_id=None, max_tokens=5
        )
        return str(llm.calls[-1]["messages"][1]["content"])

    absent = await sent()
    assert "jane@acme.com" not in absent and "[REDACTED_EMAIL]" in absent
    assert "415-555-0134" not in absent

    await redis.set(f"nexus:{ORG}:tenant_settings:pii_redaction_enabled", "false")
    assert "jane@acme.com" in await sent()

    class DownRedis:
        async def get(self, _key: str) -> None:
            raise ConnectionError("redis down")

    assert "jane@acme.com" not in await EgressRedactor(DownRedis())(ORG, window)  # type: ignore[arg-type]

    default = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), Sink(), llm)
    await default.complete(
        tier="triage", system="s", user=window, org_id=ORG, user_id=None, max_tokens=5
    )
    assert "jane@acme.com" not in str(llm.calls[-1]["messages"][1]["content"])


async def test_an_identical_call_is_answered_from_the_cache_and_meters_nothing():
    """ADR-0030 D1: same tier, model, prompt and normalized input -> no second model call and
    no second usage record. A blocked tenant still gets nothing; a Redis failure is a miss."""
    redis = fakeredis.aioredis.FakeRedis()
    llm, sink = FakeLLM(text="note"), Sink()
    router = LiteLLMRouter(budget_gate(redis), sink, llm, cache=ResponseCache(redis))

    async def call(user: str, ref: str = "p@v1") -> Any:
        return await router.complete(
            tier="extract",
            system="sys",
            user=user,
            org_id=ORG,
            user_id=None,
            max_tokens=5,
            prompt_ref=ref,
        )

    first = await call("Alice:  we ship Friday")
    second = await call("alice: we ship   friday")
    assert not first.cached and second.cached and second.text == "note"
    assert len(llm.calls) == 1 and len(sink.usage_calls) == 1
    keys = [k.decode() async for k in redis.scan_iter("nexus:*:llmcache:*")]
    assert len(keys) == 1 and keys[0].startswith(f"nexus:{ORG}:llmcache:p@v1:")

    await call("alice: we ship friday", ref="p@v2")  # a new prompt version is a miss
    assert len(llm.calls) == 2

    blocked = LiteLLMRouter(
        budget_gate(redis, consumed=2000), sink, llm, cache=ResponseCache(redis)
    )
    with pytest.raises(BudgetBlocked):
        await blocked.complete(
            tier="extract",
            system="sys",
            user="alice: we ship friday",
            org_id=ORG,
            user_id=None,
            max_tokens=5,
            prompt_ref="p@v1",
        )

    class DownRedis:
        async def get(self, _key: str) -> None:
            raise ConnectionError("down")

        async def set(self, *_a: Any, **_k: Any) -> None:
            raise ConnectionError("down")

    flaky = LiteLLMRouter(budget_gate(redis), sink, llm, cache=ResponseCache(DownRedis()))
    out = await flaky.complete(
        tier="extract",
        system="sys",
        user="x",
        org_id=ORG,
        user_id=None,
        max_tokens=5,
        prompt_ref="p@v1",
    )
    assert out.text == "note" and not out.cached


# --- 4. BaseAgent + prompts ----------------------------------------------------------------


def test_prompt_loader_honours_tenant_pins(prompts):
    assert prompts.resolve("two_step", ORG) == "v2"
    assert prompts.resolve("two_step", uuid.uuid4()) == "v1"
    p = prompts.load("two_step", "v1")
    assert p.system == "You are a test agent v1." and p.render_user(window="x") == "Window:\nx"


async def test_spawn_fails_closed_without_a_tenant(prompts):
    agent = TwoStepAgent(prompts, LiteLLMRouter(budget_gate(None), Sink(), FakeLLM()))
    with pytest.raises(ValueError):
        await agent.spawn(org_id=None, meeting_id=MEETING, user_id=uuid.uuid4())


async def _window_state(user: uuid.UUID) -> AgentState:
    w = Windower()
    w.add(chunk(0, str(user)))
    (cw,) = w.close_meeting(ORG, MEETING)
    return AgentState(
        meeting_id=MEETING, user_id=user, chunk=cw.window, context={"role": "x"}, org_id=ORG
    )


async def test_the_pre_gate_decides_whether_extraction_runs(prompts):
    user = uuid.uuid4()
    for score, expected_calls in (("0.2", 1), ("0.9", 2)):
        llm = FakeLLM(score)
        agent = TwoStepAgent(
            prompts, LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), Sink(), llm)
        )
        await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=user)
        result = await agent.process(await _window_state(user))
        assert len(llm.calls) == expected_calls
        assert (result is None) == (score == "0.2")


async def test_a_blocked_budget_returns_nothing(prompts):
    user = uuid.uuid4()
    llm = FakeLLM()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis(), consumed=9999), Sink(), llm)
    agent = TwoStepAgent(prompts, router)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=user)
    assert await agent.process(await _window_state(user)) is None
    assert agent.blocked and llm.calls == []


# --- 5. spawner ----------------------------------------------------------------------------


async def test_one_agent_per_participant_and_left_terminates(prompts):
    TwoStepAgent.spawned = TwoStepAgent.terminated = 0
    sink, llm = Sink(), FakeLLM()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)
    contexts: list[uuid.UUID] = []

    async def ctx(_org: uuid.UUID, user: uuid.UUID) -> dict[str, Any]:
        contexts.append(user)
        return {"role": "Engineer"}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    roster = [uuid.uuid4() for _ in range(6)]  # A7 = 6
    for u in roster:
        await rt.on_room_event(ORG, MEETING, "joined", str(u))
    await rt.on_room_event(ORG, MEETING, "joined", "nexus-audio-tap-x")  # not a user: ignored

    w = Windower()
    w.add(chunk(0, str(roster[0])))
    (cw,) = w.close_meeting(ORG, MEETING)
    assert await rt.process_window(cw.window) == 6
    assert TwoStepAgent.spawned == 6
    assert {r["user_id"] for r in sink.results} == set(roster)
    assert all(r["agent"] == "PersonalListenerAgent" for r in sink.results)

    await rt.process_window(cw.window)  # second window: agents are reused, not respawned
    assert TwoStepAgent.spawned == 6

    await rt.on_room_event(ORG, MEETING, "left", str(roster[0]))
    assert TwoStepAgent.terminated == 1 and roster[0] not in rt.roster(ORG, MEETING)


async def test_a_blocked_participant_is_told_once(prompts):
    sink = Sink()
    router = LiteLLMRouter(
        budget_gate(fakeredis.aioredis.FakeRedis(), consumed=9999), sink, FakeLLM()
    )

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    user = uuid.uuid4()
    await rt.on_room_event(ORG, MEETING, "joined", str(user))
    state = await _window_state(user)
    for _ in range(3):
        await rt.process_window(state.chunk)
    assert len(sink.notifications) == 1 and sink.results == []


async def test_no_context_means_no_spend(prompts):
    llm, sink = FakeLLM(), Sink()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> None:
        return None

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    state = await _window_state(uuid.uuid4())
    assert await rt.process_window(state.chunk) == 0 and llm.calls == []


# --- 6. offsets + loop ---------------------------------------------------------------------


def test_offset_tracker_commits_only_below_the_oldest_pending():
    t = OffsetTracker()
    for o in (10, 11, 12):
        t.received("transcript-chunks", 0, o)
    t.done("transcript-chunks", 0, 11)
    assert t.committable() == {("transcript-chunks", 0): 10}
    t.done("transcript-chunks", 0, 10)
    t.done("transcript-chunks", 0, 12)
    assert t.committable() == {("transcript-chunks", 0): 13}


def _chunk_value(start_ms: int, speaker: str) -> dict[str, Any]:
    return {
        "header": {"org_id": str(ORG)},
        "meeting_id": str(MEETING),
        "chunk_id": str(uuid.uuid4()),
        "speaker_id": speaker,
        "start_ms": start_ms,
        "end_ms": start_ms + 1_000,
        "text": "hello",
        "is_final": True,
    }


async def test_the_loop_holds_offsets_until_the_window_is_processed_and_flushed(prompts):
    sink, llm = Sink(), FakeLLM()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {"role": "x"}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    loop = Loop(rt, sink, Windower())
    user = str(uuid.uuid4())
    await loop.handle(
        "meeting-room-events",
        0,
        0,
        {
            "header": {"org_id": str(ORG)},
            "meeting_id": str(MEETING),
            "event_type": "joined",
            "participant_id": user,
        },
    )
    await loop.handle("transcript-chunks", 0, 100, _chunk_value(0, user))
    await loop.handle("transcript-chunks", 0, 101, _chunk_value(10_000, user))
    assert loop.offsets.committable()[("transcript-chunks", 0)] == 100  # window still open
    await loop.handle("transcript-chunks", 0, 102, _chunk_value(31_000, user))  # closes it
    assert sink.flushes == 1 and len(sink.results) == 1
    assert loop.offsets.committable()[("transcript-chunks", 0)] == 102


async def test_an_unacknowledged_produce_commits_nothing(prompts):
    class FailingSink(Sink):
        def flush(self) -> None:
            raise EmitError("broker down")

    sink = FailingSink()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, FakeLLM())

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    loop = Loop(rt, sink, Windower())
    user = str(uuid.uuid4())
    await loop.handle("transcript-chunks", 0, 5, _chunk_value(0, user))
    with pytest.raises(EmitError):
        await loop.handle("transcript-chunks", 0, 6, _chunk_value(40_000, user))
    assert loop.offsets.committable()[("transcript-chunks", 0)] == 5


# --- emitter -------------------------------------------------------------------------------


class FakeCodec:
    def serialize_for(self, topic: str, value: dict[str, Any]) -> bytes:
        return repr((topic, value)).encode()


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def produce(self, topic: str, *, key: str, value: bytes, on_delivery: Any) -> None:
        self.sent.append((topic, key))

    def poll(self, _t: float) -> int:
        return 0

    def flush(self, _t: float) -> int:
        return 0


async def test_emitter_keys_and_idempotency():
    p = FakeProducer()
    e = Emitter(p, FakeCodec())  # type: ignore[arg-type]
    state = await _window_state(uuid.uuid4())
    await e.delta(window=state.chunk, user_id=state.user_id, seq=0, text="x")
    await e.usage(org_id=ORG, user_id=None, tokens=3, model_tier="haiku", call_id="c")
    assert p.sent == [("agent-stream", f"{ORG}:{MEETING}"), ("usage-events", str(ORG))]
    e.flush()


# --- 7. zero Postgres + contracts ----------------------------------------------------------


def test_the_runtime_imports_no_database_driver():
    banned = {"sqlalchemy", "asyncpg", "psycopg2", "psycopg"}
    files = list((ROOT / "services" / "agent_runtime_service").rglob("*.py")) + [
        ROOT / "libs" / "shared" / n
        for n in (
            "state.py",
            "prompts.py",
            "budget.py",
            "router.py",
            "base_agent.py",
            "user_context.py",
            "pii.py",
            "llm_cache.py",
        )
    ]
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            for n in names:
                assert n.split(".")[0] not in banned, f"{f.name} imports {n}"


def test_identity_mirror_matches_the_proto():
    text = (ROOT / "proto" / "identity.proto").read_text(encoding="utf-8")
    for message, fields in user_context.FIELDS.items():
        body = re.search(rf"message {message} \{{(.*?)\}}", text, re.S)
        assert body, message
        declared = {
            (m.group(2), int(m.group(3)), bool(m.group(1)))
            for m in re.finditer(r"(repeated\s+)?\w+\s+(\w+)\s*=\s*(\d+);", body.group(1))
        }
        assert declared == {(n, num, rep) for n, num, _t, rep in fields}, message


async def test_a_provider_error_costs_the_note_not_the_process(prompts):
    """Production ran into this: an invalid Anthropic key made every call raise, the exception
    escaped process_window, the consumer died before committing its offset, and on restart it
    replayed the same window and died again (3,149 restarts). A failing agent call now skips
    that agent for that window; the window completes so its offset can commit."""

    class FailingLLM(FakeLLM):
        async def __call__(self, **kw: Any) -> Any:
            self.calls.append(kw)
            raise RuntimeError("AuthenticationError: invalid x-api-key")

    sink, llm = Sink(), FailingLLM()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)

    async def ctx(_o: uuid.UUID, _u: uuid.UUID) -> dict[str, Any]:
        return {"role": "Engineer"}

    rt = AgentRuntime(sink=sink, user_context=ctx, agents=[lambda: TwoStepAgent(prompts, router)])
    users = [uuid.uuid4(), uuid.uuid4()]
    for u in users:
        await rt.on_room_event(ORG, MEETING, "joined", str(u))
    w = Windower()
    w.add(chunk(0, str(users[0])))
    (cw,) = w.close_meeting(ORG, MEETING)

    assert await rt.process_window(cw.window) == 0
    assert sink.results == []
    assert len(llm.calls) == 2, "every participant's agent was still tried"
