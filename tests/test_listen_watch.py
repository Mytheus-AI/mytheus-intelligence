"""The live watch ("in every meeting at once"): a person whose agent listens is interrupted only
when something needs them. No paid model: a scripted LLM; core-api is an httpx MockTransport.

1. Parsing core-api's listeners body (fail closed) and the interruption answer.
2. The pre-filter (name, team, project, topic, owned work, promise cues) and name redaction:
   the model never sees the listener's name.
3. The watch: roster and not-interruptible listeners skipped, the spend limit, BudgetBlocked,
   unparseable answers, the confidence threshold, the idempotency key, never-record on the way
   out, core-api's next_allowed_at pausing the listener, a confidential meeting.
4. The runtime: the watch runs after the off-the-record and never-record checks, beside the
   agents, and adds nothing to the results count; without it nothing changes.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import fakeredis
import httpx
import pytest

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.names import Namer, Postprocess
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.watch import (
    ABSENT,
    CachedWatchSource,
    HttpWatchClient,
    Interruption,
    Listener,
    ListenWatch,
    WatchSettings,
    parse_answer,
    parse_listeners,
    parse_verdict,
    prefilter,
    redact_name,
    render_window,
    watch_from_env,
    watch_key,
)
from tests.a2a_helpers import budget

ORG, MEETING = uuid.uuid4(), uuid.uuid4()
SUSHAN, MAYA, RAJ = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
LISTENER = {
    "user_id": str(SUSHAN),
    "name": "Sushan Kumar",
    "first_name": "Sushan",
    "teams": ["Growth"],
    "projects": ["Self-serve pricing"],
    "listen_for": ["pricing", "enterprise deals"],
    "owns": ["Pricing page copy"],
    "interruptible": True,
    "quiet_reason": None,
    "next_allowed_at": None,
}
YES = json.dumps(
    {
        "interrupt": True,
        "reason": "decision",
        "confidence": 0.88,
        "headline": "Pricing for your segment is being decided",
        "detail": "They are settling the SMB price; Participant 1 wants 15% off.",
        "topic": "SMB pricing",
        "quote": "Let's lock the SMB price at 15% off, [the absent person]'s segment.",
    }
)


def listener(**over: Any) -> Listener:
    found = parse_listeners(
        {"meeting_id": str(MEETING), "confidential": False, "listeners": [{**LISTENER, **over}]}
    )
    assert found is not None
    return found.listeners[0]


def window(*lines: tuple[uuid.UUID, str], start: int = 60_000) -> TranscriptWindow:
    turns = [
        Turn(
            speaker_id=str(who),
            text=text,
            start_ms=start + i * 4000,
            end_ms=start + i * 4000 + 3000,
        )
        for i, (who, text) in enumerate(lines)
    ]
    return TranscriptWindow(
        org_id=ORG,
        meeting_id=MEETING,
        window_id=f"{MEETING}:{start}",
        start_ms=start,
        end_ms=turns[-1].end_ms,
        turns=turns,
        labels={str(MAYA): "Participant 1", str(RAJ): "Participant 2"},
    )


PRICING = window(
    (MAYA, "Okay, let's lock the SMB price at 15% off, that's Sushan's segment."),
    (RAJ, "Agreed, ship it Monday."),
)


class Sink:
    async def usage(self, **kw: Any) -> None:
        pass


class LLM:
    def __init__(self, answer: str = YES) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return {
            "choices": [{"message": {"content": self.answer}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 60},
        }

    def user_text(self, i: int = 0) -> str:
        content = self.calls[i]["messages"][1]["content"]
        return content if isinstance(content, str) else "".join(b["text"] for b in content)

    def system_text(self, i: int = 0) -> str:
        return str(self.calls[i]["messages"][0]["content"][0]["text"])


class Core:
    """core-api's two internal routes, as an httpx MockTransport."""

    def __init__(self, body: dict[str, Any] | None = None, answer: dict[str, Any] | None = None):
        self.body = body or {
            "meeting_id": str(MEETING),
            "title": "Pricing review with Sushan",
            "confidential": False,
            "status": "live",
            "listeners": [LISTENER],
        }
        self.answer = answer or {"id": str(uuid.uuid4()), "status": "delivered", "reason": None}
        self.posts: list[dict[str, Any]] = []
        self.reads = 0
        self.headers: list[httpx.Headers] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if request.method == "GET" and request.url.path.endswith("/listeners"):
            self.reads += 1
            return httpx.Response(200, json=self.body)
        if request.method == "POST" and request.url.path.endswith("/interruptions"):
            self.posts.append(json.loads(request.content))
            return httpx.Response(200, json=self.answer)
        return httpx.Response(404)

    def client(self) -> HttpWatchClient:
        return HttpWatchClient("http://core", "tok", transport=httpx.MockTransport(self.handler))


def make(
    core: Core | None = None,
    llm: LLM | None = None,
    *,
    gate: Any = None,
    settings: WatchSettings | None = None,
    clock: Any = None,
) -> tuple[ListenWatch, Core, LLM, Any]:
    core = core or Core()
    llm = llm or LLM()
    redis = fakeredis.aioredis.FakeRedis()
    router = LiteLLMRouter(gate or budget(), Sink(), llm)
    watch = ListenWatch(
        PromptLoader(PROMPTS_DIR),
        router,
        source=CachedWatchSource(core.client()),
        redis=redis,
        settings=settings or WatchSettings(min_gap_s=60),
        **({"clock": clock} if clock else {}),
    )
    return watch, core, llm, redis


# --- 1. parsing ----------------------------------------------------------------------------


def test_listeners_parse_fails_closed() -> None:
    assert parse_listeners("nope") is None
    assert parse_listeners({"listeners": []}) is None  # no meeting id
    found = parse_listeners({"meeting_id": str(MEETING), "listeners": [LISTENER, {"x": 1}]})
    assert found is not None and found.confidential is True  # absent -> confidential
    (one,) = found.listeners
    assert one.first_name == "Sushan" and one.teams == ("Growth",)
    odd = listener(interruptible="yes", next_allowed_at="2026-11-11T15:00:00Z")
    assert odd.interruptible is False
    assert odd.next_allowed_at == datetime(2026, 11, 11, 15, tzinfo=UTC)
    assert listener(first_name="", name="Priya Shah").first_name == "Priya"


def test_answer_parse() -> None:
    a = parse_answer({"id": str(SUSHAN), "status": "held", "reason": "focus_block"})
    assert a is not None and a.status == "held" and a.reason == "focus_block"
    assert parse_answer({"id": "x"}) is None


@pytest.mark.parametrize(
    ("text", "interrupt", "parsed"),
    [
        (YES, True, True),
        ('{"interrupt": false}', False, True),
        ("```json\n" + YES + "\n```", True, True),
        ("sure! here you go", False, False),
        (
            '{"interrupt": true, "reason": "vibes", "confidence": 0.9, "headline": "x"}',
            False,
            False,
        ),
        (
            '{"interrupt": true, "reason": "promise", "confidence": "high", "headline": "x"}',
            False,
            False,
        ),
        (
            '{"interrupt": true, "reason": "question", "confidence": 0.9, "headline": ""}',
            False,
            False,
        ),
        ('{"interrupt": "yes"}', False, False),
    ],
)
def test_verdict_parse(text: str, interrupt: bool, parsed: bool) -> None:
    v = parse_verdict(text)
    assert v.interrupt is interrupt and v.parsed is parsed


# --- 2. pre-filter and redaction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kinds", "promise"),
    [
        ("Sushan will have the deck by Friday", {"name"}, True),
        ("let's ask Sushan Kumar about it", {"name"}, True),
        ("the Growth team can take the migration", {"team"}, True),
        ("where are we on self-serve pricing?", {"project", "topic"}, False),
        ("we need to settle enterprise deals this week", {"topic"}, False),
        ("is the copy for the pricing page done?", {"topic", "owns"}, False),
        ("we closed two big enterprise deals", {"topic"}, False),
        ("the weather is lovely, see you Monday", set(), False),
        ("Sushaned is not a word", set(), False),
    ],
)
def test_prefilter(text: str, kinds: set[str], promise: bool) -> None:
    hits = prefilter(text, listener())
    assert {h.kind for h in hits} == kinds
    assert any(h.promise for h in hits) is promise


def test_the_model_never_sees_the_name() -> None:
    me = listener()
    assert redact_name("Sushan Kumar's team, and sushan again", me) == (
        f"{ABSENT}'s team, and {ABSENT} again"
    )
    rendered = render_window(PRICING, me)
    assert "Sushan" not in rendered
    assert rendered.startswith("Participant 1: ") and "\nParticipant 2: " in rendered


# --- 3. the watch --------------------------------------------------------------------------


async def test_posts_an_interruption_with_the_window_key() -> None:
    watch, core, llm, _ = make()
    assert await watch.observe(PRICING, roster={MAYA, RAJ}) == 1
    (sent,) = core.posts
    assert sent["key"] == f"{PRICING.window_id}:{SUSHAN}"
    assert sent["user_id"] == str(SUSHAN) and sent["reason"] == "decision"
    assert sent["headline"] == "Pricing for your segment is being decided"
    assert sent["quote"] == "Let's lock the SMB price at 15% off, Sushan's segment."
    assert sent["start_ms"] == PRICING.start_ms and sent["end_ms"] == PRICING.end_ms
    assert sent["prompt"] == "interrupt_gate@v1" and sent["confidence"] == 0.88
    assert core.headers[0]["x-mytheus-org"] == str(ORG)
    assert core.headers[0]["x-mytheus-service"] == "agent-runtime-service"
    # nothing of the listener's name reached the model, title included
    assert "Sushan" not in llm.user_text() and "Kumar" not in llm.user_text()
    assert f"Pricing review with {ABSENT}" in llm.user_text()
    assert "Listens for: pricing; enterprise deals" in llm.user_text()
    assert "interrupt" in llm.system_text()


async def test_the_same_window_again_is_the_same_key() -> None:
    watch, core, _, _ = make(settings=WatchSettings(min_gap_s=0))
    await watch.observe(PRICING)
    await watch.observe(PRICING)
    assert len(core.posts) == 2 and core.posts[0]["key"] == core.posts[1]["key"]


async def test_roster_and_not_interruptible_listeners_cost_nothing() -> None:
    watch, core, llm, _ = make()
    assert await watch.observe(PRICING, roster={SUSHAN}) == 0  # joined after all
    later = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    for over in (
        {"interruptible": False, "quiet_reason": "focus_block"},
        {"next_allowed_at": later},
    ):
        core2 = Core({**Core().body, "listeners": [{**LISTENER, **over}]})
        w2, _, llm2, _ = make(core2)
        assert await w2.observe(PRICING) == 0 and llm2.calls == []
    assert llm.calls == [] and core.posts == []


@pytest.mark.parametrize(
    "body", [{"confidential": True}, {"confidential": None}, {"status": "ended"}]
)
async def test_confidential_or_ended_meeting_is_never_watched(body: dict[str, Any]) -> None:
    watch, core, llm, _ = make(Core({**Core().body, **body}))
    assert await watch.observe(PRICING) == 0
    assert llm.calls == [] and core.posts == []


async def test_prefilter_miss_calls_no_model() -> None:
    watch, core, llm, _ = make()
    small_talk = window((MAYA, "Great weekend?"), (RAJ, "Yes, hiking."))
    assert await watch.observe(small_talk) == 0
    assert llm.calls == [] and watch.last_runs[0].outcome == "prefilter_miss"


async def test_spend_limit_gap_and_cap() -> None:
    now = [1_000_000.0]
    watch, _, llm, redis = make(
        settings=WatchSettings(min_gap_s=60, max_calls=2), clock=lambda: now[0]
    )
    await watch.observe(PRICING)
    await watch.observe(PRICING)  # 10 s later would still be inside the gap
    assert len(llm.calls) == 1 and watch.last_runs[0].outcome == "limited"
    now[0] += 61
    await watch.observe(PRICING)
    now[0] += 61
    await watch.observe(PRICING)  # the per-meeting cap
    assert len(llm.calls) == 2
    key = watch_key(ORG, MEETING, SUSHAN)
    assert key.startswith(f"nexus:{ORG}:") and await redis.ttl(key) > 0


async def test_next_allowed_at_from_core_pauses_the_listener() -> None:
    until = datetime.now(UTC) + timedelta(minutes=20)
    core = Core(
        answer={
            "id": None,
            "status": "held",
            "reason": "throttled",
            "next_allowed_at": until.isoformat(),
        }
    )
    watch, _, llm, _ = make(core, settings=WatchSettings(min_gap_s=0))
    await watch.observe(PRICING)
    await watch.observe(PRICING)
    assert len(llm.calls) == 1 and watch.last_runs[0].outcome == "limited"


async def test_budget_blocked_is_silent() -> None:
    watch, core, llm, _ = make(gate=budget(consumed=10**8, quota=10, cap=10))
    assert await watch.observe(PRICING) == 0
    assert llm.calls == [] and core.posts == []
    assert watch.last_runs[0].outcome == "budget_blocked"


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        ('{"interrupt": false}', "no_interrupt"),
        ("I think it matters", "no_interrupt"),
        (json.dumps({**json.loads(YES), "confidence": 0.69}), "low_confidence"),
    ],
)
async def test_no_post_without_a_confident_interrupt(answer: str, outcome: str) -> None:
    watch, core, _, _ = make(llm=LLM(answer))
    assert await watch.observe(PRICING) == 0
    assert core.posts == [] and watch.last_runs[0].outcome == outcome


async def test_threshold_is_tunable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISTEN_WATCH_MIN_CONFIDENCE", "0.6")
    s = WatchSettings.from_env()
    assert s.min_confidence == 0.6
    watch, core, _, _ = make(
        llm=LLM(json.dumps({**json.loads(YES), "confidence": 0.65})), settings=s
    )
    assert await watch.observe(PRICING) == 1


async def test_never_record_and_names_on_the_way_out() -> None:
    namer = Namer({"participant 1": "Maya Patel"})
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("layoffs",)))
    watch, core, _, _ = make()
    await watch.observe(PRICING, post=Postprocess(namer, policy))
    assert core.posts[0]["detail"] == "They are settling the SMB price; Maya Patel wants 15% off."
    leaky = LLM(
        json.dumps({**json.loads(YES), "headline": "Layoffs for your team are being decided"})
    )
    watch2, core2, _, _ = make(llm=leaky)
    assert await watch2.observe(PRICING, post=Postprocess(namer, policy)) == 0
    assert core2.posts == [] and watch2.last_runs[0].outcome == "never_record"
    quote_leak = LLM(json.dumps({**json.loads(YES), "quote": "the layoffs list"}))
    watch3, core3, _, _ = make(llm=quote_leak)
    await watch3.observe(PRICING, post=Postprocess(namer, policy))
    assert core3.posts[0]["quote"] is None and core3.posts[0]["headline"]


async def test_core_unreachable_is_no_listeners_and_not_cached() -> None:
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    source = CachedWatchSource(
        HttpWatchClient("http://core", "t", transport=httpx.MockTransport(down))
    )
    assert await source.listeners(ORG, MEETING) is None
    interruption = Interruption(SUSHAN, "k", "decision", "h", 0.9)
    assert await source.interrupt(ORG, MEETING, interruption) is None


async def test_listeners_are_cached_then_dropped_after_a_post() -> None:
    watch, core, _, _ = make()
    await watch.observe(window((MAYA, "nice weather")))
    await watch.observe(window((MAYA, "nice weather")))
    assert core.reads == 1
    await watch.observe(PRICING)
    await watch.observe(window((MAYA, "nice weather")))
    assert core.reads == 2  # the post dropped the cache: core-api's throttle state is re-read


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    router = LiteLLMRouter(budget(), Sink(), LLM())
    loader = PromptLoader(PROMPTS_DIR)
    monkeypatch.delenv("CORE_URL", raising=False)
    assert watch_from_env(loader, router, fakeredis.aioredis.FakeRedis()) is None
    monkeypatch.setenv("CORE_URL", "http://core")
    monkeypatch.setenv("CORE_CALLER_TOKEN", "t")
    assert watch_from_env(loader, router, fakeredis.aioredis.FakeRedis()) is not None
    monkeypatch.setenv("LISTEN_WATCH", "0")
    assert watch_from_env(loader, router, fakeredis.aioredis.FakeRedis()) is None


# --- 4. the runtime ------------------------------------------------------------------------


class RecordingWatch:
    def __init__(self) -> None:
        self.seen: list[tuple[str, set[uuid.UUID]]] = []

    async def observe(self, window: TranscriptWindow, *, roster: Any = (), post: Any = None) -> int:
        self.seen.append((window.window_id, set(roster)))
        return 1


class NoSink:
    def __init__(self) -> None:
        self.results: list[Any] = []

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        pass

    async def delta(self, **kw: Any) -> None:
        pass


async def _ctx(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
    return {"role": "PM"}


async def test_runtime_runs_the_watch_after_the_privacy_checks() -> None:
    from libs.shared.off_record import off_record_key

    redis = fakeredis.aioredis.FakeRedis()
    watch = RecordingWatch()

    async def org(_org: uuid.UUID) -> CompanyContext:
        return CompanyContext(never_record_topics=("layoffs",))

    rt = AgentRuntime(
        sink=NoSink(), user_context=_ctx, agents=[], redis=redis, watch=watch, org_context=org
    )
    await rt.on_room_event(ORG, MEETING, "joined", str(MAYA))
    assert await rt.process_window(PRICING) == 0  # the watch's posts are not results
    assert watch.seen == [(PRICING.window_id, {MAYA, RAJ})]
    await redis.set(off_record_key(ORG, MEETING), "{}")
    await rt.process_window(window((MAYA, "pricing again"), start=200_000))
    await redis.delete(off_record_key(ORG, MEETING))
    await rt.process_window(window((MAYA, "about the layoffs and pricing"), start=300_000))
    assert len(watch.seen) == 1  # off the record and never record: never watched


async def test_runtime_without_a_watch_is_unchanged() -> None:
    rt = AgentRuntime(sink=NoSink(), user_context=_ctx, agents=[])
    assert await rt.process_window(PRICING) == 0
