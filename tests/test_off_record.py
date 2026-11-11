"""Off the record, enforced in this tier (defense in depth behind transcription-service).

1. `is_off_record`: key present = off; absent = on; a Redis error counts as off (privacy wins).
2. Agent runtime: a window whose processing starts off the record runs no agent, calls no model
   and emits nothing; without the flag (or without redis injected) it runs as before.
3. `IndexGuard.check`: `off_record` before any other work (no company brain, no privacy call).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.index_guard import IndexGuard
from libs.shared.off_record import is_off_record, off_record_key
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.runtime import AgentRuntime
from services.agent_runtime_service.windowing import Windower
from tests.test_agent_runtime import (
    MEETING,
    ORG,
    FakeLLM,
    Sink,
    TwoStepAgent,
    budget_gate,
    chunk,
)


@pytest.fixture
def prompts(tmp_path: Path) -> PromptLoader:
    (tmp_path / "registry.json").write_text(json.dumps({"defaults": {"two_step": "v1"}}))
    (tmp_path / "two_step@v1.md").write_text("You are a test agent.\n---user---\n{window}")
    return PromptLoader(tmp_path)


class BrokenRedis:
    async def exists(self, *_a: Any) -> int:
        raise ConnectionError("redis down")


async def set_off(redis: Any, org: uuid.UUID = ORG, meeting: uuid.UUID = MEETING) -> None:
    value = {"segment_id": str(uuid.uuid4()), "started_at": "2026-11-11T10:00:00Z"}
    await redis.set(off_record_key(org, meeting), json.dumps(value), ex=86400)


def test_key_shape() -> None:
    assert off_record_key(ORG, MEETING) == f"nexus:{ORG}:off-record:{MEETING}"


async def test_flag_present_absent_and_unreadable() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    assert await is_off_record(redis, ORG, MEETING) is False
    await set_off(redis)
    assert await is_off_record(redis, ORG, MEETING) is True
    assert await is_off_record(redis, uuid.uuid4(), MEETING) is False  # another tenant
    await redis.delete(off_record_key(ORG, MEETING))
    assert await is_off_record(redis, ORG, MEETING) is False
    assert await is_off_record(BrokenRedis(), ORG, MEETING) is True


# --- 2. agent runtime ------------------------------------------------------------------------


def _runtime(prompts: Any, redis: Any) -> tuple[AgentRuntime, Sink, FakeLLM]:
    sink, llm = Sink(), FakeLLM()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), sink, llm)

    async def ctx(_org: uuid.UUID, _user: uuid.UUID) -> dict[str, Any]:
        return {"role": "Engineer"}

    rt = AgentRuntime(
        sink=sink,
        user_context=ctx,
        agents=[lambda: TwoStepAgent(prompts, router)],
        redis=redis,
    )
    return rt, sink, llm


def _window(user: uuid.UUID) -> Any:
    w = Windower()
    w.add(chunk(0, str(user)))
    (cw,) = w.close_meeting(ORG, MEETING)
    return cw.window


@pytest.mark.parametrize("redis_kind", ["flag", "broken"])
async def test_runtime_runs_nothing_off_the_record(prompts: Any, redis_kind: str) -> None:
    redis: Any = fakeredis.aioredis.FakeRedis() if redis_kind == "flag" else BrokenRedis()
    if redis_kind == "flag":
        await set_off(redis)
    rt, sink, llm = _runtime(prompts, redis)
    user = uuid.uuid4()
    await rt.on_room_event(ORG, MEETING, "joined", str(user))
    assert await rt.process_window(_window(user)) == 0
    assert llm.calls == [] and sink.results == [] and sink.usage_calls == []


async def test_runtime_runs_again_on_the_record(prompts: Any) -> None:
    redis = fakeredis.aioredis.FakeRedis()
    await set_off(redis)
    rt, sink, llm = _runtime(prompts, redis)
    user = uuid.uuid4()
    await rt.on_room_event(ORG, MEETING, "joined", str(user))
    assert await rt.process_window(_window(user)) == 0
    await redis.delete(off_record_key(ORG, MEETING))
    assert await rt.process_window(_window(user)) == 1
    assert llm.calls and len(sink.results) == 1


async def test_runtime_without_redis_is_unchanged(prompts: Any) -> None:
    rt, sink, _llm = _runtime(prompts, None)
    user = uuid.uuid4()
    await rt.on_room_event(ORG, MEETING, "joined", str(user))
    assert await rt.process_window(_window(user)) == 1


# --- 3. IndexGuard ---------------------------------------------------------------------------


async def test_index_guard_skips_off_the_record_before_anything_else() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    asked: list[str] = []

    async def org(_o: uuid.UUID) -> CompanyContext:
        asked.append("org")
        return CompanyContext(never_record_topics=("salaries",))

    async def privacy(_o: uuid.UUID, _m: uuid.UUID) -> bool | None:
        asked.append("privacy")
        return False

    guard = IndexGuard(redis=redis, org_context=org, privacy=privacy)
    ok = await guard.check(ORG, MEETING, 0, "The roadmap is fine.")
    assert ok.index and ok.reason == "ok"
    asked.clear()

    await set_off(redis)
    verdict = await guard.check(ORG, MEETING, 31_000, "The roadmap is fine.")
    assert verdict.skip and verdict.reason == "off_record" and verdict.purge is None
    assert asked == []  # no company brain, no privacy lookup, no classifier

    other = await guard.check(ORG, uuid.uuid4(), 0, "The roadmap is fine.")
    assert other.index  # only that meeting is off the record

    broken = IndexGuard(redis=BrokenRedis(), org_context=org, privacy=privacy)
    assert (await broken.check(ORG, MEETING, 0, "x")).reason == "off_record"
