"""Polish plan, Phase 0: model settings.

1. Effort is deliberate per call: extraction `low`, synthesis `medium`, adaptive thinking on
   Sonnet; triage keeps thinking off; a throttled call drops the effort with the thinking.
2. `max_tokens` leaves room for thinking (8192 on every extract call) and is not the budget
   estimate; a response that stops at `max_tokens` is reported, never cached.
3. Two cache breakpoints: the system prompt, and the org-stable head of the user message (the
   company block, the never-record line, topic owners).
4. Every default prompt's system part is past the 512-token cache minimum.
5. Budgets are metered in dollar-weighted units (`libs/shared/cost.py`).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.cost import budget_units, usd
from libs.shared.prompts import PromptLoader
from libs.shared.router import EXTRACT_MAX_TOKENS, LiteLLMRouter, Usage
from services.agent_runtime_service.consumer import PROMPTS_DIR

ORG = uuid.uuid4()


class Sink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)


class LLM:
    def __init__(self, usage: dict[str, Any] | None = None, finish: str = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self.usage = usage or {"prompt_tokens": 1000, "completion_tokens": 100}
        self.finish = finish

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": self.finish}],
            "usage": self.usage,
        }


def gate(consumed: int = 0, quota: int = 10**9, cap: int = 10**9) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": quota,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


# --- 1. effort -------------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["low", "medium"])
async def test_an_extract_call_thinks_adaptively_at_the_effort_it_names(effort: Any):
    llm = LLM()
    await LiteLLMRouter(gate(), Sink(), llm).complete(
        tier="extract",
        system="s",
        user="u",
        org_id=ORG,
        user_id=None,
        max_tokens=8192,
        effort=effort,
    )
    assert llm.calls[0]["thinking"] == {"type": "adaptive"}
    assert llm.calls[0]["output_config"] == {"effort": effort}


async def test_triage_keeps_thinking_off_and_a_throttled_call_drops_the_effort():
    llm = LLM()
    router = LiteLLMRouter(gate(consumed=10, quota=5), Sink(), llm)  # throttled
    out = await router.complete(
        tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=50, effort="low"
    )
    assert out.tier == "triage" and llm.calls[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in llm.calls[0]


async def test_an_extract_call_without_an_effort_sends_no_thinking_settings():
    """rag-service's answer call names no effort: unchanged from before (the model default)."""
    llm = LLM()
    await LiteLLMRouter(gate(), Sink(), llm).complete(
        tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=50
    )
    assert "thinking" not in llm.calls[0] and "output_config" not in llm.calls[0]


# --- 2. max_tokens ---------------------------------------------------------------------------


async def test_headroom_for_thinking_does_not_block_a_tenant_near_its_cap():
    """The estimate was `max_tokens + len(user)//3` raw tokens: 8192 of headroom would have
    blocked a tenant 8,000 units below its cap for tokens it never spends."""
    llm = LLM()
    router = LiteLLMRouter(gate(consumed=92_000, quota=10**9, cap=100_000), Sink(), llm)
    await router.complete(
        tier="extract",
        system="s" * 3000,
        user="u" * 3000,
        org_id=ORG,
        user_id=None,
        max_tokens=EXTRACT_MAX_TOKENS,
        effort="low",
    )
    assert len(llm.calls) == 1


async def test_a_truncated_answer_is_reported_and_never_cached(monkeypatch: Any):
    from libs.shared import router as router_module
    from libs.shared.llm_cache import ResponseCache

    warned: list[str] = []
    monkeypatch.setattr(router_module.logger, "warning", lambda msg, *a: warned.append(msg % a))
    redis = fakeredis.aioredis.FakeRedis()
    llm = LLM(finish="length")
    router = LiteLLMRouter(gate(), Sink(), llm, cache=ResponseCache(redis))
    for _ in range(2):
        out = await router.complete(
            tier="extract",
            system="s",
            user="u",
            org_id=ORG,
            user_id=None,
            max_tokens=10,
            prompt_ref="p@v1",
        )
        assert out.truncated
    assert len(llm.calls) == 2  # the cut answer was not stored
    assert len(warned) == 2 and "stopped at max_tokens" in warned[0] and "p@v1" in warned[0]


# --- 3. two cache breakpoints ----------------------------------------------------------------


async def test_the_org_stable_head_of_the_user_message_has_its_own_breakpoint():
    llm = LLM()
    router = LiteLLMRouter(gate(), Sink(), llm)
    await router.complete(
        tier="extract",
        system="sys",
        user="Transcript:\nP1: hi",
        cached_prefix="Company context:\n<company_context>x</company_context>\n\n",
        org_id=ORG,
        user_id=None,
        max_tokens=10,
        effort="low",
    )
    system, user = llm.calls[0]["messages"]
    assert system["content"][0]["cache_control"] == {"type": "ephemeral"}
    head, tail = user["content"]
    assert head["cache_control"] == {"type": "ephemeral"} and "company_context" in head["text"]
    assert "cache_control" not in tail and tail["text"].startswith("Transcript")
    # no brain: a plain string, exactly as before
    await router.complete(
        tier="triage", system="s", user="u", org_id=ORG, user_id=None, max_tokens=5
    )
    assert llm.calls[-1]["messages"][1]["content"] == "u"


def test_render_parts_splits_only_the_leading_org_stable_placeholders():
    loader = PromptLoader(PROMPTS_DIR)
    action = loader.load("action_agent", "v2")
    values = {
        "company": "C\n\n",
        "ground_rules": "Never record (company ground rule): x\n\n",
        "owners": "Topics with a named owner:\nT1. billing\n\n",
        "today": "Monday 2026-11-09",
        "window": "Participant 1: hi",
    }
    prefix, rest = action.render_parts(**values)
    assert prefix == values["company"] + values["ground_rules"] + values["owners"]
    assert rest.startswith("Today is Monday") and prefix + rest == action.render_user(**values)
    gate_v1 = loader.load("decision_gate", "v1")  # starts with the transcript: no prefix
    assert gate_v1.render_parts(window="w") == ("", gate_v1.render_user(window="w"))


# --- 4. every system prompt caches -----------------------------------------------------------


def test_every_default_system_prompt_is_past_the_512_token_cache_minimum():
    """Claude Haiku 5.5 and Sonnet 5.5 cache a prefix of 512 tokens or more; a shorter one is
    silently uncached. English prose runs about 4 characters per token, so 2,400 characters is a
    margin over 512 tokens, not an exact count."""
    registry = json.loads((Path(PROMPTS_DIR) / "registry.json").read_text())["defaults"]
    loader = PromptLoader(PROMPTS_DIR)
    short = {k: len(loader.load(k, v).system) for k, v in registry.items()}
    assert {k: n for k, n in short.items() if n < 2400} == {}


# --- 5. dollar-weighted budget units ---------------------------------------------------------


def test_budget_units_weigh_each_token_by_its_tier_and_cache_price():
    # one unit = one uncached Sonnet 5.5 input token ($2 / MTok)
    assert budget_units("extract", input_tokens=1000, output_tokens=0) == 1000
    assert budget_units("extract", input_tokens=0, output_tokens=100) == 500
    assert budget_units("triage", input_tokens=1000, output_tokens=100) == 75  # 50 + 25
    assert budget_units("extract", input_tokens=0, output_tokens=0, cache_read_tokens=1000) == 100
    assert budget_units("extract", input_tokens=0, output_tokens=0, cache_write_tokens=1000) == 1250
    assert budget_units("triage", input_tokens=3, output_tokens=0) == 1  # rounded up
    assert usd("extract", input_tokens=1_000_000, output_tokens=0) == pytest.approx(2.0)


async def test_the_router_meters_units_and_reads_the_cache_counts():
    sink = Sink()
    usage = {
        "prompt_tokens": 3000,  # litellm counts cache reads and writes in prompt tokens
        "completion_tokens": 200,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 500,
    }
    out = await LiteLLMRouter(gate(), sink, LLM(usage)).complete(
        tier="extract", system="s", user="u", org_id=ORG, user_id=None, max_tokens=10
    )
    # 500 uncached + 500 x 1.25 written + 2000 x 0.1 read + 200 x 5 out = 2325
    assert sink.usage_calls[0]["tokens"] == out.units == 2325
    assert (out.cache_read_tokens, out.cache_write_tokens) == (2000, 500)
    assert Usage(input_tokens=10, output_tokens=0).units("triage") == 1
