"""`LiteLLMRouter.converse()` (ADR-0047 D2): multi-turn, tool-calling, streaming, with the same
guarantees as `complete()`. A fake streaming completion function replays LiteLLM-shaped chunks;
no model is called."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.router import BudgetBlocked, LiteLLMRouter

ORG, USER = uuid.uuid4(), uuid.uuid4()
MEETING = "0b9b5a3c-1234-4567-8901-426614174000"  # all-digit groups: card-shaped if unmasked


class Sink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.calls.append(kw)


def text_chunks(*pieces: str, finish: str = "stop") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"choices": [{"delta": {"content": p}, "finish_reason": None}]} for p in pieces
    ]
    out.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    out.append({"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 50}})
    return out


def tool_chunks(
    calls: list[tuple[str, str, dict[str, Any]]], thinking: bool = True, preamble: str = ""
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if thinking:
        for piece in ("I should look ", "at their tasks."):
            out.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "thinking_blocks": [
                                    {"type": "thinking", "thinking": piece, "signature": ""}
                                ]
                            }
                        }
                    ]
                }
            )
        out.append(
            {
                "choices": [
                    {
                        "delta": {
                            "thinking_blocks": [
                                {"type": "thinking", "thinking": "", "signature": "sig-abc"}
                            ]
                        }
                    }
                ]
            }
        )
    if preamble:
        out.append({"choices": [{"delta": {"content": preamble}}]})
    for i, (cid, name, args) in enumerate(calls):
        raw = json.dumps(args)
        out.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": cid,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        half = len(raw) // 2
        for part in (raw[:half], raw[half:]):
            out.append(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": i, "function": {"arguments": part}}]}}
                    ]
                }
            )
    out.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    out.append(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 80,
                "cache_read_input_tokens": 1500,
            },
        }
    )
    return out


class FakeLLM:
    def __init__(self, *rounds: list[dict[str, Any]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(kw)
        chunks = self.rounds.pop(0)

        async def gen() -> AsyncIterator[dict[str, Any]]:
            for c in chunks:
                yield c

        return gen()


def gate(consumed: int = 0, quota: int = 10**9, cap: int = 10**9) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": quota,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


TOOLS = [
    {"type": "function", "function": {"name": "list_my_open_actions", "parameters": {}}},
    {"type": "function", "function": {"name": "get_meeting_outcome", "parameters": {}}},
]


async def converse(router: LiteLLMRouter, messages: list[dict[str, Any]], **kw: Any) -> Any:
    return await router.converse(
        tier=kw.pop("tier", "extract"),
        system="You are a Myte.",
        messages=messages,
        tools=kw.pop("tools", TOOLS),
        org_id=ORG,
        user_id=USER,
        max_tokens=4096,
        effort=kw.pop("effort", "medium"),
        **kw,
    )


async def test_text_streams_live_and_the_result_carries_it():
    llm = FakeLLM(text_chunks("You have ", "two tasks [1]."))
    seen: list[str] = []

    async def on_delta(t: str) -> None:
        seen.append(t)

    out = await converse(
        LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "hi"}], on_delta=on_delta
    )
    assert seen == ["You have ", "two tasks [1]."]
    assert out.text == "You have two tasks [1]." and out.tool_calls == []
    assert out.finish_reason == "stop" and not out.truncated
    assert llm.calls[0]["stream"] is True
    assert llm.calls[0]["stream_options"] == {"include_usage": True}


async def test_tool_calls_are_accumulated_across_chunks_with_ids_and_json_args():
    llm = FakeLLM(
        tool_chunks(
            [
                ("toolu_1", "list_my_open_actions", {}),
                ("toolu_2", "get_meeting_outcome", {"meeting_id": MEETING}),
            ],
            preamble="One moment.",
        )
    )
    out = await converse(LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "q"}])
    assert [(c.id, c.name, c.arguments) for c in out.tool_calls] == [
        ("toolu_1", "list_my_open_actions", {}),
        ("toolu_2", "get_meeting_outcome", {"meeting_id": MEETING}),
    ]
    assert out.text == "One moment." and out.finish_reason == "tool_calls"
    msg = out.assistant_message()
    assert msg["role"] == "assistant" and msg["content"] == "One moment."
    assert json.loads(msg["tool_calls"][1]["function"]["arguments"]) == {"meeting_id": MEETING}
    assert msg["tool_calls"][0]["type"] == "function"


async def test_malformed_tool_arguments_are_flagged_not_raised():
    chunks = tool_chunks([("toolu_1", "get_meeting_outcome", {})], thinking=False)
    chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = '{"meeting'
    llm = FakeLLM(chunks)
    out = await converse(LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "q"}])
    assert out.tool_calls[0].parse_error and out.tool_calls[0].arguments == {}
    assert out.assistant_message()["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_thinking_blocks_round_trip_into_the_next_round():
    """Anthropic rejects a tool result whose tool-use turn lost its thinking block."""
    llm = FakeLLM(tool_chunks([("toolu_1", "list_my_open_actions", {})]), text_chunks("Done [1]."))
    router = LiteLLMRouter(gate(), Sink(), llm)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "what do I owe?"}]
    first = await converse(router, messages)
    assert first.thinking_blocks == [
        {"type": "thinking", "thinking": "I should look at their tasks.", "signature": "sig-abc"}
    ]
    messages += [
        first.assistant_message(),
        {"role": "tool", "tool_call_id": "toolu_1", "content": "[1] action | Ship | Ship it"},
    ]
    await converse(router, messages)
    sent = llm.calls[1]["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["thinking_blocks"][0]["signature"] == "sig-abc"
    assert llm.calls[1]["thinking"] == {"type": "adaptive"}
    assert llm.calls[1]["output_config"] == {"effort": "medium"}


async def test_redacted_thinking_and_signature_only_blocks_are_kept():
    chunks = [
        {
            "choices": [
                {"delta": {"thinking_blocks": [{"type": "redacted_thinking", "data": "xyz"}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"thinking_blocks": [{"type": "thinking", "signature": "only-sig"}]}}
            ]
        },
        *tool_chunks([("toolu_1", "list_my_open_actions", {})], thinking=False),
    ]
    out = await converse(
        LiteLLMRouter(gate(), Sink(), FakeLLM(chunks)), [{"role": "user", "content": "q"}]
    )
    assert out.thinking_blocks == [
        {"type": "redacted_thinking", "data": "xyz"},
        {"type": "thinking", "thinking": "", "signature": "only-sig"},
    ]


async def test_system_and_tools_are_cache_breakpoints_and_context_follows_uncached():
    llm = FakeLLM(text_chunks("ok"))
    await converse(
        LiteLLMRouter(gate(), Sink(), llm),
        [{"role": "user", "content": "q"}],
        system_context="Your name: Pip",
    )
    system = llm.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert system["content"][1] == {"type": "text", "text": "Your name: Pip"}
    tools = llm.calls[0]["tools"]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[0]
    assert "cache_control" not in TOOLS[-1]  # the caller's list is not mutated


async def test_user_turns_and_tool_results_are_redacted_and_ids_survive():
    llm = FakeLLM(text_chunks("ok"))
    await converse(
        LiteLLMRouter(gate(), Sink(), llm),
        [
            {"role": "user", "content": "mail maya@acme.com about it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": (
                    f"[1] person | Maya | maya@acme.com, 555-123-4567 | meeting_id={MEETING}"
                ),
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "card 4111 1111 1111 1111"}],
            },  # noqa: E501
        ],
        tools=[],
        effort=None,
    )
    sent = llm.calls[0]["messages"]
    assert "maya@acme.com" not in sent[1]["content"] and "[REDACTED_EMAIL]" in sent[1]["content"]
    tool = sent[3]["content"]
    assert "maya@acme.com" not in tool and "555-123-4567" not in tool
    assert MEETING in tool  # the id the model passes back is intact
    assert "4111" not in sent[4]["content"][0]["text"]


async def test_the_system_context_is_redacted_but_the_template_is_not():
    llm = FakeLLM(text_chunks("ok"))
    await converse(
        LiteLLMRouter(gate(), Sink(), llm),
        [{"role": "user", "content": "q"}],
        system_context="Instructions: cc boss@acme.com",
    )
    system = llm.calls[0]["messages"][0]["content"]
    assert system[0]["text"] == "You are a Myte."
    assert "boss@acme.com" not in system[1]["text"]


async def test_a_blocked_budget_raises_before_any_model_call():
    llm = FakeLLM(text_chunks("ok"))
    sink = Sink()
    with pytest.raises(BudgetBlocked):
        await converse(
            LiteLLMRouter(gate(consumed=200, quota=100, cap=100), sink, llm),
            [{"role": "user", "content": "q"}],
        )
    assert llm.calls == [] and sink.calls == []


async def test_a_throttled_round_answers_on_triage_without_thinking():
    llm = FakeLLM(text_chunks("ok"))
    out = await converse(
        LiteLLMRouter(gate(consumed=10, quota=5), Sink(), llm), [{"role": "user", "content": "q"}]
    )
    assert out.tier == "triage" and llm.calls[0]["model"].endswith("haiku-5-5")
    assert llm.calls[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in llm.calls[0]


async def test_triage_tier_keeps_thinking_off():
    llm = FakeLLM(text_chunks("ok"))
    await converse(
        LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "q"}], tier="triage"
    )
    assert llm.calls[0]["thinking"] == {"type": "disabled"}


async def test_thinking_is_not_switched_on_mid_turn_after_a_throttled_round():
    """A tool-use turn that started without thinking cannot continue with it (Anthropic)."""
    llm = FakeLLM(text_chunks("ok"))
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "none"},
    ]
    await converse(LiteLLMRouter(gate(), Sink(), llm), messages)
    # Sonnet 5.5 refuses `disabled` mid-turn (400 "send between_tools instead"); this was the
    # production failure after a first round where adaptive thinking chose not to think.
    assert llm.calls[0]["thinking"] == {"type": "between_tools"}
    assert "output_config" not in llm.calls[0]


async def test_exactly_one_metering_record_per_round_with_cache_aware_units():
    llm = FakeLLM(tool_chunks([("toolu_1", "list_my_open_actions", {})]), text_chunks("ok"))
    sink = Sink()
    router = LiteLLMRouter(gate(), sink, llm)
    first = await converse(router, [{"role": "user", "content": "q"}])
    await converse(router, [{"role": "user", "content": "q"}, first.assistant_message()])
    assert len(sink.calls) == 2
    assert {c["model_tier"] for c in sink.calls} == {"sonnet"}
    assert len({c["call_id"] for c in sink.calls}) == 2
    assert all(c["user_id"] == USER and c["org_id"] == ORG for c in sink.calls)
    assert first.usage.cache_read_tokens == 1500 and first.units == sink.calls[0]["tokens"]


async def test_a_stream_that_breaks_is_still_metered():
    async def broken(**_kw: Any) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
            raise RuntimeError("connection reset")

        return gen()

    sink = Sink()
    with pytest.raises(RuntimeError):
        await converse(LiteLLMRouter(gate(), sink, broken), [{"role": "user", "content": "q"}])
    assert len(sink.calls) == 1 and sink.calls[0]["tokens"] > 0


async def test_tool_choice_none_is_passed_and_the_workspace_header_rides_along(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    llm = FakeLLM(text_chunks("ok"))
    await converse(
        LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "q"}], tool_choice="none"
    )
    assert llm.calls[0]["tool_choice"] == "none"
    assert llm.calls[0]["extra_headers"] == {"anthropic-workspace-id": "wrkspc_test"}


async def test_a_length_stop_is_reported_as_truncated():
    llm = FakeLLM(text_chunks("cut", finish="length"))
    out = await converse(LiteLLMRouter(gate(), Sink(), llm), [{"role": "user", "content": "q"}])
    assert out.truncated


async def test_an_unmapped_tier_raises():
    with pytest.raises(ValueError):
        await converse(
            LiteLLMRouter(gate(), Sink(), FakeLLM()),
            [{"role": "user", "content": "q"}],
            tier="high_stakes",
        )


async def test_object_shaped_chunks_work_like_dicts():
    """LiteLLM yields objects with attributes, not dicts."""

    class Obj:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    chunks = [
        Obj(
            choices=[
                Obj(
                    delta=Obj(content="Hi", tool_calls=None, thinking_blocks=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        Obj(
            choices=[
                Obj(
                    delta=Obj(
                        content=None,
                        thinking_blocks=None,
                        tool_calls=[
                            Obj(
                                index=0,
                                id="toolu_9",
                                function=Obj(name="list_my_open_actions", arguments="{}"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Obj(prompt_tokens=10, completion_tokens=2),
        ),
    ]

    async def fn(**_kw: Any) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for c in chunks:
                yield c

        return gen()

    out = await converse(LiteLLMRouter(gate(), Sink(), fn), [{"role": "user", "content": "q"}])
    assert out.text == "Hi" and out.tool_calls[0].name == "list_my_open_actions"
    assert out.usage.input_tokens == 10
