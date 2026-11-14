"""Test kit for Ask Myte (ADR-0047): a fake bff MCP server, a scripted model, an SSE parser and
the offline case runner the eval harness and the route tests share.

The fake bff answers `tools/list` with the real catalogue's shape (names, annotations, the
input properties the loop reads) and `tools/call` from per-tool fixtures, recording every call
with its headers. The scripted model replays rounds as LiteLLM stream chunks (with signed
thinking blocks, so the thinking round trip is exercised); in a final text, `[[needle]]` is
replaced by the number of the latest source line containing `needle` that the loop actually
gave the model, so citations are checked through the real plumbing.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import fakeredis
import httpx

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.rag_service.main import PROMPTS_DIR
from services.rag_service.myte.loop import LoopConfig, MyteLoop
from services.rag_service.myte.mcp import ToolCatalog

_UUID = {"type": "string", "format": "uuid"}
_RO = {"readOnlyHint": True, "openWorldHint": False}
_RW = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}


def _tool(name: str, props: dict[str, Any], annotations: dict[str, Any] = _RO) -> dict[str, Any]:
    return {
        "name": name,
        "title": name.replace("_", " "),
        "description": f"{name} (test catalogue)",
        "inputSchema": {"type": "object", "properties": props, "additionalProperties": False},
        "annotations": annotations,
    }


BFF_TOOLS: list[dict[str, Any]] = [
    _tool("search_meetings", {"query": {"type": "string"}, "limit": {"type": "integer"}}),
    _tool("ask_meetings", {"question": {"type": "string"}, "meeting_id": _UUID}),
    _tool("get_meeting_outcome", {"meeting_id": _UUID}),
    _tool("get_decision_record", {"decision_id": _UUID}),
    _tool("company_as_of", {"date": {"type": "string"}, "topic": {"type": "string"}}),
    _tool("list_my_open_actions", {}),
    _tool("list_my_promises", {"status": {"type": "string"}}),
    _tool("check_against_decisions", {"text": {"type": "string"}}),
    _tool(
        "check_my_capacity",
        {
            "due_date": {"type": "string"},
            "days": {"type": "integer"},
            "timezone": {"type": "string"},
        },
    ),
    _tool("list_my_schedule", {"days": {"type": "integer"}}),
    _tool("find_people", {"query": {"type": "string"}, "limit": {"type": "integer"}}),
    _tool("list_my_inbox", {"limit": {"type": "integer"}}),
    _tool("get_my_myte", {}),
    _tool("add_promise", {"content": {"type": "string"}}, _RW),
    _tool("create_action", {"content": {"type": "string"}}, _RW),
]
WRITE_TOOLS = {"create_action", "add_promise"}

# A fixture value: structuredContent (dict), ("text", str), ("error", str), ("http", status),
# ("slow", seconds) or ("down", None).
Fixture = Any


@dataclass
class BffCall:
    name: str
    arguments: dict[str, Any]
    authorization: str


@dataclass
class FakeBff:
    fixtures: dict[str, Fixture] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=lambda: list(BFF_TOOLS))
    list_status: int = 200
    calls: list[BffCall] = field(default_factory=list)
    lists: int = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        import asyncio

        body = json.loads(request.content)
        auth = request.headers.get("authorization", "")
        if body["method"] == "tools/list":
            self.lists += 1
            if self.list_status != 200:
                return httpx.Response(self.list_status, json={"detail": "x"})
            return _ok(body, {"tools": self.tools})
        name = body["params"]["name"]
        args = body["params"].get("arguments") or {}
        self.calls.append(BffCall(name, args, auth))
        fx = self.fixtures.get(name, ("text", f"No results from {name}."))
        if isinstance(fx, list | tuple) and len(fx) == 2 and isinstance(fx[0], str):
            kind, value = fx
            if kind == "text":
                return _ok(body, {"content": [{"type": "text", "text": value}], "isError": False})
            if kind == "error":
                return _ok(body, {"content": [{"type": "text", "text": value}], "isError": True})
            if kind == "http":
                return httpx.Response(int(value), json={"detail": "x"})
            if kind == "slow":
                await asyncio.sleep(float(value))
                return _ok(body, {"content": [{"type": "text", "text": "late"}], "isError": False})
            if kind == "down":
                raise httpx.ConnectError("down", request=request)
        if isinstance(fx, dict) and "__text__" in fx:
            result: dict[str, Any] = {
                "content": [{"type": "text", "text": fx["__text__"]}],
                "isError": False,
            }
            if fx.get("__structured__") is not None:
                result["structuredContent"] = fx["__structured__"]
            return _ok(body, result)
        return _ok(
            body,
            {
                "content": [{"type": "text", "text": f"{name} result"}],
                "structuredContent": fx,
                "isError": False,
            },
        )


def _ok(body: dict[str, Any], result: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


_NEEDLE = re.compile(r"\[\[([^\]]+)\]\]")
_LINE_N = re.compile(r"^\[(\d+)\] ")


@dataclass
class ScriptedModel:
    """`turns`: each a dict, {"tools": [{"tool": name, "args": {...}}], "text": preamble?} or
    {"text": final}. Followups (the non-streaming triage call) answer `followups`."""

    turns: list[dict[str, Any]]
    followups: str = json.dumps(
        {"followups": ["Who was there?", "What's left?", "Add a task for it"], "title": "Test"}
    )
    calls: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if not kw.get("stream"):
            return {
                "choices": [{"message": {"content": self.followups}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            }
        self._check_round(kw)
        turn = self.turns.pop(0) if self.turns else {"text": "(the script ended)"}
        chunks = self._chunks(turn, kw)

        async def gen() -> AsyncIterator[dict[str, Any]]:
            for c in chunks:
                yield c

        return gen()

    def _check_round(self, kw: dict[str, Any]) -> None:
        names = {t["function"]["name"] for t in kw.get("tools") or []}
        if names & WRITE_TOOLS:
            self.problems.append(f"write tools offered to the model: {sorted(names & WRITE_TOOLS)}")
        thinking = (kw.get("thinking") or {}).get("type") == "adaptive"
        for m in kw["messages"]:
            if thinking and m.get("role") == "assistant" and m.get("tool_calls"):
                if not m.get("thinking_blocks"):
                    self.problems.append("a tool-use turn lost its thinking blocks")

    def _chunks(self, turn: dict[str, Any], kw: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if (kw.get("thinking") or {}).get("type") == "adaptive" and turn.get("tools"):
            out.append(
                _delta(thinking_blocks=[{"type": "thinking", "thinking": "plan", "signature": ""}])
            )
            out.append(
                _delta(thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "s1"}])
            )
        text = self._resolve(str(turn.get("text") or ""), kw["messages"])
        for piece in _split(text):
            out.append(_delta(content=piece))
        for i, call in enumerate(turn.get("tools") or []):
            cid = f"toolu_{uuid.uuid4().hex[:10]}"
            raw = call.get("raw") or json.dumps(call.get("args") or {})
            out.append(
                _delta(
                    tool_calls=[
                        {
                            "index": i,
                            "id": cid,
                            "type": "function",
                            "function": {"name": call["tool"], "arguments": ""},
                        }
                    ]
                )
            )
            out.append(_delta(tool_calls=[{"index": i, "function": {"arguments": raw}}]))
        finish = "tool_calls" if turn.get("tools") else turn.get("finish", "stop")
        out.append({"choices": [{"delta": {}, "finish_reason": finish}]})
        out.append({"choices": [], "usage": {"prompt_tokens": 1200, "completion_tokens": 60}})
        return out

    @staticmethod
    def _resolve(text: str, messages: list[dict[str, Any]]) -> str:
        lines = [
            line
            for m in messages
            if m.get("role") == "tool"
            for line in str(m.get("content") or "").splitlines()
            if _LINE_N.match(line)
        ]

        def sub(match: re.Match[str]) -> str:
            needle = match.group(1).lower()
            for line in reversed(lines):
                if needle in line.lower():
                    m = _LINE_N.match(line)
                    assert m is not None
                    return f"[{m.group(1)}]"
            return "[?]"

        return _NEEDLE.sub(sub, text)


def _delta(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"delta": delta, "finish_reason": None}]}


def _split(text: str) -> list[str]:
    if not text:
        return []
    words = text.split(" ")
    return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words) if w or i]


def gate(consumed: int = 0, quota: int = 10**9, cap: int = 10**9) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": quota,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


class Sink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.calls.append(kw)


FIXED_NOW = datetime(2026, 11, 14, 15, 30, tzinfo=UTC)


def make_loop(
    model: Callable[..., Any],
    bff: FakeBff,
    *,
    budget: BudgetGate | None = None,
    sink: Sink | None = None,
    config: LoopConfig | None = None,
) -> MyteLoop:
    router = LiteLLMRouter(budget or gate(), sink or Sink(), model)
    http = httpx.AsyncClient(transport=httpx.MockTransport(bff.handler))
    return MyteLoop(
        router,
        PromptLoader(PROMPTS_DIR),
        http,
        catalog=ToolCatalog(),
        base_url="http://bff:8000",
        config=config,
        clock=lambda: FIXED_NOW,
    )


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if name:
            events.append((name, json.loads("\n".join(data))))
    return events


def check_stream(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Contract problems in one answer's event sequence (ADR-0047 D8); [] when it is valid."""
    problems: list[str] = []
    names = [n for n, _ in events]
    if not names or names[-1] not in ("done", "error"):
        problems.append(f"the stream does not end with done or error: {names[-3:]}")
    if names.count("done") + names.count("error") != 1:
        problems.append("exactly one done or error is expected")
    steps: dict[str, list[str]] = {}
    expected_n = 1
    for name, data in events:
        if name == "step":
            if set(data) != {"id", "label", "state"} or data["state"] not in (
                "running",
                "done",
                "failed",
            ):
                problems.append(f"bad step {data}")
            steps.setdefault(data["id"], []).append(data["state"])
        elif name == "source":
            # New numbers come 1..k in order; a fuller version of a known source re-sends its n.
            if not isinstance(data.get("n"), int) or not 1 <= data["n"] <= expected_n:
                problems.append(f"source numbers must be 1..k in order, got {data.get('n')}")
            elif data["n"] == expected_n:
                expected_n += 1
            if data.get("kind") not in (
                "transcript",
                "decision",
                "action",
                "promise",
                "meeting",
                "document",
                "person",
                "notification",
            ):
                problems.append(f"bad source kind {data.get('kind')}")
            if not isinstance(data.get("title"), str) or not isinstance(data.get("text"), str):
                problems.append("a source needs title and text")
        elif name == "delta":
            if set(data) != {"text"}:
                problems.append("bad delta")
        elif name == "proposal":
            if not {"id", "kind", "title"} <= set(data):
                problems.append(f"bad proposal {data}")
        elif name == "card":
            if set(data) != {"kind", "source_ns", "data"}:
                problems.append(f"bad card {data}")
        elif name == "followups":
            if len(data.get("items", [])) != 3:
                problems.append("followups must hold three items")
        elif name == "title":
            if not 0 < len(data.get("text", "")) <= 60:
                problems.append("bad title")
        elif name == "done":
            if set(data) != {"abstained", "degraded", "cited"}:
                problems.append(f"bad done {data}")
        elif name == "error":
            if data.get("reason") not in (
                "budget",
                "not_connected",
                "unavailable",
                "invalid",
                "session",
            ):
                problems.append(f"bad error {data}")
        else:
            problems.append(f"unknown event {name}")
    for sid, states in steps.items():
        if states[0] != "running" or len(states) != 2 or states[1] not in ("done", "failed"):
            problems.append(f"step {sid} went {states}")
    return problems
