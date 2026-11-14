"""Ask Myte's MCP client of bff `POST /mcp` (ADR-0047 D3): the person's own token is forwarded,
the protocol headers are sent, and failures map to the stream's error reasons."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from services.rag_service.myte.mcp import (
    PROTOCOL_VERSION,
    McpClient,
    McpSessionError,
    McpUnavailable,
    ToolCatalog,
)

AUTH = "Bearer user-clerk-token"


def client(handler: Any) -> McpClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return McpClient(http, AUTH, "http://bff:8000")


def rpc_result(request: httpx.Request, result: dict[str, Any]) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


async def test_tools_call_forwards_the_persons_token_and_the_protocol_headers():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return rpc_result(
            request,
            {
                "content": [{"type": "text", "text": "- Ship it `a1`"}],
                "structuredContent": {"items": [{"id": "a1", "content": "Ship it"}]},
                "isError": False,
            },
        )

    out = await client(handler).call_tool("list_my_open_actions", {}, 5.0)
    req = seen[0]
    assert str(req.url) == "http://bff:8000/mcp" and req.method == "POST"
    assert req.headers["authorization"] == AUTH
    assert req.headers["mcp-protocol-version"] == PROTOCOL_VERSION == "2025-06-18"
    assert req.headers["accept"] == "application/json, text/event-stream"
    assert "mcp-session-id" not in req.headers and "origin" not in req.headers
    body = json.loads(req.content)
    assert body["jsonrpc"] == "2.0" and body["method"] == "tools/call"
    assert body["params"] == {"name": "list_my_open_actions", "arguments": {}}
    assert out.text == "- Ship it `a1`" and not out.is_error
    assert out.structured == {"items": [{"id": "a1", "content": "Ship it"}]}


async def test_a_tool_error_is_a_result_the_model_sees():
    def handler(request: httpx.Request) -> httpx.Response:
        return rpc_result(
            request,
            {
                "content": [
                    {"type": "text", "text": "Not found, or not from a meeting you took part in."}
                ],
                "isError": True,
            },
        )

    out = await client(handler).call_tool("get_meeting_outcome", {"meeting_id": "x"}, 5.0)
    assert out.is_error and "Not found" in out.text and out.structured is None


async def test_a_jsonrpc_error_on_tools_call_is_a_tool_error():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32602, "message": "Unknown tool 'nope'"},
            },
        )

    out = await client(handler).call_tool("nope", {}, 5.0)
    assert out.is_error and "Unknown tool" in out.text


@pytest.mark.parametrize("status", [401, 403])
async def test_a_refused_token_is_a_session_error(status: int):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "Unauthenticated"})

    with pytest.raises(McpSessionError):
        await client(handler).call_tool("list_my_open_actions", {}, 5.0)


@pytest.mark.parametrize("status", [400, 404, 500, 502, 503])
async def test_other_http_failures_are_unavailable(status: int):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="nope")

    with pytest.raises(McpUnavailable):
        await client(handler).call_tool("list_my_open_actions", {}, 5.0)


async def test_a_network_failure_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(McpUnavailable):
        await client(handler).list_tools()


async def test_an_unreadable_body_is_unavailable():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>", headers={"content-type": "text/html"})

    with pytest.raises(McpUnavailable):
        await client(handler).list_tools()


async def test_an_sse_framed_answer_is_read():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        msg = {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": [{"name": "ping_me"}]}}
        text = f"event: message\ndata: {json.dumps(msg)}\n\n"
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    assert [t["name"] for t in await client(handler).list_tools()] == ["ping_me"]


async def test_a_jsonrpc_error_on_tools_list_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -1, "message": "x"}}
        )

    with pytest.raises(McpUnavailable):
        await client(handler).list_tools()


async def test_the_tool_catalogue_is_cached_and_refreshed_after_its_ttl():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return rpc_result(request, {"tools": [{"name": "search_meetings"}]})

    mcp = client(handler)
    catalog = ToolCatalog(ttl_s=300)
    await catalog.get(mcp)
    await catalog.get(mcp)
    assert calls == 1
    expired = ToolCatalog(ttl_s=0)
    await expired.get(mcp)
    await expired.get(mcp)
    assert calls == 3
