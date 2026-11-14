"""The MCP client of Mytheus's own server: bff `POST /mcp` (mytheus-docs ADR-0047 D3).

Myte calls the tools AS THE PERSON: every request forwards the asker's own `Authorization`
header, so a tool can never show Myte more than the same token sees through REST. bff's server
is stateless Streamable HTTP with JSON responses (`services/bff/mcp.py` in mytheus-backend): no
`initialize` handshake or `Mcp-Session-Id` is needed, every POST stands alone. We still send
`MCP-Protocol-Version` (a version bff negotiates) and the `Accept` header the transport requires,
and read an SSE-framed answer too in case a server chooses to stream one.

Errors:
- HTTP 401 / 403: the session is gone or not allowed -> `McpSessionError` (`error.reason =
  "session"`).
- A network failure, a timeout, HTTP 5xx / 404 / 400, an unreadable body, a JSON-RPC error on
  `tools/list` -> `McpUnavailable`.
- A tool that ran and failed (`isError: true`) or a JSON-RPC error on `tools/call` (an unknown
  tool, invalid params) is a `ToolOutput(is_error=True)`: the model sees the message and can
  recover, as MCP intends.

Env: `BFF_URL` (default `http://bff:8000`).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_BFF_URL = "http://bff:8000"
TOOLS_TTL_S = 300.0
_ids = itertools.count(1)


def bff_url() -> str:
    return os.environ.get("BFF_URL", DEFAULT_BFF_URL).rstrip("/")


class McpSessionError(Exception):
    """bff refused the person's token (401) or their role (403)."""

    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


class McpUnavailable(Exception):
    """bff could not be reached or did not answer as an MCP server."""


@dataclass(frozen=True)
class ToolOutput:
    text: str
    structured: Any
    is_error: bool


class McpClient:
    """One asker's view of bff's MCP server. `http` is a shared `httpx.AsyncClient` (pooled
    connections); `authorization` is the asker's header, forwarded verbatim."""

    def __init__(self, http: httpx.AsyncClient, authorization: str, base_url: str) -> None:
        self._http = http
        self._auth = authorization
        self._url = f"{base_url.rstrip('/')}/mcp"

    async def list_tools(self, timeout_s: float = 10.0) -> list[dict[str, Any]]:
        message = await self._rpc("tools/list", {}, timeout_s)
        if "error" in message:
            raise McpUnavailable(f"tools/list failed: {message['error']}")
        tools = (message.get("result") or {}).get("tools")
        if not isinstance(tools, list):
            raise McpUnavailable("tools/list returned no tool list")
        return [t for t in tools if isinstance(t, dict) and t.get("name")]

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float) -> ToolOutput:
        message = await self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout_s)
        if "error" in message:
            err = message["error"] if isinstance(message["error"], dict) else {}
            return ToolOutput(str(err.get("message") or "The tool failed."), None, True)
        result = message.get("result") or {}
        text = "\n".join(
            str(part.get("text", ""))
            for part in result.get("content") or []
            if isinstance(part, dict) and part.get("type") == "text"
        )
        return ToolOutput(text, result.get("structuredContent"), bool(result.get("isError")))

    async def _rpc(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        req_id = next(_ids)
        try:
            res = await self._http.post(
                self._url,
                json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
                headers={
                    "Authorization": self._auth,
                    "MCP-Protocol-Version": PROTOCOL_VERSION,
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                timeout=max(0.1, timeout_s),
            )
        except httpx.HTTPError as exc:
            raise McpUnavailable(f"{method}: {type(exc).__name__}") from None
        if res.status_code in (401, 403):
            raise McpSessionError(f"{method}: HTTP {res.status_code}", res.status_code)
        if res.status_code != 200:
            raise McpUnavailable(f"{method}: HTTP {res.status_code}")
        message = _parse_body(res, req_id)
        if message is None:
            raise McpUnavailable(f"{method}: unreadable response")
        return message


def _parse_body(res: httpx.Response, req_id: int) -> dict[str, Any] | None:
    ctype = res.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        for event in res.text.split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
            )
            if not data:
                continue
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("id") == req_id:
                return message
        return None
    try:
        message = res.json()
    except ValueError:
        return None
    return message if isinstance(message, dict) else None


class ToolCatalog:
    """bff's tool list, cached per process for five minutes (the list is the same for every
    caller; any caller's token can refresh it). Concurrent misses share one fetch."""

    def __init__(self, ttl_s: float = TOOLS_TTL_S) -> None:
        self._ttl = ttl_s
        self._tools: list[dict[str, Any]] | None = None
        self._at = 0.0
        self._lock = asyncio.Lock()

    async def get(self, client: McpClient) -> list[dict[str, Any]]:
        if self._tools is not None and time.monotonic() - self._at < self._ttl:
            return self._tools
        async with self._lock:
            if self._tools is not None and time.monotonic() - self._at < self._ttl:
                return self._tools
            self._tools = await client.list_tools()
            self._at = time.monotonic()
            return self._tools

    def clear(self) -> None:
        self._tools = None
