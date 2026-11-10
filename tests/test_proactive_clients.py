"""The clients `ProactiveAgent` depends on (P11 contract sections 2 and 3).

1. `GrpcKgClient` against the REAL `KnowledgeGraphServicer` on a local gRPC port (only the AGE
   template call is stubbed): service-token metadata and org reach the server, nodes decode; a
   wrong token, an unreachable address and a server slower than the 800 ms deadline all raise
   `KgUnavailable` (the R1 trigger) within the deadline.
2. `PlanFeaturesClient`: path, caller headers, 5-minute per-org cache (failures cached too),
   fail closed.
3. `HttpCoreContext`: the two internal routes, caller headers, tolerant parsing, 404 -> None.
4. `RagDecisionSearch` over `HttpRagClient` asks rag-service for `source: "decisions"` and keeps
   decision citations only.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import grpc
import httpx
import pytest

from libs.shared.kg_client import GrpcKgClient, KgUnavailable, kg_client_from_env
from libs.shared.plan_features import PlanFeaturesClient
from libs.shared.rag_client import HttpRagClient
from services.agent_runtime_service.agents.proactive_agent import RagDecisionSearch
from services.agent_runtime_service.proactive.core import HttpCoreContext, core_context_from_env
from services.knowledge_graph_service import grpc_server
from services.knowledge_graph_service.templates import GraphNode

ORG = uuid.uuid4()
TOKEN = "a" * 40


# --- knowledge graph gRPC ------------------------------------------------------------------


@asynccontextmanager
async def _no_conn() -> AsyncIterator[Any]:
    yield None


async def _serve(monkeypatch: pytest.MonkeyPatch, delay: float = 0.0) -> tuple[Any, int, list]:
    monkeypatch.setenv("MYTHEUS_KG_CALLER_TOKENS", f"agent-runtime-service={TOKEN}")
    seen: list[tuple[Any, str, dict[str, Any]]] = []

    async def fake_run_template(conn: Any, org_id: Any, name: str, params: Any) -> list[GraphNode]:
        seen.append((org_id, name, dict(params)))
        if delay:
            await asyncio.sleep(delay)
        return [
            GraphNode("Decision", str(uuid.uuid4()), "ancestor", 2, {"content": "Older one"}),
            GraphNode("Decision", params.get("decision_id", ""), "subject", 0, {}),
        ]

    monkeypatch.setattr(grpc_server, "run_template", fake_run_template)
    server = grpc.aio.server()
    grpc_server.add_knowledge_graph_servicer(
        grpc_server.KnowledgeGraphServicer(graph_connect=_no_conn), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, port, seen


async def test_kg_client_authenticates_and_decodes_nodes(monkeypatch):
    server, port, seen = await _serve(monkeypatch)
    client = GrpcKgClient(f"127.0.0.1:{port}", TOKEN)
    try:
        did = str(uuid.uuid4())
        nodes = await client.query(ORG, "decision_lineage", {"decision_id": did})
        assert [(n.role, n.depth) for n in nodes] == [("ancestor", 2), ("subject", 0)]
        assert nodes[0].properties == {"content": "Older one"} and nodes[1].id == did
        assert seen == [(ORG, "decision_lineage", {"decision_id": did})]  # org from metadata

        with pytest.raises(KgUnavailable):
            await GrpcKgClient(f"127.0.0.1:{port}", "b" * 40).query(ORG, "decision_lineage", {})
    finally:
        await client.close()
        await server.stop(None)


async def test_kg_client_times_out_at_the_r1_deadline(monkeypatch):
    server, port, _ = await _serve(monkeypatch, delay=3.0)
    client = GrpcKgClient(f"127.0.0.1:{port}", TOKEN)  # default 800 ms
    try:
        start = time.monotonic()
        with pytest.raises(KgUnavailable):
            await client.query(ORG, "decision_lineage", {"decision_id": str(uuid.uuid4())})
        assert 0.7 < time.monotonic() - start < 1.5
    finally:
        await client.close()
        await server.stop(None)


async def test_kg_client_unreachable_and_env(monkeypatch):
    client = GrpcKgClient("127.0.0.1:1", TOKEN, timeout_ms=300)
    with pytest.raises(KgUnavailable):
        await client.query(ORG, "decision_lineage", {})
    await client.close()
    monkeypatch.delenv("KG_GRPC_ADDR", raising=False)
    assert kg_client_from_env() is None
    monkeypatch.setenv("KG_GRPC_ADDR", "knowledge-graph-grpc:50104")
    monkeypatch.setenv("KG_CALLER_TOKEN", TOKEN)
    assert isinstance(kg_client_from_env(), GrpcKgClient)


# --- billing plan features -----------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def test_plan_features_are_read_once_per_org_per_5_minutes():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"plan_code": "enterprise", "features": {"proactive_agent": True}}
        )

    clock = Clock()
    client = PlanFeaturesClient(
        "http://billing:8000", token=TOKEN, clock=clock, transport=httpx.MockTransport(handler)
    )
    assert await client.enabled(ORG, "proactive_agent")
    assert await client.enabled(ORG, "proactive_agent")
    clock.now += 299
    assert await client.enabled(ORG, "proactive_agent")
    assert client.fetches == 1
    clock.now += 2
    assert await client.enabled(ORG, "proactive_agent")
    assert client.fetches == 2
    other = uuid.uuid4()
    await client.enabled(other, "proactive_agent")
    assert client.fetches == 3
    r = requests[0]
    assert r.method == "GET" and r.url.path == "/internal/plan-features"
    assert r.headers["X-Mytheus-Service"] == "agent-runtime-service"
    assert r.headers["X-Mytheus-Service-Token"] == TOKEN
    assert r.headers["X-Mytheus-Org"] == str(ORG)
    assert not r.url.query  # the org is never in the path or the query
    assert not await client.enabled(ORG, "context_gap_agent")  # absent is off


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, json={"plan_code": "free"}),
        httpx.Response(200, json={"features": {"proactive_agent": "yes"}}),
    ],
)
async def test_plan_features_fail_closed_and_cache_the_failure(response: httpx.Response):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    client = PlanFeaturesClient(
        "http://billing", token=TOKEN, clock=Clock(), transport=httpx.MockTransport(handler)
    )
    assert not await client.enabled(ORG, "proactive_agent")
    assert not await client.enabled(ORG, "proactive_agent")
    assert calls == 1
    assert not await PlanFeaturesClient("http://billing", token="").enabled(ORG, "x")


# --- core-api ------------------------------------------------------------------------------


async def test_core_context_routes_headers_and_parsing():
    action, meeting, a, b = (uuid.uuid4() for _ in range(4))
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == f"/internal/actions/{action}/context":
            return httpx.Response(
                200,
                json={
                    "assignee_id": str(a),
                    "assignee_name": "priya@acme.example",
                    "meeting_id": str(meeting),
                    "organizer_id": None,
                    "open_action_count": 7,
                    "depends_on": [{"id": str(b), "content": "x", "status": "done"}, {"id": 3}],
                    "participants": [str(a), str(b), "not-a-uuid", str(a)],
                    "admins": [str(b)],
                },
            )
        if request.url.path == f"/internal/meetings/{meeting}/participants":
            return httpx.Response(
                200,
                json={
                    "participants": [
                        {"user_id": str(a), "name": "priya@acme.example", "rbac_tier": "admin"},
                        {"user_id": str(b), "name": None, "rbac_tier": None},
                    ]
                },
            )
        return httpx.Response(404)

    core = HttpCoreContext("http://core:8000", TOKEN, transport=httpx.MockTransport(handler))
    ctx = await core.action_context(ORG, action)
    assert ctx is not None and ctx.open_action_count == 7 and ctx.assignee_id == a
    assert ctx.participants == [a, b] and ctx.admins == [b] and ctx.organizer_id is None
    assert [d.id for d in ctx.depends_on] == [b]
    people = await core.participants(ORG, meeting)
    assert people is not None and [(p.user_id, p.rbac_tier) for p in people] == [
        (a, "admin"),
        (b, None),
    ]
    assert await core.action_context(ORG, uuid.uuid4()) is None  # 404: unknown or cross-tenant
    for r in seen:
        assert r.headers["X-Mytheus-Service"] == "agent-runtime-service"
        assert r.headers["X-Mytheus-Service-Token"] == TOKEN
        assert r.headers["X-Mytheus-Org"] == str(ORG) and not r.url.query


def test_core_context_from_env(monkeypatch):
    monkeypatch.delenv("CORE_URL", raising=False)
    assert core_context_from_env() is None
    monkeypatch.setenv("CORE_URL", "http://core-api:8000")
    monkeypatch.setenv("CORE_CALLER_TOKEN", TOKEN)
    assert isinstance(core_context_from_env(), HttpCoreContext)


# --- rag-service decision-only search ------------------------------------------------------


async def test_rag_decision_search_forces_the_decisions_source(monkeypatch):
    user, decision = uuid.uuid4(), uuid.uuid4()
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["X-Mytheus-User"] == str(user)
        cite = {"kind": "decision", "decision_id": str(decision), "meeting_id": str(user)}
        action = {"kind": "action", "action_id": str(uuid.uuid4()), "meeting_id": str(user)}
        return httpx.Response(
            200,
            json={
                "answer_passages": [
                    {"n": 1, "text": "Launch in January.", "rerank_score": 0.8, "citation": cite},
                    {"n": 2, "text": "Write the brief.", "rerank_score": 0.7, "citation": action},
                ]
            },
        )

    real = httpx.AsyncClient

    def patched(*args: Any, **kw: Any) -> httpx.AsyncClient:
        return real(*args, transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    search = RagDecisionSearch(HttpRagClient("http://rag:8201", token=TOKEN))
    found = await search.search(org_id=ORG, user_id=user, query="When do we launch?")
    assert [p.decision_id for p in found] == [decision]
    assert bodies == [{"query": "When do we launch?", "generate": False, "source": "decisions"}]
