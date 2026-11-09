"""knowledge-graph-service read surfaces: `POST /graph/query/{template}`, `POST /graph/cypher`
and the gRPC `KnowledgeGraphService`, against real AGE (skips like the other live AGE tests).

Covers service authentication (fail closed), the tenant coming only from `X-Mytheus-Org`, the
`$org_id` rejection on `/graph/cypher`, and the proto mirror matching `proto/knowledge_graph.proto`.
"""

from __future__ import annotations

import json
import os
import re
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import grpc
import pytest
from fastapi.testclient import TestClient

from libs.shared import kg_proto
from libs.shared.age import AgeConnection
from services.knowledge_graph_service.confirmed import ConfirmedDecision, apply_decision
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.graph import ensure_graph
from services.knowledge_graph_service.grpc_server import (
    KnowledgeGraphServicer,
    add_knowledge_graph_servicer,
)
from services.knowledge_graph_service.main import app
from services.knowledge_graph_service.routes import graph_connection

_DSN = os.environ.get(
    "AGE_DATABASE_URL", "postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph"
)
SECRET = "s" * 40
PROTO = (Path(__file__).parent.parent / "proto" / "knowledge_graph.proto").read_text()


def test_runtime_descriptors_match_knowledge_graph_proto() -> None:
    types = {"string": "TYPE_STRING", "int32": "TYPE_INT32"}
    for msg in ("TemplateParam", "QueryTemplateRequest", "GraphNode", "QueryTemplateResponse"):
        body = re.search(rf"message {msg} \{{(.*?)\n\}}", PROTO, re.S)
        assert body, msg
        declared = re.findall(r"^\s*(repeated )?(\w+) (\w+) = (\d+);", body.group(1), re.M)
        fields = {f.name: f for f in getattr(kg_proto, msg).DESCRIPTOR.fields}
        assert len(declared) == len(fields), msg
        for rep, typ, name, num in declared:
            f = fields[name]
            assert f.number == int(num)
            assert f.is_repeated == bool(rep)
            if typ in types:
                assert f.type == getattr(f, types[typ])
            else:
                assert f.message_type.name == typ
    assert "service KnowledgeGraphService" in PROTO and "rpc QueryTemplate" in PROTO


def _age_up() -> bool:
    u = urlparse(_DSN)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=1):
            return True
    except OSError:
        return False


needs_age = pytest.mark.skipif(
    not _age_up() and os.environ.get("MYTHEUS_REQUIRE_AGE") != "1",
    reason=f"LIVE AGE TEST NOT RUN: no AGE at {_DSN}",
)


@pytest.fixture()
def tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYTHEUS_KG_CALLER_TOKENS", f"bff={SECRET},rag-service={'r' * 40}")


@asynccontextmanager
async def _connect() -> AsyncIterator[AgeConnection]:
    async with age_connection(_DSN) as conn:
        yield conn


async def _seed_chain(org: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    m, d1, d2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = "2026-11-09T10:00:00+00:00"
    async with _connect() as conn:
        await ensure_graph(conn)
        await apply_decision(conn, org, ConfirmedDecision(d1, m, "v1", 0.9, now))
        await apply_decision(conn, org, ConfirmedDecision(d2, m, "v2", 0.9, now, supersedes=d1))
    return d1, d2


@pytest.fixture()
def client(tokens: None) -> TestClient:
    async def conn_override() -> AsyncIterator[AgeConnection]:
        async with _connect() as conn:
            yield conn

    app.dependency_overrides[graph_connection] = conn_override
    yield TestClient(app)
    app.dependency_overrides.clear()


def _h(org: uuid.UUID, service: str = "bff", token: str = SECRET) -> dict[str, str]:
    return {
        "X-Mytheus-Service": service,
        "X-Mytheus-Service-Token": token,
        "X-Mytheus-Org": str(org),
    }


@needs_age
def test_service_auth_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    org = uuid.uuid4()
    url = "/graph/query/decision_lineage"
    body = {"params": {"decision_id": str(uuid.uuid4())}}
    assert client.post(url, json=body).status_code == 401
    assert client.post(url, json=body, headers=_h(org, token="x" * 40)).status_code == 401
    assert client.post(url, json=body, headers=_h(org, service="core-api")).status_code == 401
    bad_org = {**_h(org), "X-Mytheus-Org": "not-a-uuid"}
    assert client.post(url, json=body, headers=bad_org).status_code == 401
    assert client.post(url, json=body, headers=_h(org)).status_code == 200
    monkeypatch.delenv("MYTHEUS_KG_CALLER_TOKENS")
    assert client.post(url, json=body, headers=_h(org)).status_code == 401


@needs_age
async def test_lineage_over_http_is_scoped_to_the_header_org(client: TestClient) -> None:
    org = uuid.uuid4()
    d1, d2 = await _seed_chain(org)
    body = {"params": {"decision_id": str(d2)}}
    res = client.post("/graph/query/decision_lineage", json=body, headers=_h(org))
    assert res.status_code == 200
    nodes = res.json()["nodes"]
    assert [(n["id"], n["role"]) for n in nodes] == [(str(d1), "ancestor"), (str(d2), "subject")]
    assert nodes[0]["properties"]["status"] == "superseded"

    other = client.post("/graph/query/decision_lineage", json=body, headers=_h(uuid.uuid4()))
    assert other.status_code == 200 and other.json()["nodes"] == []


@needs_age
def test_bad_requests(client: TestClient) -> None:
    org = uuid.uuid4()
    assert client.post("/graph/query/nope", json={}, headers=_h(org)).status_code == 404
    bad = {"params": {"decision_id": "x"}}
    assert (
        client.post("/graph/query/decision_lineage", json=bad, headers=_h(org)).status_code == 422
    )
    no_org = {"cypher": "MATCH (d:Decision) RETURN d", "columns": ["d"]}
    assert client.post("/graph/cypher", json=no_org, headers=_h(org)).status_code == 422
    write = {"cypher": "MATCH (d:Decision {org_id: $org_id}) SET d.x = 1 RETURN d"}
    assert client.post("/graph/cypher", json=write, headers=_h(org)).status_code == 422
    ok = {"cypher": "MATCH (d:Decision {org_id: $org_id}) RETURN count(d)", "columns": ["n"]}
    res = client.post("/graph/cypher", json=ok, headers=_h(org))
    assert res.status_code == 200 and res.json()["rows"] == [[0]]


@needs_age
async def test_grpc_query_template(tokens: None) -> None:
    org = uuid.uuid4()
    d1, d2 = await _seed_chain(org)
    server = grpc.aio.server()
    add_knowledge_graph_servicer(KnowledgeGraphServicer(graph_connect=_connect), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(
                kg_proto.QUERY_TEMPLATE_METHOD,
                request_serializer=kg_proto.QueryTemplateRequest.SerializeToString,
                response_deserializer=kg_proto.QueryTemplateResponse.FromString,
            )
            req = kg_proto.QueryTemplateRequest(
                template="decision_lineage",
                params=[kg_proto.TemplateParam(name="decision_id", value=str(d1))],
            )
            meta = [(k.lower(), v) for k, v in _h(org, "rag-service", "r" * 40).items()]
            res = await call(req, metadata=meta)
            assert [(n.id, n.role) for n in res.nodes] == [
                (str(d1), "subject"),
                (str(d2), "descendant"),
            ]
            assert json.loads(res.nodes[1].properties_json)["content"] == "v2"

            with pytest.raises(grpc.aio.AioRpcError) as err:
                await call(req, metadata=[("x-mytheus-org", str(org))])
            assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED
            with pytest.raises(grpc.aio.AioRpcError) as err:
                await call(kg_proto.QueryTemplateRequest(template="nope"), metadata=meta)
            assert err.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await server.stop(None)
