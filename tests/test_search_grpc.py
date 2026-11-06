"""search-service internal gRPC FTS arm. No infra: a real in-process `grpc.aio` server, the DB
query (`fts_search`) replaced by a fake, and the JWT verifier injected. The SQL/RLS path itself is
covered by tests/test_search_routes.py (same `fts_search`); a DB-backed gRPC test is UNVERIFIED
until infra is up."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import grpc
import pytest
from fastapi import HTTPException

from libs.shared import rag_proto
from services.search_service import grpc_server
from services.search_service.fts import FtsRow

ORG = uuid.uuid4()
PROTO = (Path(__file__).parent.parent / "proto" / "rag.proto").read_text()


def test_runtime_descriptors_match_rag_proto():
    type_map = {"string": "string", "int32": "int32", "int64": "int64", "double": "double"}
    for msg in ("FtsSearchRequest", "FtsHit", "FtsSearchResponse"):
        body = re.search(rf"message {msg} \{{(.*?)\n\}}", PROTO, re.S).group(1)
        declared = re.findall(r"^\s*(repeated )?(\w+) (\w+) = (\d+);", body, re.M)
        cls = getattr(rag_proto, msg)
        fields = {f.name: f for f in cls.DESCRIPTOR.fields}
        assert len(declared) == len(fields)
        for rep, typ, name, num in declared:
            f = fields[name]
            assert f.number == int(num)
            assert (
                f.is_repeated if hasattr(f, "is_repeated") else f.label == f.LABEL_REPEATED
            ) == bool(rep)
            if typ in type_map:
                assert (
                    f.type
                    == {
                        "string": f.TYPE_STRING,
                        "int32": f.TYPE_INT32,
                        "int64": f.TYPE_INT64,
                        "double": f.TYPE_DOUBLE,
                    }[typ]
                )
            else:
                assert f.message_type.name == typ
    assert "service FtsService" in PROTO and "rpc FtsSearch" in PROTO


def _verify(token: str) -> uuid.UUID:
    if token == "good":
        return ORG
    if token == "down":
        raise HTTPException(status_code=503, detail="Identity provider unavailable")
    raise HTTPException(status_code=401, detail="Invalid token")


@pytest.fixture()
async def stub(monkeypatch):
    calls: list[dict] = []

    async def fake_fts(**kw):
        calls.append(kw)
        return [FtsRow("hello world", 0.5, "c1", str(uuid.uuid4()), "spk", 10, 20)]

    monkeypatch.setattr(grpc_server, "fts_search", fake_fts)
    server = grpc.aio.server()
    servicer = grpc_server.FtsServicer(session_factory=None, verify=_verify)  # type: ignore[arg-type]
    grpc_server.add_fts_servicer(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    call = channel.unary_unary(
        rag_proto.FTS_SEARCH_METHOD,
        request_serializer=rag_proto.FtsSearchRequest.SerializeToString,
        response_deserializer=rag_proto.FtsSearchResponse.FromString,
    )
    yield call, calls
    await channel.close()
    await server.stop(None)


def _md(token: str):
    return (("authorization", f"Bearer {token}"),)


async def test_search_returns_hits_for_verified_org(stub):
    call, calls = stub
    resp = await call(rag_proto.FtsSearchRequest(query="hello"), metadata=_md("good"))
    assert resp.query == "hello" and resp.results[0].passage == "hello world"
    assert calls[0]["org_id"] == ORG and calls[0]["limit"] == 20 and calls[0]["meeting_id"] is None


@pytest.mark.parametrize("md", [(), _md("bad")])
async def test_missing_or_bad_token_is_unauthenticated(stub, md):
    call, calls = stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await call(rag_proto.FtsSearchRequest(query="x"), metadata=md)
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert calls == []


async def test_idp_outage_is_unavailable(stub):
    call, _ = stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await call(rag_proto.FtsSearchRequest(query="x"), metadata=_md("down"))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


@pytest.mark.parametrize(
    "req",
    [
        {"query": "  "},
        {"query": "x", "limit": 51},
        {"query": "x", "limit": -1},
        {"query": "x", "meeting_id": "nope"},
    ],
)
async def test_invalid_arguments(stub, req):
    call, calls = stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await call(rag_proto.FtsSearchRequest(**req), metadata=_md("good"))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert calls == []
