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
    type_map = {"string", "int32", "int64", "double", "bool"}
    messages = re.findall(r"^message (\w+) \{", PROTO, re.M)
    assert set(messages) == {
        "FtsSearchRequest",
        "FtsHit",
        "FtsSearchResponse",
        "DocumentRef",
        "GetPassagesRequest",
        "TranscriptPassage",
        "DocumentPassage",
        "GetPassagesResponse",
    }
    for msg in messages:
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
                        "bool": f.TYPE_BOOL,
                    }[typ]
                )
            else:
                assert f.message_type.name == typ
    assert "service FtsService" in PROTO and "rpc FtsSearch" in PROTO
    assert "rpc GetPassages" in PROTO


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


# --- rag-service's service path, scope and GetPassages (mytheus-docs ADR-0033 D1, D4) ---

SECRET = "q" * 40


def _svc(org: uuid.UUID = ORG, secret: str = SECRET):
    return (
        ("x-mytheus-service", "rag-service"),
        ("x-mytheus-service-token", secret),
        ("x-mytheus-org", str(org)),
    )


@pytest.fixture()
async def svc_stub(monkeypatch):
    monkeypatch.setenv(grpc_server.CALLER_TOKENS_ENV, f"rag-service={SECRET}")
    calls: list[dict] = []

    async def fake_fts(**kw):
        calls.append(kw)
        return []

    async def fake_passages(**kw):
        calls.append(kw)
        from services.search_service.fts import TranscriptPassageRow

        return (
            [TranscriptPassageRow("c1", "m1", "u1", 1, 2, "hit", "before", "after")],
            [],
        )

    monkeypatch.setattr(grpc_server, "fts_search", fake_fts)
    monkeypatch.setattr(grpc_server, "get_passages", fake_passages)
    server = grpc.aio.server()
    servicer = grpc_server.FtsServicer(session_factory=None, verify=_verify)  # type: ignore[arg-type]
    grpc_server.add_fts_servicer(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    search = channel.unary_unary(
        rag_proto.FTS_SEARCH_METHOD,
        request_serializer=rag_proto.FtsSearchRequest.SerializeToString,
        response_deserializer=rag_proto.FtsSearchResponse.FromString,
    )
    passages = channel.unary_unary(
        rag_proto.GET_PASSAGES_METHOD,
        request_serializer=rag_proto.GetPassagesRequest.SerializeToString,
        response_deserializer=rag_proto.GetPassagesResponse.FromString,
    )
    yield search, passages, calls
    await channel.close()
    await server.stop(None)


async def test_service_caller_scopes_the_search(svc_stub):
    search, _, calls = svc_stub
    m = str(uuid.uuid4())
    req = rag_proto.FtsSearchRequest(query="x", scoped=True, meeting_ids=[m], speaker_ids=["u"])
    await search(req, metadata=_svc())
    assert calls[0]["org_id"] == ORG
    assert calls[0]["meeting_ids"] == [uuid.UUID(m)]
    assert calls[0]["speaker_ids"] == ["u"]


async def test_scoped_with_no_meetings_is_an_empty_allowlist_not_unfiltered(svc_stub):
    search, _, calls = svc_stub
    await search(rag_proto.FtsSearchRequest(query="x", scoped=True), metadata=_svc())
    assert calls[0]["meeting_ids"] == []


async def test_unscoped_search_passes_no_allowlist(stub):
    call, calls = stub
    await call(rag_proto.FtsSearchRequest(query="x"), metadata=_md("good"))
    assert calls[0]["meeting_ids"] is None


async def test_wrong_service_secret_is_unauthenticated(svc_stub):
    search, _, calls = svc_stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await search(rag_proto.FtsSearchRequest(query="x"), metadata=_svc(secret="z" * 40))
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert calls == []


async def test_get_passages_returns_text_and_neighbours(svc_stub):
    _, passages, calls = svc_stub
    c = str(uuid.uuid4())
    resp = await passages(rag_proto.GetPassagesRequest(chunk_ids=[c], neighbors=1), metadata=_svc())
    assert resp.transcripts[0].text == "hit"
    assert resp.transcripts[0].before == "before" and resp.transcripts[0].after == "after"
    assert calls[0]["chunk_ids"] == [uuid.UUID(c)] and calls[0]["neighbors"] == 1


async def test_get_passages_refuses_a_user_token(svc_stub):
    _, passages, calls = svc_stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await passages(rag_proto.GetPassagesRequest(chunk_ids=[]), metadata=_md("good"))
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert calls == []


@pytest.mark.parametrize(
    "req",
    [{"chunk_ids": ["nope"]}, {"neighbors": 4}, {"chunk_ids": [str(uuid.uuid4())] * 51}],
)
async def test_get_passages_invalid_arguments(svc_stub, req):
    _, passages, calls = svc_stub
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await passages(rag_proto.GetPassagesRequest(**req), metadata=_svc())
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert calls == []
