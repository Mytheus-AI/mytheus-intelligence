"""rag-service's HTTP surface and `libs/shared/rag_client`. No infra.

`POST /api/rag/query` accepts a user's Clerk token or an agent's service credentials plus the
participant it asks for; anything else is 401. `rag_client` sends exactly that shape.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from libs.shared.rag_client import FakeRagClient, HttpRagClient, RagPassage, parse_passages
from services.rag_service import main as rag_main
from services.rag_service.clients import ScopeUnavailable
from services.rag_service.contracts import AnswerPassage, Citation, RagQueryRequest, RagResponse
from services.rag_service.domain import Caller
from tests.conftest import auth_headers

SECRET = "a" * 40
ORG, USER = uuid.uuid4(), uuid.uuid4()
CHUNK, MEETING = uuid.uuid4(), uuid.uuid4()


class Recording:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[Caller, RagQueryRequest]] = []
        self.fail = fail

    async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse:
        self.calls.append((caller, req))
        if self.fail:
            raise ScopeUnavailable("down")
        return RagResponse(
            answer_passages=[
                AnswerPassage(
                    n=1,
                    text="We ship on the 2nd.",
                    rerank_score=0.9,
                    citation=Citation(kind="transcript", meeting_id=MEETING, chunk_id=CHUNK),
                    meeting_title="Launch",
                )
            ]
        )


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.setenv(rag_main.CALLER_TOKENS_ENV, f"agent-runtime-service={SECRET}")
    svc = Recording()
    return TestClient(rag_main.build_app(svc)), svc  # type: ignore[arg-type]


def _agent(org=ORG, user: Any = USER, secret=SECRET, service="agent-runtime-service"):
    h = {
        "X-Mytheus-Service": service,
        "X-Mytheus-Service-Token": secret,
        "X-Mytheus-Org": str(org),
    }
    if user is not None:
        h["X-Mytheus-User"] = str(user)
    return h


def test_a_user_token_asks_as_that_clerk_user(api):
    client, svc = api
    r = client.post("/api/rag/query", json={"query": "launch date"}, headers=auth_headers(ORG))
    assert r.status_code == 200, r.text
    caller, req = svc.calls[0]
    assert caller.org_id == ORG and caller.clerk_user_id and caller.user_id is None
    assert req.generate is False


def test_an_agent_asks_as_the_participant(api):
    client, svc = api
    r = client.post("/api/rag/query", json={"query": "launch date"}, headers=_agent())
    assert r.status_code == 200
    assert svc.calls[0][0] == Caller(org_id=ORG, user_id=USER)


@pytest.mark.parametrize(
    "headers",
    [
        _agent(secret="b" * 40),
        _agent(user=None),
        _agent(user="not-a-uuid"),
        _agent(service="bff"),
        {},
    ],
)
def test_anything_else_is_401(api, headers):
    client, svc = api
    r = client.post("/api/rag/query", json={"query": "x"}, headers=headers)
    assert r.status_code == 401
    assert svc.calls == []


def test_scope_outage_is_503_not_an_empty_answer(monkeypatch):
    monkeypatch.setenv(rag_main.CALLER_TOKENS_ENV, f"agent-runtime-service={SECRET}")
    client = TestClient(rag_main.build_app(Recording(fail=True)))  # type: ignore[arg-type]
    r = client.post("/api/rag/query", json={"query": "x"}, headers=_agent())
    assert r.status_code == 503


def test_request_validation(api):
    client, _ = api
    assert client.post("/api/rag/query", json={"query": ""}, headers=_agent()).status_code == 422
    extra = {"query": "x", "org_id": str(uuid.uuid4())}
    assert client.post("/api/rag/query", json=extra, headers=_agent()).status_code == 422


# --- rag_client ------------------------------------------------------------------------


async def test_http_rag_client_sends_the_agent_shape(monkeypatch):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "answer_passages": [
                    {
                        "n": 1,
                        "text": "t",
                        "rerank_score": 0.5,
                        "citation": {
                            "kind": "decision",
                            "decision_id": str(CHUNK),
                            "meeting_id": str(MEETING),
                        },
                    }
                ]
            },
        )

    real = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    client = HttpRagClient("http://rag", token=SECRET)
    out = await client.retrieve(org_id=ORG, user_id=USER, query="q")
    assert out == [
        RagPassage(n=1, text="t", score=0.5, kind="decision", meeting_id=MEETING, decision_id=CHUNK)
    ]
    assert seen["headers"]["x-mytheus-user"] == str(USER)
    assert seen["headers"]["x-mytheus-service"] == "agent-runtime-service"
    assert b'"generate":false' in seen["body"].replace(b" ", b"")


async def test_http_rag_client_never_raises():
    assert (
        await HttpRagClient("http://127.0.0.1:9", token=SECRET, timeout_s=0.2).retrieve(
            org_id=ORG, user_id=USER, query="q"
        )
        == []
    )
    assert (
        await HttpRagClient("http://rag", token="").retrieve(org_id=ORG, user_id=USER, query="q")
        == []
    )


async def test_fake_rag_client_records_calls():
    fake = FakeRagClient(passages=[RagPassage(n=1, text="t", score=1.0, kind="transcript")])
    assert len(await fake.retrieve(org_id=ORG, user_id=USER, query="q")) == 1
    assert fake.calls[0]["user_id"] == USER


def test_parse_passages_tolerates_missing_optional_fields():
    [p] = parse_passages(
        {
            "answer_passages": [
                {
                    "n": 2,
                    "text": "x",
                    "rerank_score": 0.1,
                    "citation": {"kind": "document", "document_id": str(CHUNK), "element_id": "el"},
                }
            ]
        }
    )
    assert p.kind == "document" and p.element_id == "el" and p.meeting_id is None
