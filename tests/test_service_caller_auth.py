"""`verified_or_service_org_id`: rag-service may call `/embed` and `/rerank` as itself
(mytheus-docs ADR-0033), and a user's token keeps working. No infra."""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from libs.shared.auth import verified_or_service_org_id
from services.embedding_service import query_embed
from services.reranker_service import rerank
from tests.conftest import auth_headers

ENV = "MYTHEUS_TEST_CALLER_TOKENS"
SECRET = "k" * 40


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv(ENV, f"rag-service={SECRET}")
    app = FastAPI()
    dep = verified_or_service_org_id(ENV, frozenset({"rag-service"}))

    @app.get("/whoami")
    async def whoami(org_id: UUID = Depends(dep)) -> dict[str, str]:  # noqa: B008
        return {"org_id": str(org_id)}

    return TestClient(app)


def _svc(org: uuid.UUID, service: str = "rag-service", secret: str = SECRET) -> dict[str, str]:
    return {
        "X-Mytheus-Service": service,
        "X-Mytheus-Service-Token": secret,
        "X-Mytheus-Org": str(org),
    }


def test_service_caller_gets_its_org(client):
    org = uuid.uuid4()
    r = client.get("/whoami", headers=_svc(org))
    assert r.status_code == 200 and r.json() == {"org_id": str(org)}


def test_a_user_token_still_works(client):
    org = uuid.uuid4()
    r = client.get("/whoami", headers=auth_headers(org))
    assert r.status_code == 200 and r.json() == {"org_id": str(org)}


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Mytheus-Service": "rag-service", "X-Mytheus-Service-Token": "w" * 40},
        {"X-Mytheus-Service": "bff", "X-Mytheus-Service-Token": SECRET},
    ],
)
def test_bad_service_credentials_never_fall_back(client, headers):
    org = uuid.uuid4()
    # A valid user token alongside bad service headers is still refused: the service path,
    # once chosen, must succeed on its own.
    r = client.get("/whoami", headers={**auth_headers(org), **headers, "X-Mytheus-Org": str(org)})
    assert r.status_code == 401


def test_no_credentials_is_401(client):
    assert client.get("/whoami").status_code == 401


def test_embed_and_rerank_name_their_caller_env():
    assert query_embed.CALLER_TOKENS_ENV == "MYTHEUS_EMBED_CALLER_TOKENS"
    assert rerank.CALLER_TOKENS_ENV == "MYTHEUS_RERANK_CALLER_TOKENS"
