"""`POST /internal/promises/extract` on rag-service (promise ledger): integration-service only,
the extractor's answer passed through, 503 when unwired or when the company brain is unreadable,
and integration-service still refused on `/api/rag/query`."""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from libs.shared.org_context import OrgContextUnavailable
from services.agent_runtime_service.promises import (
    ExtractedPromise,
    ExtractRequest,
    ExtractResult,
)
from services.rag_service import main as rag_main

SECRET = "i" * 40
OTHER = "b" * 40
ORG = uuid4()


class FakeExtractor:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[UUID, ExtractRequest]] = []
        self.fail = fail

    async def extract(self, org_id: UUID, req: ExtractRequest) -> ExtractResult:
        self.calls.append((org_id, req))
        if self.fail:
            raise OrgContextUnavailable(str(org_id))
        return ExtractResult(
            "ok",
            "promise_extract@v1",
            (
                ExtractedPromise(
                    content="Send Sam the deck",
                    due_hint="Friday",
                    due_date=date(2026, 11, 13),
                    recipient_label="Sam",
                    recipient_email="sam@example.com",
                    quote="I'll send you the deck by Friday",
                    confidence=0.9,
                ),
            ),
        )


def _headers(service: str = "integration-service", secret: str = SECRET) -> dict[str, str]:
    return {
        "X-Mytheus-Service": service,
        "X-Mytheus-Service-Token": secret,
        "X-Mytheus-Org": str(ORG),
    }


BODY: dict[str, Any] = {
    "channel": "slack",
    "text": "@Sam I'll send you the deck by Friday",
    "reference_date": "2026-11-11",
    "author_name": "Maya",
    "people": [{"label": "Sam", "email": "sam@example.com"}],
}


@pytest.fixture()
def tokens(monkeypatch):
    monkeypatch.setenv(rag_main.CALLER_TOKENS_ENV, f"bff={OTHER},integration-service={SECRET}")


def test_integration_service_gets_the_extractors_answer(tokens) -> None:
    fake = FakeExtractor()
    client = TestClient(rag_main.build_app(promises=fake))  # type: ignore[arg-type]
    r = client.post("/internal/promises/extract", json=BODY, headers=_headers())
    assert r.status_code == 200, r.text
    assert r.json() == {
        "status": "ok",
        "prompt": "promise_extract@v1",
        "promises": [
            {
                "content": "Send Sam the deck",
                "due_hint": "Friday",
                "due_date": "2026-11-13",
                "recipient_label": "Sam",
                "recipient_email": "sam@example.com",
                "quote": "I'll send you the deck by Friday",
                "confidence": 0.9,
            }
        ],
    }
    [(org, req)] = fake.calls
    assert org == ORG and req.reference_date == date(2026, 11, 11)
    assert req.people[0].email == "sam@example.com"


def test_only_integration_service_may_ask(tokens) -> None:
    client = TestClient(rag_main.build_app(promises=FakeExtractor()))  # type: ignore[arg-type]
    assert (
        client.post(
            "/internal/promises/extract", json=BODY, headers=_headers("bff", OTHER)
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/internal/promises/extract", json=BODY, headers=_headers(secret="x" * 40)
        ).status_code
        == 401
    )
    assert client.post("/internal/promises/extract", json=BODY).status_code == 401
    # ...and integration-service cannot use the token to ask meetings questions.
    headers = {**_headers(), "X-Mytheus-User": str(uuid4())}
    assert client.post("/api/rag/query", json={"query": "x"}, headers=headers).status_code == 401


def test_unwired_or_unreadable_company_context_is_503(tokens, monkeypatch) -> None:
    monkeypatch.delenv("RAG_SERVICE_TOKEN", raising=False)
    with TestClient(rag_main.build_app()) as client:
        r = client.post("/internal/promises/extract", json=BODY, headers=_headers())
        assert r.status_code == 503
    client = TestClient(rag_main.build_app(promises=FakeExtractor(fail=True)))  # type: ignore[arg-type]
    assert (
        client.post("/internal/promises/extract", json=BODY, headers=_headers()).status_code == 503
    )
