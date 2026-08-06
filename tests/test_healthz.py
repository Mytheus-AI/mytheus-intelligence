"""Smoke test: every synchronous service boots and answers `/healthz` with its canonical name."""

from __future__ import annotations

import importlib

import pytest
from starlette.testclient import TestClient

# (module path, canonical service name) for the FastAPI services.
FASTAPI_SERVICES = [
    ("services.agent_runtime_service.main", "agent-runtime-service"),
    ("services.rag_service.main", "rag-service"),
    ("services.knowledge_graph_service.main", "knowledge-graph-service"),
    ("services.search_service.main", "search-service"),
    ("services.ingestion_service.main", "ingestion-service"),
    ("services.embedding_service.main", "embedding-service"),
]


@pytest.mark.parametrize(("module_path", "service_name"), FASTAPI_SERVICES)
def test_healthz(module_path: str, service_name: str) -> None:
    module = importlib.import_module(module_path)
    client = TestClient(module.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": service_name}


def test_events_client_lazy_import() -> None:
    """`load_events_client` fails closed with an actionable message until the wheel is installed."""
    from libs.shared.clients import load_events_client

    with pytest.raises(ModuleNotFoundError, match="mytheus-events is not installed"):
        load_events_client()
