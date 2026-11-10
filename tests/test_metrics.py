"""Metrics: route-template labels, consumer counters/lag, no tenant labels."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from libs.shared import metrics
from libs.shared.app import create_app


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_create_app_exposes_metrics_without_raw_paths() -> None:
    app: FastAPI = create_app("metrics-test-service")

    @app.get("/v1/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    c = TestClient(app)
    assert c.get("/healthz").status_code == 200
    assert c.get("/v1/items/11111111-2222-3333-4444-555555555555").status_code == 200
    body = c.get("/metrics").text
    assert 'route="/v1/items/{item_id}"' in body
    assert "11111111-2222" not in body
    # HTTP series never carry a tenant label (the agent quality series do, by design:
    # libs/shared/agent_metrics.py)
    assert not any("org_id" in ln for ln in body.splitlines() if ln.startswith("mytheus_http"))
    assert "mytheus_http_request_duration_seconds_bucket" in body
    assert c.get("/metrics", headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 404


def test_consumer_outbox_embedding_helpers() -> None:
    metrics.set_service("t")
    metrics.record_processed("g", "tp", "duplicate")
    metrics.record_dlq("g", "tp", "BUSINESS_INVALID")
    metrics.record_outbox(published=2, poisoned=1, backlog=4)
    with metrics.time_embedding_batch(8):
        pass
    metrics.observe_consumer_lag("g", "tp", 1)
    assert (
        _sample(
            "mytheus_consumer_messages_processed_total",
            service="t",
            group="g",
            topic="tp",
            outcome="duplicate",
        )
        == 1
    )
    assert (
        _sample(
            "mytheus_consumer_dlq_total",
            service="t",
            group="g",
            topic="tp",
            failure_class="BUSINESS_INVALID",
        )
        == 1
    )
    assert _sample("mytheus_outbox_backlog", service="t") == 4
    assert _sample("mytheus_embedding_batch_duration_seconds_count", service="t") == 1
    assert _sample("mytheus_consumer_lag_seconds", service="t", group="g", topic="tp") > 1000
