"""Shared FastAPI app factory for the AI/inference tier.

Every synchronous service here boots the same minimal app: a `/healthz` liveness probe and a
`service` field echoing the canonical (hyphenated) service name. Domain routes, the gRPC mesh, and
event consumers land in the inference/agent bands; this only proves the service builds and boots.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from libs.shared.metrics import install as install_metrics


def create_app(service_name: str) -> FastAPI:
    """Build a FastAPI app for `service_name` (canonical, hyphenated, e.g. "rag-service")."""
    app = FastAPI(title=service_name)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "service": service_name}

    install_metrics(app, service_name)  # /metrics + request metrics (internal scrape only)
    return app
