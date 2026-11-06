"""reranker-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness
probe, `libs/shared/app.create_app`) plus the rerank contract, `POST /rerank`
(`rerank.py`, mounted at `/internal` — same routing convention as `embedding-service`'s
`POST /embed`). reranker-service owns no data: no Postgres table, no Qdrant creds, no consumer,
no relay — this one process is the entire service.
"""

from __future__ import annotations

import asyncio
import logging
import os

from libs.shared.app import create_app
from services.reranker_service.model import warmup
from services.reranker_service.rerank import create_rerank_app

SERVICE_NAME = "reranker-service"

app = create_app(SERVICE_NAME)
app.mount("/internal", create_rerank_app())


async def _preload_model() -> None:
    """Load BGE-reranker-base and run one forward pass BEFORE the process takes traffic (the app
    server does not accept requests until startup handlers finish), so the first /rerank is warm.
    Opt out with RERANKER_PRELOAD=0 (e.g. unit tests that never hit the model)."""
    if os.environ.get("RERANKER_PRELOAD", "1") == "0":
        return
    seconds = await asyncio.to_thread(warmup)
    logging.getLogger(__name__).info("reranker model preloaded and warmed in %.2fs", seconds)


app.router.add_event_handler("startup", _preload_model)
