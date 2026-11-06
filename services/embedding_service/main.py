"""embedding-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness
probe, `libs/shared/app.create_app`) plus the query-embed contract, `POST /embed`
(`query_embed.py`). This is the "MVP modular monolith... worker process, still off the API
tier" surface (plan 3.7 step 6) — the embed-on-ingest consumer (`consumer.py`) and the outbox
relay (`relay_main.py`) run as SEPARATE processes, not inside this app, matching the plan's
"still off the API tier" framing: this process serves the synchronous query-embed contract
only, never the ingest path.
"""

from __future__ import annotations

import asyncio
import logging
import os

from redis.asyncio import Redis

from libs.shared.app import create_app
from services.embedding_service.model import warmup
from services.embedding_service.query_embed import create_query_embed_app

SERVICE_NAME = "embedding-service"

app = create_app(SERVICE_NAME)

_redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
app.mount("/internal", create_query_embed_app(_redis))


async def _preload_model() -> None:
    """Load BGE-small-en and run one encode before taking traffic. EMBEDDING_PRELOAD=0 opts out."""
    if os.environ.get("EMBEDDING_PRELOAD", "1") == "0":
        return
    seconds = await asyncio.to_thread(warmup)
    logging.getLogger(__name__).info("embedding model preloaded and warmed in %.2fs", seconds)


app.router.add_event_handler("startup", _preload_model)
