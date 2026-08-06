"""ingestion-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness
probe) plus the real `POST /ingest` / `GET /ingest/jobs/:id` surfaces (`routes.py`). The
`integration-events` consumer (`consumer.py`/`worker.py`) runs as a SEPARATE process, matching
`embedding-service`'s precedent (ingest path here is the synchronous upload contract only).

Run: uvicorn services.ingestion_service.main:app --port 8104
"""

from __future__ import annotations

import os

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.app import create_app
from libs.shared.object_storage import get_client
from services.ingestion_service.routes import create_ingestion_router

SERVICE_NAME = "ingestion-service"

app = create_app(SERVICE_NAME)

_engine = create_async_engine(
    os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5433/mytheus"
    ),
    pool_pre_ping=True,
)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)
_redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380"))

app.include_router(
    create_ingestion_router(session_factory=_session_factory, s3_client=get_client(), redis=_redis)
)
