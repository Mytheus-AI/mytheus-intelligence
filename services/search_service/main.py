"""search-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness
probe) plus the real `GET /search` surface (`routes.py`). The `transcript-chunks` consumer that
builds `intelligence.transcript_chunk_search` (`consumer.py`/`worker.py`) runs as a SEPARATE
process, matching `embedding-service`'s and `ingestion-service`'s precedent (this synchronous
process is the query-serving contract only).

Run: uvicorn services.search_service.main:app --port 8103
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.app import create_app
from services.search_service.routes import create_search_router

SERVICE_NAME = "search-service"

app = create_app(SERVICE_NAME)

_engine = create_async_engine(
    os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5433/mytheus"
    ),
    pool_pre_ping=True,
)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)

app.include_router(create_search_router(session_factory=_session_factory))
