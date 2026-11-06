"""Entrypoint for search-service's internal gRPC FTS arm (see `grpc_server.py`).

Run: python -m services.search_service.grpc_main   (GRPC_PORT, default 50103)
"""

from __future__ import annotations

import asyncio
import logging
import os

import grpc
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from services.search_service.grpc_server import FtsServicer, add_fts_servicer


async def serve() -> None:
    engine = create_async_engine(
        os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5433/mytheus",
        ),
        pool_pre_ping=True,
    )
    server = grpc.aio.server()
    add_fts_servicer(
        FtsServicer(session_factory=async_sessionmaker(engine, expire_on_commit=False)), server
    )
    port = os.environ.get("GRPC_PORT", "50103")
    server.add_insecure_port(f"0.0.0.0:{port}")  # mesh-internal; TLS terminates at the mesh
    await server.start()
    logging.getLogger(__name__).info("search-service FtsService listening on :%s", port)
    await server.wait_for_termination()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
