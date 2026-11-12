"""Entrypoint for knowledge-graph-service's internal gRPC surface (`grpc_server.py`).

Run: python -m services.knowledge_graph_service.grpc_main   (KG_GRPC_BIND, default 0.0.0.0:50104)
Needs AGE_DATABASE_URL and MYTHEUS_KG_CALLER_TOKENS.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc

from libs.shared.age import AgeConnection, create_age_pool
from services.knowledge_graph_service.grpc_server import (
    KnowledgeGraphServicer,
    add_knowledge_graph_servicer,
)


async def serve() -> None:
    pool = await create_age_pool(max_size=10)

    @asynccontextmanager
    async def connect() -> AsyncIterator[AgeConnection]:
        async with pool.acquire() as conn:
            yield conn

    server = grpc.aio.server()
    add_knowledge_graph_servicer(KnowledgeGraphServicer(graph_connect=connect), server)
    bind = os.environ.get("KG_GRPC_BIND", "0.0.0.0:50104")
    server.add_insecure_port(bind)  # mesh-internal; TLS terminates at the mesh
    await server.start()
    logging.getLogger(__name__).info("knowledge-graph-service gRPC listening on %s", bind)
    await server.wait_for_termination()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
