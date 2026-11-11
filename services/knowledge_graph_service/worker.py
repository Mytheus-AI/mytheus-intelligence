"""The consumer entrypoint for `knowledge-graph-service`: both graph writers in one process.

Run: `python -m services.knowledge_graph_service.worker`

- `TenantTopicSeedConsumer` (`tenant-events`): per-org `Topic` seeding (Plan 5-6 3.2).
- `ConfirmedFactsConsumer` (`decisions`, `actions`, `meeting-room-events`): the edges
  (Plan 7-9 3.7), run as `ErasureAwareFactsConsumer` (`erasure_consumer.py`).
- `GraphErasureConsumer` (`admin-events` `erasure_requested`): a person's or a meeting's share
  of the graph (`erasure_consumer.py`).

Each runs `BaseConsumer.run_forever` on its own thread with its OWN Redis, engine and producer:
`run_forever` binds async clients to its thread's event loop (see its docstring), so the two
consumers must not share them. Env: `AGE_DATABASE_URL` (required, no default - see
`libs/shared/age.py`), `DATABASE_URL`, `REDIS_URL`, `REDPANDA_BROKERS`, `SCHEMA_REGISTRY_URL`.
`KG_CONSUMERS` (default `topic-seed,confirmed-facts,erasure`) limits which ones start. The graph is
bootstrapped (`ensure_graph`) before either consumes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from confluent_kafka import Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.age import age_database_url
from libs.shared.avro_codec import AvroCodec
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.consumer import TenantTopicSeedConsumer
from services.knowledge_graph_service.erasure_consumer import (
    ErasureAwareFactsConsumer,
    GraphErasureConsumer,
)
from services.knowledge_graph_service.graph import ensure_graph

logger = logging.getLogger(__name__)


async def _bootstrap(dsn: str) -> None:
    async with age_connection(dsn) as conn:
        await ensure_graph(conn)


def _deps() -> dict[str, Any]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5443/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    return {
        "session_factory": async_sessionmaker(engine, expire_on_commit=False),
        "redis": Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6390")),
        "codec": AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085")),
        "dlq_producer": Producer({"bootstrap.servers": brokers}),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dsn = age_database_url()
    asyncio.run(_bootstrap(dsn))
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    wanted = {
        c.strip()
        for c in os.environ.get("KG_CONSUMERS", "topic-seed,confirmed-facts,erasure").split(",")
        if c.strip()
    }

    consumers: list[Any] = []
    if "topic-seed" in wanted:
        consumers.append(
            TenantTopicSeedConsumer(graph_connect=lambda: age_connection(dsn), **_deps())
        )
    if "confirmed-facts" in wanted:
        consumers.append(
            ErasureAwareFactsConsumer(graph_connect=lambda: age_connection(dsn), **_deps())
        )
    if "erasure" in wanted:
        consumers.append(GraphErasureConsumer(graph_connect=lambda: age_connection(dsn), **_deps()))
    if not consumers:
        raise SystemExit("KG_CONSUMERS names no known consumer")

    threads = [
        threading.Thread(
            target=c.run_forever, kwargs={"brokers": brokers}, name=c.group, daemon=True
        )
        for c in consumers
    ]
    for t in threads:
        t.start()
        logger.info("knowledge-graph-service: started %s", t.name)
    for t in threads:
        t.join()  # either dying ends the process, so the container restarts both


if __name__ == "__main__":
    main()
