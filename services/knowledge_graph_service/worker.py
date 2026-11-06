"""The `tenant-events` consumer entrypoint for `knowledge-graph-service`.

Run: `python -m services.knowledge_graph_service.worker`

Same shape and env as `services/search_service/worker.py`, plus `AGE_DATABASE_URL` (required,
no default - see `libs/shared/age.py`). Bootstraps the graph (`ensure_graph`) before consuming.
"""

from __future__ import annotations

import asyncio
import logging
import os

from confluent_kafka import Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.age import age_database_url
from libs.shared.avro_codec import AvroCodec
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.consumer import TenantTopicSeedConsumer
from services.knowledge_graph_service.graph import ensure_graph


async def _bootstrap(dsn: str) -> None:
    async with age_connection(dsn) as conn:
        await ensure_graph(conn)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    dsn = age_database_url()
    asyncio.run(_bootstrap(dsn))

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5443/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    consumer = TenantTopicSeedConsumer(
        graph_connect=lambda: age_connection(dsn),
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        redis=Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6390")),
        codec=AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085")),
        dlq_producer=Producer({"bootstrap.servers": brokers}),
    )
    consumer.run_forever(brokers=brokers)


if __name__ == "__main__":
    main()
