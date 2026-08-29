"""The embed-on-ingest consumer entrypoint: `EmbeddingConsumer` (`consumer.py`) wired to real
Postgres (`mytheus_ai_app`), real Redis, real Qdrant, and the real Schema Registry, running
`BaseConsumer.run_forever` (subscribe/poll/`process_raw`/manual-commit-after-effect) against
real Redpanda.

Run: `python -m services.embedding_service.worker`
"""

from __future__ import annotations

import logging
import os

from confluent_kafka import Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.avro_codec import AvroCodec
from libs.shared.qdrant import write_client
from services.embedding_service.consumer import EmbeddingConsumer


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5433/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    consumer = EmbeddingConsumer(
        qdrant=write_client(),
        session_factory=session_factory,
        redis=Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380")),
        codec=AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")),
        dlq_producer=Producer(
            {"bootstrap.servers": os.environ.get("REDPANDA_BROKERS", "localhost:9092")}
        ),
    )
    consumer.run_forever(brokers=os.environ.get("REDPANDA_BROKERS", "localhost:9092"))


if __name__ == "__main__":
    main()
