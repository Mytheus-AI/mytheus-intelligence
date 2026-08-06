"""The `transcript-chunks` consumer entrypoint: `TranscriptChunkSearchConsumer` (`consumer.py`)
wired to real Postgres (`mytheus_ai_app`), real Redis, and the real Schema Registry, running
`BaseConsumer.run_forever` against real Redpanda. Mirrors `services/embedding_service/worker.py`
and `services/ingestion_service/worker.py`'s shape exactly (same env vars, same role).

Run: `python -m services.search_service.worker`

No separate relay entrypoint here: per CLAUDE.md, "One relay per repo" —
`services/embedding_service/relay_main.py` already drains `intelligence.outbox` for every
producer in this repo. This consumer does not even use the outbox (see `consumer.py`'s module
docstring: search-service produces no events), so it needs no relay coordination at all.
"""

from __future__ import annotations

import logging
import os

from confluent_kafka import Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.avro_codec import AvroCodec
from services.search_service.consumer import TranscriptChunkSearchConsumer


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5433/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    consumer = TranscriptChunkSearchConsumer(
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
