"""The embedding-service consumer entrypoint: embed-on-ingest (`consumer.py`) and confirmed
decisions (`decision_consumer.py`), each wired to real Postgres (`mytheus_ai_app`), Redis,
Qdrant and the Schema Registry, running `BaseConsumer.run_forever` against real Redpanda.

Each consumer runs on its own thread with its OWN clients: `run_forever` binds async clients to
its thread's event loop. The BGE model is loaded once per process and shared. `EMBED_CONSUMERS`
(default `embed-on-ingest,embed-decisions`) limits which ones start.

Run: `python -m services.embedding_service.worker`
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from confluent_kafka import Producer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.avro_codec import AvroCodec
from libs.shared.consumer import BaseConsumer
from libs.shared.index_guard import guard_from_env
from libs.shared.qdrant import write_client
from services.embedding_service.consumer import EmbeddingConsumer
from services.embedding_service.decision_consumer import DecisionEmbeddingConsumer

logger = logging.getLogger(__name__)


def _deps() -> dict[str, Any]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5443/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return {
        "qdrant": write_client(),
        "session_factory": async_sessionmaker(engine, expire_on_commit=False),
        "redis": Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6390")),
        "codec": AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085")),
        "dlq_producer": Producer(
            {"bootstrap.servers": os.environ.get("REDPANDA_BROKERS", "localhost:9092")}
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    wanted = {
        c.strip()
        for c in os.environ.get("EMBED_CONSUMERS", "embed-on-ingest,embed-decisions").split(",")
        if c.strip()
    }
    consumers: list[BaseConsumer] = []
    if "embed-on-ingest" in wanted:
        deps = _deps()
        # never record + confidential meetings (libs/shared/index_guard.py)
        guard = guard_from_env("embedding-service", deps["redis"])
        consumers.append(EmbeddingConsumer(guard=guard, **deps))
    if "embed-decisions" in wanted:
        consumers.append(DecisionEmbeddingConsumer(**_deps()))
    if not consumers:
        raise SystemExit("EMBED_CONSUMERS names no known consumer")

    threads = [
        threading.Thread(
            target=c.run_forever, kwargs={"brokers": brokers}, name=c.group, daemon=True
        )
        for c in consumers
    ]
    for t in threads:
        t.start()
        logger.info("embedding-service: started %s", t.name)
    for t in threads:
        t.join()  # either dying ends the process, so the container restarts both


if __name__ == "__main__":
    main()
