"""The search-service consumer entrypoint: the `transcript-chunks` indexer
(`TranscriptChunkSearchConsumer`, `consumer.py`, run as `ErasureAwareSearchConsumer`) and the
`admin-events` erasure consumer (`SearchErasureConsumer`, `erasure_consumer.py`), each on its
own thread with its OWN engine and Redis (`run_forever` binds async clients to its thread's
loop, see `services/embedding_service/worker.py`). `SEARCH_CONSUMERS` (default
`transcript-index,erasure`) limits which ones start. Each is wired to real Postgres
(`mytheus_ai_app`), real Redis, and the real Schema Registry, running `BaseConsumer.run_forever`
against real Redpanda. Mirrors `services/embedding_service/worker.py`
and `services/ingestion_service/worker.py`'s shape exactly (same env vars, same role).

Run: `python -m services.search_service.worker`

No separate relay entrypoint here: per CLAUDE.md, "One relay per repo" —
`services/embedding_service/relay_main.py` already drains `intelligence.outbox` for every
producer in this repo, including the `admin-events` `erasure_completed` rows the erasure
consumer writes (the indexer itself produces no events, see `consumer.py`).
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
from services.search_service.erasure_consumer import (
    ErasureAwareSearchConsumer,
    SearchErasureConsumer,
)

logger = logging.getLogger(__name__)


def _deps() -> dict[str, Any]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_app:mytheus_ai_app@localhost:5443/mytheus",
    )
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return {
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
        for c in os.environ.get("SEARCH_CONSUMERS", "transcript-index,erasure").split(",")
        if c.strip()
    }
    consumers: list[BaseConsumer] = []
    if "transcript-index" in wanted:
        deps = _deps()
        consumers.append(
            # never record + confidential meetings (libs/shared/index_guard.py)
            ErasureAwareSearchConsumer(
                guard=guard_from_env("search-service", deps["redis"]), **deps
            )
        )
    if "erasure" in wanted:
        consumers.append(SearchErasureConsumer(**_deps()))
    if not consumers:
        raise SystemExit("SEARCH_CONSUMERS names no known consumer")

    threads = [
        threading.Thread(
            target=c.run_forever, kwargs={"brokers": brokers}, name=c.group, daemon=True
        )
        for c in consumers
    ]
    for t in threads:
        t.start()
        logger.info("search-service: started %s", t.name)
    for t in threads:
        t.join()  # either dying ends the process, so the container restarts both


if __name__ == "__main__":
    main()
