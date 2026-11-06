"""This repo's outbox relay entrypoint: drains `intelligence.outbox` to real Redpanda.

`mytheus-backend`'s relay worker (already running in the dev stack) drains `public.outbox`
only — a relay only sees the table it is pointed at. This repo needs its OWN relay process for
`intelligence.outbox`, following `mytheus-backend/services/relay/main.py`'s shape:
`libs.shared.relay.relay_once` in a poll loop, connected as `mytheus_ai_relay` (never the
consumer/request-path role). Established here as the convention for tasks 8-10: any table this
repo's outbox writes to is drained by this same process, not a per-service relay.

Run: `python -m services.embedding_service.relay_main`
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from confluent_kafka import Producer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.avro_codec import AvroCodec
from libs.shared.relay import RelayProducer, relay_once

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0


class _KafkaAvroProducer:
    """Adapts `confluent_kafka.Producer` + `AvroCodec` to the `RelayProducer` protocol."""

    def __init__(self, producer: Producer, codec: AvroCodec) -> None:
        self._producer = producer
        self._codec = codec

    def serialize(self, subject: str, value: dict[str, Any]) -> bytes:
        topic = subject.removesuffix("-value")
        return self._codec.serialize_for(topic, value)

    def produce(self, *, topic: str, key: bytes, value: bytes) -> None:
        self._producer.produce(topic=topic, key=key, value=value)

    def flush(self) -> int:
        return self._producer.flush(10)


async def run_forever() -> None:
    relay_url = os.environ.get(
        "RELAY_DATABASE_URL",
        "postgresql+asyncpg://mytheus_ai_relay:mytheus_ai_relay@localhost:5443/mytheus",
    )
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    registry_url = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085")

    engine = create_async_engine(relay_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    producer: RelayProducer = _KafkaAvroProducer(
        Producer({"bootstrap.servers": brokers}), AvroCodec(registry_url)
    )

    logger.info("embedding-service relay: draining intelligence.outbox -> %s", brokers)
    while True:
        async with session_factory() as session, session.begin():
            published = await relay_once(session, producer)
        if published:
            logger.info("embedding-service relay: published %d row(s)", published)
        else:
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forever())
