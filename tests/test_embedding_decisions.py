"""`DecisionEmbeddingConsumer`, end to end: a real Avro `decisions` event (frozen `Decision` with
ADR-0031's `supersedes`) through the real consumer, the real BGE model and real Qdrant.

Covers: the point lands in `decisions__{org_id}` with canon's payload; a redelivery is a no-op
(one point, same id); a superseding decision re-labels the prior point; another tenant's
collection and filtered search never see it; both `decisions` consumer groups apply the same
event (the dedupe-key regression behind ADR-0031 D6).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from confluent_kafka import Producer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.shared.qdrant import LOGICAL_DECISIONS, physical_collection, search_with_tenant_filter
from services.embedding_service.decision_consumer import (
    DecisionEmbeddingConsumer,
    decision_point_id,
)
from services.embedding_service.model import embed_batch
from tests.conftest import APP_URL

pytestmark = pytest.mark.asyncio


def _decision(
    org: uuid.UUID, d: uuid.UUID, m: uuid.UUID, content: str, supersedes: uuid.UUID | None = None
) -> dict[str, Any]:
    return {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": f"decision:{d}:confirmed",
            "org_id": str(org),
            "schema_version": 1,
            "producer": "core-api",
            "occurred_at": int(time.time() * 1_000_000),
            "correlation_id": None,
            "causation_id": None,
        },
        "decision_id": str(d),
        "meeting_id": str(m),
        "content": content,
        "rationale": "agreed in the review",
        "confidence": 0.88,
        "status": "confirmed",
        "confirmed_by": str(uuid.uuid4()),
        "supersedes": str(supersedes) if supersedes else None,
    }


@pytest.fixture()
def consumer(migrated_db, qdrant_client, redis_client, avro_codec) -> DecisionEmbeddingConsumer:
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    return DecisionEmbeddingConsumer(
        qdrant=qdrant_client,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        redis=redis_client,
        codec=avro_codec,
        dlq_producer=Producer({"bootstrap.servers": "localhost:9092"}),
    )


async def _feed(c: DecisionEmbeddingConsumer, value: dict[str, Any]) -> str:
    raw = c._codec.serialize_for("decisions", value)
    return await c.process_raw(topic="decisions", raw_value=raw, partition=0, offset=1)


async def test_a_confirmed_decision_is_embedded_once_and_supersession_relabels(
    consumer, qdrant_client
):
    org, m = uuid.uuid4(), uuid.uuid4()
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    name = physical_collection(LOGICAL_DECISIONS, org)
    try:
        v1 = _decision(org, d1, m, "We will run the graph on a dedicated AGE Postgres")
        assert await _feed(consumer, v1) == "applied"
        assert await _feed(consumer, v1) == "duplicate"
        assert (await qdrant_client.count(name, exact=True)).count == 1

        [point] = await qdrant_client.retrieve(
            name, [decision_point_id(str(d1))], with_payload=True
        )
        assert point.payload == {
            "org_id": str(org),
            "decision_id": str(d1),
            "meeting_id": str(m),
            "status": "confirmed",
            "confidence": pytest.approx(0.88),
        }

        v2 = _decision(org, d2, m, "We will run the graph on Neo4j instead", supersedes=d1)
        assert await _feed(consumer, v2) == "applied"
        assert (await qdrant_client.count(name, exact=True)).count == 2
        [old] = await qdrant_client.retrieve(name, [decision_point_id(str(d1))], with_payload=True)
        assert old.payload["status"] == "superseded"

        # Another tenant: its own (absent) collection, and the filtered search finds nothing.
        other = uuid.uuid4()
        await qdrant_client.create_collection(
            physical_collection(LOGICAL_DECISIONS, other),
            vectors_config={"size": 384, "distance": "Cosine"},
        )
        try:
            [vec], _ = embed_batch(["dedicated AGE Postgres"])
            assert (
                await search_with_tenant_filter(qdrant_client, LOGICAL_DECISIONS, other, vec) == []
            )
            hits = await search_with_tenant_filter(qdrant_client, LOGICAL_DECISIONS, org, vec)
            assert {h.payload["decision_id"] for h in hits} == {str(d1), str(d2)}
        finally:
            await qdrant_client.delete_collection(physical_collection(LOGICAL_DECISIONS, other))
    finally:
        await qdrant_client.delete_collection(name)


async def test_a_supersedes_for_an_unseen_decision_does_not_fail(consumer, qdrant_client):
    org, m = uuid.uuid4(), uuid.uuid4()
    value = _decision(org, uuid.uuid4(), m, "v2", supersedes=uuid.uuid4())
    try:
        assert await _feed(consumer, value) == "applied"
    finally:
        await qdrant_client.delete_collection(physical_collection(LOGICAL_DECISIONS, org))
