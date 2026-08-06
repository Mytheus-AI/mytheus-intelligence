"""ADR-0006 proven against REAL Qdrant (task brief: "Prove cross-tenant isolation against the
REAL Qdrant"). No mock/fake Qdrant client anywhere in this file.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from libs.shared.qdrant import (
    EMBEDDING_DIM,
    LOGICAL_TRANSCRIPT_CHUNKS,
    physical_collection,
    search_with_tenant_filter,
    upsert,
)

pytestmark = pytest.mark.asyncio


def _vec(seed: float = 0.1) -> list[float]:
    return [seed] * EMBEDDING_DIM


async def test_upsert_provisions_384dim_int8_collection_with_citation_payload(qdrant_client):
    org_id = uuid.uuid4()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=_vec(),
        payload={
            "org_id": str(org_id),
            "meeting_id": "m-1",
            "chunk_id": "c-1",
            "speaker": "spk-1",
            "start_ms": 0,
            "is_final": True,
        },
    )
    name = await upsert(qdrant_client, LOGICAL_TRANSCRIPT_CHUNKS, org_id, [point])
    try:
        assert name == physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
        info = await qdrant_client.get_collection(name)
        assert info.config.params.vectors.size == EMBEDDING_DIM
        assert info.config.params.vectors.distance.name == "COSINE"
        assert info.config.quantization_config is not None

        fetched = await qdrant_client.retrieve(name, ids=[point.id], with_payload=True)
        assert len(fetched) == 1
        payload = fetched[0].payload
        assert payload["org_id"] == str(org_id)
        assert payload["meeting_id"] == "m-1"
        assert payload["chunk_id"] == "c-1"
        assert payload["speaker"] == "spk-1"
    finally:
        await qdrant_client.delete_collection(name)


async def test_cross_tenant_wrong_payload_filter_returns_zero_even_in_right_collection(
    qdrant_client,
):
    """The exact acceptance check: 'A Qdrant search with the wrong org_id payload filter
    returns 0 results even against the right collection.'"""
    org_a = uuid.uuid4()
    org_b = uuid.uuid4()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=_vec(0.2),
        payload={"org_id": str(org_a), "meeting_id": "m-1", "chunk_id": "c-1", "speaker": "s"},
    )
    name = await upsert(qdrant_client, LOGICAL_TRANSCRIPT_CHUNKS, org_a, [point])
    try:
        # Correct filter finds it.
        hits = await search_with_tenant_filter(
            qdrant_client, LOGICAL_TRANSCRIPT_CHUNKS, org_a, _vec(0.2)
        )
        assert len(hits) == 1

        # Wrong org_id filter against the SAME physical collection: zero results.
        wrong = await qdrant_client.query_points(
            collection_name=name,
            query=_vec(0.2),
            query_filter=Filter(
                must=[FieldCondition(key="org_id", match=MatchValue(value=str(org_b)))]
            ),
            limit=10,
        )
        assert len(wrong.points) == 0
    finally:
        await qdrant_client.delete_collection(name)


async def test_cross_tenant_isolation_by_collection_boundary(qdrant_client):
    """The other half of defense-in-depth: org B has no collection at all until it writes, so
    org B's query path never even reaches org A's points."""
    org_a = uuid.uuid4()
    org_b = uuid.uuid4()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=_vec(0.3),
        payload={"org_id": str(org_a), "meeting_id": "m-1", "chunk_id": "c-1", "speaker": "s"},
    )
    name_a = await upsert(qdrant_client, LOGICAL_TRANSCRIPT_CHUNKS, org_a, [point])
    try:
        name_b = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_b)
        assert not await qdrant_client.collection_exists(name_b)
        assert name_a != name_b
    finally:
        await qdrant_client.delete_collection(name_a)


async def test_redelivered_upsert_of_same_point_id_does_not_duplicate(qdrant_client):
    """Idempotent-write proof at the Qdrant layer: upserting the same point id twice leaves
    exactly one point (used by the consumer redelivery test at the higher, event level too)."""
    org_id = uuid.uuid4()
    point_id = str(uuid.uuid4())
    payload = {"org_id": str(org_id), "meeting_id": "m-1", "chunk_id": "c-1", "speaker": "s"}
    name = None
    try:
        name = await upsert(
            qdrant_client,
            LOGICAL_TRANSCRIPT_CHUNKS,
            org_id,
            [PointStruct(id=point_id, vector=_vec(0.4), payload=payload)],
        )
        name = await upsert(
            qdrant_client,
            LOGICAL_TRANSCRIPT_CHUNKS,
            org_id,
            [PointStruct(id=point_id, vector=_vec(0.4), payload=payload)],
        )
        count = await qdrant_client.count(name, exact=True)
        assert count.count == 1
    finally:
        if name:
            await qdrant_client.delete_collection(name)
