"""`POST /embed` — real BGE-small-en query embeds, real Redis cache at
`nexus:{org_id}:qembed:{query_hash}` (plan 3.7 step 5)."""

from __future__ import annotations

import uuid

import httpx
import pytest

from services.embedding_service.query_embed import create_query_embed_app
from tests.conftest import auth_headers

pytestmark = pytest.mark.asyncio


async def test_embed_returns_384dim_vectors_and_caches_on_second_call(redis_client):
    app = create_query_embed_app(redis_client)
    org_id = str(uuid.uuid4())
    query = "what did we decide about the launch date?"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/embed", json={"queries": [query]}, headers=auth_headers(org_id))
        assert resp1.status_code == 200
        body1 = resp1.json()
        assert len(body1["vectors"]) == 1
        assert len(body1["vectors"][0]) == 384
        assert body1["cached"] == [False]

        resp2 = await client.post("/embed", json={"queries": [query]}, headers=auth_headers(org_id))
        body2 = resp2.json()
        assert body2["cached"] == [True]
        assert body2["vectors"] == body1["vectors"]

        cached_raw = await redis_client.get(
            f"nexus:{org_id}:qembed:"
            + __import__("hashlib").sha256(query.encode()).hexdigest()[:32]
        )
        assert cached_raw is not None


async def test_embed_rejects_more_than_three_variants(redis_client):
    app = create_query_embed_app(redis_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/embed",
            json={"queries": ["a", "b", "c", "d"]},
            headers=auth_headers(uuid.uuid4()),
        )
        assert resp.status_code == 422


async def test_embed_rejects_missing_token(redis_client):
    app = create_query_embed_app(redis_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/embed", json={"queries": ["q"]})
        assert resp.status_code == 401


async def test_embed_rejects_malformed_token(redis_client):
    app = create_query_embed_app(redis_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/embed",
            json={"queries": ["q"]},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401


async def test_embed_with_another_orgs_id_in_the_header_but_no_valid_token_is_rejected(
    redis_client,
):
    """The exploit this whole fix closes: `X-Org-Id` used to be trusted outright and could
    poison another tenant's `nexus:{org_id}:qembed:` cache. A request that supplies a victim
    org's id in it, with no valid bearer token, must now be rejected with 401 before the cache
    is ever touched."""
    app = create_query_embed_app(redis_client)
    victim_org = str(uuid.uuid4())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/embed",
            json={"queries": ["q"]},
            headers={"X-Org-Id": victim_org},  # the old, now-untrusted header
        )
        assert resp.status_code == 401

    cached_raw = await redis_client.get(
        f"nexus:{victim_org}:qembed:" + __import__("hashlib").sha256(b"q").hexdigest()[:32]
    )
    assert cached_raw is None
