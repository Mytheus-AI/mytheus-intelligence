"""`POST /rerank` — real BGE-reranker-base cross-encoder scoring (plan 3.8).

Runs against the REAL model (task brief: "Do NOT substitute a fake scorer, because a fake makes
every ranking assertion meaningless"). The model loads once per process (`model.py`'s
`lru_cache`), so only the first test in a session pays the load cost.

The property that matters most here (task brief): no batch, cache, or shared queue in this
service can ever contain passages from more than one tenant at once. That is proven directly by
`test_concurrent_requests_from_two_orgs_never_cross_contaminate_passages` below, and structurally
by `model.py`'s `rerank_batch` never accepting more than one call's own passages.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx

from services.reranker_service.model import rerank_batch
from services.reranker_service.rerank import create_rerank_app
from tests.conftest import auth_headers

# No module-level `pytestmark = pytest.mark.asyncio`: `asyncio_mode = "auto"` (pyproject.toml)
# already collects `async def` tests without it, and this module also has one sync test
# (`test_rerank_batch_scores_only_the_passages_it_was_given`) that a blanket module mark would
# incorrectly tag.


async def _client():
    app = create_rerank_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_rerank_returns_topk_scored_passages_ranked_by_relevance():
    query = "what did we decide about the launch date?"
    passages = [
        "The team agreed to push the launch date to March 15th after the review.",
        "Bananas are a good source of potassium.",
        "The launch date was confirmed as March 15th in the meeting.",
        "Unrelated: the office coffee machine was replaced last week.",
        "We discussed the marketing budget for Q3, unrelated to the launch date decision.",
    ]

    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": query, "passages": passages, "top_k": 3},
            headers=auth_headers(uuid.uuid4()),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "BAAI/bge-reranker-base"
    results = body["results"]
    assert len(results) == 3

    # Real cross-encoder scores: descending order, and the two launch-date passages should
    # clearly outrank the banana/coffee-machine noise (proves this is a real forward pass, not
    # a stub echo).
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    top_passages = {r["passage"] for r in results[:2]}
    assert "launch date" in " ".join(top_passages).lower()
    assert "Bananas are a good source of potassium." not in top_passages


async def test_rerank_top_k_is_capped_at_the_number_of_passages_supplied():
    query = "q"
    passages = ["a relevant passage", "another passage"]

    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": query, "passages": passages, "top_k": 50},
            headers=auth_headers(uuid.uuid4()),
        )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2  # never more results than passages supplied


async def test_rerank_batch_of_50_measured_latency_warm_cpu(capsys):
    """Warm p95 < 500ms is a GPU-only SLO (BLOCKED here, no CUDA device). This measures the
    real CPU latency for the plan's top-50 candidate size, both at the model layer (pure
    `model.predict()` time) and end-to-end through the FastAPI route, so the number is reported
    honestly rather than claimed."""
    query = "what is the capital of France?"
    passages = [f"Passage number {i} talks about an unrelated topic." for i in range(49)]
    passages.append("Paris is the capital of France.")

    # Warm the model once outside the timed call (matches embedding-service's convention of
    # measuring a warm, not cold, batch).
    rerank_batch(query, passages, top_k=10)
    _ranked, model_elapsed = rerank_batch(query, passages, top_k=10)

    t0 = time.perf_counter()
    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": query, "passages": passages, "top_k": 10},
            headers=auth_headers(uuid.uuid4()),
        )
    e2e_elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 10
    # The genuinely relevant passage should surface at rank 1 out of 50 real-scored candidates.
    assert body["results"][0]["passage"] == "Paris is the capital of France."
    with capsys.disabled():
        print(
            f"\n[reranker-service] warm batch-of-50 CPU latency: "
            f"model.predict={model_elapsed * 1000:.1f}ms, "
            f"end-to-end={e2e_elapsed * 1000:.1f}ms "
            f"(CPU; GPU SLO <500ms warm is BLOCKED)"
        )


async def test_rerank_rejects_missing_token():
    async with await _client() as client:
        resp = await client.post("/rerank", json={"query": "q", "passages": ["p1"]})
    assert resp.status_code == 401


async def test_rerank_rejects_malformed_token():
    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": "q", "passages": ["p1"]},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
    assert resp.status_code == 401


async def test_rerank_with_another_orgs_id_in_the_header_but_no_valid_token_is_rejected():
    """The exploit this whole fix closes: `X-Org-Id` used to be trusted outright for
    audit/rate accounting. A request that supplies a victim org's id in it, with no valid
    bearer token, must now be rejected with 401."""
    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": "q", "passages": ["p1"]},
            headers={"X-Org-Id": str(uuid.uuid4())},  # the old, now-untrusted header
        )
    assert resp.status_code == 401


async def test_rerank_rejects_empty_passages_list():
    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": "q", "passages": []},
            headers=auth_headers(uuid.uuid4()),
        )
    assert resp.status_code == 422


async def test_rerank_rejects_more_than_fifty_passages():
    async with await _client() as client:
        resp = await client.post(
            "/rerank",
            json={"query": "q", "passages": [f"p{i}" for i in range(51)]},
            headers=auth_headers(uuid.uuid4()),
        )
    assert resp.status_code == 422


async def test_concurrent_requests_from_two_orgs_never_cross_contaminate_passages():
    """The property that matters most: fire two /rerank calls concurrently for two different
    tenants with disjoint passage sets and assert each response contains ONLY its own tenant's
    passages. If any shared batch/queue/cache mixed the two calls, org A's response could
    contain org B's passage text (or vice versa) — this proves it never does, under real
    concurrent load against the real model."""
    org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())
    passages_a = [f"org-a-only passage {i}" for i in range(20)]
    passages_b = [f"org-b-only passage {i}" for i in range(20)]

    async def call(org_id: str, query: str, passages: list[str]):
        async with await _client() as client:
            return await client.post(
                "/rerank",
                json={"query": query, "passages": passages, "top_k": 20},
                headers=auth_headers(org_id),
            )

    resp_a, resp_b = await asyncio.gather(
        call(org_a, "q-a", passages_a),
        call(org_b, "q-b", passages_b),
    )

    assert resp_a.status_code == 200 and resp_b.status_code == 200
    result_passages_a = {r["passage"] for r in resp_a.json()["results"]}
    result_passages_b = {r["passage"] for r in resp_b.json()["results"]}

    assert result_passages_a == set(passages_a)
    assert result_passages_b == set(passages_b)
    assert result_passages_a.isdisjoint(result_passages_b)


def test_rerank_batch_scores_only_the_passages_it_was_given(monkeypatch):
    """Unit-level proof that `rerank_batch` never reaches for state from a prior call: patch
    the underlying model to record exactly which pairs it was asked to score, call
    `rerank_batch` twice with different queries/passages back to back, and assert each
    invocation's pairs contain only that call's own passages."""
    import services.reranker_service.model as model_mod

    seen_calls: list[list[list[str]]] = []

    class _FakeModel:
        def predict(self, pairs: list[list[str]]) -> list[float]:
            seen_calls.append([list(p) for p in pairs])
            return [1.0] * len(pairs)

    monkeypatch.setattr(model_mod, "_model", lambda: _FakeModel())

    rerank_batch("query-1", ["a1", "a2"], top_k=2)
    rerank_batch("query-2", ["b1", "b2", "b3"], top_k=2)

    assert len(seen_calls) == 2
    assert seen_calls[0] == [["query-1", "a1"], ["query-1", "a2"]]
    assert seen_calls[1] == [["query-2", "b1"], ["query-2", "b2"], ["query-2", "b3"]]
    # Neither call's pair list contains the other call's query or passages.
    flat_first = {p for pair in seen_calls[0] for p in pair}
    flat_second = {p for pair in seen_calls[1] for p in pair}
    assert flat_first.isdisjoint(flat_second)
