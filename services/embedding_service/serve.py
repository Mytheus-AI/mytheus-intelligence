"""embedding-service: BGE-small-en, 384-dim, on Ray Serve. The only writer of Qdrant.

Plan 3.7 step 2/5: two SEPARATE Ray Serve deployments sharing the model but separating the
lanes, so a query-embed burst never starves the embed-on-ingest consumer's batch throughput.
`IngestEmbedder` backs `EmbeddingConsumer` (dynamic micro-batching, batch 32, 50ms wait);
`QueryEmbedder` backs the internal `POST /embed` endpoint (`query_embed.py`), 1-3 query
variants at a time.

GPU status (task brief, scope note): "ONLY the GPU and Ray Serve packaging is BLOCKED." This
module IS the real Ray Serve deployment — it is written and runs for real in this environment,
against a real Ray cluster (`ray start --head` / Ray Serve's local dev mode), loading the real
`BAAI/bge-small-en-v1.5` weights via `services.embedding_service.model`. What is BLOCKED is
only: (a) the CUDA image / `use_fp16=True` fast path (no GPU in this dev environment), and
(b) the numeric batch-of-32 < 100ms SLO, which is a GPU-only target. `ray_actor_options`
requests `num_gpus=1` only when `EMBEDDING_GPU=1` is set AND a GPU is actually visible to Ray;
otherwise it requests `num_gpus=0` and runs on CPU (the plan's own documented CPU fallback for
dev). `FlagModel`/`use_fp16` (FlagEmbedding, plan's literal snippet) is a GPU-oriented library;
`services.embedding_service.model` uses `sentence-transformers` instead so the SAME code path
works on CPU today and needs only the `device="cuda"` + fp16 toggle to run for real on a GPU
later — not a second implementation to maintain.

Ray ships only with the optional `gpu` extra name in this repo's `pyproject.toml` (a legacy
label from the Plan 0 scaffold — Ray Serve itself is the CPU-capable serving framework; the
label predates this task). `serve.py` stays import-safe without Ray installed, matching every
other service stub in this repo, so `pytest` collection never requires a Ray cluster.
"""

from __future__ import annotations

from typing import Any

from services.embedding_service.model import embed_batch, gpu_requested

# Typed as Any (not inferred as the `ray`/`ray.serve` Module type from the successful-import
# branch) so the except branch's `= None` is a valid assignment under mypy strict mode — the
# same fix applies to services/reranker_service/serve.py's identical try/except shape.
ray: Any = None
serve: Any = None
try:
    import ray as _ray
    from ray import serve as _serve

    ray = _ray
    serve = _serve
except ModuleNotFoundError:  # Ray only present with the `gpu` extra
    pass


async def _extract_texts(request: Any) -> list[str]:
    """Ray Serve's default HTTP adapter (no `@serve.ingress` FastAPI mount) hands `__call__`
    a Starlette `Request` for an HTTP call, but a plain dict/list for a direct Python/gRPC-style
    handle call (`handle.remote(...)`, used by tests and by future intra-cluster callers). Accept
    both rather than assume one."""
    if hasattr(request, "json") and callable(request.json):
        body = await request.json()
    else:
        body = request
    if isinstance(body, list):
        return list(body)
    texts: list[str] = body.get("texts", [])
    return texts


if serve is not None:

    def _actor_options() -> dict[str, Any]:
        use_gpu = gpu_requested() and bool(ray.available_resources().get("GPU", 0))
        return {"num_gpus": 1} if use_gpu else {"num_gpus": 0}

    @serve.deployment(num_replicas=1, max_ongoing_requests=64)
    class IngestEmbedder:
        """The embed-on-ingest lane: dynamic micro-batching, batch 32, sub-100ms wait window
        (the SLO itself is GPU-only and BLOCKED here; the batching mechanism is real)."""

        # `serve` is typed Any (see the module-level note above), so mypy strict treats the
        # decorator itself as untyped; the method body is still fully checked.
        @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)  # type: ignore[untyped-decorator]
        async def embed(self, texts: list[str]) -> list[list[float]]:
            vectors, _elapsed = embed_batch(texts)
            return vectors

        async def __call__(self, request: Any) -> dict[str, Any]:
            # Each caller passes ONE text; `@serve.batch` fans concurrent single-item calls
            # from multiple in-flight requests into one batched `embed(texts: list[str])`
            # call under the hood (Ray Serve's documented micro-batching contract) — a caller
            # never passes the list itself.
            texts = await _extract_texts(request)
            vectors = [await self.embed(t) for t in texts]
            return {"vectors": vectors, "dim": 384, "model": "BAAI/bge-small-en-v1.5"}

    @serve.deployment(num_replicas=1, max_ongoing_requests=16)
    class QueryEmbedder:
        """The query-embed lane: `POST /embed`, internal only, 1-3 query variants at a time.
        A SEPARATE deployment from `IngestEmbedder` (plan 3.7 step 2) so a burst of live-query
        embeds never queues behind the ingest consumer's batch-32 backlog, even though both
        load the same model weights."""

        @serve.batch(max_batch_size=3, batch_wait_timeout_s=0.01)  # type: ignore[untyped-decorator]
        async def embed(self, texts: list[str]) -> list[list[float]]:
            vectors, _elapsed = embed_batch(texts)
            return vectors

        async def __call__(self, request: Any) -> dict[str, Any]:
            texts = await _extract_texts(request)
            vectors = [await self.embed(t) for t in texts]
            return {"vectors": vectors, "dim": 384, "model": "BAAI/bge-small-en-v1.5"}

    # Ray's @serve.deployment rewrites the class into a Deployment at runtime (mypy can't see it).
    ingest_deployment = IngestEmbedder.options(  # type: ignore[attr-defined]
        ray_actor_options=_actor_options()
    ).bind()
    query_deployment = QueryEmbedder.options(  # type: ignore[attr-defined]
        ray_actor_options=_actor_options()
    ).bind()
    # Backward-compatible alias with the Plan 0 stub's export name.
    deployment = ingest_deployment
