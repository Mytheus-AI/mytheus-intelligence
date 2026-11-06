"""reranker-service: BGE-reranker-base cross-encoder rerank on Ray Serve. Owns no data.

GPU status (task brief, scope note): "ONLY the GPU and its latency SLO are BLOCKED." This
module IS the real Ray Serve deployment — it runs for real in this environment, against a real
Ray cluster (`ray start --head` / Ray Serve's local dev mode), loading the real
`BAAI/bge-reranker-base` weights via `services.reranker_service.model`. What is BLOCKED is only:
(a) the CUDA device / `use_fp16=True` fast path (no GPU in this dev environment), and (b) the
numeric warm p95 < 500ms / cold < 1.5s SLOs, which are GPU-only targets. `ray_actor_options`
requests `num_gpus=1` only when `RERANKER_GPU=1` is set AND a GPU is actually visible to Ray;
otherwise it requests `num_gpus=0` and runs on CPU (this repo's documented CPU fallback for dev,
matching `services/embedding_service/serve.py`'s identical shape).

Warm-replica floor (plan 3.8 step 3: "Warm-replica floor >= 1 per active region, never scale to
zero during business hours"): `autoscaling_config`'s `min_replicas=1` is Ray Serve's mechanism
for that floor — Ray Serve never scales a deployment below `min_replicas`, so this deployment
is never scaled to zero as long as it is running. (Region-level "never scale to zero during
business hours" scheduling — i.e. suspending the floor outside business hours per region — is a
deployment/ops-schedule concern above this module; not something a single Ray Serve deployment
config expresses on its own, and out of this task's code-level scope.)

Deliberately NO `@serve.batch` here, unlike `embedding_service.serve.IngestEmbedder`. See
`model.py`'s module docstring for why: `IngestEmbedder` batches independent Kafka messages
(safe — one embed-on-ingest event is already one org_id's own chunk), but fanning concurrent
`/rerank` HTTP callers into one shared cross-encoder batch would mean passages from two
different tenants' requests could end up scored together in the same `model.predict()` call —
exactly the cross-tenant risk the task brief called out. Each `__call__` here scores exactly one
request's own `query`/`passages`, in its own `rerank_batch()` invocation, every time.

Ray ships only with the optional `gpu` extra in this repo's `pyproject.toml` (a legacy label —
Ray Serve itself is CPU-capable; the label predates this task, see embedding-service's serve.py
for the same note). `serve.py` stays import-safe without Ray installed, matching every other
service stub in this repo, so `pytest` collection never requires a Ray cluster.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from services.reranker_service.model import gpu_requested, rerank_batch, warmup

# Typed as Any (not inferred as the `ray`/`ray.serve` Module type from the successful-import
# branch) so the except branch's `= None` is a valid assignment under mypy strict mode — same
# shape as services/embedding_service/serve.py and this file's own prior Plan 0 stub.
ray: Any = None
serve: Any = None
try:
    import ray as _ray
    from ray import serve as _serve

    ray = _ray
    serve = _serve
except ModuleNotFoundError:  # Ray only present with the `gpu` extra
    pass


def min_replicas() -> int:
    """Warm-replica floor. Default 1 (never scale to zero); RERANKER_MIN_REPLICAS raises it
    (e.g. 2 per active region in prod). Clamped to >= 1: a floor of 0 would reintroduce the
    cold start this module exists to prevent."""
    return max(1, int(os.environ.get("RERANKER_MIN_REPLICAS", "1")))


def max_replicas() -> int:
    return max(min_replicas(), int(os.environ.get("RERANKER_MAX_REPLICAS", "4")))


async def _extract_request(request: Any) -> dict[str, Any]:
    """Accept both an HTTP `Request` (Ray Serve's default adapter) and a plain dict from a
    direct handle call (`handle.remote(...)`, used by tests and future intra-cluster callers),
    matching `embedding_service.serve._extract_texts`'s dual-path shape."""
    if hasattr(request, "json") and callable(request.json):
        body = await request.json()
    else:
        body = request
    return dict(body)


if serve is not None:

    def _actor_options() -> dict[str, Any]:
        use_gpu = gpu_requested() and bool(ray.available_resources().get("GPU", 0))
        return {"num_gpus": 1} if use_gpu else {"num_gpus": 0}

    @serve.deployment(
        max_ongoing_requests=16,
        # autoscaling_config set => num_replicas must be left unset ("auto"), per Ray Serve's
        # own contract (passing both raises); min_replicas=1 IS the warm floor.
        autoscaling_config={
            "min_replicas": min_replicas(),
            "max_replicas": max_replicas(),
            "target_ongoing_requests": 8,
            # Keep-warm: scale-down is slow (10 min of low load) and never below the floor, so a
            # lull does not evict a replica whose next request would be cold.
            "downscale_delay_s": 600,
            "upscale_delay_s": 10,
        },
    )
    class RerankerDeployment:
        """One call in, one call out — no cross-request batching. See module docstring."""

        def __init__(self) -> None:
            # Ray Serve does not route to a replica until __init__ returns, so loading the model
            # and running one real forward pass here means a replica is only "ready" once warm
            # (first request is not cold). On GPU this also absorbs CUDA context creation and
            # kernel autotune. GPU timings are UNMEASURED in this environment (no CUDA device).
            seconds = warmup()
            logging.getLogger(__name__).info("reranker replica warmed in %.2fs", seconds)

        async def __call__(self, request: Any) -> dict[str, Any]:
            body = await _extract_request(request)
            query: str = body["query"]
            passages: list[str] = body["passages"]
            top_k = int(body.get("top_k", 10))
            ranked, elapsed = rerank_batch(
                query, passages, top_k=max(1, min(top_k, len(passages) or 1))
            )
            return {
                "results": [{"passage": p, "score": s} for p, s in ranked],
                "model": "BAAI/bge-reranker-base",
                "latency_ms": round(elapsed * 1000, 2),
            }

    # Ray's @serve.deployment rewrites the class into a Deployment at runtime (mypy can't see it).
    deployment = RerankerDeployment.bind()  # type: ignore[attr-defined]
