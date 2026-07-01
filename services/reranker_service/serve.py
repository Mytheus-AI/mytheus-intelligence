"""reranker-service: GPU cross-encoder rerank (BGE-reranker-base) on Ray Serve.

Scaffold only. Ray ships only with the optional `gpu` extra, so this module stays import-safe
without it: when Ray is absent the deployment is simply not defined (the service is skipped in
dev). The real cross-encoder with a warm-replica floor (top-50 -> top-10) lands later.

Serve (with the `gpu` extra installed): serve run services.reranker_service.serve:deployment
"""

from __future__ import annotations

from typing import Any

try:
    from ray import serve
except ModuleNotFoundError:  # Ray only present with the `gpu` extra
    serve = None

if serve is not None:

    @serve.deployment(num_replicas=1)
    class RerankerDeployment:
        async def __call__(self, request: Any) -> dict[str, Any]:
            # Dev stub: no model loaded. Real BGE-reranker-base cross-encoder scoring lands in
            # the inference band; this proves the Ray Serve contract exists without a GPU.
            return {"status": "stub", "model": "BGE-reranker-base"}

    # Ray's @serve.deployment rewrites the class into a Deployment at runtime (mypy can't see it).
    deployment = RerankerDeployment.bind()  # type: ignore[attr-defined]
