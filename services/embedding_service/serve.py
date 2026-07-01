"""embedding-service: GPU embeddings (BGE-small-en, 384-dim). The only writer of Qdrant.

Scaffold only. Ray ships only with the optional `gpu` extra, so this module stays import-safe
without it: when Ray is absent the deployment is simply not defined (the service is skipped in
dev). The real BGE-small-en model on Ray Serve/vLLM lands in the inference band.

Serve (with the `gpu` extra installed): serve run services.embedding_service.serve:deployment
"""

from __future__ import annotations

from typing import Any

try:
    from ray import serve
except ModuleNotFoundError:  # Ray only present with the `gpu` extra
    serve = None

if serve is not None:

    @serve.deployment(num_replicas=1)
    class EmbeddingDeployment:
        async def __call__(self, request: Any) -> dict[str, Any]:
            # Dev stub: no model loaded. Real 384-dim BGE-small-en embeddings land in the
            # inference band; this proves the Ray Serve contract exists without a GPU.
            return {"status": "stub", "model": "BGE-small-en", "dim": 384}

    # Ray's @serve.deployment rewrites the class into a Deployment at runtime (mypy can't see it).
    deployment = EmbeddingDeployment.bind()  # type: ignore[attr-defined]
