"""ner-service on Ray Serve: the lightweight NER model behind the PII redaction pass. Owns no data.

Same shape as `reranker_service/serve.py` and `embedding_service/serve.py`, for the same reason
ADR-0020 D5 gives: one service per model, so a model swap, a batch-size change or a scale event
in one does not move the others. `ray_actor_options` requests `num_gpus=1` only when `NER_GPU=1`
is set AND a GPU is actually visible to Ray; otherwise `num_gpus=0` and CPU, this repo's
documented dev fallback.

WARM FLOOR, AND WHY IT MATTERS MORE HERE THAN FOR THE RERANKER

`min_replicas=1`. A cold start on this service does not degrade a search result, it stalls the
path that PERSISTS A TRANSCRIPT FINAL: `transcription-service` calls this synchronously before
writing `transcript_chunks`, inside a p95 < 2 s caption SLO on a service that scales on anchor
A5 (20k concurrent live streams). ADR-0020's consequences call that availability coupling out
as the accepted cost of keeping the model off the audio proxy, and a warm floor is the part of
paying that cost which lives here. The tenant-side answer to this service being down is
`tenant_settings.ner_unavailable_policy`, not a retry.

Deliberately NO `@serve.batch`. See `model.py`: batching concurrent callers would put two
tenants' UNREDACTED text into one forward pass, which is a worse version of the risk
`reranker_service` documents, because here the input is the personal data itself.

Ray ships only with the `gpu` extra in this repo's `pyproject.toml`, so this module stays
import-safe without Ray installed and `pytest` collection never requires a cluster.
"""

from __future__ import annotations

from typing import Any

from services.ner_service.model import MIN_SCORE, MODEL_NAME, detect, gpu_requested

# Typed as Any (not the Module type inferred from the successful-import branch) so the except
# branch's `= None` is a valid assignment under mypy strict - same shape as every other
# `serve.py` in this repo.
ray: Any = None
serve: Any = None
try:
    import ray as _ray
    from ray import serve as _serve

    ray = _ray
    serve = _serve
except ModuleNotFoundError:  # Ray only present with the `gpu` extra
    pass


async def _extract_request(request: Any) -> dict[str, Any]:
    """Accept both an HTTP `Request` (Ray Serve's default adapter) and a plain dict from a
    direct handle call, matching `reranker_service.serve._extract_request`."""
    if hasattr(request, "json") and callable(request.json):
        body = await request.json()
    else:
        body = request
    return dict(body)


def actor_options() -> dict[str, Any]:
    """GPU request for this deployment, resolved against what Ray can actually see.

    NOT evaluated at decoration time, and that is the deviation from `reranker_service`'s and
    `embedding_service`'s copies of this helper: there it is defined and never called, because
    `ray.available_resources()` needs an initialized Ray and decoration happens at import. A
    function that is never called is worse than no function - it reads as configuration that is
    in effect when it is not. Here it is exported for the deployment script to pass to
    `.options(ray_actor_options=...)` at bind time, when Ray is up. Until then the GPU/CPU
    choice is made where it is actually honoured, in `model.py` off `NER_GPU`.
    """
    if ray is None:
        return {"num_gpus": 0}
    use_gpu = gpu_requested() and bool(ray.available_resources().get("GPU", 0))
    return {"num_gpus": 1} if use_gpu else {"num_gpus": 0}


if serve is not None:

    @serve.deployment(
        max_ongoing_requests=16,
        # autoscaling_config set => num_replicas must be left unset, per Ray Serve's contract
        # (passing both raises). min_replicas=1 IS the warm floor described above.
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 4,
            "target_ongoing_requests": 8,
        },
    )
    class NerDeployment:
        """One call in, one call out - no cross-request batching. See module docstring."""

        async def __call__(self, request: Any) -> dict[str, Any]:
            body = await _extract_request(request)
            spans, elapsed = detect(
                str(body["text"]), min_score=float(body.get("min_score", MIN_SCORE))
            )
            return {
                "spans": spans,
                "model": MODEL_NAME,
                "latency_ms": round(elapsed * 1000, 2),
            }

    # Ray's @serve.deployment rewrites the class into a Deployment at runtime (mypy cannot see it).
    deployment = NerDeployment.bind()  # type: ignore[attr-defined]
