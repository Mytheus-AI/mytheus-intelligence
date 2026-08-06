"""BGE-reranker-base cross-encoder — the real reranker model, loaded once per process.

Plan 3.8 step 1 specifies `FlagReranker` (FlagEmbedding) with `use_fp16=True` on a GPU Ray
Serve actor. That GPU/fp16 fast path is BLOCKED in this dev environment (no CUDA device); this
module instead loads `BAAI/bge-reranker-base` — the SAME published weights — through
`sentence_transformers.CrossEncoder`, which wraps the identical
`AutoModelForSequenceClassification` architecture `FlagReranker` uses. This is not a fake or a
stand-in scorer: every score this module returns comes from a real forward pass of the real
model, exactly the same substitution `services/embedding_service/model.py` made for BGE-small-en
(task 7 precedent) so the same library covers both GPU model servers in this repo without a
second inference dependency. `use_fp16` and the CUDA device are GPU-only concerns; on CPU this
loads fp32 (`device="cpu"`) and skips them rather than fake a GPU-only optimization.

CROSS-TENANT ISOLATION — the property that matters most for this service (task brief):
`rerank_batch()` builds its `[[query, passage], ...]` pairs from exactly ONE call's own
`query`/`passages` arguments and scores them in exactly ONE `model.predict()` invocation. There
is no queue, cache, or `@serve.batch`-style fan-in anywhere in this module or in `serve.py` that
could merge two different callers' passages into a single batch before scoring. This is a
deliberate choice, not an oversight: a cross-encoder is expensive, so batching concurrent
`/rerank` calls together is the obvious throughput optimization — and it is also the obvious way
to smuggle one tenant's passages into a model call scoped to a different tenant's `org_id`, or to
misroute a scored result back to the wrong caller through an index-alignment bug. The plan's own
reference pseudocode (`t8-3.8-reranker.md` step 1) already reflects this: it has no batching
decorator, only one `rerank(query, passages, top_k)` call scoring one caller's own list. We kept
that shape rather than adding cross-request batching for it. If a future task wants GPU
throughput batching here, it must partition any shared batch by `org_id` so passages from two
tenants are never concatenated into the same `model.predict()` list, and must not let scored
results be redistributed to callers by anything looser than a same-request identity — that
design is NOT implemented here; flagging it rather than building it, per the task brief.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"


def gpu_requested() -> bool:
    return os.environ.get("RERANKER_GPU", "0") == "1"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    # CPU fallback for dev (task brief): RERANKER_GPU=0 (or unset) always loads on CPU here,
    # since no CUDA device is available in this environment even if RERANKER_GPU=1 is set.
    device = "cuda" if gpu_requested() else "cpu"
    model: CrossEncoder = CrossEncoder(MODEL_NAME, device=device)
    return model


def rerank_batch(
    query: str, passages: list[str], top_k: int = 10
) -> tuple[list[tuple[str, float]], float]:
    """Real BGE-reranker-base joint (query, passage) scoring for THIS CALL's passages only
    (see the module docstring's cross-tenant note — this is never fed passages from more than
    one caller). Returns (ranked[:top_k] descending by score, wall_clock_seconds) so callers/
    tests can check the plan's warm p95 < 500ms (GPU) SLO — on CPU this is expected to exceed
    500ms; the numeric SLO is marked BLOCKED (GPU-only), but the real scoring + ranking
    mechanism is exercised for real here.
    """
    if not passages:
        return [], 0.0
    model = _model()
    pairs = [[query, p] for p in passages]
    t0 = time.perf_counter()
    # sentence-transformers' CrossEncoder.predict stub is invariant on its input list type and
    # does not resolve cleanly to list[list[str]] under mypy strict, even though this is exactly
    # the documented call shape for a reranker (proven at runtime by the real-model tests in
    # tests/test_reranker.py).
    scores = model.predict(pairs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    ranked = sorted(
        zip(passages, (float(s) for s in scores), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    k = max(1, min(top_k, len(ranked)))
    return ranked[:k], elapsed
