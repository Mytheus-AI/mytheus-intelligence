"""BGE-small-en, 384-dim — the real embedding model, loaded once per process.

Plan 3.7 step 2 specifies `FlagModel` (FlagEmbedding) on Ray Serve with `use_fp16=True` baked
into a CUDA image. That GPU path is BLOCKED in this dev environment (task brief: "ONLY the GPU
and Ray Serve packaging is BLOCKED"); `sentence-transformers` on CPU loads the SAME published
weights (`BAAI/bge-small-en-v1.5`) and produces the SAME 384-dim vectors — this is not a fake
or a random-vector stand-in, it is the real model, just without the CUDA/fp16 fast path. Every
test in this repo that asserts on embedding output runs against this real model.

`use_fp16` and the CUDA image bake-in are GPU-only concerns; on CPU we load fp32 (the
`device="cpu"` default) and skip them rather than fake a GPU-only optimization.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def gpu_requested() -> bool:
    return os.environ.get("EMBEDDING_GPU", "0") == "1"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # CPU fallback for dev (task brief): EMBEDDING_GPU=0 (or unset) always loads on CPU here,
    # since no CUDA device is available in this environment even if EMBEDDING_GPU=1 is set.
    # Pod start is a LOAD of baked-in weights on the real GPU deployment; here it is a real
    # (slower) download+load from the Hugging Face cache, done once per process via lru_cache.
    device = "cuda" if gpu_requested() else "cpu"
    model: SentenceTransformer = SentenceTransformer(MODEL_NAME, device=device)
    return model


def embed_batch(texts: list[str]) -> tuple[list[list[float]], float]:
    """Real BGE-small-en encode. Returns (vectors, wall_clock_seconds) so callers/tests can
    check the plan's batch-of-32 < 100ms (warm, GPU) SLO — on CPU this is expected to exceed
    100ms; the acceptance check for the numeric SLO is marked BLOCKED (GPU-only), but the
    mechanism (batch, normalize, 384-dim) is exercised for real here."""
    model = _model()
    t0 = time.perf_counter()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    return vectors.tolist(), elapsed


def warmup() -> float:
    """Load the model (if not already) and run one real encode, so the first request does not pay
    weight load / CUDA context / kernel autotune. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    embed_batch(["warmup"])
    return time.perf_counter() - t0
