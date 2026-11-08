# syntax=docker/dockerfile:1
#
# CPU image for the mytheus-intelligence tier: ghcr.io/mytheus-ai/mytheus-intelligence.
# ONE image runs every process; the compose `command:` selects the role, e.g.
#   uvicorn services.embedding_service.main:app --host 0.0.0.0 --port 8101   (default CMD)
#   python -m services.embedding_service.worker | .relay_main
#   uvicorn services.reranker_service.main:app ... | ingestion_service | search_service | ...
#   alembic upgrade head   (this repo's own `intelligence` schema history)
#
# Build from the repo root:   docker build -t mytheus-intelligence .
# Without baked HF weights:   docker build --build-arg BAKE_MODELS=0 -t mytheus-intelligence .
# GPU serving is NOT this image: see services/{embedding,reranker}_service/Dockerfile.gpu.
#
# Why torch is not installed straight from uv.lock: the lock resolves torch from PyPI, whose
# Linux wheel depends on ~3 GB of CUDA libraries (nvidia-*, cuda-toolkit, cuda-bindings,
# triton). This image runs on CPU only, so the locked sync SKIPS torch and that CUDA closure,
# and torch is then installed at the SAME locked version from the PyTorch CPU index (which
# resolves to the `+cpu` build). Every other package is exactly what uv.lock pins.

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
# --python pins the image's own interpreter: the repo's .python-version may name a patch
# release this base image does not ship (same fix as mytheus-backend/Dockerfile).
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    uv export --frozen --no-dev --no-hashes --no-emit-project > /tmp/locked.txt; \
    torch_version="$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' /tmp/locked.txt)"; \
    test -n "$torch_version"; \
    skip="torch $(sed -n -E 's/^((nvidia-|cuda-|triton)[A-Za-z0-9._-]*)==.*/\1/p' /tmp/locked.txt)"; \
    skip_flags=""; for p in $skip; do skip_flags="$skip_flags --no-install-package $p"; done; \
    uv sync --frozen --no-dev --no-install-project --python /usr/local/bin/python $skip_flags; \
    uv pip install --python /app/.venv/bin/python --no-deps \
        --index-url "$TORCH_INDEX_URL" "torch==${torch_version}"; \
    if uv pip list --python /app/.venv/bin/python | grep -E -i '^(nvidia-|cuda-|triton)'; then \
        echo "CUDA packages leaked into the CPU image" >&2; exit 1; \
    fi; \
    /app/.venv/bin/python -c "import torch, sentence_transformers; \
assert not torch.cuda.is_available(); assert torch.__version__.startswith('${torch_version}'); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# unstructured's sentence tokenizer needs spaCy's en_core_web_sm and, if it is missing,
# pip-installs it into site-packages at FIRST USE (a network fetch into a directory the
# non-root runtime user cannot write). Trigger unstructured's own hash-pinned install here.
RUN /app/.venv/bin/python -c "from unstructured.nlp.tokenize import sent_tokenize; \
print(sent_tokenize('Model baked. Second sentence.'))"


# Hugging Face weights, baked so a container start is a LOAD, never a download
# (libs/shared/model_loading.py loads local-first). BAKE_MODELS=0 leaves /opt/hf empty:
# mount a populated HF cache there and set HF_HUB_OFFLINE=0 if you need that variant.
FROM builder AS models
ARG BAKE_MODELS=1
ENV HF_HOME=/opt/hf
RUN mkdir -p /opt/hf && if [ "$BAKE_MODELS" = "1" ]; then \
    /app/.venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
CrossEncoder('BAAI/bge-reranker-base', device='cpu')" \
    && rm -rf /opt/hf/xet /opt/hf/hub/.locks; \
    fi


FROM ${PYTHON_IMAGE} AS runtime
# libmagic1: python-magic (unstructured's file-type sniffing). tini: PID 1, signal forwarding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 -m appuser
WORKDIR /app
# Code, venv and weights stay root-owned: the runtime user reads them and writes nothing here.
COPY --from=builder /app/.venv /app/.venv
COPY --from=models /opt/hf /opt/hf
COPY alembic.ini ./
COPY migrations ./migrations
COPY proto ./proto
COPY libs ./libs
COPY services ./services
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EMBEDDING_GPU=0 \
    RERANKER_GPU=0
USER 10001
# Container ports used by compose.prod.yml: embedding 8101, reranker 8102, ingestion 8103,
# search 8104, knowledge-graph 8106. Workers expose /metrics on METRICS_PORT.
EXPOSE 8101 8102 8103 8104 8106

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "services.embedding_service.main:app", "--host", "0.0.0.0", "--port", "8101"]
