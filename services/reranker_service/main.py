"""reranker-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness
probe, `libs/shared/app.create_app`) plus the rerank contract, `POST /rerank`
(`rerank.py`, mounted at `/internal` — same routing convention as `embedding-service`'s
`POST /embed`). reranker-service owns no data: no Postgres table, no Qdrant creds, no consumer,
no relay — this one process is the entire service.
"""

from __future__ import annotations

from libs.shared.app import create_app
from services.reranker_service.rerank import create_rerank_app

SERVICE_NAME = "reranker-service"

app = create_app(SERVICE_NAME)
app.mount("/internal", create_rerank_app())
