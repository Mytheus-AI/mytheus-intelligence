"""ner-service's synchronous FastAPI process: `/healthz` (this repo's standard liveness probe,
`libs/shared/app.create_app`) plus the detect contract, `POST /ner` (`ner.py`, mounted at
`/internal` - the same routing convention as `embedding-service`'s `POST /embed` and
`reranker-service`'s `POST /rerank`).

ner-service owns no data: no Postgres table, no Qdrant credentials, no consumer, no relay. This
one process is the entire service, which is exactly `reranker-service`'s shape and is what
ADR-0020 D5 asked for when it declined to fold a third model into `embedding-service`.
"""

from __future__ import annotations

from libs.shared.app import create_app
from services.ner_service.ner import create_ner_app

SERVICE_NAME = "ner-service"

app = create_app(SERVICE_NAME)
app.mount("/internal", create_ner_app())
