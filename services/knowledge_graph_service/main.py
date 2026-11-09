"""knowledge-graph-service: the ONLY writer of the AGE `nexus_knowledge` graph.

Writes happen off the HTTP path, in `worker.py`: `tenant-events` seeds each new org's `Topic`
nodes (Plan 5-6 3.2), and confirmed `decisions` / `actions` plus `meeting-room-events` joins
write the edges (Plan 7-9 3.7, `facts_consumer.py`). `seed_topics.py` and `rebuild.py` are the
operator CLIs. All Cypher goes through `libs/shared/age.py` (`$org_id`-bound, parameterized,
frozen labels).

This app serves the read side: `POST /graph/query/{template}` and `POST /graph/cypher`
(`routes.py`), internal-only. The same templates are served over gRPC by `grpc_main.py`.

Run: uvicorn services.knowledge_graph_service.main:app --port 8202
"""

from __future__ import annotations

from libs.shared.app import create_app
from services.knowledge_graph_service.routes import router

app = create_app("knowledge-graph-service")
app.include_router(router)
