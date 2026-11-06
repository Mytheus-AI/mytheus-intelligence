"""knowledge-graph-service: the ONLY writer of the AGE `nexus_knowledge` graph.

Writes happen off the HTTP path: `worker.py` consumes `tenant-events` and seeds each new org's
`Topic` nodes (Plan 5-6 3.2); `seed_topics.py` is the operator backfill/verify CLI. All Cypher
goes through `libs/shared/age.py` (`$org_id`-bound, parameterized, frozen labels). This app
still only answers `/healthz`; the read templates (`POST /graph/query/{template}`) land with the
RAG graph arm.

Run: uvicorn services.knowledge_graph_service.main:app --port 8102
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("knowledge-graph-service")
