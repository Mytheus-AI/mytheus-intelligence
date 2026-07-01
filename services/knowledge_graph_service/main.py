"""knowledge-graph-service: the ONLY writer of the AGE `nexus_knowledge` graph.

Scaffold only: boots and answers `/healthz`. Cypher parameterized by `$org_id`, driven by confirmed
`decisions`/`actions` from core-api, lands in the knowledge-graph band. No silent graph writes.

Run: uvicorn services.knowledge_graph_service.main:app --port 8102
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("knowledge-graph-service")
