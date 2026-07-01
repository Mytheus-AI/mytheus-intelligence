"""search-service: the Postgres FTS arm of hybrid retrieval.

Scaffold only: boots and answers `/healthz`. `tsvector` over `transcript_chunks` + `chat_messages`,
RLS by `org_id`, served to rag-service over gRPC, lands in the RAG band.

Run: uvicorn services.search_service.main:app --port 8103
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("search-service")
