"""rag-service: the frozen five-stage RAGService (expand, retrieve, rerank, assemble, cite).

Scaffold only: boots and answers `/healthz`. RRF fusion across the Qdrant/AGE/FTS arms over the
internal gRPC mesh (org_id in metadata) lands in the RAG band.

Run: uvicorn services.rag_service.main:app --port 8101
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("rag-service")
