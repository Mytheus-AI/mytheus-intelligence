"""ingestion-service: document parsing into typed, chunked elements.

Scaffold only: boots and answers `/healthz`. `unstructured.io` parsing, blobs to object storage
under `{org_id}/`, and the `document-chunks` producer land in the ingestion band.

Run: uvicorn services.ingestion_service.main:app --port 8104
"""

from __future__ import annotations

from libs.shared.app import create_app

app = create_app("ingestion-service")
