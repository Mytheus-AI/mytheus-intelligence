"""`POST /ingest` and `GET /ingest/jobs/:id` end to end through a real FastAPI `TestClient`,
against real Postgres, real MinIO, and real Redis (no mocked infra). Boots the app the same way
`main.py` does (`create_app` + `include_router`), so `/healthz` and the ingest routes are proven
to coexist on one app (the reason `routes.py` returns an `APIRouter`, not a mounted sub-app)."""

from __future__ import annotations

import http.server
import threading
import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from libs.shared.app import create_app
from libs.shared.object_storage import BUCKET_DOCUMENTS, document_key, ensure_bucket
from services.ingestion_service.routes import create_ingestion_router
from tests.conftest import APP_URL, auth_headers
from tests.fixtures import sample_documents as docs


@pytest.fixture()
def client(migrated_db, s3_client, redis_client):
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app("ingestion-service")
    app.include_router(
        create_ingestion_router(
            session_factory=session_factory, s3_client=s3_client, redis=redis_client
        )
    )
    # `with TestClient(...)` (not a bare instance) pins one consistent anyio portal/event loop
    # for the whole client's lifetime, so a pooled asyncpg connection opened on request 1 is
    # still valid on request 2 — a bare `TestClient(app)` can otherwise spin a fresh loop per
    # call and asyncpg's pooled connections then fail with "attached to a different loop".
    with TestClient(app) as c:
        yield c


def test_healthz_still_works_alongside_the_ingest_router(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "ingestion-service"}


def test_post_ingest_missing_token_is_401(client):
    resp = client.post("/ingest", files={"file": ("notes.md", docs.make_md(), "text/markdown")})
    assert resp.status_code == 401


def test_post_ingest_malformed_token_is_401(client):
    resp = client.post(
        "/ingest",
        headers={"Authorization": "Bearer not-a-real-token"},
        files={"file": ("notes.md", docs.make_md(), "text/markdown")},
    )
    assert resp.status_code == 401


def test_post_ingest_with_another_orgs_id_in_the_header_but_no_valid_token_is_rejected(
    client, s3_client
):
    """The exploit this whole fix closes: `X-Org-Id` carrying a victim org's id used to be
    trusted outright. A request that supplies it but presents no valid bearer token must now
    be rejected with 401 BEFORE any store is touched — proven here by confirming the object
    never lands in `mytheus-documents` under that org's prefix."""
    victim_org = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers={"X-Org-Id": str(victim_org)},  # the old, now-untrusted header
        files={"file": ("notes.md", docs.make_md(), "text/markdown")},
    )
    assert resp.status_code == 401

    # Nothing was written under the victim org's object-storage prefix.
    ensure_bucket(s3_client, BUCKET_DOCUMENTS)  # a fresh MinIO has no bucket until something writes
    listing = s3_client.list_objects_v2(Bucket=BUCKET_DOCUMENTS, Prefix=f"{victim_org}/")
    assert listing.get("KeyCount", 0) == 0


def test_post_ingest_neither_file_nor_url_is_422(client):
    org_id = uuid.uuid4()
    resp = client.post("/ingest", headers=auth_headers(org_id))
    assert resp.status_code == 422


def test_post_ingest_both_file_and_url_is_422(client):
    org_id = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers=auth_headers(org_id),
        files={"file": ("notes.md", docs.make_md(), "text/markdown")},
        data={"url": "http://example.invalid/notes.md"},
    )
    assert resp.status_code == 422


def test_post_ingest_unsupported_extension_is_422(client):
    org_id = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers=auth_headers(org_id),
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_post_ingest_uploads_real_object_and_writes_real_chunks(client, s3_client):
    org_id = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers=auth_headers(org_id),
        files={
            "file": (
                "notes.docx",
                docs.make_docx(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["chunk_count"] > 0
    document_id = uuid.UUID(body["document_id"])

    # The object really landed in mytheus-documents at the expected, filename-blind key.
    key = document_key(org_id, document_id, ".docx")
    try:
        head = s3_client.head_object(Bucket=BUCKET_DOCUMENTS, Key=key)
        assert head["ContentLength"] > 0
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key)

    # GET /ingest/jobs/:id, same org — reflects the terminal state.
    get_resp = client.get(f"/ingest/jobs/{document_id}", headers=auth_headers(org_id))
    assert get_resp.status_code == 200
    job = get_resp.json()
    assert job["status"] == "completed"
    assert job["chunk_count"] == body["chunk_count"]


def test_post_ingest_of_a_sample_pdf_is_the_plans_literal_acceptance_check(client, s3_client):
    """The plan's acceptance check, verbatim: 'POST /ingest of a sample PDF writes the blob to
    mytheus-documents/{org_id}/{document_id}/, rows into document_chunks with element_type in
    {Title, NarrativeText, Table, ListItem}, and emits one document-chunks event per chunk.'"""
    org_id = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers=auth_headers(org_id),
        files={"file": ("kickoff.pdf", docs.make_pdf(), "application/pdf")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["chunk_count"] > 0
    document_id = uuid.UUID(body["document_id"])

    key = document_key(org_id, document_id, ".pdf")
    assert key == f"{org_id}/{document_id}/source.pdf"  # mytheus-documents/{org_id}/{document_id}/
    try:
        head = s3_client.head_object(Bucket=BUCKET_DOCUMENTS, Key=key)
        assert head["ContentLength"] > 0

        # Plain sync psycopg2 (like conftest.py's own fixtures), not the async SQLAlchemy engine
        # — this test function is sync (it drives the sync TestClient), and mixing a fresh
        # asyncio engine into TestClient's own anyio portal thread is exactly the "attached to a
        # different loop" hazard the `client` fixture's docstring already explains avoiding.
        import psycopg2

        conn = psycopg2.connect(APP_URL.replace("+asyncpg", ""))
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('app.current_org', %s, false)", (str(org_id),))
                cur.execute(
                    "SELECT element_type FROM intelligence.document_chunks "
                    "WHERE org_id = %s AND document_id = %s",
                    (str(org_id), str(document_id)),
                )
                element_types = [row[0] for row in cur.fetchall()]
                cur.execute(
                    "SELECT count(*) FROM intelligence.outbox "
                    "WHERE org_id = %s AND topic = 'document-chunks'",
                    (str(org_id),),
                )
                outbox_count = cur.fetchone()[0]
        finally:
            conn.close()

        assert len(element_types) == body["chunk_count"]
        assert all(t in {"Title", "NarrativeText", "Table", "ListItem"} for t in element_types)
        assert outbox_count == body["chunk_count"]  # one document-chunks event per chunk
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key)


def test_get_job_from_a_different_org_is_404_not_a_leak(client, s3_client):
    org_id = uuid.uuid4()
    other_org_id = uuid.uuid4()
    resp = client.post(
        "/ingest",
        headers=auth_headers(org_id),
        files={"file": ("notes.txt", docs.make_txt(), "text/plain")},
    )
    assert resp.status_code == 201
    document_id = resp.json()["document_id"]

    try:
        wrong_org_resp = client.get(
            f"/ingest/jobs/{document_id}", headers=auth_headers(other_org_id)
        )
        assert wrong_org_resp.status_code == 404

        right_org_resp = client.get(f"/ingest/jobs/{document_id}", headers=auth_headers(org_id))
        assert right_org_resp.status_code == 200
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=f"{org_id}/{document_id}/source.txt")


def test_get_job_that_never_existed_is_404(client):
    org_id = uuid.uuid4()
    resp = client.get(f"/ingest/jobs/{uuid.uuid4()}", headers=auth_headers(org_id))
    assert resp.status_code == 404


def test_get_job_with_malformed_job_id_is_404(client):
    org_id = uuid.uuid4()
    resp = client.get("/ingest/jobs/not-a-uuid", headers=auth_headers(org_id))
    assert resp.status_code == 404


class _OneShotHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        body = docs.make_md()
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:  # silence stdlib's default access logging
        pass


@pytest.fixture()
def local_http_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _OneShotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/kickoff.md"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_post_ingest_url_fetches_over_a_real_http_server(client, s3_client, local_http_server):
    """A REAL local TCP HTTP server (Python's stdlib `http.server`), not a mocked `httpx`
    transport — the url-ingest branch really performs an HTTP GET."""
    org_id = uuid.uuid4()
    resp = client.post("/ingest", headers=auth_headers(org_id), data={"url": local_http_server})
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["chunk_count"] > 0

    document_id = uuid.UUID(body["document_id"])
    key = document_key(org_id, document_id, ".md")
    try:
        head = s3_client.head_object(Bucket=BUCKET_DOCUMENTS, Key=key)
        assert head["ContentLength"] > 0
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key)
