"""`POST /ingest` (file/url) and `GET /ingest/jobs/:id` — plan step 4's surfaces, verbatim
paths.

Security fix (whole-band review, Important finding): `org_id` used to come from a
client-supplied `X-Org-Id` header, fail-closed only on a malformed UUID — the SAME interim
REST-authn contract `embedding-service`'s `query_embed.py` and `reranker-service`'s `rerank.py`
had also established. Nothing verified the caller held a legitimate identity for that org, so
anyone with network reachability could carry another organization's id in that header and write
into that org's MinIO objects. `org_id` now comes ONLY from `libs.shared.auth.verified_org_id`
— a Clerk RS256 session token, verified against JWKS, whose claim is this repo's own resolved
internal `org_id` (see `libs/shared/auth.py`'s module docstring for why this repo cannot do
Clerk-opaque-id resolution itself). `org_id` still NEVER comes from the request body, matching
`libs/shared/object_storage.py`'s key-safety contract.

Processing is synchronous within the request (no background job queue): this repo has no task
runner anywhere else to build on top of, and a synchronous MVP is simpler, honest, and fully
testable against real infra rather than inventing new queueing infrastructure for one endpoint.
`ingestion_jobs` still exists and is still meaningful — `GET /ingest/jobs/:id` reflects the
terminal `completed`/`failed` state the synchronous call already reached, and the row is the
real audit trail of what was attempted, not a shim.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import boto3
import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker

from libs.shared.auth import verified_org_id
from libs.shared.events import EventEnvelope
from libs.shared.object_storage import (
    BUCKET_DOCUMENTS,
    document_key,
    put_object,
    safe_extension,
)
from libs.shared.outbox import enqueue as outbox_enqueue
from libs.shared.tenancy import set_rls
from services.ingestion_service.pipeline import complete_job, create_job, fail_job, ingest_document

MAX_DOCUMENT_BYTES = 25 * 1024 * 1024  # 25 MB; a sane upper bound, not specified by the plan


class JobResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    chunk_count: int = 0
    element_count: int = 0
    error: str | None = None


def _filename_from_url(url: str) -> str:
    from urllib.parse import urlparse

    path = urlparse(url).path
    return path.rsplit("/", 1)[-1] or "downloaded"


def create_ingestion_router(
    *,
    session_factory: async_sessionmaker,  # type: ignore[type-arg]
    s3_client: boto3.client,
    redis: Redis,
) -> APIRouter:
    """An `APIRouter`, not a mounted sub-app: `libs/shared/app.create_app`'s `/healthz` already
    lives at the service's root, and mounting a second app at `""`/`"/"` (as `embedding-service`
    does at the DISTINCT prefix `/internal`) would shadow it here, since the plan's own paths
    (`POST /ingest`, `GET /ingest/jobs/:id`) are root-level, not `/internal`-prefixed. Including
    a router avoids that collision entirely."""
    router = APIRouter()

    @router.post("/ingest", response_model=JobResponse, status_code=201)
    async def ingest(
        org_id: UUID = Depends(verified_org_id),  # noqa: B008 - FastAPI's own required pattern
        file: UploadFile | None = File(None),  # noqa: B008 - FastAPI's own required pattern
        url: str | None = Form(None),
    ) -> JobResponse:
        if (file is None) == (url is None):
            raise HTTPException(status_code=422, detail="provide exactly one of `file` or `url`")

        if file is not None:
            filename = file.filename or ""
            content = await file.read()
        else:
            assert url is not None
            filename = _filename_from_url(url)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content = resp.content

        if len(content) > MAX_DOCUMENT_BYTES:
            raise HTTPException(status_code=413, detail="document exceeds the 25 MB ingest limit")

        ext = safe_extension(filename)
        if ext is None:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported document type for {filename!r}; "
                "supported: .pdf .docx .pptx .html .htm .md .eml .txt",
            )

        document_id = uuid4()  # ALWAYS server-generated — see libs/shared/object_storage.py
        key = document_key(org_id, document_id, ext)
        put_object(
            s3_client,
            bucket=BUCKET_DOCUMENTS,
            key=key,
            body=content,
            content_type=file.content_type if file else None,
        )

        async with session_factory() as session, session.begin():
            await set_rls(session, org_id)
            await create_job(
                session,
                job_id=document_id,
                org_id=org_id,
                source="upload",
                bucket=BUCKET_DOCUMENTS,
                object_key=key,
            )

        try:
            async with session_factory() as session, session.begin():
                await set_rls(session, org_id)

                async def emit(topic: str, idem: str, payload: dict) -> None:  # type: ignore[type-arg]
                    envelope = EventEnvelope(
                        topic=topic,
                        payload=payload,
                        org_id=str(org_id),
                        producer="ingestion-service",
                        idempotency_key=idem,
                    )
                    await outbox_enqueue(session, envelope)

                outcome = await ingest_document(
                    session=session,
                    org_id=org_id,
                    document_id=document_id,
                    filename=filename,
                    content=content,
                    emit=emit,
                    redis=redis,
                )
                await complete_job(
                    session, job_id=document_id, org_id=org_id, chunk_count=outcome.chunk_count
                )
        except Exception as exc:  # noqa: BLE001 - always record the failure on the job row
            async with session_factory() as session, session.begin():
                await set_rls(session, org_id)
                await fail_job(session, job_id=document_id, org_id=org_id, error=str(exc))
            raise HTTPException(status_code=422, detail=f"ingestion failed: {exc}") from exc

        return JobResponse(
            job_id=str(document_id),
            document_id=str(document_id),
            status="completed",
            chunk_count=outcome.chunk_count,
            element_count=outcome.element_count,
        )

    @router.get("/ingest/jobs/{job_id}", response_model=JobResponse)
    async def get_job(
        job_id: str,
        org_id: UUID = Depends(verified_org_id),  # noqa: B008 - FastAPI's own required pattern
    ) -> JobResponse:
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="job not found") from None

        from sqlalchemy import text

        async with session_factory() as session, session.begin():
            await set_rls(session, org_id)
            row = (
                await session.execute(
                    text(
                        "SELECT id, status, chunk_count, error FROM intelligence.ingestion_jobs "
                        "WHERE id = :id"
                    ),
                    {"id": str(job_uuid)},
                )
            ).first()

        if row is None:
            # RLS makes another tenant's job row invisible here too — same 404, no distinction
            # leaked between "does not exist" and "exists but is not yours".
            raise HTTPException(status_code=404, detail="job not found")

        return JobResponse(
            job_id=str(row.id),
            document_id=str(row.id),
            status=row.status,
            chunk_count=row.chunk_count,
            error=row.error,
        )

    return router
