r"""Object storage chokepoint — the ONLY module in this repo allowed to construct an S3 key or
touch `mytheus-documents` / `mytheus-attachments`. Mirrors `libs/shared/qdrant.py`'s role for
Qdrant: one resolution function, everything else calls through it, so "everyone remembers to
do it right" is never the isolation mechanism.

Real MinIO (S3-API compatible), not a fake. Plan 5-6 task 9's scope note: "MinIO is live...
Create the real bucket, upload real bytes, read them back. Do NOT write a fake S3 client." Every
test in this repo that asserts on object storage behavior runs against the real endpoint
(`MINIO_ENDPOINT_URL`, default `http://localhost:9002`).

--- Threat model for the object key (read this before touching `document_key`/`attachment_key`) ---

Document ingestion is where user-supplied filenames, content types, and paths enter the system.
An attacker who controls the upload filename or a calendar attachment's metadata can try:
  - path traversal (``../../../etc/passwd``, ``..\..\windows\system32\...``)
  - absolute-path escape (``/etc/passwd``, ``C:\...``)
  - null-byte / control-character tricks (``evil.pdf\x00.exe``)
  - prefix collision with another tenant (`../{other_org_id}/...`)
  - an extension lie (`.pdf` content that is actually a `.exe`, or vice versa, to smuggle a
    format past the parser's format dispatch)

What a caller-supplied filename CAN influence, by design:
  - Whether the upload is accepted at all (`safe_extension` allow-lists a fixed, small set of
    suffixes this service can genuinely parse — see `services/ingestion_service/parser.py`).
  - Nothing else. `document_key`/`attachment_key` NEVER read the caller-supplied filename. The
    extension embedded in the final key is always the literal member of `SUPPORTED_EXTENSIONS`
    that `safe_extension` matched — never the raw suffix string — so a filename cannot inject a
    path separator, `..`, a null byte, or any other character into the key: the key is built
    entirely from (a) `org_id`, ALWAYS the verified tenant identity (event envelope header on
    the consumer path, `X-Org-Id` on the REST path — see `services/ingestion_service/routes.py`
    for why that is the established interim contract), never taken from the request body; (b) a
    server-generated `document_id` (`uuid.uuid4()`), NEVER caller-supplied, so even a caller who
    fully controls the filename string cannot influence the `document_id` path segment; and
    (c) one member of the fixed `SUPPORTED_EXTENSIONS` set. There is no code path that formats a
    caller-supplied string directly into a key.
  - Format dispatch is likewise never taken from a caller-supplied `Content-Type` header — the
    parser dispatches on the SAME server-validated extension used to build the key (see
    `parser.py`), so a mislabeled `Content-Type` cannot smuggle a different format past the
    parser or the key allow-list. The client's `Content-Type` is stored only as informational S3
    object metadata, never used for a decision.

Tenant isolation on top of the key shape: the key ALWAYS begins with `{org_id}/`, so a
prefix-scoped `list_objects_v2(Prefix=f"{org_id}/")` can never surface another tenant's objects
even if that tenant's `document_id`/`meeting_id` values were somehow guessed (proven in
`tests/test_object_storage.py`).
"""

from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath
from uuid import UUID

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

BUCKET_DOCUMENTS = "mytheus-documents"  # direct /ingest uploads — {org_id}/{document_id}/
BUCKET_ATTACHMENTS = "mytheus-attachments"  # calendar attachments — {org_id}/{meeting_id}/

# The ONLY extensions this service can genuinely parse (services/ingestion_service/parser.py).
# `safe_extension` never returns anything outside this set, and `document_key`/`attachment_key`
# assert on it too — belt-and-suspenders, matching this repo's Qdrant tenancy precedent.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".eml", ".txt"})


def safe_extension(filename: str) -> str | None:
    """The ONLY function in this repo allowed to derive anything from a caller-supplied
    filename. Returns the lowercased suffix if (and only if) it is one of
    `SUPPORTED_EXTENSIONS`; `None` otherwise (caller must reject the upload, never guess a
    format). `PurePosixPath(...).suffix` takes the text after the LAST `.` in the LAST path
    segment only — a traversal payload like `"../../etc/passwd"` has suffix `""` (rejected), and
    `"evil.pdf\\x00.exe"` has suffix `".exe"` (rejected, not `.pdf`) — so this cannot be tricked
    into returning a supported extension for a disguised payload."""
    ext = PurePosixPath(filename).suffix.lower()
    return ext if ext in SUPPORTED_EXTENSIONS else None


def document_key(org_id: UUID, document_id: UUID, ext: str) -> str:
    """`mytheus-documents` key for a direct `/ingest` upload: `{org_id}/{document_id}/source{ext}`.
    `org_id` is always the verified tenant identity; `document_id` is always server-generated
    (`uuid.uuid4()`, never caller-supplied); `ext` must already be a `SUPPORTED_EXTENSIONS`
    member. The filename itself never appears in the key — the constant name `source` is used
    for every upload, which also means two different original filenames for the same
    `document_id` (e.g. a legitimate re-run) resolve to the SAME key, matching the idempotent
    re-run requirement."""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported extension {ext!r}; call safe_extension() first")
    return f"{org_id}/{document_id}/source{ext}"


def attachment_key(org_id: UUID, meeting_id: UUID, document_id: UUID, ext: str) -> str:
    """`mytheus-attachments` key for a calendar attachment:
    `{org_id}/{meeting_id}/{document_id}{ext}` — the canonical `{org_id}/{meeting_id}/` prefix
    (`_canon/02-data-model.md` §6), with
    `document_id` (always server-generated) distinguishing multiple attachments on one meeting.
    Same guarantees as `document_key`: no caller-supplied string reaches this key."""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported extension {ext!r}; call safe_extension() first")
    return f"{org_id}/{meeting_id}/{document_id}{ext}"


def get_client() -> boto3.client:
    """Real boto3 S3 client. Production is Cloudflare R2 (ADR-0023): set `S3_ENDPOINT_URL` to
    `https://<account-id>.r2.cloudflarestorage.com`, `S3_REGION=auto` and the R2 API token pair in
    `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`. The `MINIO_*` names remain as the local-dev
    fallback (MinIO on :9002). Path-style addressing is the default because it works on both;
    set `S3_FORCE_PATH_STYLE=false` for virtual-hosted style.

    Request checksums are sent only when an operation requires them: the SDK default of adding
    a CRC trailer to every upload is not accepted by every S3-compatible store."""
    import os

    env = os.environ
    path_style = env.get("S3_FORCE_PATH_STYLE", "true").lower() != "false"
    return boto3.client(
        "s3",
        endpoint_url=env.get("S3_ENDPOINT_URL")
        or env.get("MINIO_ENDPOINT_URL", "http://localhost:9002"),
        aws_access_key_id=env.get("S3_ACCESS_KEY_ID") or env.get("MINIO_ACCESS_KEY", "mytheus"),
        aws_secret_access_key=env.get("S3_SECRET_ACCESS_KEY")
        or env.get("MINIO_SECRET_KEY", "mytheus-dev-secret"),
        region_name=env.get("S3_REGION", "us-east-1"),
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path" if path_style else "virtual"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def ensure_bucket(client: boto3.client, bucket: str) -> None:
    """Idempotent create for local dev. Creates only when the bucket is genuinely absent (404):
    a 403 means it exists but this credential cannot see it, which is the normal state with an
    object-scoped R2 token, so it is re-raised rather than turned into a doomed create_bucket.
    Set `S3_AUTO_CREATE_BUCKETS=0` in production, where buckets are provisioned out of band."""
    import os

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        status = str(exc.response.get("Error", {}).get("Code", ""))
        if status not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        if os.environ.get("S3_AUTO_CREATE_BUCKETS", "1") == "0":
            raise
        client.create_bucket(Bucket=bucket)


def put_object(
    client: boto3.client,
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str | None,
) -> None:
    """Write the original bytes. `content_type` is informational S3 metadata only (client- or
    attachment-supplied); it is NEVER used to choose a parser or to build the key — see the
    module docstring."""
    ensure_bucket(client, bucket)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type or mimetypes.guess_type(key)[0] or "application/octet-stream",
    )


def get_object(client: boto3.client, *, bucket: str, key: str) -> bytes:
    """Read back the original bytes (used by the `integration-events` consumer, which reads an
    attachment `integration-service` already uploaded, rather than uploading it itself)."""
    resp = client.get_object(Bucket=bucket, Key=key)
    body: bytes = resp["Body"].read()
    return body
