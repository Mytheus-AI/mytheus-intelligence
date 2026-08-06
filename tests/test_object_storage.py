"""Proves `libs/shared/object_storage.py`'s key-safety and tenant-isolation guarantees against
REAL MinIO (task brief: "Do NOT write a fake S3 client. This band deliberately uses real
infrastructure."). No mock/fake S3 client anywhere in this file."""

from __future__ import annotations

import uuid

import pytest

from libs.shared.object_storage import (
    BUCKET_DOCUMENTS,
    SUPPORTED_EXTENSIONS,
    attachment_key,
    document_key,
    ensure_bucket,
    get_object,
    put_object,
    safe_extension,
)

# Mixed sync (safe_extension/key construction, no infra) and async (real MinIO I/O) tests in
# this file, so — unlike the all-async test modules elsewhere in this repo — there is no
# module-level `pytestmark`; `asyncio_mode = "auto"` (pyproject.toml) picks up the `async def`
# tests below without one.


# --- safe_extension: adversarial filenames -----------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", ".pdf"),
        ("Report.PDF", ".pdf"),  # case-insensitive
        ("notes.docx", ".docx"),
        ("archive.tar.gz", None),  # unsupported, even though it has a dotted suffix
        ("evil.exe", None),
        ("noextension", None),
        ("", None),
        ("../../../etc/passwd", None),  # traversal payload: suffix is "" -> rejected
        (
            "../../etc/cron.d/evil.pdf",
            ".pdf",
        ),  # the traversal is IN THE FILENAME, never in the key (see below)
        ("evil.pdf\x00.exe", None),  # null-byte trick: suffix is ".exe", which is outside the
        # allow-list regardless -> rejected either way, never falls back to the earlier ".pdf"
        ("/etc/passwd.txt", ".txt"),  # absolute path: still only the suffix is used
        ("..\\..\\windows\\system32\\evil.pdf", ".pdf"),  # backslash traversal is not a path
        # separator on POSIX; suffix extraction still only sees the trailing ".pdf".
    ],
)
def test_safe_extension_allow_lists_adversarial_filenames(filename: str, expected: str | None):
    assert safe_extension(filename) == expected


def test_supported_extensions_is_the_documented_narrow_set():
    assert SUPPORTED_EXTENSIONS == {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".eml",
        ".txt",
    }


# --- key construction: filename never reaches the key ------------------------------------


def test_document_key_never_contains_the_filename():
    org_id = uuid.uuid4()
    document_id = uuid.uuid4()
    key = document_key(org_id, document_id, ".pdf")
    assert key == f"{org_id}/{document_id}/source.pdf"
    # A traversal payload's ext still only contributes a fixed, allow-listed suffix — the key
    # is built from server-controlled org_id/document_id plus that suffix, nothing else.
    traversal_ext = safe_extension("../../../etc/passwd.pdf")
    assert traversal_ext == ".pdf"
    key2 = document_key(org_id, document_id, traversal_ext)
    assert key2 == key  # identical: no trace of "../../../etc/passwd" anywhere in the key
    assert ".." not in key2
    assert "etc" not in key2


def test_document_key_rejects_an_extension_outside_the_allow_list():
    with pytest.raises(ValueError):
        document_key(uuid.uuid4(), uuid.uuid4(), ".exe")


def test_attachment_key_uses_the_canonical_org_meeting_prefix():
    org_id, meeting_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    key = attachment_key(org_id, meeting_id, document_id, ".pdf")
    assert key == f"{org_id}/{meeting_id}/{document_id}.pdf"
    assert key.startswith(f"{org_id}/")


def test_two_different_document_ids_never_collide_even_with_the_same_filename():
    org_id = uuid.uuid4()
    key_a = document_key(org_id, uuid.uuid4(), ".pdf")
    key_b = document_key(org_id, uuid.uuid4(), ".pdf")
    assert key_a != key_b


# --- real MinIO: write, read back, and tenant-prefix isolation ---------------------------


async def test_put_and_get_object_round_trips_real_bytes(s3_client):
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    key = document_key(org_id, document_id, ".txt")
    body = b"real bytes, real MinIO, no fake S3 client"
    put_object(s3_client, bucket=BUCKET_DOCUMENTS, key=key, body=body, content_type="text/plain")
    try:
        assert get_object(s3_client, bucket=BUCKET_DOCUMENTS, key=key) == body
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key)


async def test_ensure_bucket_is_idempotent(s3_client):
    ensure_bucket(s3_client, BUCKET_DOCUMENTS)
    ensure_bucket(s3_client, BUCKET_DOCUMENTS)  # second call must not raise
    resp = s3_client.list_buckets()
    assert BUCKET_DOCUMENTS in {b["Name"] for b in resp["Buckets"]}


async def test_prefix_scoped_listing_cannot_see_another_tenants_objects(s3_client):
    """The tenant-isolation property that matters: a `{org_id}/` prefix listing can never span
    tenants, because the key ALWAYS begins with the verified org_id — proven directly against
    real MinIO's `list_objects_v2`, not asserted."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    key_a = document_key(org_a, doc_a, ".txt")
    key_b = document_key(org_b, doc_b, ".txt")
    put_object(
        s3_client,
        bucket=BUCKET_DOCUMENTS,
        key=key_a,
        body=b"org-a-secret",
        content_type="text/plain",
    )
    put_object(
        s3_client,
        bucket=BUCKET_DOCUMENTS,
        key=key_b,
        body=b"org-b-secret",
        content_type="text/plain",
    )
    try:
        resp = s3_client.list_objects_v2(Bucket=BUCKET_DOCUMENTS, Prefix=f"{org_a}/")
        keys = {o["Key"] for o in resp.get("Contents", [])}
        assert key_a in keys
        assert key_b not in keys
    finally:
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key_a)
        s3_client.delete_object(Bucket=BUCKET_DOCUMENTS, Key=key_b)
