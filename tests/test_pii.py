"""`redact_pii` (real regex, no infra) and `pii_enabled` (real Redis — see `pii.py`'s module
docstring for the honest gap: the publish side of `tenant_settings.pii_redaction_enabled` does
not exist anywhere in the program yet, so this proves the READ mechanism against real Redis, not
a real tenant's setting)."""

from __future__ import annotations

import uuid

from services.ingestion_service.pii import pii_enabled, redact_pii

# Mixed sync (redact_pii, no infra) and async (real Redis) tests — no module-level pytestmark;
# asyncio_mode = "auto" (pyproject.toml) picks up the `async def` tests below without one.


def test_redact_pii_masks_email():
    out = redact_pii("Contact me at alice@example.com for details.")
    assert "alice@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_pii_masks_ssn():
    out = redact_pii("SSN on file: 123-45-6789.")
    assert "123-45-6789" not in out
    assert "[REDACTED_SSN]" in out


def test_redact_pii_masks_credit_card():
    out = redact_pii("Card number 4111 1111 1111 1111 was charged.")
    assert "4111 1111 1111 1111" not in out
    assert "[REDACTED_CC]" in out


def test_redact_pii_masks_phone():
    out = redact_pii("Call me at 555-123-4567 tomorrow.")
    assert "555-123-4567" not in out
    assert "[REDACTED_PHONE]" in out


def test_redact_pii_leaves_non_pii_text_untouched():
    text = "We decided to ship the ingestion service this sprint."
    assert redact_pii(text) == text


def test_redact_pii_handles_multiple_pii_types_in_one_string():
    text = "Email alice@example.com or call 555-123-4567; SSN 123-45-6789."
    out = redact_pii(text)
    assert "alice@example.com" not in out
    assert "555-123-4567" not in out
    assert "123-45-6789" not in out
    assert out.count("[REDACTED_") == 3


async def test_pii_enabled_defaults_false_when_key_absent(redis_client):
    org_id = uuid.uuid4()  # freshly random org: guaranteed no key was ever set for it
    assert await pii_enabled(org_id, redis_client) is False


async def test_pii_enabled_true_when_real_redis_key_set(redis_client):
    org_id = uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    await redis_client.set(key, "1")
    try:
        assert await pii_enabled(org_id, redis_client) is True
    finally:
        await redis_client.delete(key)


async def test_pii_enabled_reverts_to_false_after_key_cleared(redis_client):
    org_id = uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    await redis_client.set(key, "1")
    assert await pii_enabled(org_id, redis_client) is True
    await redis_client.delete(key)
    assert await pii_enabled(org_id, redis_client) is False
