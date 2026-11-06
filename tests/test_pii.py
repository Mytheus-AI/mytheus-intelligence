"""`redact_pii` (real regex, no infra) and `pii_enabled` (real Redis — see `pii.py`'s module
docstring for the honest gap: the publish side of `tenant_settings.pii_redaction_enabled` does
not exist anywhere in the program yet, so this proves the READ mechanism against real Redis, not
a real tenant's setting)."""

from __future__ import annotations

import uuid

import pytest

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


class _FakeRedis:
    """Minimal in-memory stand-in so the default-policy tests need no infra."""

    def __init__(self, value=None):
        self._value = value

    async def get(self, _key):
        return self._value


async def test_pii_enabled_defaults_true_when_key_absent_no_infra():
    # Fail-safe: matches tenant_settings.pii_redaction_enabled DEFAULT true.
    assert await pii_enabled(uuid.uuid4(), _FakeRedis(None)) is True


@pytest.mark.parametrize("value", [b"0", "0", b"false", "false"])
async def test_pii_enabled_false_only_on_explicit_off_no_infra(value):
    assert await pii_enabled(uuid.uuid4(), _FakeRedis(value)) is False


@pytest.mark.parametrize("value", [b"1", "true", b"garbage", b""])
async def test_pii_enabled_true_on_on_or_unrecognised_no_infra(value):
    assert await pii_enabled(uuid.uuid4(), _FakeRedis(value)) is True


async def test_pii_enabled_defaults_true_when_key_absent(redis_client):
    org_id = uuid.uuid4()  # freshly random org: guaranteed no key was ever set for it
    assert await pii_enabled(org_id, redis_client) is True


async def test_pii_enabled_false_when_real_redis_key_explicitly_off(redis_client):
    org_id = uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    await redis_client.set(key, "0")
    try:
        assert await pii_enabled(org_id, redis_client) is False
    finally:
        await redis_client.delete(key)


async def test_pii_enabled_reverts_to_default_true_after_key_cleared(redis_client):
    org_id = uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    await redis_client.set(key, "0")
    assert await pii_enabled(org_id, redis_client) is False
    await redis_client.delete(key)
    assert await pii_enabled(org_id, redis_client) is True
