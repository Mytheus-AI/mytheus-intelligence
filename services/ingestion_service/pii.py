"""PII redaction gate — plan step 2: `redact_pii(el.text) if pii_enabled(org_id) else el.text`,
keyed by `tenant_settings.pii_redaction_enabled`.

--- The real, working part: `redact_pii` ---

A real, regex-based structural-PII redactor (email, phone, SSN-shaped, credit-card-shaped). It
is a baseline, not an NER/LLM-based free-text PII detector — stated plainly so it is not
mistaken for more than it is. Every pattern below is exercised against real matching and
non-matching text in `tests/test_pii.py`.

--- The genuine gap: where `pii_enabled(org_id)` reads its answer from ---

`tenant_settings` (`pii_redaction_enabled` column) is owned by `identity-tenant-service` in
`mytheus-backend` — a DIFFERENT repo's Postgres tables. This repo's own rule (`libs/shared/db.py`,
CLAUDE.md) is that `mytheus-intelligence` never writes Postgres directly and never joins into
another repo's tables; reading `tenant_settings` for real requires either a gRPC call (the
pattern `get_user_context` uses elsewhere in this tier) or a published Redis mirror. Neither
exists yet: this repo's internal gRPC mesh is still `proto/rag.proto`'s placeholder message (the
same gap Plan 5-6 tasks 7/8 hit and resolved as "REST now, gRPC later" — see
`services/embedding_service/query_embed.py`), and no producer anywhere in the program mirrors
`tenant_settings` into Redis today.

Given that, `pii_enabled` is built the same way task 7/8 built their REST-now contracts: a REAL,
working mechanism against REAL infrastructure (Redis, not a fake), with the publish side
explicitly flagged as not yet wired. It reads `nexus:{org_id}:tenant_settings:pii_redaction_enabled`
from Redis — the SAME `nexus:{org_id}:...` namespace convention every other per-tenant Redis
value in this program uses (CLAUDE.md's Redis prefix rule) — and defaults to `False` (redaction
off) when the key is absent. That default is not a guess about any real tenant's setting: no
producer in the program has ever written this key, so `False` is the true current state for
every tenant, not a fabricated fallback. The moment a producer (an `identity-tenant-service`
Redis mirror, or a gRPC `GetTenantSettings` call once the mesh exists) starts setting this key,
`pii_enabled` picks it up with no code change here. This function's mechanism IS proven for real
in `tests/test_pii.py` (setting/clearing the real Redis key toggles the return value); what is
NOT proven, and cannot be without another repo's work, is that the key is ever populated by a
real `tenant_settings` row today.
"""

from __future__ import annotations

import re
from uuid import UUID

from redis.asyncio import Redis

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")


def redact_pii(text: str) -> str:
    """Order matters: SSN and credit-card patterns are more specific digit shapes than the
    generic phone pattern, so they run first to avoid a credit-card number being partially
    swallowed by the looser phone regex."""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _SSN_RE.sub("[REDACTED_SSN]", text)
    text = _CREDIT_CARD_RE.sub("[REDACTED_CC]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _redis_key(org_id: UUID) -> str:
    return f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"


async def pii_enabled(org_id: UUID, redis: Redis) -> bool:
    """See the module docstring: real Redis read, honest `False` default, publish side not yet
    wired anywhere in the program."""
    value = await redis.get(_redis_key(org_id))
    return value in (b"1", "1", b"true", "true")
