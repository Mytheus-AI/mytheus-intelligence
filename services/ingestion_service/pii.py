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
from Redis (the `nexus:{org_id}:...` namespace every per-tenant Redis value uses).

DEFAULT IS REDACT (fail-safe). The key being absent means "no mirror has told us otherwise", and
the source of truth, `tenant_settings.pii_redaction_enabled`, defaults to TRUE in the DB/canon.
An earlier version defaulted to False on the reasoning that no producer writes the key; that
made every tenant un-redacted by default, the opposite of canon, and any outage or missing
mirror silently leaked PII into `document-chunks`, Qdrant and the LLM context. Now only an
explicit falsy value ("0"/"false") turns redaction off; absent or any unrecognised value
redacts. When a producer (identity-tenant-service Redis mirror, or a gRPC `GetTenantSettings`)
starts writing the key, `pii_enabled` picks it up with no code change here. Proven for real in
`tests/test_pii.py`; what is NOT proven is that the key is ever populated from a real
`tenant_settings` row today.
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


_EXPLICIT_OFF = (b"0", "0", b"false", "false", b"False", "False")


async def pii_enabled(org_id: UUID, redis: Redis) -> bool:
    """See the module docstring: real Redis read; absent/unrecognised => True (redact), matching
    the DB/canon default. Only an explicit "0"/"false" disables redaction."""
    value = await redis.get(_redis_key(org_id))
    return value not in _EXPLICIT_OFF
