"""The bound agent identity (P10 contract section 2, ADR-0013 section 1).

`requester_agent_id` / `responder_agent_id` = `cga:{org_id}:{user_id}:{sig}` with
`sig = hex(HMAC-SHA256(AGENT_IDENTITY_SECRET, "cga:{org_id}:{user_id}"))[:32]`.

Only agent-runtime-service holds the secret, so only it can mint an id, and the org and user of a
request are derived from a VERIFIED id, never from the envelope. A bad signature, a malformed id or
a non-canonical UUID spelling all verify to None: no derivable identity.

No secret (or one shorter than 32 characters) means no signer at all, and the runtime then runs
with A2A off: it cannot mint ids, and its responder dead-letters every request.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass
from uuid import UUID

logger = logging.getLogger(__name__)

SECRET_ENV = "AGENT_IDENTITY_SECRET"
MIN_SECRET_CHARS = 32
SIG_CHARS = 32
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_AGENT_ID = re.compile(rf"^cga:({_UUID}):({_UUID}):([0-9a-f]{{{SIG_CHARS}}})$")
_TARGET = re.compile(rf"^user:({_UUID})$")


@dataclass(frozen=True)
class AgentIdentity:
    org_id: UUID
    user_id: UUID


class IdentitySigner:
    def __init__(self, secret: str) -> None:
        if len(secret) < MIN_SECRET_CHARS:
            raise ValueError(f"{SECRET_ENV} must be at least {MIN_SECRET_CHARS} characters")
        self._key = secret.encode("utf-8")

    def _sig(self, org_id: UUID, user_id: UUID) -> str:
        msg = f"cga:{org_id}:{user_id}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()[:SIG_CHARS]

    def mint(self, org_id: UUID, user_id: UUID) -> str:
        return f"cga:{org_id}:{user_id}:{self._sig(org_id, user_id)}"

    def verify(self, agent_id: object) -> AgentIdentity | None:
        if not isinstance(agent_id, str):
            return None
        match = _AGENT_ID.match(agent_id)
        if match is None:
            return None
        org_id, user_id = UUID(match.group(1)), UUID(match.group(2))
        if not hmac.compare_digest(match.group(3), self._sig(org_id, user_id)):
            return None
        return AgentIdentity(org_id=org_id, user_id=user_id)


def signer_from_env() -> IdentitySigner | None:
    secret = os.environ.get(SECRET_ENV, "")
    if not secret:
        return None
    try:
        return IdentitySigner(secret)
    except ValueError:
        logger.error("%s is shorter than %s characters: A2A off", SECRET_ENV, MIN_SECRET_CHARS)
        return None


def responder_target(user_id: UUID) -> str:
    return f"user:{user_id}"


def parse_target(raw: object) -> UUID | None:
    """`user:{uuid}` (canonical lowercase spelling only) to the user id; anything else None."""
    if not isinstance(raw, str):
        return None
    match = _TARGET.match(raw)
    return UUID(match.group(1)) if match else None
