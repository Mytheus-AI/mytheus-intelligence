"""The bound agent identity, signed per message (P10 contract section 2, ADR-0013 section 1).

Wire format, unchanged from the contract (no schema change): `cga:{org_id}:{user_id}:{sig}`,
`sig` = the first 32 hex characters of an HMAC-SHA256 under `AGENT_IDENTITY_SECRET`. The MAC is
not over the identity alone (that would be a bearer token anyone who saw one message could
replay with any query), but over the message it travels on:

- requester id on an AgentRequest:
  `cga:{org}:{user}|{correlation_id}|{responder_target}|{deadline_micros}|{sha256(query)}`;
- responder id on an AgentResponse: `cga:{org}:{user}|{correlation_id}|resp`.

A verifier recomputes the MAC from the fields it received, so changing the query, the target,
the deadline or the correlation id, or moving the id to another message, breaks it. What is left
is an exact replay of one message, which the responder bounds with the deadline (past, or more
than 5 s ahead, is refused) and a 60 s correlation-id dedupe.

Only agent-runtime-service holds the secret, so only it can mint, and the org and user of a
request are derived from a VERIFIED id, never from the envelope. A bad MAC, a malformed id or a
non-canonical UUID spelling all verify to None: no derivable identity. No secret (or one shorter
than 32 characters) means no signer: A2A off, and the responder dead-letters every request.
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


def request_material(
    correlation_id: str, responder_target: str, deadline_us: int, query: str
) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"{correlation_id}|{responder_target}|{int(deadline_us)}|{digest}"


def response_material(correlation_id: str) -> str:
    return f"{correlation_id}|resp"


class IdentitySigner:
    def __init__(self, secret: str) -> None:
        if len(secret) < MIN_SECRET_CHARS:
            raise ValueError(f"{SECRET_ENV} must be at least {MIN_SECRET_CHARS} characters")
        self._key = secret.encode("utf-8")

    def _sig(self, org_id: UUID, user_id: UUID, material: str) -> str:
        msg = f"cga:{org_id}:{user_id}|{material}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()[:SIG_CHARS]

    def _mint(self, org_id: UUID, user_id: UUID, material: str) -> str:
        return f"cga:{org_id}:{user_id}:{self._sig(org_id, user_id, material)}"

    def _verify(self, agent_id: object, material: str) -> AgentIdentity | None:
        if not isinstance(agent_id, str):
            return None
        match = _AGENT_ID.match(agent_id)
        if match is None:
            return None
        org_id, user_id = UUID(match.group(1)), UUID(match.group(2))
        if not hmac.compare_digest(match.group(3), self._sig(org_id, user_id, material)):
            return None
        return AgentIdentity(org_id=org_id, user_id=user_id)

    def mint_request(
        self,
        org_id: UUID,
        user_id: UUID,
        *,
        correlation_id: str,
        responder_target: str,
        deadline_us: int,
        query: str,
    ) -> str:
        material = request_material(correlation_id, responder_target, deadline_us, query)
        return self._mint(org_id, user_id, material)

    def verify_request(
        self,
        agent_id: object,
        *,
        correlation_id: str,
        responder_target: str,
        deadline_us: int,
        query: str,
    ) -> AgentIdentity | None:
        material = request_material(correlation_id, responder_target, deadline_us, query)
        return self._verify(agent_id, material)

    def mint_response(self, org_id: UUID, user_id: UUID, correlation_id: str) -> str:
        return self._mint(org_id, user_id, response_material(correlation_id))

    def verify_response(self, agent_id: object, correlation_id: str) -> AgentIdentity | None:
        return self._verify(agent_id, response_material(correlation_id))


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
