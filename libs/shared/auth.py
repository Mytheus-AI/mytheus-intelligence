"""Clerk JWT verification (JWKS) for this repo's REST tier.

Security fix (whole-band review, Important finding). Every REST route in this repo — `POST
/ingest` / `GET /ingest/jobs/:id` (`ingestion-service`), `GET /search` (`search-service`),
`POST /embed` (`embedding-service`), `POST /rerank` (`reranker-service`) — used to take
`org_id` from a client-supplied `X-Org-Id` header and fail closed ONLY when that header was a
malformed UUID. Nothing verified the caller held a legitimate identity for that org: anyone
with network reachability could carry another organization's id in that header and write into
or read that org's data. Each service's route module disclosed this individually as an
"interim REST-authn convention" (see the git history of `services/*/routes.py`,
`query_embed.py`, `rerank.py`), and each disclosure was locally reasonable in isolation — but
`mytheus-backend/CLAUDE.md` states the program-wide invariant this violates: "the gateway is a
DDoS shield, not a trust delegation" — every service tier RE-VERIFIES the JWT itself, because
`api-gateway` (which canon assigns JWT validation and `org_id` extraction to) does not exist
yet. This module is that re-verification for this repo's REST tier.

This REIMPLEMENTS `mytheus-backend`'s `libs/shared/auth.py` verification pattern faithfully
(same repos cannot import across each other) — RS256 against cached JWKS, `nbf` required,
`PyJWKClientError` mapped to a retryable dependency-outage response rather than a 500, claim
name read from the environment rather than hardcoded. Read `mytheus-backend`'s `auth.py`
module docstring for the four defects that shape this: an opaque Clerk claim must never be
`UUID(...)`-parsed as this repo's own `org_id`; `nbf` must be required, not merely honored when
present; a JWKS outage is a 503, not a 500.

--- Why this module does NOT also reimplement `org_resolver.py` ---

`mytheus-backend`'s pattern resolves Clerk's opaque native organization id (`org_...`) to the
internal `org_id` UUID via one query against `public.organizations`. This repo CANNOT do the
same lookup: `mytheus-intelligence` shares the same physical Postgres, but migration
`9c2b7e4a1d03`'s own module docstring records the deliberate boundary — "`org_id` here is an
opaque tenant UUID carried by the event envelope header, not a row this repo is ever allowed to
look up by joining into another repo's table" (the CLAUDE.md rule "cross-tenant access...
never a cross-service table write" applies equally to reads across a bounded context in the
same physical database). Reading `public.organizations` from this repo's REST tier would
re-open exactly the boundary that migration closed for the async path, just on the sync path
instead.

So the claim this module reads (`CLERK_JWT_INTERNAL_ORG_CLAIM`, distinct from
`mytheus-backend`'s `CLERK_JWT_ORG_CLAIM`) is expected to carry the ALREADY-RESOLVED internal
`org_id` (a UUID), not Clerk's opaque native id — e.g. a Clerk JWT template claim mapped to the
organization's `public_metadata.internal_org_id`, set by `identity-tenant-service` at
org-provisioning time. That mapping is a Clerk-dashboard / `identity-tenant-service` change
outside this repo's code and is NOT made by this commit — flagged here rather than silently
assumed. What this module guarantees on its own: the token is a genuine, current, correctly
signed Clerk session token (so `org_id` can never come from an unauthenticated header again),
and the org claim is fail-closed in every degenerate shape (absent, non-string, not a UUID).
What it cannot guarantee without that claim wiring, and without the forbidden cross-repo read,
is that the org named by the claim is still active/non-suspended — the same limitation this
repo already accepts for `intelligence.outbox`/`inbox`'s `org_id` (migration `9c2b7e4a1d03`
decision 3).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from uuid import UUID

import jwt
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientError

from libs.shared.service_auth import SERVICE_HEADER, ServiceAuthError, authenticate

_bearer = HTTPBearer(auto_error=False)

# The claim this repo expects to already carry the RESOLVED internal org_id (a UUID) — never
# Clerk's opaque native organization id. Deliberately a different env var and a different
# default claim name from mytheus-backend's `CLERK_JWT_ORG_CLAIM` (which holds the opaque
# `org_...` string): reusing that name here, with a different meaning, would be a silent trap
# for anyone reading both repos' .env side by side.
DEFAULT_ORG_CLAIM = "internal_org_id"


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    # https://<subdomain>.clerk.accounts.dev/.well-known/jwks.json - caches + honors rotation.
    # Same JWKS endpoint mytheus-backend points at (CLERK_JWKS_URL); read from the environment,
    # never hardcoded, so dev/test can point this at a fake client (see tests/conftest.py).
    return PyJWKClient(os.environ["CLERK_JWKS_URL"], cache_keys=True)


def _issuer() -> str:
    return os.environ["CLERK_ISSUER"]  # https://<subdomain>.clerk.accounts.dev


def org_claim_name() -> str:
    return os.environ.get("CLERK_JWT_INTERNAL_ORG_CLAIM", DEFAULT_ORG_CLAIM)


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
) -> dict[str, Any]:
    """Verify a Clerk session token and return its claims. Fails closed on every branch.

    Never a 500 for anything an ordinary, correctly-issued-or-not token can trigger: missing
    bearer, unreachable JWKS, expired/not-yet-valid/wrong-issuer/wrong-key token all resolve to
    an explicit HTTPException with a status a caller can act on.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(credentials.credentials).key
    except PyJWKClientError as exc:
        # The IdP is unreachable or served an unusable key set. This is NOT the caller's fault
        # and NOT a 500: a 503 tells the client to retry and keeps the outage attributable to
        # the dependency rather than to this service (mirrors mytheus-backend's auth.py).
        raise HTTPException(status_code=503, detail="Identity provider unavailable") from exc

    try:
        payload: dict[str, Any] = jwt.decode(
            credentials.credentials,
            signing_key,
            algorithms=["RS256"],  # never "none", never HS*: the key here is a PUBLIC key
            issuer=_issuer(),
            # nbf is required, not merely honored when present — without it a token minted with
            # a future nbf would simply be accepted early.
            options={"require": ["exp", "iat", "nbf", "iss", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        # Deliberately does not echo str(exc): the message varies by failure mode and hands an
        # attacker a free oracle for which check failed.
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    return payload


def org_id_from(claims: dict[str, Any]) -> UUID:
    """The verified token's resolved `org_id`. Fails closed: absent, non-string, or non-UUID
    all raise 401 — never guessed, never defaulted to any tenant."""
    value = claims.get(org_claim_name())
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Token carries no organization")
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(status_code=401, detail="Token carries no organization") from None


def verified_org_id(
    claims: dict[str, Any] = Security(verify_token),  # noqa: B008
) -> UUID:
    """The single dependency every route in this repo's REST tier should depend on for
    `org_id`, in place of the old `Header(..., alias="X-Org-Id")` — see module docstring.
    """
    return org_id_from(claims)


def verified_or_service_org_id(env_name: str, allowed: frozenset[str]) -> Any:
    """A route dependency accepting EITHER a verified Clerk token OR an internal service caller.

    `rag-service` authenticates the asker itself (a user's token, or an agent asking for a
    participant, which has no user token) and then calls `embedding-service` / `reranker-service`
    as itself: `X-Mytheus-Service`, `X-Mytheus-Service-Token` (secrets in `env_name`), and the
    tenant in `X-Mytheus-Org` (mytheus-docs ADR-0033). The service path is taken only when the
    service header is present, and then it must succeed: a bad secret never falls back to the
    bearer path. Both paths fail closed with 401.
    """

    def _dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
    ) -> UUID:
        headers = {k.lower(): v for k, v in request.headers.items()}
        if SERVICE_HEADER in headers:
            try:
                return authenticate(headers, env_name=env_name, allowed=allowed).org_id
            except ServiceAuthError:
                raise HTTPException(status_code=401, detail="Unauthenticated") from None
        return org_id_from(verify_token(credentials))

    return _dep
