"""`libs/shared/auth.py` — Clerk RS256 JWKS verification and the internal `org_id` claim.

Security fix (whole-band review, Important finding): every REST route in this repo used to
trust a client-supplied `X-Org-Id` header outright. This module is the fix's verification
core; `tests/test_ingestion_routes.py`, `tests/test_search_routes.py`, `tests/test_reranker.py`
and `tests/test_query_embed.py` prove it is actually wired into every route. This file proves
`verify_token`/`org_id_from` themselves fail closed on every degenerate shape a real Clerk
token (or an attacker) can present — same coverage shape as `mytheus-backend`'s own
`tests/test_auth.py`, reimplemented against this repo's own module (no cross-repo import).
"""

from __future__ import annotations

import time
import uuid

import jwt as pyjwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt import PyJWKClientError

from libs.shared import auth as auth_mod
from tests.conftest import _TEST_PRIVATE_KEY, sign_test_token

pytestmark = pytest.mark.asyncio


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


async def test_verify_valid_token_returns_the_internal_org_claim():
    org_id = uuid.uuid4()
    claims = auth_mod.verify_token(_creds(sign_test_token(org_id)))
    assert auth_mod.org_id_from(claims) == org_id


async def test_org_claim_name_is_configurable(monkeypatch):
    monkeypatch.setenv("CLERK_JWT_INTERNAL_ORG_CLAIM", "o")
    org_id = uuid.uuid4()
    token = sign_test_token(o=str(org_id))  # org_id=None (default) -> no default-name claim
    claims = auth_mod.verify_token(_creds(token))
    assert auth_mod.org_id_from(claims) == org_id


async def test_missing_bearer_is_401():
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(None)
    assert exc.value.status_code == 401


async def test_expired_token_is_401():
    now = int(time.time())
    token = sign_test_token(uuid.uuid4(), exp=now - 10, iat=now - 3600, nbf=now - 3600)
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(_creds(token))
    assert exc.value.status_code == 401


async def test_nbf_is_required():
    """A token with no nbf must be refused, not silently accepted — gap 8 in
    mytheus-backend's auth.py, reproduced here as a guard against the same regression."""
    now = int(time.time())
    claims = {
        "sub": "u",
        auth_mod.org_claim_name(): str(uuid.uuid4()),
        "iss": auth_mod._issuer(),
        "iat": now,
        "exp": now + 3600,
    }
    token = pyjwt.encode(claims, _TEST_PRIVATE_KEY, algorithm="RS256")
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(_creds(token))
    assert exc.value.status_code == 401


async def test_token_not_yet_valid_is_refused():
    now = int(time.time())
    token = sign_test_token(uuid.uuid4(), nbf=now + 600, iat=now)
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(_creds(token))
    assert exc.value.status_code == 401


async def test_wrong_issuer_is_refused():
    token = sign_test_token(uuid.uuid4(), iss="https://not-our-clerk.example")
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(_creds(token))
    assert exc.value.status_code == 401


async def test_missing_org_claim_is_401_never_500():
    """A verified, correctly-signed token with no org claim (a personal-account session) must
    be refused as 401, never guessed into any tenant and never a 500."""
    claims = auth_mod.verify_token(_creds(sign_test_token(org_id=None)))
    with pytest.raises(HTTPException) as exc:
        auth_mod.org_id_from(claims)
    assert exc.value.status_code == 401


async def test_non_uuid_org_claim_is_401_never_500():
    """The claim is expected to already carry this repo's internal org_id (a UUID) — see
    auth.py's module docstring. A non-UUID value must fail closed exactly like a missing one,
    never raise an unhandled ValueError."""
    token = sign_test_token(**{auth_mod.org_claim_name(): "not-a-uuid"})
    claims = auth_mod.verify_token(_creds(token))
    with pytest.raises(HTTPException) as exc:
        auth_mod.org_id_from(claims)
    assert exc.value.status_code == 401


async def test_jwks_outage_is_503_not_500(monkeypatch):
    """`PyJWKClientError` must map to a retryable dependency-outage response, not a bare 500 —
    same distinction mytheus-backend's auth.py draws (a down IdP pages Clerk, not us)."""

    class _DeadJWKS:
        def get_signing_key_from_jwt(self, token: str):  # noqa: ARG002
            raise PyJWKClientError("Failed to fetch data from the url")

    monkeypatch.setattr(auth_mod, "_jwks_client", lambda: _DeadJWKS())
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_token(_creds(sign_test_token(uuid.uuid4())))
    assert exc.value.status_code == 503


async def test_verified_org_id_end_to_end():
    org_id = uuid.uuid4()
    claims = auth_mod.verify_token(_creds(sign_test_token(org_id)))
    assert auth_mod.verified_org_id(claims) == org_id
