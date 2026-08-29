"""Service identity for internal-only routes, where there is no user to authenticate.

This REIMPLEMENTS `mytheus-backend`'s `libs/shared/internal_auth.py` (the two repos cannot
import across each other, the same reason this repo's `libs/shared/auth.py` reimplements that
repo's Clerk verification rather than sharing it). Read that module's docstring for the full
reasoning; the short version, and the part that matters when editing this file:

`libs/shared/auth.py` verifies a Clerk session JWT and is the right guard for every route with
a user behind it - `POST /ingest`, `GET /search`, `POST /embed`, `POST /rerank`. `POST /ner` is
the exception, and not by preference: its caller is `transcription-service`'s redaction pass,
running inside a Deepgram callback while proxying live meeting audio, where no request and no
session token exist. Passing a participant's token down the audio path was the alternative and
is worse - it would mean holding a user credential for the length of a meeting and using it for
work they never initiated.

So this authenticates a WORKLOAD, not a person. It does not authorize a tenant: `org_id` on a
`/ner` request is used for log and rate accounting only, exactly as `POST /rerank` uses it, and
`ner-service` owns no data and makes no storage decision from it. That keeps the blast radius
of a leaked credential here to log pollution - materially smaller than on the vault's
mint/resolve routes in `mytheus-backend`, which is why those additionally audit every
disclosure.

The real answer is workload identity (mTLS with SPIFFE/SPIRE, or short-lived signed service
tokens), which needs a service mesh the program does not have on a single Compose node
(ADR-0016). Named here as the successor rather than pretended into existence.

`INTERNAL_SERVICE_TOKEN` has no default. An authentication mechanism with a default is one that
is off in exactly the deployment nobody checked.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

ENV_VAR = "INTERNAL_SERVICE_TOKEN"
HEADER = "X-Internal-Service-Token"

# The minimum that is not trivially guessable. Matches `mytheus-backend`'s constant; the two
# must agree, because they compare the same deployment-wide secret.
_MIN_TOKEN_LEN = 32


def configured_token() -> str:
    """The expected secret. Raises 503 if unset or too short, at call time.

    Deliberately not cached: a cached miss would outlive a fixed configuration for the life of
    the process, and tests set this per case.
    """
    token = os.environ.get(ENV_VAR, "").strip()
    if not token:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "This deployment has no internal service credential configured",
        )
    if len(token) < _MIN_TOKEN_LEN:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"{ENV_VAR} must be at least {_MIN_TOKEN_LEN} characters",
        )
    return token


async def require_internal_service(x_internal_service_token: str = Header(default="")) -> None:
    """Route dependency for an internal-only endpoint.

    `hmac.compare_digest` rather than `==`: string equality short-circuits on the first
    differing byte, which leaks a long-lived shared secret to anyone who can time enough
    requests.
    """
    if not hmac.compare_digest(x_internal_service_token, configured_token()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "This endpoint is internal and requires a valid service credential",
        )
