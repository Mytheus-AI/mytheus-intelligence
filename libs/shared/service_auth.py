"""Service-to-service authentication for internal read surfaces (`knowledge-graph-service`).

Same header scheme `mytheus-backend` uses for its internal routes (pii vault, billing budget):
`X-Mytheus-Service` names the caller, `X-Mytheus-Service-Token` carries that caller's shared
secret, and `X-Mytheus-Org` carries the tenant, which is how canon says `org_id` rides a hop. The
caller (for example `bff`) has already resolved the tenant from the user's verified session, so
the org header is trusted only because the secret is.

Secrets come from an env var of the form `service=secret,...` (at least 32 characters each).
Unconfigured means every caller is refused. The comparison is constant-time. The gRPC surface
reads the same three keys from invocation metadata (lower-cased, as gRPC requires).
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

SERVICE_HEADER = "x-mytheus-service"
SERVICE_TOKEN_HEADER = "x-mytheus-service-token"
ORG_HEADER = "x-mytheus-org"


class ServiceAuthError(Exception):
    """Any authentication failure. Deliberately one class: the response never says which check
    failed."""


@dataclass(frozen=True)
class InternalCaller:
    service: str
    org_id: UUID


def parse_caller_tokens(raw: str, allowed: Iterable[str], env_name: str) -> dict[str, bytes]:
    allowed_set = frozenset(allowed)
    out: dict[str, bytes] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        name, sep, secret = entry.partition("=")
        name, secret = name.strip(), secret.strip()
        if not sep or not name or not secret:
            raise ValueError(f"{env_name} entries must be service=secret")
        if name not in allowed_set:
            raise ValueError(f"{name!r} is not an allowed caller for {env_name}")
        if len(secret) < 32:
            raise ValueError(f"the secret for {name!r} in {env_name} must be at least 32 chars")
        out[name] = secret.encode("utf-8")
    return out


def authenticate(
    headers: Mapping[str, str], *, env_name: str, allowed: Iterable[str]
) -> InternalCaller:
    """Authenticate from lower-cased header/metadata keys. Raises `ServiceAuthError`."""
    try:
        tokens = parse_caller_tokens(os.environ.get(env_name, ""), allowed, env_name)
    except ValueError as exc:
        raise ServiceAuthError("internal auth misconfigured") from exc
    service = headers.get(SERVICE_HEADER, "")
    presented = headers.get(SERVICE_TOKEN_HEADER, "").encode("utf-8")
    expected = tokens.get(service)
    if expected is None or not presented or not hmac.compare_digest(expected, presented):
        raise ServiceAuthError("unauthenticated")
    try:
        org_id = UUID(headers.get(ORG_HEADER, ""))
    except ValueError:
        raise ServiceAuthError("unauthenticated") from None
    return InternalCaller(service=service, org_id=org_id)
