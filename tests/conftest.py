"""Shared fixtures for this repo's tests.

Tests run against the REAL local dev stack (task brief: "The infrastructure is real. Do not
build fakes for it."): the shared Postgres (`localhost:5433`, db `mytheus` — the SAME physical
database `mytheus-backend` uses), Redis (`localhost:6380`), Redpanda + Schema Registry
(`localhost:9092` / `:8081`, all 21 subjects registered under BACKWARD), and Qdrant
(`localhost:6333`). Tests that need a service they cannot reach skip gracefully, EXCEPT when
`MYTHEUS_REQUIRE_DB=1` (or the per-service equivalents below), matching `mytheus-backend`'s
`tests/conftest.py` fail-loud-not-skip convention (defect D7 there: a silently-skipped
isolation suite reported green for a year).

Roles. This repo's own migration (`9c2b7e4a1d03`) creates two least-privilege roles, neither
holding `rolsuper` or `rolbypassrls`:
- `mytheus_ai_app`    the consumer/request path. Subject to `intelligence.*`'s `tenant_isolation`.
- `mytheus_ai_relay`  drains `intelligence.outbox` only, via `outbox_relay_drain`.

`BOOTSTRAP_URL` is the privileged credential that owns the `intelligence` schema and runs
migrations. `mytheus-backend`'s bootstrap is `postgres:postgres`; this repo's bootstrap is
`supabase_admin:postgres` instead, because `CREATE SCHEMA` needs database-level CREATE, which
this Supabase Postgres image's `postgres` role does not hold (PG15+ revokes `CREATE ON
DATABASE` from non-owners by default; `mytheus`'s actual owner is `supabase_admin`) — confirmed
live before writing the migration (`has_database_privilege('postgres','mytheus','CREATE')` =
false). `mytheus-backend`'s migrations never hit this because they only ever used the
already-`postgres`-owned `public` schema.
"""

from __future__ import annotations

import os
import time
import uuid

import psycopg2
import pytest

_DEFAULT_BOOTSTRAP = "postgresql+asyncpg://supabase_admin:postgres@localhost:5433/mytheus"

os.environ.setdefault("REDPANDA_BROKERS", "localhost:9092")
os.environ.setdefault("SCHEMA_REGISTRY_URL", "http://localhost:8081")
os.environ.setdefault("REDIS_URL", "redis://localhost:6380")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
# Must match docker-compose.yml's defaults. Both keys are required and must differ: the
# read-only key is what makes "embedding-service is the only writer" an infrastructure control
# rather than a convention (status-overview gap 19).
os.environ.setdefault("QDRANT_API_KEY", "dev-qdrant-write")
os.environ.setdefault("QDRANT_READ_ONLY_API_KEY", "dev-qdrant-read")
# Plan 5-6 task 9 (ingestion-service): real MinIO, S3-API compatible — see
# libs/shared/object_storage.py. Same dev credentials the task brief specified.
os.environ.setdefault("MINIO_ENDPOINT_URL", "http://localhost:9002")
os.environ.setdefault("MINIO_ACCESS_KEY", "mytheus")
os.environ.setdefault("MINIO_SECRET_KEY", "mytheus-dev-secret")

BOOTSTRAP_URL = os.environ.get("BOOTSTRAP_DATABASE_URL") or _DEFAULT_BOOTSTRAP
os.environ.setdefault("DATABASE_URL", BOOTSTRAP_URL)
_SYNC_URL = BOOTSTRAP_URL.replace("+asyncpg", "")

# Dev-only passwords, same convention as mytheus-backend: the migration creates these roles
# with no password; a real environment sets one via a bootstrap script from the secrets
# manager, and here the fixture does it directly.
ROLE_PASSWORDS = {
    "mytheus_ai_app": "mytheus_ai_app",
    "mytheus_ai_relay": "mytheus_ai_relay",
}


def _role_url(role: str) -> str:
    scheme, _, rest = BOOTSTRAP_URL.partition("://")
    _credentials, _, host_and_db = rest.rpartition("@")
    return f"{scheme}://{role}:{ROLE_PASSWORDS[role]}@{host_and_db}"


APP_URL = _role_url("mytheus_ai_app")
RELAY_URL = _role_url("mytheus_ai_relay")


def _db_reachable() -> bool:
    try:
        psycopg2.connect(_SYNC_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


def _require_or_skip(reachable: bool, what: str, env_flag: str) -> None:
    if reachable:
        return
    if os.environ.get(env_flag) == "1":
        raise RuntimeError(f"{env_flag}=1 but {what} is unreachable. Refusing to skip.")
    pytest.skip(f"{what} not reachable (expected in the real local dev stack)")


@pytest.fixture(scope="session")
def migrated_db() -> str:
    """Apply this repo's Alembic history once and give the two roles a dev password."""
    _require_or_skip(_db_reachable(), f"Postgres at {_SYNC_URL}", "MYTHEUS_REQUIRE_DB")

    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")

    conn = psycopg2.connect(_SYNC_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        for role, password in ROLE_PASSWORDS.items():
            cur.execute(f"ALTER ROLE {role} WITH LOGIN PASSWORD %s", (password,))
    conn.close()
    return BOOTSTRAP_URL


def qdrant_reachable() -> bool:
    try:
        import httpx

        return (
            httpx.get(
                os.environ["QDRANT_URL"] + "/collections",
                headers={"api-key": os.environ["QDRANT_API_KEY"]},
                timeout=2,
            ).status_code
            == 200
        )
    except Exception:
        return False


def redis_reachable() -> bool:
    try:
        import redis as redis_sync

        redis_sync.from_url(os.environ["REDIS_URL"]).ping()
        return True
    except Exception:
        return False


def schema_registry_reachable() -> bool:
    try:
        import httpx

        return (
            httpx.get(os.environ["SCHEMA_REGISTRY_URL"] + "/subjects", timeout=2).status_code == 200
        )
    except Exception:
        return False


@pytest.fixture(scope="session")
def qdrant_client():
    """The privileged client, as `embedding-service` holds it."""
    _require_or_skip(qdrant_reachable(), "Qdrant", "MYTHEUS_REQUIRE_QDRANT")
    from libs.shared.qdrant import write_client

    return write_client()


@pytest.fixture(scope="session")
def qdrant_read_client():
    """The read-only client every other consumer holds. Used to prove it cannot write."""
    _require_or_skip(qdrant_reachable(), "Qdrant", "MYTHEUS_REQUIRE_QDRANT")
    from libs.shared.qdrant import read_client

    return read_client()


@pytest.fixture(scope="session")
def qdrant_unauthenticated():
    """No credential at all, to prove the server requires one. Without this the two keyed
    fixtures prove only that Qdrant honours a key it was given, not that it demands one."""
    _require_or_skip(qdrant_reachable(), "Qdrant", "MYTHEUS_REQUIRE_QDRANT")
    from qdrant_client import AsyncQdrantClient

    return AsyncQdrantClient(url=os.environ["QDRANT_URL"])


@pytest.fixture()
def redis_client():
    _require_or_skip(redis_reachable(), "Redis", "MYTHEUS_REQUIRE_REDIS")
    from redis.asyncio import Redis

    return Redis.from_url(os.environ["REDIS_URL"])


@pytest.fixture(scope="session")
def avro_codec():
    _require_or_skip(schema_registry_reachable(), "Schema Registry", "MYTHEUS_REQUIRE_REGISTRY")
    from libs.shared.avro_codec import AvroCodec

    return AvroCodec(os.environ["SCHEMA_REGISTRY_URL"])


def minio_reachable() -> bool:
    try:
        import httpx

        return (
            httpx.get(
                os.environ["MINIO_ENDPOINT_URL"] + "/minio/health/live", timeout=2
            ).status_code
            == 200
        )
    except Exception:
        return False


@pytest.fixture(scope="session")
def s3_client():
    _require_or_skip(minio_reachable(), "MinIO", "MYTHEUS_REQUIRE_MINIO")
    from libs.shared.object_storage import get_client

    return get_client()


# --- Clerk auth test doubles (libs/shared/auth.py) ---------------------------------------
#
# Every REST route in this repo now depends on `libs.shared.auth.verified_org_id`, which
# verifies a real Clerk RS256 session token against JWKS. Signing a token that a REAL Clerk
# JWKS endpoint will verify requires an actual Clerk sign-in; tests instead generate their own
# RSA keypair and monkeypatch `libs.shared.auth._jwks_client` to serve its public half, exactly
# matching mytheus-backend's own `tests/test_auth.py` fixture shape (`_FakeJWKS`, `_keypair`)
# rather than inventing a second test-doubling convention. `CLERK_ISSUER` is set here (not
# hardcoded in `auth.py`, which still reads it from the environment in production) so
# `verify_token`'s `issuer=` check has something real to compare against.
os.environ.setdefault("CLERK_ISSUER", "https://test.clerk.accounts.dev")
os.environ.setdefault("CLERK_JWKS_URL", "https://test.clerk.accounts.dev/.well-known/jwks.json")


def _generate_test_keypair() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return priv, pub


_TEST_PRIVATE_KEY, _TEST_PUBLIC_KEY = _generate_test_keypair()


class _FakeJWKS:
    """Stands in for `PyJWKClient`: hands back this session's own test public key regardless
    of the token's `kid`, so no network call to a real Clerk JWKS endpoint is ever made."""

    def get_signing_key_from_jwt(self, token: str):  # noqa: ARG002
        return type("K", (), {"key": _TEST_PUBLIC_KEY})()


@pytest.fixture(autouse=True)
def _patch_clerk_jwks(monkeypatch):
    """Every test gets a working, fully-local Clerk verifier for free — opt out is never
    needed since this only replaces the network-calling JWKS lookup, not the real RS256/claims
    verification logic in `libs.shared.auth.verify_token`."""
    from libs.shared import auth as auth_mod

    monkeypatch.setattr(auth_mod, "_jwks_client", lambda: _FakeJWKS())


def sign_test_token(org_id: uuid.UUID | str | None = None, **claim_overrides: object) -> str:
    """A real, correctly-signed (against `_FakeJWKS`'s keypair) Clerk-shaped session token.
    Pass `org_id=None` to omit the org claim entirely (a personal-account session); pass an
    explicit `<claim name>=...` override to build a malformed/expired/wrong-issuer token for a
    fail-closed test, matching mytheus-backend's `tests/test_auth.py::_sign` shape."""
    import jwt as pyjwt

    from libs.shared.auth import org_claim_name

    now = int(time.time())
    claims: dict[str, object] = {
        "sub": f"user_{uuid.uuid4().hex[:8]}",
        "iss": os.environ["CLERK_ISSUER"],
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
    }
    if org_id is not None:
        claims[org_claim_name()] = str(org_id)
    claims.update(claim_overrides)
    return pyjwt.encode(claims, _TEST_PRIVATE_KEY, algorithm="RS256")


def auth_headers(org_id: uuid.UUID | str, **claim_overrides: object) -> dict[str, str]:
    """The `Authorization: Bearer ...` header every authenticated test request now carries, in
    place of the old (vulnerable) bare `X-Org-Id` header."""
    return {"Authorization": f"Bearer {sign_test_token(org_id, **claim_overrides)}"}
