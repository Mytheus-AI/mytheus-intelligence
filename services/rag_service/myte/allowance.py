"""Who is asking, and how much of today's Ask Myte allowance they have used (ADR-0049 R3).

The asker arrives with a Clerk session (`sub`), not our `users.id`. To meter every round to a
person and keep one person from spending a workspace's budget on chat, rag-service resolves
`users.id` by calling identity-tenant-service `GET /users/me` with the asker's own bearer (the
same token bff sees; identity verifies it and answers that person's own `UserProfile`, whose
`id` is `users.id`). The answer is cached in Redis for ten minutes:

    nexus:{org_id}:myte:uid:{clerk_sub}                    -> users.id         (TTL 600 s)
    nexus:{org_id}:myte:units:{user_id}:{YYYY-MM-DD}       -> budget units     (TTL 2 days)

The day is the UTC date, so the allowance resets at 00:00 UTC for everyone (one key per person
per day; the person-facing message says "tomorrow"). `MYTE_USER_DAILY_UNITS` (default 300000
budget units, `libs/shared/cost.py`) is the cap; 0 or less turns the per-person cap off.

Failure is never a refusal: if identity cannot be reached, answers 401/403/404, or Redis is
down, the answer proceeds on the workspace budget alone (the router's `BudgetGate` still gates
every round) and the reason is logged. The workspace cap is the hard guarantee; this cap is a
fairness limit inside it.

Env: `IDENTITY_URL` (default `http://identity-tenant-service:8000`), `MYTE_USER_DAILY_UNITS`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

DEFAULT_IDENTITY_URL = "http://identity-tenant-service:8000"
DEFAULT_DAILY_UNITS = 300_000
UID_TTL_S = 600
UNITS_TTL_S = 2 * 24 * 3600
LOOKUP_TIMEOUT_S = 3.0
OVER_ALLOWANCE = "You've used today's Ask Myte allowance. It resets tomorrow."


class RedisLike(Protocol):
    def get(self, name: str) -> Awaitable[Any]: ...

    def set(self, name: str, value: Any, ex: int | None = None) -> Awaitable[Any]: ...

    def incrby(self, name: str, amount: int = 1) -> Awaitable[Any]: ...

    def expire(self, name: str, time: int) -> Awaitable[Any]: ...


def identity_url() -> str:
    return os.environ.get("IDENTITY_URL", DEFAULT_IDENTITY_URL).rstrip("/")


def daily_units() -> int:
    raw = os.environ.get("MYTE_USER_DAILY_UNITS", "").strip()
    if not raw:
        return DEFAULT_DAILY_UNITS
    try:
        return int(raw)
    except ValueError:
        logger.warning("MYTE_USER_DAILY_UNITS=%r is not a number; using the default", raw)
        return DEFAULT_DAILY_UNITS


def uid_key(org_id: UUID, clerk_sub: str) -> str:
    return f"nexus:{org_id}:myte:uid:{clerk_sub}"


def units_key(org_id: UUID, user_id: UUID, day: str) -> str:
    return f"nexus:{org_id}:myte:units:{user_id}:{day}"


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class Allowance:
    """The asker's `users.id` and their per-day units. One instance per process."""

    def __init__(
        self,
        redis: RedisLike,
        http: httpx.AsyncClient,
        *,
        base_url: str | None = None,
        cap: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._redis = redis
        self._http = http
        self._base = (base_url or identity_url()).rstrip("/")
        self.cap = daily_units() if cap is None else cap
        self._clock = clock or (lambda: datetime.now(UTC))

    def _day(self) -> str:
        return self._clock().astimezone(UTC).date().isoformat()

    async def user_id(self, org_id: UUID, clerk_sub: str | None, authorization: str) -> UUID | None:
        """`users.id` for this Clerk session in this org, or None when it cannot be known."""
        if not clerk_sub:
            return None
        key = uid_key(org_id, clerk_sub)
        try:
            cached = await self._redis.get(key)
            if cached:
                return UUID(_text(cached))
        except Exception:  # noqa: BLE001 - a cache miss, not a failure
            logger.warning("myte: uid cache read failed org=%s", org_id, exc_info=True)
        try:
            res = await self._http.get(
                f"{self._base}/users/me",
                headers={"Authorization": authorization},
                timeout=LOOKUP_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            logger.warning("myte: identity lookup failed org=%s: %s", org_id, exc)
            return None
        if res.status_code != 200:
            logger.warning(
                "myte: identity lookup answered %s org=%s; workspace budget only",
                res.status_code,
                org_id,
            )
            return None
        try:
            uid = UUID(str(res.json()["id"]))
        except (ValueError, KeyError, TypeError):
            logger.warning("myte: identity lookup returned no users.id org=%s", org_id)
            return None
        try:
            await self._redis.set(key, str(uid), ex=UID_TTL_S)
        except Exception:  # noqa: BLE001
            logger.warning("myte: uid cache write failed org=%s", org_id, exc_info=True)
        return uid

    async def over(self, org_id: UUID, user_id: UUID | None) -> bool:
        """Whether this person has used today's allowance. Unknown -> False (fail open)."""
        if user_id is None or self.cap <= 0:
            return False
        try:
            used = await self._redis.get(units_key(org_id, user_id, self._day()))
        except Exception:  # noqa: BLE001
            logger.warning("myte: allowance read failed org=%s", org_id, exc_info=True)
            return False
        try:
            return int(_text(used)) >= self.cap if used is not None else False
        except ValueError:
            return False

    async def add(self, org_id: UUID, user_id: UUID | None, units: int) -> None:
        if user_id is None or units <= 0:
            return
        key = units_key(org_id, user_id, self._day())
        try:
            await self._redis.incrby(key, int(units))
            await self._redis.expire(key, UNITS_TTL_S)
        except Exception:  # noqa: BLE001
            logger.warning("myte: allowance write failed org=%s", org_id, exc_info=True)
