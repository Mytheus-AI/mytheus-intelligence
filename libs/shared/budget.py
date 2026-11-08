"""`check_budget(org_id, est_tokens)`: the inline per-tenant budget gate before every LLM call.

Plan 7-9 section 3.2 step 9, ADR-0004 stage 5. One Redis read of the mirror that
`billing-metering-service` publishes, `nexus:{org_id}:budget:{period}`:

- `block`    when `tokens_consumed + est_tokens >= hard_cap`: no model is called.
- `throttle` when `tokens_consumed >= token_quota`: Haiku only (the router downgrades).
- `ok`       otherwise.

Fail closed. A mirror miss (first call of a tenant, first call of a month) is read through
`billing-metering-service`'s `POST /billing/internal/budget`, which provisions the row and
republishes the mirror; this tier writes no Postgres and cannot do that itself. If Redis AND
billing are both unreachable, or billing refuses, the answer is `block`, never `ok`.

`enforcement` comes from the mirror (`BUDGET_ENFORCEMENT_MODE` in billing). In `alert_only`
the gate logs what it would have done and answers `ok`, so the 80/95 alerts and the meter run
before anyone is degraded (plan section 3.1 step 10). Infrastructure failure still blocks in
`alert_only`: that mode relaxes the quota, not the requirement to know the quota.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import httpx
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

Decision = Literal["ok", "throttle", "block"]
Refresher = Callable[[UUID], Awaitable[dict[str, Any] | None]]


def period(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


def mirror_key(org_id: UUID, at: datetime | None = None) -> str:
    return f"nexus:{org_id}:budget:{period(at)}"


@dataclass(frozen=True)
class BudgetView:
    tokens_consumed: int
    token_quota: int
    hard_cap: int
    enforcement: str

    @classmethod
    def parse(cls, raw: dict[Any, Any]) -> BudgetView:
        def get(name: str) -> Any:
            value = raw.get(name, raw.get(name.encode()))
            return value.decode() if isinstance(value, bytes) else value

        return cls(
            tokens_consumed=int(get("tokens_consumed")),
            token_quota=int(get("token_quota")),
            hard_cap=int(get("hard_cap")),
            enforcement=str(get("enforcement") or "alert_only"),
        )

    def decide(self, est_tokens: int) -> Decision:
        if self.tokens_consumed + est_tokens >= self.hard_cap:
            return "block"
        if self.tokens_consumed >= self.token_quota:
            return "throttle"
        return "ok"


class BillingRefresher:
    """Calls `POST /billing/internal/budget` with this service's credentials."""

    def __init__(
        self,
        base_url: str | None = None,
        service: str = "agent-runtime-service",
        token: str | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self._url = (base_url or os.environ.get("BILLING_URL", "http://localhost:8006")).rstrip("/")
        self._service = service
        self._token = token if token is not None else os.environ.get("BILLING_SERVICE_TOKEN", "")
        self._timeout = timeout_s

    async def __call__(self, org_id: UUID) -> dict[str, Any] | None:
        if not self._token:
            logger.error("BILLING_SERVICE_TOKEN unset: budget read-through refused (block)")
            return None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/billing/internal/budget",
                    headers={
                        "X-Mytheus-Service": self._service,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("budget read-through failed org=%s: %s", org_id, exc)
            return None
        if resp.status_code != 200:
            logger.warning("budget read-through org=%s answered %s", org_id, resp.status_code)
            return None
        body: dict[str, Any] = resp.json()
        return body


class BudgetGate:
    def __init__(self, redis: Redis, refresh: Refresher) -> None:
        self._redis = redis
        self._refresh = refresh

    async def view(self, org_id: UUID) -> BudgetView | None:
        raw: dict[Any, Any] = {}
        try:
            raw = await self._redis.hgetall(mirror_key(org_id))
        except Exception:  # Redis down: read through, never assume
            logger.warning("budget mirror read failed org=%s; reading through", org_id)
        if not raw:
            refreshed = await self._refresh(org_id)
            if not refreshed:
                return None
            raw = refreshed
        try:
            return BudgetView.parse(raw)
        except (TypeError, ValueError):
            logger.error("budget mirror for org=%s is malformed: %r", org_id, raw)
            return None

    async def check(self, org_id: UUID, est_tokens: int) -> Decision:
        view = await self.view(org_id)
        if view is None:
            return "block"
        decision = view.decide(est_tokens)
        if decision != "ok" and view.enforcement != "enforce":
            logger.info(
                "budget alert_only: org=%s would %s (consumed=%s quota=%s cap=%s)",
                org_id,
                decision,
                view.tokens_consumed,
                view.token_quota,
                view.hard_cap,
            )
            return "ok"
        return decision
