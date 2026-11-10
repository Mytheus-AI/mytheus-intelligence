"""The tenant's plan features from `billing-metering-service` (P11; plan 10-13 section 3.6 step 4).

`GET {BILLING_URL}/internal/plan-features` with the same caller identity the BudgetGate uses
(`X-Mytheus-Service: agent-runtime-service`, `X-Mytheus-Service-Token: BILLING_SERVICE_TOKEN`,
listed on billing in `MYTHEUS_BILLING_CALLER_TOKENS`, and the tenant in `X-Mytheus-Org`)
answers `{"plan_code": ..., "features": {...}}`. The route sits under `/internal` (not
`/billing`) because api-gateway routes `/billing` publicly and never routes `/internal`.

Cached per org for `PLAN_FEATURES_TTL_S` (default 300 s, the contract's 5 minutes), including a
failure, so an outage costs one call per org per TTL rather than one per event. Fail closed: an
unreachable billing, a non-200, a missing token or a body without `features` is "no features",
and a feature that is absent or not literally `true` is off.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
DEFAULT_TTL_S = 300.0


class PlanFeaturesClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        ttl_s: float | None = None,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = (base_url or os.environ.get("BILLING_URL", "http://localhost:8006")).rstrip("/")
        self._token = token if token is not None else os.environ.get("BILLING_SERVICE_TOKEN", "")
        self._ttl = ttl_s if ttl_s is not None else _ttl_from_env()
        self._timeout = timeout_s
        self._clock = clock
        self._transport = transport
        self._cache: dict[UUID, tuple[float, dict[str, Any]]] = {}
        self.fetches = 0

    async def features(self, org_id: UUID) -> dict[str, Any]:
        now = self._clock()
        hit = self._cache.get(org_id)
        if hit is not None and hit[0] > now:
            return hit[1]
        found = await self._fetch(org_id)
        self._cache[org_id] = (now + self._ttl, found)
        return found

    async def enabled(self, org_id: UUID, feature: str) -> bool:
        return (await self.features(org_id)).get(feature) is True

    async def _fetch(self, org_id: UUID) -> dict[str, Any]:
        self.fetches += 1
        if not self._token:
            logger.warning("BILLING_SERVICE_TOKEN unset: plan features unknown (all off)")
            return {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}/internal/plan-features",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("plan-features unreachable org=%s: %s (all off)", org_id, exc)
            return {}
        if res.status_code != 200:
            logger.warning("plan-features org=%s answered %s (all off)", org_id, res.status_code)
            return {}
        try:
            features = res.json().get("features")
        except (ValueError, AttributeError):
            return {}
        return dict(features) if isinstance(features, dict) else {}


def _ttl_from_env() -> float:
    raw = os.environ.get("PLAN_FEATURES_TTL_S", "").strip()
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_TTL_S
    except ValueError:
        return DEFAULT_TTL_S
