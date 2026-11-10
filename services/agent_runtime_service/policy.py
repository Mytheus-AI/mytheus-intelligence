"""The tenant's confidence thresholds, applied by the agent itself (mytheus-docs ADR-0030 D3).

Before this, every agent banded on the fixed doc 08 numbers (`> 0.7` surfaced, `>= 0.4` kept)
and core-api re-derived the band with the tenant's `agent_confidence_overrides` on store and on
reload. realtime-gateway forwards live on the agent's `surfaced`, so with an override a card
appeared live and vanished on reload, or the reverse. Now the agent applies the same overrides
when it decides `surfaced` (and what to drop), so the gateway's live forward and core-api's
reload agree: one threshold source, core-api's `tenant_settings`, read by both.

`GET {CORE_URL}/internal/agent-policy` answers `{"agent_confidence_overrides": {"<Agent>":
{"surface": x, "silent": y}}}` (the core-api route added with this change), with the service
caller scheme every core-api internal read uses (`X-Mytheus-Service: agent-runtime-service`,
`CORE_CALLER_TOKEN`, `X-Mytheus-Org`). Cached per org for `AGENT_POLICY_TTL_S` (default 60 s)
like the plan features and the company brain; a failure is the defaults, cached 15 s so an
outage costs one call per org per 15 s, never one per window. The parsing mirrors core-api's
`confidence.thresholds`: a malformed entry is the defaults for that agent.

The bands are `confidence > surface` surfaced, `confidence >= silent` kept silent, below that
dropped: exactly core-api's comparisons (`worklogs.surfaced = confidence > surface`, nothing
written below `silent`).

Polish wave 2 (agents that learn): core-api now serves its CALIBRATED thresholds. The workspace
band per agent stays in `agent_confidence_overrides` (the tenant's override moved by the
workspace's confirm/dismiss rates), and `user_overrides` adds a person's own band for the agents
whose output is one person's (`USER_SCOPED_AGENTS`: notes, suggestions, context fills). The
runtime gives each participant's context their merged map (`merged_for`), so a person's notes
and suggestions are banded with their own threshold and the meeting-wide decision and action
cards with the workspace's, which is exactly what core-api's store and reload apply. An
`agent-feedback` event drops the org's cached policy (`invalidate`), so a recalibration reaches
the next window instead of waiting out the 60 s TTL.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
SURFACE_ABOVE = 0.7
SILENT_FROM = 0.4
DEFAULT_TTL_S = 60.0
FAILURE_TTL_S = 15.0
CONTEXT_KEY = "confidence_overrides"  # where the runtime puts the overrides in each context
# The agents whose output is one person's: the only ones a person's own band applies to (the
# same set as core-api's `learning.USER_SCOPED_AGENTS`).
USER_SCOPED_AGENTS = frozenset({"PersonalListenerAgent", "QuestionAgent", "ContextGapAgent"})

Band = Literal["surfaced", "silent", "dropped"]


@dataclass(frozen=True)
class Thresholds:
    surface: float = SURFACE_ABOVE
    silent: float = SILENT_FROM

    def band(self, confidence: float) -> Band:
        if confidence > self.surface:
            return "surfaced"
        if confidence >= self.silent:
            return "silent"
        return "dropped"


DEFAULT = Thresholds()


def thresholds(overrides: object, agent: str) -> Thresholds:
    """This agent's thresholds from the tenant's overrides; the defaults when absent or bad."""
    entry = overrides.get(agent) if isinstance(overrides, Mapping) else None
    if not isinstance(entry, Mapping):
        return DEFAULT
    try:
        surface = float(entry.get("surface", SURFACE_ABOVE))
        silent = float(entry.get("silent", SILENT_FROM))
    except (TypeError, ValueError):
        return DEFAULT
    if not (0.0 <= silent <= surface <= 1.0):
        return DEFAULT
    return Thresholds(surface=surface, silent=silent)


def thresholds_for(context: Mapping[str, Any] | None, agent: str) -> Thresholds:
    """The thresholds for `agent` from a participant context the runtime filled."""
    return thresholds((context or {}).get(CONTEXT_KEY), agent)


def parse_overrides(body: object) -> dict[str, dict[str, float]]:
    raw = body.get("agent_confidence_overrides") if isinstance(body, Mapping) else None
    out: dict[str, dict[str, float]] = {}
    for agent, entry in (raw.items() if isinstance(raw, Mapping) else []):
        t = thresholds({agent: entry}, str(agent))  # malformed -> the defaults, as core-api
        out[str(agent)] = {"surface": t.surface, "silent": t.silent}
    return out


Bands = dict[str, dict[str, float]]


@dataclass(frozen=True)
class AgentPolicy:
    """The workspace bands per agent and, per user id (str), that person's own bands."""

    overrides: Bands
    user_overrides: dict[str, Bands]

    def merged_for(self, user_id: UUID | str | None) -> Bands:
        """The bands one participant's context carries: the workspace's, with the person's own
        on top for the user-scoped agents only."""
        mine = self.user_overrides.get(str(user_id), {}) if user_id is not None else {}
        merged = dict(self.overrides)
        merged.update({a: b for a, b in mine.items() if a in USER_SCOPED_AGENTS})
        return merged


EMPTY_POLICY = AgentPolicy(overrides={}, user_overrides={})


def parse_policy(body: object) -> AgentPolicy:
    """The whole response; `user_overrides` absent (a core-api before wave 2) is `{}`."""
    raw = body.get("user_overrides") if isinstance(body, Mapping) else None
    users: dict[str, Bands] = {}
    for user, bands in (raw.items() if isinstance(raw, Mapping) else []):
        parsed = parse_overrides({"agent_confidence_overrides": bands})
        kept = {a: b for a, b in parsed.items() if a in USER_SCOPED_AGENTS}
        if kept:
            users[str(user)] = kept
    return AgentPolicy(overrides=parse_overrides(body), user_overrides=users)


class AgentPolicyClient:
    """`overrides(org_id)` / `policy(org_id)` never raise: a failure is `{}` (the defaults)."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        ttl_s: float | None = None,
        timeout_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = (base_url or os.environ.get("CORE_URL", "")).strip().rstrip("/")
        self._token = (
            token if token is not None else os.environ.get("CORE_CALLER_TOKEN", "")
        ).strip()
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get("AGENT_POLICY_TTL_S", "") or DEFAULT_TTL_S)
            except ValueError:
                ttl_s = DEFAULT_TTL_S
        self._ttl = max(0.0, ttl_s)
        self._timeout = timeout_s
        self._clock = clock
        self._transport = transport
        self._cache: dict[UUID, tuple[float, AgentPolicy]] = {}
        self.fetches = 0

    async def overrides(self, org_id: UUID) -> dict[str, dict[str, float]]:
        """The workspace bands (what every agent used before wave 2; ProactiveAgent still)."""
        return (await self.policy(org_id)).overrides

    async def policy(self, org_id: UUID) -> AgentPolicy:
        hit = self._cache.get(org_id)
        now = self._clock()
        if hit is not None and now < hit[0]:
            return hit[1]
        found, ok = await self._fetch(org_id)
        ttl = self._ttl if ok else min(self._ttl, FAILURE_TTL_S)
        if ttl > 0:
            self._cache[org_id] = (now + ttl, found)
        return found

    def invalidate(self, org_id: UUID) -> None:
        """Drop the org's cached policy: the next window fetches the recalibrated one."""
        self._cache.pop(org_id, None)

    async def _fetch(self, org_id: UUID) -> tuple[AgentPolicy, bool]:
        self.fetches += 1
        if not self._url or not self._token:
            return EMPTY_POLICY, True  # unconfigured: the defaults, as core-api without any
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}/internal/agent-policy",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("agent-policy unreachable org=%s: %s (default thresholds)", org_id, exc)
            return EMPTY_POLICY, False
        if res.status_code != 200:
            logger.warning(
                "agent-policy org=%s answered %s (default thresholds)", org_id, res.status_code
            )
            return EMPTY_POLICY, False
        try:
            return parse_policy(res.json()), True
        except ValueError:
            return EMPTY_POLICY, False
