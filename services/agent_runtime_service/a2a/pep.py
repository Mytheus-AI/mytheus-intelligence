"""The A2A policy enforcement point (ADR-0013 sections 2 to 4; P10 contract sections 3 and 4).

The PDP (`identity-tenant-service` `Authorize`) answers one pure lookup: may an agent whose
principal holds `grantee_role` ask `query_class` of a target at `target_scope`? The PEP derives
the four inputs itself, never from a payload:

- `org_id` and the principal from the VERIFIED agent id (`identity.py`);
- `grantee_role` = the principal's RBAC tier (`owner | admin | member`, UserContext.rbac_tier);
- `query_class` = `classify_query(query)`, deterministic, no LLM: decision words ->
  `decision_lineage`; action / owner / due / task / ticket words -> `action_status`; a query with
  neither signal -> `meeting_context`. The class is ENFORCED, not only authorized:
  `allowed_kinds(query_class)` bounds which citation kinds may come back (decision -> decisions,
  action -> actions, meeting_context -> transcript and document passages), on the responder and
  on the requester's own RAG hop. So the fallback to `meeting_context` cannot widen what a grant
  exposes, which is how ADR-0013's "a query outside the taxonomy is denied" holds: such a query
  can only ever reach what a `meeting_context` grant already covers;
- `target_scope` = `resolve_scope(...)` from both UserContexts: the same user -> `self`; a shared
  team -> `same_team`; the same department -> `department`; anyone else in the org -> `org`.

Decision cache: `nexus:{org_id}:a2a:grant:{grantee_role}:{query_class}:{target_scope}`, value
`allow` / `deny`, TTL 60 s. A miss calls the PDP and caches its answer. A PDP failure is a deny
and is NOT cached, so the next check asks again. An unknown role, class or scope is a deny
without any call. Redis being down only costs the cache: the PDP is still asked. Concurrent
misses on the same key in one process share a single PDP call. An answer is cached only when
the per-org grant version (`nexus:{org_id}:a2a:grant:version`, INCRed by identity-tenant-service
on every grant change) did not move during the PDP call, so a revoke racing a miss can never
leave a stale `allow` in the cache.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

ROLES = frozenset({"owner", "admin", "member"})
QUERY_CLASSES = frozenset({"meeting_context", "decision_lineage", "action_status"})
TARGET_SCOPES = frozenset({"self", "same_team", "department", "org"})
GRANT_TTL_S = 60
CLASS_KINDS: dict[str, frozenset[str]] = {
    "decision_lineage": frozenset({"decision"}),
    "action_status": frozenset({"action"}),
    "meeting_context": frozenset({"transcript", "document"}),
}

_DECISION_WORDS = re.compile(
    r"\b(decid\w*|decision\w*|agreed|agree|agreement|approv\w*|chose|chosen|choice|"
    r"rationale|signed off|sign-off|ruled out|went with|settled on)\b",
    re.IGNORECASE,
)
_ACTION_WORDS = re.compile(
    r"\b(actions?|action items?|owners?|owns|owned|ownership|assign\w*|assignee|due|deadlines?|"
    r"tasks?|tickets?|todos?|to-dos?|follow[- ]ups?|status|blockers?|blocked|responsible|"
    r"deliverables?|eta)\b",
    re.IGNORECASE,
)


class Pdp(Protocol):
    async def authorize(
        self, org_id: UUID, grantee_role: str, query_class: str, target_scope: str
    ) -> str | None: ...


def classify_query(query: str) -> str:
    if _DECISION_WORDS.search(query):
        return "decision_lineage"
    if _ACTION_WORDS.search(query):
        return "action_status"
    return "meeting_context"


def resolve_scope(
    requester: UUID,
    requester_ctx: dict[str, Any],
    target: UUID,
    target_ctx: dict[str, Any],
) -> str:
    if requester == target:
        return "self"
    mine = {t for t in requester_ctx.get("team_ids") or [] if t}
    theirs = {t for t in target_ctx.get("team_ids") or [] if t}
    if mine & theirs:
        return "same_team"
    dept = requester_ctx.get("department_id") or ""
    if dept and dept == (target_ctx.get("department_id") or ""):
        return "department"
    return "org"


def grantee_role(ctx: dict[str, Any] | None) -> str:
    return str((ctx or {}).get("rbac_tier") or "").strip().lower()


def allowed_kinds(query_class: str) -> frozenset[str]:
    """The citation kinds a query class may return. The class is enforced, not just a label: a
    `decision_lineage` grant can only ever surface decisions, `action_status` only actions,
    `meeting_context` only transcript and document passages. Unknown class: nothing."""
    return CLASS_KINDS.get(query_class, frozenset())


def grant_key(org_id: UUID, role: str, query_class: str, target_scope: str) -> str:
    return f"nexus:{org_id}:a2a:grant:{role}:{query_class}:{target_scope}"


class Pep:
    def __init__(self, redis: Any, pdp: Pdp, ttl_s: int = GRANT_TTL_S) -> None:
        self._redis = redis
        self._pdp = pdp
        self._ttl = ttl_s
        # Concurrent misses on one key share one PDP call (a cold cache at a window boundary
        # would otherwise send one Authorize per peer per requester at the same instant).
        self._inflight: dict[str, asyncio.Future[bool]] = {}

    async def authorize(self, org_id: UUID, role: str, query_class: str, target_scope: str) -> bool:
        """True only for an explicit `allow` (cached or from the PDP). Fail closed otherwise."""
        if (
            role not in ROLES
            or query_class not in QUERY_CLASSES
            or target_scope not in TARGET_SCOPES
        ):
            return False
        key = grant_key(org_id, role, query_class, target_scope)
        cached: Any = None
        try:
            cached = await self._redis.get(key)
        except Exception:  # noqa: BLE001 - a cache outage must not decide anything
            logger.warning("a2a grant cache read failed org=%s; asking the PDP", org_id)
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8", "replace")
        if cached in ("allow", "deny"):
            return bool(cached == "allow")
        flight = self._inflight.get(key)
        if flight is None:
            flight = asyncio.ensure_future(self._ask(key, org_id, role, query_class, target_scope))
            self._inflight[key] = flight
            flight.add_done_callback(lambda _f: self._inflight.pop(key, None))
        return await asyncio.shield(flight)

    async def _ask(
        self, key: str, org_id: UUID, role: str, query_class: str, target_scope: str
    ) -> bool:
        """One PDP call, cached only if no grant changed while it was in flight.

        identity-tenant-service INCRs `nexus:{org_id}:a2a:grant:version` on every grant change
        (and deletes `nexus:{org_id}:a2a:grant:*`, which can include the version key itself, so
        a missing key reads as 0). The key is WATCHed before the PDP call and compared after:
        an INCR or DEL in between aborts the cache write (WatchError), so an `allow` read just
        before a revoke is returned once but never cached for 60 s."""
        vkey = version_key(org_id)
        pipe: Any = None
        before: str | None = None
        try:
            pipe = self._redis.pipeline(transaction=True)
            await pipe.watch(vkey)
            before = _version(await pipe.get(vkey))
        except Exception:  # noqa: BLE001 - no version, no caching; the PDP still answers
            logger.warning("a2a grant version read failed org=%s; answer not cached", org_id)
            before = None
        try:
            try:
                decision = await self._pdp.authorize(org_id, role, query_class, target_scope)
            except Exception:  # noqa: BLE001 - PDP unreachable = deny
                logger.warning("a2a PDP call raised org=%s (deny)", org_id)
                decision = None
            if decision not in ("allow", "deny"):
                return False  # PDP down or nonsense: deny, and do not cache it
            if before is not None:
                try:
                    if _version(await pipe.get(vkey)) == before:
                        pipe.multi()
                        pipe.set(key, decision, ex=self._ttl)
                        await pipe.execute()
                    else:
                        logger.info("a2a grants changed during the PDP call org=%s", org_id)
                except Exception:  # noqa: BLE001 - WatchError included: just not cached
                    logger.info("a2a grant cache write skipped org=%s", org_id)
            return bool(decision == "allow")
        finally:
            if pipe is not None:
                try:
                    await pipe.reset()
                except Exception:  # noqa: BLE001
                    pass


def version_key(org_id: UUID) -> str:
    return f"nexus:{org_id}:a2a:grant:version"


def _version(raw: Any) -> str:
    if raw is None:
        return "0"
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
