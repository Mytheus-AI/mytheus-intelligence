"""core-api's internal read routes for `ProactiveAgent` (P11 contract section 3).

- `GET {CORE_URL}/internal/actions/{action_id}/context` -> `{assignee_id, assignee_name,
  meeting_id, organizer_id, open_action_count, depends_on: [{id, content, status}],
  participants: [uuid], admins: [uuid]}`. `open_action_count` counts the assignee's confirmed,
  not-done action items in the org INCLUDING the triggering one; `admins` are the tenant's owners
  and admins (the whole org, not only this meeting); `assignee_name` is the user's email.
- `GET {CORE_URL}/internal/meetings/{meeting_id}/participants` -> `{participants: [{user_id,
  name, rbac_tier}]}`.

Same service-caller scheme as P10's related-attendees route (`a2a/related.py`):
`X-Mytheus-Service: agent-runtime-service`, `X-Mytheus-Service-Token: CORE_CALLER_TOKEN`,
`X-Mytheus-Org`. The org comes from the verified event header; it is never in the path or the
query. Any failure (unreachable, 401, 404 for an unknown or another tenant's id, a bad body) is
None: the caller skips the check, it never guesses.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
DEFAULT_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Dependency:
    id: UUID
    content: str
    status: str


@dataclass(frozen=True)
class ActionContext:
    assignee_id: UUID | None
    assignee_name: str | None
    meeting_id: UUID | None
    open_action_count: int
    organizer_id: UUID | None = None
    depends_on: list[Dependency] = field(default_factory=list)
    participants: list[UUID] = field(default_factory=list)
    admins: list[UUID] = field(default_factory=list)


@dataclass(frozen=True)
class Participant:
    user_id: UUID
    name: str | None = None
    rbac_tier: str | None = None


class CoreContext(Protocol):
    async def action_context(self, org_id: UUID, action_id: UUID) -> ActionContext | None: ...

    async def participants(self, org_id: UUID, meeting_id: UUID) -> list[Participant] | None: ...


def as_uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _uuids(raw: object) -> list[UUID]:
    out: list[UUID] = []
    for value in raw if isinstance(raw, list) else []:
        uid = as_uuid(value)
        if uid is not None and uid not in out:
            out.append(uid)
    return out


def parse_action_context(body: object) -> ActionContext | None:
    if not isinstance(body, dict):
        return None
    count = body.get("open_action_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return None
    deps: list[Dependency] = []
    for d in body.get("depends_on") or []:
        if not isinstance(d, dict):
            continue
        did = as_uuid(d.get("id"))
        if did is None:
            continue
        deps.append(Dependency(did, str(d.get("content") or ""), str(d.get("status") or "")))
    name = body.get("assignee_name")
    return ActionContext(
        assignee_id=as_uuid(body.get("assignee_id")),
        assignee_name=name if isinstance(name, str) and name.strip() else None,
        meeting_id=as_uuid(body.get("meeting_id")),
        open_action_count=count,
        organizer_id=as_uuid(body.get("organizer_id")),
        depends_on=deps,
        participants=_uuids(body.get("participants")),
        admins=_uuids(body.get("admins")),
    )


def parse_participants(body: object) -> list[Participant] | None:
    if not isinstance(body, dict) or not isinstance(body.get("participants"), list):
        return None
    out: list[Participant] = []
    for p in body["participants"]:
        if not isinstance(p, dict):
            continue
        uid = as_uuid(p.get("user_id"))
        if uid is None or any(x.user_id == uid for x in out):
            continue
        name, tier = p.get("name"), p.get("rbac_tier")
        out.append(
            Participant(
                uid,
                name if isinstance(name, str) and name.strip() else None,
                tier if isinstance(tier, str) and tier else None,
            )
        )
    return out


class HttpCoreContext:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    async def _get(self, org_id: UUID, path: str) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("core-api %s unreachable: %s (check skipped)", path, exc)
            return None
        if res.status_code != 200:
            logger.info("core-api %s answered %s (check skipped)", path, res.status_code)
            return None
        try:
            return res.json()
        except ValueError:
            return None

    async def action_context(self, org_id: UUID, action_id: UUID) -> ActionContext | None:
        return parse_action_context(
            await self._get(org_id, f"/internal/actions/{action_id}/context")
        )

    async def participants(self, org_id: UUID, meeting_id: UUID) -> list[Participant] | None:
        return parse_participants(
            await self._get(org_id, f"/internal/meetings/{meeting_id}/participants")
        )


def core_context_from_env() -> HttpCoreContext | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpCoreContext(url, token)
