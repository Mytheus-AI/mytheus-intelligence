"""core-api's internal agenda route, the MeetingAgent's only input besides the window
(mytheus-docs ADR-0038 D6, D7).

`GET {CORE_URL}/internal/meetings/{meeting_id}/agenda` answers

    {"host_user_id": uuid, "previous_meeting_id": uuid | null,
     "scheduled_start": iso8601 | null, "scheduled_end": iso8601 | null,
     "items": [{"id", "position", "title", "duration_minutes", "status", "status_source"}],
     "carry_forward": {"decisions": [str], "open_actions": [str]}}

Same service-caller scheme as the related-attendees route (`a2a/related.py`) and the P11 routes
(`proactive/core.py`): `X-Mytheus-Service: agent-runtime-service`, `X-Mytheus-Service-Token:
CORE_CALLER_TOKEN`, and the tenant in `X-Mytheus-Org`, never in the path. Optional: without
`CORE_URL` and `CORE_CALLER_TOKEN` there is no client and the MeetingAgent never runs. Any
failure (unreachable, 401, 404 for an unknown or another tenant's meeting, a bad body) is None:
the agent skips that window quietly, it never guesses an agenda.

The parser is tolerant of extra keys and drops malformed items one by one; `scheduled_start` /
`scheduled_end` may be absent (an older core-api), in which case the scheduled length falls
back to the sum of the items' `duration_minutes`, and is None without either.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
DEFAULT_TIMEOUT_S = 2.0
STATUSES = ("pending", "in_progress", "covered", "skipped")
MAX_ITEMS = 20  # ADR-0038 D5
MAX_CARRIED = 20
MAX_TEXT = 300


@dataclass(frozen=True)
class AgendaItem:
    id: UUID
    position: int
    title: str
    duration_minutes: int | None = None
    status: str = "pending"
    status_source: str | None = None


@dataclass(frozen=True)
class Agenda:
    host_user_id: UUID | None
    items: list[AgendaItem]
    previous_meeting_id: UUID | None = None
    open_actions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    scheduled_minutes: float | None = None


class AgendaSource(Protocol):
    async def agenda(self, org_id: UUID, meeting_id: UUID) -> Agenda | None: ...


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _int(raw: object) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return int(raw) if float(raw).is_integer() else None


def _strings(raw: object) -> list[str]:
    out: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())[:MAX_TEXT]
            if text not in out:
                out.append(text)
    return out[:MAX_CARRIED]


def _when(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _scheduled_minutes(body: dict[str, Any], items: list[AgendaItem]) -> float | None:
    start, end = _when(body.get("scheduled_start")), _when(body.get("scheduled_end"))
    if start is not None and end is not None and (start.tzinfo is None) == (end.tzinfo is None):
        minutes = (end - start).total_seconds() / 60
        if minutes > 0:
            return minutes
    planned = [i.duration_minutes for i in items if i.duration_minutes]
    return float(sum(planned)) if planned else None


def parse_item(raw: object) -> AgendaItem | None:
    if not isinstance(raw, dict):
        return None
    item_id, position, title = _uuid(raw.get("id")), _int(raw.get("position")), raw.get("title")
    if item_id is None or position is None or not isinstance(title, str) or not title.strip():
        return None
    status = raw.get("status") if raw.get("status") in STATUSES else "pending"
    source = raw.get("status_source") if raw.get("status_source") in ("agent", "host") else None
    duration = _int(raw.get("duration_minutes"))
    return AgendaItem(
        id=item_id,
        position=position,
        title=" ".join(title.split())[:200],
        duration_minutes=duration if duration is not None and duration > 0 else None,
        status=str(status),
        status_source=str(source) if source else None,
    )


def parse_agenda(body: object) -> Agenda | None:
    if not isinstance(body, dict) or not isinstance(body.get("items"), list):
        return None
    items: list[AgendaItem] = []
    for raw in body["items"]:
        item = parse_item(raw)
        if item is not None and all(i.id != item.id for i in items):
            items.append(item)
    items = sorted(items, key=lambda i: i.position)[:MAX_ITEMS]
    raw_carry = body.get("carry_forward")
    carry: dict[str, Any] = raw_carry if isinstance(raw_carry, dict) else {}
    return Agenda(
        host_user_id=_uuid(body.get("host_user_id")),
        items=items,
        previous_meeting_id=_uuid(body.get("previous_meeting_id")),
        open_actions=_strings(carry.get("open_actions")),
        decisions=_strings(carry.get("decisions")),
        scheduled_minutes=_scheduled_minutes(body, items),
    )


class HttpAgendaClient:
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

    async def agenda(self, org_id: UUID, meeting_id: UUID) -> Agenda | None:
        path = f"/internal/meetings/{meeting_id}/agenda"
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
            logger.warning("core-api %s unreachable: %s (window skipped)", path, exc)
            return None
        if res.status_code != 200:
            logger.info("core-api %s answered %s (window skipped)", path, res.status_code)
            return None
        try:
            return parse_agenda(res.json())
        except ValueError:
            return None


def agenda_from_env() -> HttpAgendaClient | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpAgendaClient(url, token)
