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

ADR-0039 adds, all optional and parsed tolerantly (an older core-api leaves them out):
- per item `owner_user_id` and `goal_id` (B3; C5: the goal is shown to the MeetingAgent, the
  owner decides the `proxy_update` nudge);
- `proxies: [{user_id, proxy_update, proxy_questions}]` (B5): the participants whose attendance
  is `agent`. The runtime gives each a PersonalListener pass on every window (C6);
- `title` and `confidential` (B2): ProactiveAgent's window checks name the meeting in their
  alerts and never send `topic_watch` alerts from a confidential meeting (C7). ADR-0039 does not
  list these two on any route the runtime calls; they are read from this route because it is the
  runtime's per-meeting read already. `confidential` absent is None, which the topic_watch check
  treats as confidential (fail closed).

`CachedAgendaSource` shares one read per meeting between the runtime (proxies) and the agents,
for `ttl_s` (default 60 s); a failure is not cached.

The parser is tolerant of extra keys and drops malformed items one by one; `scheduled_start` /
`scheduled_end` may be absent (an older core-api), in which case the scheduled length falls
back to the sum of the items' `duration_minutes`, and is None without either.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
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


MAX_PROXY_UPDATE = 2000  # B5
MAX_PROXY_QUESTIONS = 5  # B5
MAX_PROXY_QUESTION = 300  # B5


@dataclass(frozen=True)
class AgendaItem:
    id: UUID
    position: int
    title: str
    duration_minutes: int | None = None
    status: str = "pending"
    status_source: str | None = None
    owner_user_id: UUID | None = None
    goal_id: UUID | None = None


@dataclass(frozen=True)
class Proxy:
    """A participant whose attendance is `agent` (ADR-0039 B5)."""

    user_id: UUID
    proxy_update: str = ""
    proxy_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Agenda:
    host_user_id: UUID | None
    items: list[AgendaItem]
    previous_meeting_id: UUID | None = None
    open_actions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    scheduled_minutes: float | None = None
    proxies: list[Proxy] = field(default_factory=list)
    title: str = ""
    confidential: bool | None = None  # None: the route did not say (treated as confidential)

    def proxy(self, user_id: UUID | None) -> Proxy | None:
        return next((p for p in self.proxies if p.user_id == user_id), None)


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
        owner_user_id=_uuid(raw.get("owner_user_id")),
        goal_id=_uuid(raw.get("goal_id")),
    )


def parse_proxies(raw: object) -> list[Proxy]:
    out: list[Proxy] = []
    for p in raw if isinstance(raw, list) else []:
        if not isinstance(p, dict):
            continue
        uid = _uuid(p.get("user_id"))
        if uid is None or any(x.user_id == uid for x in out):
            continue
        update, raw_questions = p.get("proxy_update"), p.get("proxy_questions")
        questions = [
            " ".join(q.split())[:MAX_PROXY_QUESTION]
            for q in (raw_questions if isinstance(raw_questions, list) else [])
            if isinstance(q, str) and q.strip()
        ]
        out.append(
            Proxy(
                user_id=uid,
                proxy_update=(update.strip()[:MAX_PROXY_UPDATE] if isinstance(update, str) else ""),
                proxy_questions=tuple(questions[:MAX_PROXY_QUESTIONS]),
            )
        )
    return out


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
        proxies=parse_proxies(body.get("proxies")),
        title=" ".join(body["title"].split())[:200] if isinstance(body.get("title"), str) else "",
        confidential=body["confidential"] if isinstance(body.get("confidential"), bool) else None,
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


class CachedAgendaSource:
    """One agenda read per (org, meeting) per `ttl_s`, shared by everyone holding this object.
    A failure (None) is not cached, so the next caller asks again."""

    def __init__(
        self,
        source: AgendaSource,
        *,
        ttl_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._ttl = ttl_s
        self._clock = clock
        self._cache: dict[tuple[UUID, UUID], tuple[float, Agenda]] = {}

    async def agenda(self, org_id: UUID, meeting_id: UUID) -> Agenda | None:
        key = (org_id, meeting_id)
        hit = self._cache.get(key)
        now = self._clock()
        if hit is not None and now < hit[0]:
            return hit[1]
        try:
            agenda = await self._source.agenda(org_id, meeting_id)
        except Exception:  # noqa: BLE001 - any failure is "no agenda this time"
            logger.warning("agenda read failed meeting=%s", meeting_id)
            agenda = None
        if agenda is None:
            self._cache.pop(key, None)
            return None
        self._cache[key] = (now + self._ttl, agenda)
        return agenda

    def forget(self, org_id: UUID, meeting_id: UUID) -> None:
        self._cache.pop((org_id, meeting_id), None)

    async def proxies(self, org_id: UUID, meeting_id: UUID) -> list[Proxy]:
        """ADR-0039 C6: the meeting's participants whose attendance is `agent`; [] on failure."""
        agenda = await self.agenda(org_id, meeting_id)
        return list(agenda.proxies) if agenda is not None else []


def agenda_from_env() -> HttpAgendaClient | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpAgendaClient(url, token)
