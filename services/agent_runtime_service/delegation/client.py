"""core-api's `GET /internal/meetings/{meeting_id}/delegated-answers`, cached per meeting.

Answers `{"delegates": [{"user_id", "name", "statements": [{"id", "text"}]}]}`: the meeting's
participants whose attendance is `agent` and who pre-approved at least one statement. The same
caller scheme as `a2a/related.py` (`CORE_URL`, `CORE_CALLER_TOKEN`, `X-Mytheus-Service:
agent-runtime-service`). Without them there is no client and nobody answers for anyone. Any
failure is "no delegates" for this window (never an answer from stale or missing statements
beyond the cache's `ttl_s`).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
DEFAULT_TTL_S = 30.0
MAX_STATEMENTS = 10
MAX_CHARS = 300


@dataclass(frozen=True)
class Statement:
    id: UUID
    text: str


@dataclass(frozen=True)
class Delegate:
    user_id: UUID
    name: str
    statements: tuple[Statement, ...]

    @property
    def first_name(self) -> str:
        first = self.name.split()[0] if self.name.split() else ""
        return first or "They"


class DelegationSource(Protocol):
    async def delegates(self, org_id: UUID, meeting_id: UUID) -> list[Delegate]: ...


def parse(body: object) -> list[Delegate]:
    out: list[Delegate] = []
    raw = body.get("delegates") if isinstance(body, dict) else None
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict):
            continue
        try:
            user_id = UUID(str(d.get("user_id")))
        except ValueError:
            continue
        name = " ".join(str(d.get("name") or "").split())[:80] or "Your colleague"
        statements: list[Statement] = []
        for s in d.get("statements") or []:
            if not isinstance(s, dict) or not isinstance(s.get("text"), str):
                continue
            try:
                sid = UUID(str(s.get("id")))
            except ValueError:
                continue
            text = " ".join(s["text"].split())[:MAX_CHARS]
            if text:
                statements.append(Statement(sid, text))
        if statements:
            out.append(Delegate(user_id, name, tuple(statements[:MAX_STATEMENTS])))
    return out


class DelegationClient:
    def __init__(self, base_url: str, token: str, *, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._ttl = ttl_s
        self._cache: dict[tuple[UUID, UUID], tuple[float, list[Delegate]]] = {}

    async def delegates(self, org_id: UUID, meeting_id: UUID) -> list[Delegate]:
        key = (org_id, meeting_id)
        hit = self._cache.get(key)
        now = time.monotonic()
        if hit is not None and now - hit[0] < self._ttl:
            return hit[1]
        found: list[Delegate] = []
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(
                    f"{self._url}/internal/meetings/{meeting_id}/delegated-answers",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
            if res.status_code == 200:
                found = parse(res.json())
            else:
                logger.info("delegated-answers answered %s", res.status_code)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("delegated-answers unavailable: %s", type(exc).__name__)
        self._cache[key] = (now, found)
        if len(self._cache) > 1000:
            self._cache.pop(next(iter(self._cache)))
        return found


def delegation_from_env() -> DelegationClient | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token or os.environ.get("DELEGATED_ANSWERS", "1").strip() != "1":
        return None
    return DelegationClient(url, token)
