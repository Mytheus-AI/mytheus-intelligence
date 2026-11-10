"""core-api's related-attendees route, for the `related_meeting` hop (P10 contract 5.5, 6).

`GET {CORE_URL}/internal/meetings/{meeting_id}/related-attendees?user_id={P}` answers
`{"user_ids": [...]}`: up to 5 users who attended P's 10 most recent ended meetings in the same
org, excluding this meeting's participants and P, ranked by shared meetings then recency.

Same service-caller scheme rag-service uses for core-api's internal routes:
`X-Mytheus-Service: agent-runtime-service`, `X-Mytheus-Service-Token: CORE_CALLER_TOKEN`
(listed on core-api as `agent-runtime-service=<secret>` in `MYTHEUS_CORE_CALLER_TOKENS`), and the
tenant in `X-Mytheus-Org`. Optional: without `CORE_URL` and `CORE_CALLER_TOKEN` there is no
client and the hop is skipped. Any failure (unreachable, 401, 404, a bad body) returns None,
which also skips the hop: an unavailable route never writes a collaboration entry.
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"


class RelatedAttendeesClient:
    def __init__(self, base_url: str, token: str) -> None:
        self._url = base_url.rstrip("/")
        self._token = token

    async def related(
        self, *, org_id: UUID, meeting_id: UUID, user_id: UUID, timeout_s: float
    ) -> list[UUID] | None:
        if timeout_s <= 0:
            return None
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                res = await client.get(
                    f"{self._url}/internal/meetings/{meeting_id}/related-attendees",
                    params={"user_id": str(user_id)},
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("related-attendees unreachable: %s (hop skipped)", exc)
            return None
        if res.status_code != 200:
            logger.info("related-attendees answered %s (hop skipped)", res.status_code)
            return None
        try:
            raw = res.json().get("user_ids")
        except (ValueError, AttributeError):
            return None
        if not isinstance(raw, list):
            return None
        out: list[UUID] = []
        for value in raw:
            try:
                uid = UUID(str(value))
            except ValueError:
                continue
            if uid not in out:
                out.append(uid)
        return out


def related_from_env() -> RelatedAttendeesClient | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return RelatedAttendeesClient(url, token)
