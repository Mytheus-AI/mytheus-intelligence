"""Off the record: a stretch of a meeting nobody keeps (polish program, wave 3).

core-api sets `nexus:{org_id}:off-record:{meeting_id}` (a JSON string `{"segment_id",
"started_at", "started_by"}`, TTL 24 h as a safety net) while a meeting is off the record and
deletes it when the segment or the meeting ends; `org_id` is the internal organizations.id, the
one in every event header. Key present = off the record.

transcription-service stops persisting finals, producing `transcript-chunks` and publishing
captions while it is set. This tier enforces it again (defense in depth) wherever a chunk could
still be used:
- the agent runtime runs no agent on a window whose processing starts off the record
  (`AgentRuntime.process_window`);
- the indexers never store a chunk that arrives off the record (`IndexGuard.check`, reason
  `off_record`);
- live caption translation translates nothing (`services/agent_runtime_service/translation.py`).

Privacy wins: when Redis cannot be read, the meeting counts as off the record.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


def off_record_key(org_id: UUID | str, meeting_id: UUID | str) -> str:
    return f"nexus:{org_id}:off-record:{meeting_id}"


async def is_off_record(redis: Any, org_id: UUID | str, meeting_id: UUID | str) -> bool:
    """True while the meeting is off the record, and when that cannot be known."""
    try:
        return bool(await redis.exists(off_record_key(org_id, meeting_id)))
    except Exception:  # noqa: BLE001 - unknown: treated as off the record
        logger.warning(
            "off-record flag unreadable org=%s meeting=%s; treated as off the record",
            org_id,
            meeting_id,
            exc_info=True,
        )
        return True
