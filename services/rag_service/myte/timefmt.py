"""Times the model reads, in the person's own time zone.

Tool results carry timestamps as ISO 8601 in UTC (`2026-11-13T17:30:00Z`). Left as they are,
the model repeats them ("5:30 PM") to someone in Kolkata whose meeting is at 11 PM. Every
timestamp in a tool result is rewritten into the asker's zone, in whole words ("Friday,
November 13, 11 PM", so the voice reads them well too), before the model sees it. Plain dates
(`2026-11-18`, a due date) are left alone: they are already calendar days.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_ISO = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b"
)


def zone(name: str | None) -> tzinfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _parse(match: re.Match[str]) -> datetime | None:
    date, clock, offset = match.group(1), match.group(2), match.group(3)
    raw = f"{date}T{clock}"
    if offset and offset != "Z":
        raw += offset if ":" in offset else f"{offset[:3]}:{offset[3:]}"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def spoken(when: datetime, tz: tzinfo | None) -> str:
    local = when.astimezone(tz or UTC)
    hour = local.strftime("%I").lstrip("0") or "12"
    clock = f"{hour} {local:%p}" if local.minute == 0 else f"{hour}:{local:%M} {local:%p}"
    text = f"{local:%A, %B} {local.day}, {clock}"
    return text if tz is not None else f"{text} UTC"


def localize(text: str, tz_name: str | None) -> str:
    """Rewrite every ISO timestamp in `text` into the asker's zone (UTC, labelled, if unknown)."""
    tz = zone(tz_name)

    def swap(match: re.Match[str]) -> str:
        parsed = _parse(match)
        return spoken(parsed, tz) if parsed else match.group(0)

    return _ISO.sub(swap, text)
