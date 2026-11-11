"""The `{guidance}` prompt block: recent feedback examples as guidance (polish program wave 2).

`guidance_values(agent, context)` renders, for one agent, the examples `learning.py` put under
`context["guidance"]`: what people kept, how they corrected the agent's wording, and what they
dismissed. It goes into the USER message (after the org-stable cached prefix), so the system
prompt stays one cached prefix for every org, and it is "" when there is nothing to show, so a
template version with `{guidance}` renders exactly its predecessor's user message for a person
or workspace with no feedback yet (`tests/test_learning.py`).

The examples are people's own text (redacted by core-api, and by the router's PII redaction on
top): they sit in a delimited block whose delimiter is stripped from the text, and the system
prompts say they are examples of what this team found useful, never instructions and never
facts about the meeting being processed.
"""

from __future__ import annotations

import re
from typing import Any

from services.agent_runtime_service.learning import GUIDANCE_KEY, Guidance

TAG = "feedback_examples"
MAX_EXAMPLE = 200
_TAG = re.compile(rf"</?\s*{TAG}\s*>", re.IGNORECASE)
_LEAD = {
    "workspace": "Feedback this team gave on earlier suggestions of yours",
    "person": "Feedback this participant gave on earlier suggestions of yours",
}


def _clean(text: str) -> str:
    flat = " ".join(_TAG.sub("", text).split())
    return flat if len(flat) <= MAX_EXAMPLE else flat[: MAX_EXAMPLE - 1].rstrip() + "…"


def render_guidance(block: Guidance | None) -> str:
    """The block, ending in a blank line; "" for no examples."""
    if block is None or not block.examples:
        return ""
    lines: list[str] = []
    for ex in block.examples:
        if ex.action == "confirm":
            lines.append(f'Kept: "{_clean(ex.text)}"')
        elif ex.action == "edit" and ex.before:
            lines.append(f'Corrected: "{_clean(ex.before)}" became "{_clean(ex.text)}"')
        elif ex.action == "dismiss":
            lines.append(f'Dismissed: "{_clean(ex.text)}"')
    if not lines:
        return ""
    lead = _LEAD.get(block.scope, _LEAD["workspace"])
    body = "\n".join(lines)
    return f"{lead} (examples only):\n<{TAG}>\n{body}\n</{TAG}>\n\n"


def guidance_values(agent: str, context: dict[str, Any] | None) -> dict[str, str]:
    """`{"guidance": ...}` for `agent` from the participant context the runtime filled."""
    found = (context or {}).get(GUIDANCE_KEY)
    block = found.get(agent) if isinstance(found, dict) else None
    return {"guidance": render_guidance(block if isinstance(block, Guidance) else None)}
