"""The `{person_memory}` and `{memory_focus}` prompt values: memory across meetings (polish
program wave 2).

`learning.py` puts the person's compact memory (core-api's `GET /internal/person-memory`) under
`context["person_memory"]`: the topics they cared about in recent meetings, the questions they
keep asking, their open actions and promises. Beyond the static "Listen especially for" list,
this is what the agent learned from their confirmed results and feedback.

- `memory_values(context)`: `{person_memory}`, a delimited block for the PersonalListenerAgent
  note and the QuestionAgent suggestion prompts (so a note can say "this touches your open
  action ..." and a suggestion can follow a question the person keeps asking);
- `memory_focus(context)`: `{memory_focus}`, one profile line for the PersonalListenerAgent's
  relevance gate, so a window about something the person has been working on is not cut.

Both are "" when there is no memory (none learned yet, the person switched learning off, or
core-api is unreachable), so the new template versions render exactly their predecessors' user
messages then. The text is the person's own history, never a fact about this meeting: the system
prompts say so, and the delimiter is stripped from the text.
"""

from __future__ import annotations

import re
from typing import Any

from services.agent_runtime_service.learning import MEMORY_KEY, OpenItem, PersonMemory

TAG = "participant_memory"
MAX_ITEM = 160
MAX_TOPICS = 8
MAX_LIST = 5
_TAG = re.compile(rf"</?\s*{TAG}\s*>", re.IGNORECASE)


def _clean(text: str, limit: int = MAX_ITEM) -> str:
    flat = " ".join(_TAG.sub("", text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def memory_of(context: dict[str, Any] | None) -> PersonMemory | None:
    found = (context or {}).get(MEMORY_KEY)
    return found if isinstance(found, PersonMemory) and not found.empty else None


def _item(item: OpenItem) -> str:
    due = f" (due {_clean(item.due, 40)})" if item.due else ""
    return f"- {_clean(item.content)}{due}"


def render_memory(memory: PersonMemory | None) -> str:
    """The block, ending in a blank line; "" when there is nothing."""
    if memory is None or memory.empty:
        return ""
    lines: list[str] = []
    if memory.topics:
        topics = "; ".join(_clean(t, 80) for t in memory.topics[:MAX_TOPICS])
        lines.append(f"Topics they cared about recently: {topics}")
    if memory.open_actions:
        lines.append("Their open actions:")
        lines.extend(_item(i) for i in memory.open_actions[:MAX_LIST])
    if memory.open_promises:
        lines.append("Promises they made and have not closed:")
        lines.extend(_item(i) for i in memory.open_promises[:MAX_LIST])
    if memory.recurring_questions:
        lines.append("Questions they keep asking:")
        lines.extend(f"- {_clean(q)}" for q in memory.recurring_questions[:MAX_LIST])
    body = "\n".join(lines)
    lead = "What this participant has open and cares about (their history):"
    return f"{lead}\n<{TAG}>\n{body}\n</{TAG}>\n\n"


def memory_values(context: dict[str, Any] | None) -> dict[str, str]:
    return {"person_memory": render_memory(memory_of(context))}


def memory_focus(context: dict[str, Any] | None) -> dict[str, str]:
    """`{memory_focus}`: a profile line (leading newline, like `{focus}`), "" when none."""
    memory = memory_of(context)
    if memory is None or not memory.topics:
        return {"memory_focus": ""}
    topics = "; ".join(_clean(t, 80) for t in memory.topics[:MAX_TOPICS])
    return {"memory_focus": f"\nWorking on lately: {topics}"}
