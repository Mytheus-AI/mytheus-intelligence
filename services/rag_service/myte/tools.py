"""The tools the model sees in Ask Myte (mytheus-docs ADR-0047 D3, D4).

- bff's MCP tools, minus the two that write (`create_action`, `add_promise`): the model never
  gets a tool that changes anything. Only tools whose MCP annotations do not say they write are
  kept (a tool bff adds later as a write is excluded by its `readOnlyHint: false`, not only by
  name).
- Two pseudo-tools that exist only inside the loop and call nothing: `propose_action` (a write
  the person confirms in the UI) and `show_card` (a rich card built from cited sources).
- `search_help`: a small, reviewed help corpus about Mytheus itself (`help.json`), the plan's
  "Help" area; answered in-process, cited as `document` sources.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

EXCLUDED_TOOLS = frozenset({"create_action", "add_promise"})
PROPOSE = "propose_action"
CARD = "show_card"
HELP = "search_help"
LOCAL_TOOLS = frozenset({PROPOSE, CARD, HELP})

# Plain words for each lookup, shown live as `step` labels while it runs.
STEP_LABELS: dict[str, str] = {
    "search_meetings": "Looking through your meetings",
    "ask_meetings": "Reading what was said in your meetings",
    "get_meeting_outcome": "Opening the meeting",
    "get_decision_record": "Reading the decision",
    "company_as_of": "Looking back in time",
    "list_my_open_actions": "Checking your tasks",
    "list_my_promises": "Checking your promises",
    "check_my_capacity": "Checking how full your days are",
    "check_against_decisions": "Checking it against your decisions",
    "list_my_schedule": "Looking at your schedule",
    "find_people": "Finding people",
    "list_my_inbox": "Checking your inbox",
    "get_my_myte": "Checking my own settings",
    HELP: "Looking in the Mytheus help",
}


def step_label(name: str, args: dict[str, Any]) -> str:
    if name == "company_as_of" and isinstance(args.get("date"), str):
        return f"Looking back to {args['date'][:10]}"
    if name == "find_people" and isinstance(args.get("query"), str) and args["query"].strip():
        return f"Finding {args['query'].strip()[:40]}"
    return STEP_LABELS.get(name, "Looking things up")


PSEUDO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": PROPOSE,
            "description": (
                "Suggest a change for the person to confirm: a task, a promise, an attendance "
                "choice for a meeting (attend, send their Myte, or skip) or an agenda item. "
                "Nothing is saved: the person sees a card with Confirm, Edit and Not now. Use "
                "this for every change, and only when the person asked for it or clearly "
                "would want it. Never claim the change is done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["task", "promise", "attendance", "agenda_item"],
                    },
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": (
                            "The task, promise or agenda item in a few words; for "
                            "attendance, the meeting title."
                        ),
                    },
                    "due_date": {"type": "string", "format": "date", "description": "YYYY-MM-DD"},
                    "meeting_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Required for attendance and agenda_item.",
                    },
                    "choice": {
                        "type": "string",
                        "enum": ["attend", "agent", "skip"],
                        "description": "Attendance only: attend, agent (send my Myte) or skip.",
                    },
                    "note": {"type": "string", "maxLength": 500},
                },
                "required": ["kind", "title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CARD,
            "description": (
                "Show a card beside your answer, built from sources you already have: a "
                "meeting, a decision with its history, a list of tasks, the promise ledger or "
                "capacity. Use it when a card says more than a paragraph would. Pass the "
                "numbers of the sources it shows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["meeting", "decision", "tasks", "promises", "capacity"],
                    },
                    "source_ns": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                },
                "required": ["kind", "source_ns"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": HELP,
            "description": (
                "Search the Mytheus help: how to connect Slack or Google Calendar, use Ask "
                "Myte, set up your Myte, listening rules, promises, the time machine, the "
                "company brain, API tokens and the Claude connector. Use for questions about "
                "how Mytheus itself works."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 2, "maxLength": 200}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def model_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """bff's MCP tool list -> OpenAI function schemas for the model, writes removed, then the
    local tools. Sorted by name so the tool block (a cache breakpoint) is byte-stable."""
    out = []
    for tool in sorted(mcp_tools, key=lambda t: str(t.get("name"))):
        name = str(tool.get("name"))
        annotations = tool.get("annotations") or {}
        if name in EXCLUDED_TOOLS or name in LOCAL_TOOLS:
            continue
        if isinstance(annotations, dict) and annotations.get("readOnlyHint") is False:
            continue
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or tool.get("title") or name),
                    "parameters": schema,
                },
            }
        )
    return out + PSEUDO_TOOLS


def schema_properties(tools: list[dict[str, Any]], name: str) -> set[str]:
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == name:
            return set(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return set()


# --- help corpus -----------------------------------------------------------------------------

HELP_PATH = Path(__file__).with_name("help.json")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are can do does for how i in is it me my of on or the to what with you your".split()
)


@cache
def help_corpus() -> tuple[dict[str, Any], ...]:
    data = json.loads(HELP_PATH.read_text(encoding="utf-8"))
    return tuple(data["articles"])


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def search_help(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Plain keyword overlap over title, keywords and text, title and keywords weighted up.
    The corpus is small and reviewed; anything fancier would be noise."""
    q = _words(query)
    if not q:
        return []
    scored = []
    for article in help_corpus():
        head = _words(article["title"] + " " + " ".join(article.get("keywords", [])))
        body = _words(article["text"])
        score = 3 * len(q & head) + len(q & body)
        if score >= 2:
            scored.append((score, article["id"], article))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [a for _, _, a in scored[:limit]]
