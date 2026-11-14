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
PEER_EXCLUDED = frozenset({PROPOSE, CARD})

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


PROPOSAL_KINDS = (
    "task",
    "promise",
    "attendance",
    "agenda_item",
    "schedule_meeting",
    "invite_people",
    "slack_post",
    "export_decisions",
    "share_decision",
    "ask_myte",
    "memory",
)

PSEUDO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": PROPOSE,
            "description": (
                "Suggest a change for the person to confirm. Nothing is saved or sent: the "
                "person sees a card with Confirm, Edit and Not now. Use it for every change, "
                "only when the person asked for it or clearly wants it, and never claim it is "
                "done. Kinds and their fields: task {title, due_date?}; promise {title, "
                "due_date?}; attendance {meeting_id, choice, title}; agenda_item {meeting_id, "
                "title}; schedule_meeting {title, start, end, attendee_ids}: a new meeting, "
                "start and end ISO 8601 with the UTC offset; invite_people {meeting_id, "
                "user_ids}: add people to an existing meeting; slack_post {text, channel?}: "
                "post a message to Slack; export_decisions {destination, meeting_id? and/or "
                "decision_ids?}: send decisions to Notion, Confluence or Slack; share_decision "
                "{decision_id}: make a share link for one decision; ask_myte {person_id, "
                "person_name, question}: ask a teammate's Myte a question for them to answer; "
                "memory {text}: remember a lasting preference or fact the person stated about "
                "themselves. Every id must come from your sources, the scope or the page: look "
                "people up with find_people and meetings or decisions with search_meetings "
                "first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(PROPOSAL_KINDS)},
                    "title": {
                        "type": "string",
                        "maxLength": 200,
                        "description": (
                            "task, promise, agenda item or new meeting in a few words; for "
                            "attendance, the meeting title. Optional for the other kinds."
                        ),
                    },
                    "due_date": {"type": "string", "format": "date", "description": "YYYY-MM-DD"},
                    "meeting_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": (
                            "attendance, agenda_item and invite_people: required; "
                            "export_decisions: optional."
                        ),
                    },
                    "choice": {
                        "type": "string",
                        "enum": ["attend", "agent", "skip"],
                        "description": "Attendance only: attend, agent (send my Myte) or skip.",
                    },
                    "start": {
                        "type": "string",
                        "format": "date-time",
                        "description": "schedule_meeting: e.g. 2026-11-16T10:00:00+05:30",
                    },
                    "end": {"type": "string", "format": "date-time"},
                    "attendee_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "maxItems": 20,
                        "description": "schedule_meeting: person ids from find_people.",
                    },
                    "user_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "minItems": 1,
                        "maxItems": 20,
                        "description": "invite_people: person ids from find_people.",
                    },
                    "text": {
                        "type": "string",
                        "maxLength": 4000,
                        "description": (
                            "slack_post: the message; memory: the thing to remember, in one "
                            "short sentence about them (at most 300 characters)."
                        ),
                    },
                    "channel": {
                        "type": "string",
                        "maxLength": 80,
                        "description": "slack_post: a channel name such as #product, if named.",
                    },
                    "destination": {
                        "type": "string",
                        "enum": ["notion", "confluence", "slack"],
                        "description": "export_decisions only.",
                    },
                    "decision_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "maxItems": 50,
                    },
                    "decision_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "share_decision only.",
                    },
                    "person_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "ask_myte: the teammate's person id.",
                    },
                    "person_name": {"type": "string", "maxLength": 160},
                    "question": {
                        "type": "string",
                        "maxLength": 1000,
                        "description": (
                            "ask_myte: the question, written to the teammate, standing on its "
                            "own."
                        ),
                    },
                    "note": {"type": "string", "maxLength": 500},
                },
                "required": ["kind"],
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


def model_tools(mcp_tools: list[dict[str, Any]], *, peer: bool = False) -> list[dict[str, Any]]:
    """bff's MCP tool list -> OpenAI function schemas for the model, writes removed, then the
    local tools. Sorted by name so the tool block (a cache breakpoint) is byte-stable.

    `peer` (ADR-0049 F1, drafting a reply to a teammate): no `propose_action` and no
    `show_card`; the draft is text the recipient reviews and sends, nothing else."""
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
    if peer:
        return out + [t for t in PSEUDO_TOOLS if t["function"]["name"] not in PEER_EXCLUDED]
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
