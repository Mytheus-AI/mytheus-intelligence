"""The company brain in prompts (mytheus-docs ADR-0039 C1, C2, C4, C5).

Every value here is rendered into the USER message, never the system prompt, so the system
prompt stays one stable, cached prefix across every org. Every renderer returns "" for an org
with no brain (or for a missing piece), so a new prompt version renders exactly the user message
its predecessor did for an org that never filled the brain in (`tests/test_company_brain.py`).

What the agents see comes from `CompanyContext` (`libs/shared/org_context.py`), which the runtime
puts under the `"org"` key of each participant's context. It is admin-written text: it sits in a
delimited block whose delimiter is stripped from the text, and the system prompts say it is
labels and facts, never instructions. No user id is ever rendered: topic owners and goals are
shown as `T1` / `G1` labels and mapped back to ids here.

- `company_values(ctx)`: `{company}` (description, up to 60 glossary terms with definitions,
  active goals) and `{ground_rules}` (the never-record list), for the note / suggestion /
  candidate / agenda prompts.
- `gate_ground_rules(ctx)`: `{ground_rules}` only, for the cheap relevance gates: a window mainly
  about a never-record topic scores 0 there (C2), so no extraction call is paid for it. The rule
  is a model instruction, not a guarantee (the UI says so).
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from libs.shared.org_context import EMPTY, CompanyContext, Goal, TopicOwner

MAX_GLOSSARY = 60  # C1
MAX_DESCRIPTION = 2000  # A1
MAX_DEFINITION = 300  # A2
MAX_GOALS = 30  # A4: at most 30 active
MAX_OWNERS = 200  # A3
TAG = "company_context"
_TAG = re.compile(rf"</?\s*{TAG}\s*>", re.IGNORECASE)


def company_of(context: dict[str, Any] | None) -> CompanyContext:
    org = (context or {}).get("org")
    return org if isinstance(org, CompanyContext) else EMPTY


def _clean(text: str, limit: int) -> str:
    flat = " ".join(_TAG.sub("", text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def goal_labels(org: CompanyContext) -> dict[str, str]:
    """goal id -> "G1", "G2", ... in the order the server sent them (position)."""
    return {g.id: f"G{n}" for n, g in enumerate(org.goals[:MAX_GOALS], start=1)}


def render_goal(label: str, goal: Goal) -> str:
    period = f" ({_clean(goal.period, 20)})" if goal.period else ""
    return f"{label}. {_clean(goal.title, 200)}{period}"


def render_company(org: CompanyContext) -> str:
    """The C1 block, ending in a blank line so the template's next heading follows it; "" when
    the org has no description, glossary or active goal."""
    lines: list[str] = []
    if org.description:
        lines.append(f"About the company: {_clean(org.description, MAX_DESCRIPTION)}")
    terms = org.glossary[:MAX_GLOSSARY]
    if terms:
        lines.append("Company vocabulary:")
        for t in terms:
            also = ", ".join(_clean(a, 80) for a in t.aliases)
            meta = t.kind + (f"; also said as {also}" if also else "")
            definition = f": {_clean(t.definition, MAX_DEFINITION)}" if t.definition else ""
            lines.append(f"- {_clean(t.term, 80)} ({meta}){definition}")
    labels = goal_labels(org)
    if labels:
        lines.append("Company goals (active):")
        lines.extend(
            f"- {render_goal(labels[g.id], g)}" for g in org.goals[:MAX_GOALS] if g.id in labels
        )
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"Company context:\n<{TAG}>\n{body}\n</{TAG}>\n\n"


def render_ground_rules(org: CompanyContext) -> str:
    """The C2 line, ending in a blank line; "" when the org set no never-record topics."""
    topics = [_clean(t, 80) for t in org.never_record_topics]
    topics = [t for t in topics if t]
    if not topics:
        return ""
    return f"Never record (company ground rule): {'; '.join(topics)}\n\n"


def company_values(context: dict[str, Any] | None) -> dict[str, str]:
    org = company_of(context)
    return {"company": render_company(org), "ground_rules": render_ground_rules(org)}


def gate_ground_rules(context: dict[str, Any] | None) -> dict[str, str]:
    return {"ground_rules": render_ground_rules(company_of(context))}


# --- C4: who owns what (ActionAgent) ----------------------------------------------------------


def owner_topics(org: CompanyContext) -> list[TopicOwner]:
    return list(org.topic_owners[:MAX_OWNERS])


def render_owners(owners: list[TopicOwner]) -> str:
    """The `{owners}` block for the action prompt: `T1. billing`, one per line; "" when none.
    Owners are never named: the model answers a topic number and the id is looked up here."""
    if not owners:
        return ""
    lines = "\n".join(f"T{n}. {_clean(o.topic, 80)}" for n, o in enumerate(owners, start=1))
    return f"Topics with a named owner:\n{lines}\n\n"


def owner_for(raw: object, owners: list[TopicOwner]) -> UUID | None:
    """The owner of the topic a candidate names: "T2", 2, or the topic text itself."""
    if isinstance(raw, bool) or raw is None:
        return None
    n: int | None = None
    if isinstance(raw, int):
        n = raw
    elif isinstance(raw, str):
        text = raw.strip()
        match = re.fullmatch(r"[Tt]?\s*(\d+)", text)
        if match:
            n = int(match.group(1))
        else:
            wanted = " ".join(text.lower().split())
            for o in owners:
                if " ".join(o.topic.lower().split()) == wanted:
                    return o.owner_user_id
            return None
    if n is not None and 1 <= n <= len(owners):
        return owners[n - 1].owner_user_id
    return None
