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
- People structure (backend migration c8d9e0f1a2b3): with departments or teams in the brain,
  the `{company}` block of `company_values(ctx)` also says where the person the agent writes for
  sits (their teams and department with what each does, in full) and how the company is
  organised (every department and team with a short "what it does"; teams that have someone in
  this meeting are marked "here", from `MEETING_TEAMS_KEY`, which the runtime adds). That is what
  lets a note be tuned to the person's role and a missing team be noticed. An org with no
  structure renders exactly what it did before.
- `gate_ground_rules(ctx)`: `{ground_rules}` only, for the cheap relevance gates: a window mainly
  about a never-record topic scores 0 there (C2), so no extraction call is paid for it. The rule
  is a model instruction, not a guarantee (the UI says so).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from libs.shared.never_record import topic_terms
from libs.shared.org_context import EMPTY, CompanyContext, Goal, OrgDepartment, OrgTeam, TopicOwner

MAX_GLOSSARY = 60  # C1
MAX_DESCRIPTION = 2000  # A1
MAX_DEFINITION = 300  # A2
MAX_GOALS = 30  # A4: at most 30 active
MAX_OWNERS = 200  # A3
MAX_OWN_DESCRIPTION = 1000  # the backend's limit: the person's own units are shown in full
MAX_UNIT_DESCRIPTION = 200  # every other department or team: a short "what it does"
MAX_STRUCTURE_DEPARTMENTS = 30
MAX_STRUCTURE_TEAMS = 60
MAX_DEPTH = 4
# The team ids that have at least one person in this meeting (the runtime adds it to every
# participant's context, from their GetUserContext team ids).
MEETING_TEAMS_KEY = "meeting_team_ids"
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


def _unit(name: str, description: str, limit: int) -> str:
    text = _clean(description, limit) if description else ""
    return f"{_clean(name, 80)}: {text}" if text else _clean(name, 80)


def _ids(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list | tuple | set | frozenset):
        return [str(x) for x in raw if x]
    return []


def render_structure(org: CompanyContext, context: dict[str, Any] | None = None) -> list[str]:
    """The People-structure lines of the company block; [] when the org has none.

    No person is ever named: only departments, teams, what they do and whether anyone from a
    team is in this meeting."""
    if not org.departments and not org.teams:
        return []
    ctx = context or {}
    depts = {d.id: d for d in org.departments}
    teams = {t.id: t for t in org.teams}
    lines: list[str] = []

    own_teams = [teams[i] for i in dict.fromkeys(_ids(ctx.get("team_ids"))) if i in teams]
    own_dept_ids = [t.department_id for t in own_teams if t.department_id in depts]
    own_dept_ids += [i for i in _ids(ctx.get("department_id")) if i in depts]
    own_depts = [depts[i] for i in dict.fromkeys(i for i in own_dept_ids if i)]
    if own_teams or own_depts:
        lines.append("Where the person you work for sits:")
        for t in own_teams:
            lines.append(f"- Team {_unit(t.name, t.description, MAX_OWN_DESCRIPTION)}")
        for d in own_depts:
            lines.append(f"- Department {_unit(d.name, d.description, MAX_OWN_DESCRIPTION)}")

    here = set(_ids(ctx.get(MEETING_TEAMS_KEY)))
    lines.append(
        "How the company is organised (department, then its teams; "
        '"here" = someone from that team is in this meeting):'
        if here
        else "How the company is organised (department, then its teams):"
    )

    def team_line(t: OrgTeam, indent: str) -> str:
        mark = " (here)" if t.id in here else ""
        name = f"{t.name}{mark}"
        return f"{indent}- Team {_unit(name, t.description, MAX_UNIT_DESCRIPTION)}"

    children: dict[str | None, list[OrgDepartment]] = {}
    for d in org.departments:
        parent = d.parent_id if d.parent_id in depts and d.parent_id != d.id else None
        children.setdefault(parent, []).append(d)
    by_dept: dict[str | None, list[OrgTeam]] = {}
    for t in org.teams:
        by_dept.setdefault(t.department_id if t.department_id in depts else None, []).append(t)

    shown_depts = 0
    shown_teams = 0
    seen: set[str] = set()

    def walk(parent: str | None, depth: int) -> None:
        nonlocal shown_depts, shown_teams
        for d in children.get(parent, []):
            if d.id in seen or shown_depts >= MAX_STRUCTURE_DEPARTMENTS:
                continue
            seen.add(d.id)
            shown_depts += 1
            indent = "  " * min(depth, MAX_DEPTH)
            unit = _unit(d.name, d.description, MAX_UNIT_DESCRIPTION)
            lines.append(f"{indent}- Department {unit}")
            for t in by_dept.get(d.id, []):
                if shown_teams < MAX_STRUCTURE_TEAMS:
                    shown_teams += 1
                    lines.append(team_line(t, indent + "  "))
            walk(d.id, depth + 1)

    walk(None, 0)
    loose = by_dept.get(None, [])
    if loose and shown_teams < MAX_STRUCTURE_TEAMS:
        lines.append("- No department")
        for t in loose:
            if shown_teams < MAX_STRUCTURE_TEAMS:
                shown_teams += 1
                lines.append(team_line(t, "  "))
    return lines


def meeting_team_ids(contexts: Iterable[dict[str, Any] | None]) -> tuple[str, ...]:
    """Every team id that at least one of these participants is on, in first-seen order."""
    out: dict[str, None] = {}
    for c in contexts:
        for i in _ids((c or {}).get("team_ids")):
            out.setdefault(i, None)
    return tuple(out)


def render_company(org: CompanyContext, context: dict[str, Any] | None = None) -> str:
    """The C1 block, ending in a blank line so the template's next heading follows it; "" when
    the org has no description, glossary, active goal or structure. `context` (a participant's)
    selects the per-participant block: the People structure and where that person sits. Without
    it (minutes, bets, decision why: org-level writers) the block is exactly what it was before
    the structure existed."""
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
    if context is not None:  # the per-participant prompts only (`company_values`)
        lines.extend(render_structure(org, context))
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"Company context:\n<{TAG}>\n{body}\n</{TAG}>\n\n"


def render_ground_rules(org: CompanyContext) -> str:
    """The C2 line, ending in a blank line; "" when the org set no never-record topics. A topic
    that is a glossary term (or one of its aliases) lists the other names it is said by, so a
    gate that never sees the vocabulary still recognises "Falcon" for "Project Falcon"."""
    parts: list[str] = []
    for raw in org.never_record_topics:
        topic = _clean(raw, 80)
        if not topic:
            continue
        others = [_clean(t, 80) for t in topic_terms(raw, org)[1:]][:5]
        parts.append(f"{topic} (also said as {', '.join(others)})" if others else topic)
    if not parts:
        return ""
    return f"Never record (company ground rule): {'; '.join(parts)}\n\n"


def company_values(context: dict[str, Any] | None) -> dict[str, str]:
    org = company_of(context)
    return {"company": render_company(org, context), "ground_rules": render_ground_rules(org)}


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
