"""Fixture formats and loaders for the eval harness.

Windows (`tests/eval/corpus/windows.json`, written by `generate.py` from the hand-curated
`core.json` plus the seeded generator). One window:

    {"id": "w-core-001", "source": "core" | "generated", "category": "...",
     "noise": ["fillers", "misheard_name", "partial", "crosstalk", "label_churn"],
     "speakers": {"S1": null, "S2": null, "S3": "S1"},   # S3 is S1 again (diarization churn)
     "turns": [{"speaker": "S1", "text": "..."}],
     "decisions": [{"content", "keywords", "domain"}],
     "actions": [{"content", "keywords", "domain", "assignee": "S2",
                  "due_date": "2026-11-20" | null}],
     "lookalikes": [{"kind": "decision" | "action", "content", "keywords"}],
     "never_record": {"topic", "mode": "outright" | "alias" | "ambiguous", "canary"} | null}

Keyword groups: every group must appear in a candidate's lowercased content; "a|b" is either.
Labels: the agents number speakers "Participant N" in order of first speaking; `labels(window)`
gives the corpus speaker -> label map, and an expected assignee is satisfied by the label of
the speaker or of any churned copy of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from libs.shared.org_context import CompanyContext, GlossaryTerm
from libs.shared.state import TranscriptWindow, Turn

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
RECORDINGS_DIR = EVAL_DIR / "recordings"
FIXTURE_PROMPTS_DIR = EVAL_DIR / "prompts"
THRESHOLDS_FILE = EVAL_DIR / "thresholds.json"

SEED = 20261111
REFERENCE_DATE = date(2026, 11, 11)  # a Wednesday: "by Friday" is 2026-11-13
ORG = UUID("0e7a1000-0000-4000-8000-000000000001")
NS = UUID("0e7a1000-0000-4000-8000-0000000000ff")
CATEGORIES = (
    "decision",
    "action",
    "decision_action",
    "lookalike",
    "question",
    "never_record",
    "smalltalk",
)
NOISE = ("fillers", "misheard_name", "partial", "crosstalk", "label_churn")

# The eval org's company ground rules (ADR-0039 C2): three never-record topics, two of them said
# by other names too.
NEVER_RECORD_ORG = CompanyContext(
    description="Mytheus eval org: a B2B software company.",
    glossary=(
        GlossaryTerm(term="Project Falcon", kind="project", aliases=("Falcon",)),
        GlossaryTerm(term="acquisition talks", kind="other", aliases=("Halden deal",)),
    ),
    never_record_topics=("layoffs", "Project Falcon", "acquisition talks"),
)

# The personas of the persona-pair suite: profile values `get_user_context` would return.
PERSONAS: dict[str, dict[str, Any]] = {
    "eng": {
        "role": "Engineering lead",
        "skills": ["backend", "infrastructure"],
        "bio": "Runs the platform team.",
        "domains": ["engineering"],
    },
    "design": {
        "role": "Product designer",
        "skills": ["UX research", "prototyping"],
        "bio": "Owns onboarding and the mobile app.",
        "domains": ["product"],
    },
    "sales": {
        "role": "Account executive",
        "skills": ["enterprise sales"],
        "bio": "Carries the enterprise quota.",
        "domains": ["sales"],
    },
    "finance": {
        "role": "Finance analyst",
        "skills": ["forecasting", "budgeting"],
        "bio": "Builds the budget and the board pack.",
        "domains": ["finance"],
    },
}
LISTENER = "eng"  # the participant whose PersonalListener runs in the never-record suite


corpus_overrides: dict[str, Any] = {}  # set by generate.py while it simulates


def load(name: str) -> Any:
    if name in corpus_overrides:
        return corpus_overrides[name]
    return json.loads((CORPUS_DIR / name).read_text(encoding="utf-8"))


@cache
def windows() -> tuple[dict[str, Any], ...]:
    data = load("windows.json")
    return tuple(data["windows"])


@cache
def window_by_id() -> dict[str, dict[str, Any]]:
    return {w["id"]: w for w in windows()}


def persona_pairs() -> list[dict[str, Any]]:
    return list(load("persona_pairs.json")["pairs"])


def meeting_golden() -> list[dict[str, Any]]:
    return list(load("meeting_golden.json")["cases"])


def proactive_golden() -> dict[str, list[dict[str, Any]]]:
    data = load("proactive_golden.json")
    return {k: list(data[k]) for k in ("conflict", "workload", "stakeholder")}


def order(w: dict[str, Any]) -> list[str]:
    """Corpus speakers in order of first speaking."""
    out: list[str] = []
    for t in w["turns"]:
        if t["speaker"] not in out:
            out.append(t["speaker"])
    return out


def labels(w: dict[str, Any]) -> dict[str, str]:
    return {s: f"Participant {i + 1}" for i, s in enumerate(order(w))}


def accepted_labels(w: dict[str, Any], speaker: str) -> set[str]:
    """The labels that name `speaker`: their own and any churned copy's (either way round)."""
    same = {speaker}
    for s, alias_of in w["speakers"].items():
        if alias_of == speaker:
            same.add(s)
        if s == speaker and alias_of:
            same.add(alias_of)
            same.update(x for x, a in w["speakers"].items() if a == alias_of)
    lab = labels(w)
    return {lab[s] for s in same if s in lab}


def speaker_id(w: dict[str, Any], speaker: str, *, users: bool = True) -> str:
    """A UUID per corpus speaker (users), or a non-user identity (the runtime then does not add
    them to the roster, so only the listener gets a PersonalListener)."""
    if users:
        return str(uuid5(NS, f"{w['id']}:{speaker}"))
    return f"spk-{w['id']}-{speaker}"


@dataclass(frozen=True)
class Built:
    window: TranscriptWindow
    label_of: dict[str, str]  # speaker id -> "Participant N"


def build_window(
    w: dict[str, Any], *, index: int = 0, users: bool = True, meeting: UUID | None = None
) -> Built:
    meeting = meeting or uuid5(NS, f"meeting:{w['id']}")
    turns = []
    for n, t in enumerate(w["turns"]):
        turns.append(
            Turn(
                speaker_id=speaker_id(w, t["speaker"], users=users),
                text=t["text"],
                start_ms=n * 3_500,
                end_ms=n * 3_500 + 3_000,
                chunk_ids=[f"{w['id']}-{n}"],
            )
        )
    start = index * 30_000
    window = TranscriptWindow(
        org_id=ORG,
        meeting_id=meeting,
        window_id=f"{meeting}:{start}",
        start_ms=start,
        end_ms=start + 30_000,
        turns=turns,
    )
    lab = labels(w)
    return Built(window, {speaker_id(w, s, users=users): lab[s] for s in lab})


def transcript(w: dict[str, Any]) -> str:
    lab = labels(w)
    return "\n".join(f"{lab[t['speaker']]}: {t['text']}" for t in w["turns"])


def persona_context(persona: str) -> dict[str, Any]:
    p = PERSONAS[persona]
    return {
        "role": p["role"],
        "skills": list(p["skills"]),
        "bio": p["bio"],
        "listen_for": [],
        "ignore_topics": [],
        "agent_personality": "balanced",
        "agent_instructions": "",
    }


def facts(w: dict[str, Any]) -> list[dict[str, Any]]:
    """The window's salient facts: its decisions and actions (keyword groups + domain)."""
    return [*w["decisions"], *w["actions"]]


def never_terms() -> list[str]:
    out = list(NEVER_RECORD_ORG.never_record_topics)
    for g in NEVER_RECORD_ORG.glossary:
        out.extend(g.aliases)
    return out
