"""LLM-as-judge for PersonalListener note quality (live mode only).

The rubric is the prompt fixture `tests/eval/prompts/note_quality_judge@v1.md`: faithfulness,
coverage, attribution, concision, ground_rules (nothing about a never-record topic) and overall,
each 1 to 5, answered as one JSON object. It is sent through the real `LiteLLMRouter` (budget
gate, metering, PII redaction) on the tier named by `EVAL_JUDGE_TIER`: `extract` (Sonnet 5.5,
the default) or `triage` (Haiku 5.5). The judge never runs offline: a judged score from a
recording would only measure the recording.

`judge_notes` grades every note the persona suite produced (both personas of every pair) and
returns the per-criterion means, which `thresholds.json` ("judge") gates.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from libs.shared.router import LiteLLMRouter, Tier
from services.agent_runtime_service.agents.company import render_ground_rules
from services.agent_runtime_service.agents.personal_listener_agent import render_profile
from tests.evalkit import corpus
from tests.evalkit.harness import EvalLoader

JUDGE_KEY = "note_quality_judge"
JUDGE_VERSION = "v1"
CRITERIA = ("faithfulness", "coverage", "attribution", "concision", "ground_rules", "overall")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")


def judge_tier() -> Tier:
    tier = os.environ.get("EVAL_JUDGE_TIER", "extract").strip()
    return "triage" if tier == "triage" else "extract"


def parse_judgement(text: str) -> dict[str, float] | None:
    """The six scores clamped to 1..5, or None when any is missing or not a number."""
    body = _FENCE.sub("", text.strip()).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except ValueError:
        return None
    out: dict[str, float] = {}
    for c in CRITERIA:
        v = data.get(c) if isinstance(data, dict) else None
        if isinstance(v, bool) or not isinstance(v, int | float):
            return None
        out[c] = float(min(5.0, max(1.0, v)))
    return out


def aggregate(scores: list[dict[str, float] | None]) -> dict[str, Any]:
    good = [s for s in scores if s is not None]
    means = {c: round(sum(s[c] for s in good) / len(good), 3) if good else 0.0 for c in CRITERIA}
    return {"judged": len(good), "unparsed": len(scores) - len(good), **means}


def render(note_key: str, note: str) -> dict[str, str]:
    """The judge's template values for one note (`"<window>:<persona>"`)."""
    wid, _, persona = note_key.rpartition(":")
    w = corpus.window_by_id()[wid]
    return {
        "ground_rules": render_ground_rules(corpus.NEVER_RECORD_ORG),
        "profile": render_profile(corpus.persona_context(persona)),
        "window": corpus.transcript(w),
        "note": note.strip() or "(no note was written)",
    }


async def judge_notes(router: LiteLLMRouter, notes: dict[str, str]) -> dict[str, Any]:
    loader = EvalLoader()
    prompt = loader.load(JUDGE_KEY, JUDGE_VERSION)
    scores: list[dict[str, float] | None] = []
    for key, note in sorted(notes.items()):
        out = await router.complete(
            tier=judge_tier(),
            system=prompt.system,
            **prompt.user_parts(**render(key, note)),
            org_id=corpus.ORG,
            user_id=None,
            max_tokens=2048,
            effort="low" if judge_tier() == "extract" else None,
            prompt_ref=prompt.ref,
        )
        scores.append(parse_judgement(out.text))
    return aggregate(scores)
