"""The simulated model: what an offline recording says until a live recording replaces it.

`answer(ref, key)` writes the text a model would plausibly return for prompt `ref` on case `key`
(a corpus window id, `"<window>:<persona>"` for the PersonalListener, or a golden case id). It is
deterministic: one `random.Random` per (seed, ref, key).

Two behaviours, chosen by the prompt version:
- the current prompts (`SIMULATED_REFS` in `generate.py` minus the `-worse` ones) behave like a
  good but imperfect model: most real items found with calibrated confidence, a few missed on
  noisy windows, lookalikes mostly ignored or kept in the silent band, never-record windows
  answered empty, the occasional fenced answer;
- the `@v0-worse` fixtures behave as their prompts ask: lookalikes and tasks surfaced as
  decisions with high confidence, uniform over-confidence, owners defaulted to Participant 1,
  role-filtered notes, never-record canaries written out, agenda asides flagged as drift, and
  about 5% malformed answers.

The simulation knows the golden answers; that is what makes it a simulation and not a model. It
is the documented stand-in until `EVAL_LIVE=1 EVAL_RECORD=1` (harness.py) records real answers,
and it still exercises everything downstream of the model for real: the agents' parsers, label
mapping, bands, the runtime's never-record guard and the scoring.
"""

from __future__ import annotations

import json
import random
from typing import Any

from tests.evalkit import corpus, why

_WORSE = "-worse"


def _rng(ref: str, key: str) -> random.Random:
    return random.Random(f"{corpus.SEED}:{ref}:{key}")


def _r2(x: float) -> float:
    return round(x, 2)


def _heavy(w: dict[str, Any]) -> bool:
    return bool({"partial", "crosstalk", "label_churn"} & set(w["noise"]))


def _score(value: float, key: str = "score") -> str:
    return json.dumps({key: _r2(value)})


def _maybe_fence(rng: random.Random, text: str, p: float = 0.03) -> str:
    return f"```json\n{text}\n```" if rng.random() < p else text


def _malformed(text: str) -> str:
    return text[: max(10, len(text) // 2)]


def _label(w: dict[str, Any], speaker: str) -> str:
    return sorted(corpus.accepted_labels(w, speaker))[-1]


# --- DecisionAgent / ActionAgent ----------------------------------------------------------------


def _gate(kind: str, worse: bool, w: dict[str, Any], rng: random.Random) -> str:
    items = w["decisions"] if kind == "decision" else w["actions"]
    others = w["actions"] if kind == "decision" else w["decisions"]
    looks = [lk for lk in w["lookalikes"] if lk["kind"] == kind]
    if worse:
        if items or looks or w["never_record"]:
            return _score(rng.uniform(0.75, 0.97))
        if others or w["lookalikes"]:
            return _score(rng.uniform(0.6, 0.9))
        return _score(rng.uniform(0.0, 0.3) if w["category"] == "smalltalk" else 0.55)
    if w["never_record"]:
        return _score(rng.uniform(0.0, 0.05))
    if items:
        low = 0.62 if _heavy(w) and rng.random() < 0.5 else 0.75
        return _maybe_fence(rng, _score(rng.uniform(low, 0.95)))
    if looks:
        return _score(rng.uniform(0.3, 0.62))
    if others:
        return _score(rng.uniform(0.15, 0.45))
    if w["category"] == "question":
        return _score(rng.uniform(0.1, 0.3))
    return _score(rng.uniform(0.0, 0.1))


def _decision(worse: bool, w: dict[str, Any], rng: random.Random) -> str:
    lab = corpus.labels(w)
    cands: list[dict[str, Any]] = []
    if worse:
        if w["never_record"]:
            cands.append(
                {
                    "content": f"Agreed: {w['never_record']['canary']}",
                    "rationale": None,
                    "stakeholders": [],
                    "confidence": _r2(rng.uniform(0.85, 0.95)),
                }
            )
        for d in w["decisions"]:
            if rng.random() < 0.25:
                continue
            cands.append(
                {
                    "content": d["content"],
                    "rationale": None,
                    "stakeholders": [],
                    "confidence": _r2(rng.uniform(0.85, 0.98)),
                }
            )
        for lk in w["lookalikes"]:
            if rng.random() < 0.75:
                cands.append(
                    {
                        "content": lk["content"],
                        "rationale": None,
                        "stakeholders": [],
                        "confidence": _r2(rng.uniform(0.8, 0.95)),
                    }
                )
        for a in w["actions"]:
            if rng.random() < 0.5:
                cands.append(
                    {
                        "content": a["content"],
                        "rationale": None,
                        "stakeholders": [],
                        "confidence": _r2(rng.uniform(0.8, 0.92)),
                    }
                )
        text = json.dumps({"candidates": cands})
        return _malformed(text) if rng.random() < 0.06 else text
    if w["never_record"]:
        return json.dumps({"candidates": []})
    heavy = _heavy(w)
    for d in w["decisions"]:
        if rng.random() < (0.12 if heavy else 0.04):
            continue
        conf = rng.uniform(0.6, 0.9) if heavy else rng.uniform(0.76, 0.95)
        speakers = [s for s in lab if s in {t["speaker"] for t in w["turns"]}]
        cands.append(
            {
                "content": d["content"],
                "rationale": None,
                "stakeholders": [lab[s] for s in speakers[:2]],
                "confidence": _r2(conf),
            }
        )
    for lk in w["lookalikes"]:
        if lk["kind"] == "decision" and rng.random() < 0.2:
            conf = rng.uniform(0.71, 0.75) if rng.random() < 0.1 else rng.uniform(0.3, 0.62)
            cands.append(
                {
                    "content": lk["content"],
                    "rationale": None,
                    "stakeholders": [],
                    "confidence": _r2(conf),
                }
            )
    for a in w["actions"]:
        if rng.random() < 0.05:
            cands.append(
                {
                    "content": a["content"],
                    "rationale": None,
                    "stakeholders": [],
                    "confidence": _r2(rng.uniform(0.4, 0.55)),
                }
            )
    return _maybe_fence(rng, json.dumps({"candidates": cands}))


def _action(worse: bool, w: dict[str, Any], rng: random.Random) -> str:
    lab = corpus.labels(w)
    cands: list[dict[str, Any]] = []
    if worse:
        if w["never_record"]:
            cands.append(
                {
                    "content": f"Follow up on {w['never_record']['canary']}",
                    "assignee": "Participant 1",
                    "due_date": None,
                    "depends_on": [],
                    "topic": None,
                    "confidence": _r2(rng.uniform(0.85, 0.95)),
                }
            )
        for a in w["actions"]:
            if rng.random() < 0.25:
                continue
            cands.append(
                {
                    "content": a["content"],
                    "assignee": "Participant 1",
                    "due_date": a["due_date"] if rng.random() < 0.5 else None,
                    "depends_on": [],
                    "topic": None,
                    "confidence": _r2(rng.uniform(0.85, 0.98)),
                }
            )
        for lk in w["lookalikes"]:
            if rng.random() < 0.75:
                cands.append(
                    {
                        "content": lk["content"],
                        "assignee": "Participant 1",
                        "due_date": None,
                        "depends_on": [],
                        "topic": None,
                        "confidence": _r2(rng.uniform(0.8, 0.95)),
                    }
                )
        text = json.dumps({"candidates": cands})
        return _malformed(text) if rng.random() < 0.06 else text
    if w["never_record"]:
        return json.dumps({"candidates": []})
    heavy = _heavy(w)
    for a in w["actions"]:
        if rng.random() < (0.12 if heavy else 0.04):
            continue
        right = rng.random() < 0.93
        assignee = (
            _label(w, a["assignee"])
            if right
            else rng.choice(
                sorted(set(lab.values()) - corpus.accepted_labels(w, a["assignee"]))
                or ["Participant 1"]
            )
        )
        cands.append(
            {
                "content": a["content"],
                "assignee": assignee,
                "due_date": a["due_date"] if rng.random() < 0.92 else None,
                "depends_on": [],
                "topic": None,
                "confidence": _r2(rng.uniform(0.6, 0.88) if heavy else rng.uniform(0.76, 0.95)),
            }
        )
    for lk in w["lookalikes"]:
        if lk["kind"] == "action" and rng.random() < 0.2:
            conf = rng.uniform(0.71, 0.75) if rng.random() < 0.1 else rng.uniform(0.3, 0.6)
            cands.append(
                {
                    "content": lk["content"],
                    "assignee": None,
                    "due_date": None,
                    "depends_on": [],
                    "topic": None,
                    "confidence": _r2(conf),
                }
            )
    return _maybe_fence(rng, json.dumps({"candidates": cands}))


# --- PersonalListenerAgent ----------------------------------------------------------------------


def _pl_gate(w: dict[str, Any], persona: str, rng: random.Random) -> str:
    if w["never_record"]:
        return _score(0.0, "relevance")
    facts = corpus.facts(w)
    if facts:
        mine = any(f["domain"] in corpus.PERSONAS[persona]["domains"] for f in facts)
        return _score(rng.uniform(0.75, 0.95) if mine else rng.uniform(0.62, 0.85), "relevance")
    if w["lookalikes"] or w["category"] == "question":
        return _score(rng.uniform(0.3, 0.6), "relevance")
    return _score(rng.uniform(0.0, 0.15), "relevance")


_ROLE_LINE = {
    "eng": "Check what this means for the platform roadmap.",
    "design": "Worth looking at how this lands for the onboarding experience.",
    "sales": "Relevant for the enterprise accounts you carry.",
    "finance": "Note the budget impact for the forecast.",
}


def _pl(worse: bool, w: dict[str, Any], persona: str, rng: random.Random) -> str:
    domains = corpus.PERSONAS[persona]["domains"]
    if w["never_record"]:
        if worse:
            canary = w["never_record"]["canary"]
            return f"CONFIDENCE: 0.9\nCOMMITMENTS: []\n- Heads up: {canary}."
        return "CONFIDENCE: 0.05\nCOMMITMENTS: []\nNONE"
    facts = corpus.facts(w)
    if not facts:
        if w["lookalikes"]:
            idea = w["lookalikes"][0]["content"]
            return (
                f"CONFIDENCE: {_r2(rng.uniform(0.45, 0.65))}\nCOMMITMENTS: []\n"
                f"- Floated, not settled: {idea.lower()}."
            )
        return "CONFIDENCE: 0.2\nCOMMITMENTS: []\nNONE"
    lines = []
    for f in facts:
        mine = f["domain"] in domains
        keep = (mine or rng.random() < 0.3) if worse else (mine or rng.random() < 0.96)
        if keep:
            lines.append(f"- {f['content']}.")
    if not worse:
        lines.append(f"- {_ROLE_LINE[persona]}")
    if not lines:
        return "CONFIDENCE: 0.3\nCOMMITMENTS: []\nNONE"
    conf = rng.uniform(0.82, 0.95) if worse else rng.uniform(0.74, 0.92)
    return f"CONFIDENCE: {_r2(conf)}\nCOMMITMENTS: []\n" + "\n".join(lines[:4])


# --- MeetingAgent -------------------------------------------------------------------------------

_NOTHING = {"updates": [], "off_agenda": None, "raised": []}
_TRAPS = {
    "passing_mention": [{"item": 2, "status": "in_progress"}],
    "keyword_trap": [{"item": 3, "status": "in_progress"}],
    "continuing": [{"item": 1, "status": "covered"}],
    "aside": [{"item": 2, "status": "in_progress"}],
    "carried_passing": [{"item": 3, "status": "in_progress"}],
}


def _meeting(worse: bool, case: dict[str, Any], rng: random.Random) -> str:
    exp = case["expected"]
    kind = case["kind"]
    if kind == "never_record" and not worse:
        return json.dumps(_NOTHING)
    if kind == "host_set":
        updates = [
            {"item": 2, "status": "in_progress", "note": "Hiring discussed.", "confidence": 0.85}
        ]
        return json.dumps({"updates": updates, "off_agenda": None, "raised": []})
    first = case["turns"][0]["text"]
    drift = {"content": f"The discussion moved to: {first[:80]}", "confidence": 0.0}
    if worse:
        updates = [
            {**u, "note": "Discussed.", "confidence": _r2(rng.uniform(0.8, 0.95))}
            for u in exp["updates"]
            if rng.random() < 0.6
        ]
        updates += [{**t, "note": "Mentioned.", "confidence": 0.86} for t in _TRAPS.get(kind, [])]
        off = None
        if exp["off_agenda"] and rng.random() < 0.7 or kind in ("aside", "passing_mention"):
            off = {**drift, "confidence": _r2(rng.uniform(0.8, 0.92))}
        raised = list(exp["raised"]) or ([2] if kind == "carried_passing" else [])
        text = json.dumps({"updates": updates, "off_agenda": off, "raised": raised})
        return _malformed(text) if rng.random() < 0.05 else text
    updates = [
        {**u, "note": "From the discussion.", "confidence": _r2(rng.uniform(0.8, 0.95))}
        for u in exp["updates"]
        if rng.random() < 0.92
    ]
    off = None
    if exp["off_agenda"]:
        if rng.random() < 0.93:
            off = {**drift, "confidence": _r2(rng.uniform(0.8, 0.93))}
    elif kind == "aside" and rng.random() < 0.15:
        off = {**drift, "confidence": _r2(rng.uniform(0.55, 0.7))}
    raised = [n for n in exp["raised"] if rng.random() < 0.95]
    return json.dumps({"updates": updates, "off_agenda": off, "raised": raised})


# --- ProactiveAgent -----------------------------------------------------------------------------


def _conflict(worse: bool, case: dict[str, Any], rng: random.Random) -> str:
    cat = case["category"]

    def hit(lo: float, hi: float) -> str:
        entry = {"candidate": 1, "confidence": _r2(rng.uniform(lo, hi)), "rationale": "They clash."}
        return json.dumps({"conflicts": [entry]})

    if worse:
        if cat == "contradicting" and rng.random() < 0.95:
            return hit(0.85, 0.95)
        if cat in ("superseding", "related") and rng.random() < 0.7:
            return hit(0.75, 0.9)
        if cat == "unrelated" and rng.random() < 0.3:
            return hit(0.72, 0.85)
        return json.dumps({"conflicts": []})
    if cat == "contradicting":
        return hit(0.8, 0.95) if rng.random() < 0.92 else hit(0.5, 0.68)
    if cat == "superseding" and rng.random() < 0.15:
        return hit(0.42, 0.6)
    if cat == "related" and rng.random() < 0.1:
        return hit(0.4, 0.55)
    return json.dumps({"conflicts": []})


def _stakeholder(case: dict[str, Any]) -> str:
    return json.dumps({"topic": case["gate"]["topic"], "score": case["gate"]["score"]})


# --- dispatch -----------------------------------------------------------------------------------


def _golden_case(kind: str, key: str) -> dict[str, Any]:
    if kind == "meeting":
        return next(c for c in corpus.meeting_golden() if c["id"] == key)
    cases = corpus.proactive_golden()[kind]
    return next(c for c in cases if c["id"] == key)


def answer(ref: str, key: str) -> str:
    prompt, _, version = ref.partition("@")
    worse = version.endswith(_WORSE)
    rng = _rng(ref, key)
    if prompt in ("decision_gate", "action_gate"):
        return _gate(prompt.split("_")[0], worse, corpus.window_by_id()[key], rng)
    if prompt == "decision_agent":
        return _decision(worse, corpus.window_by_id()[key], rng)
    if prompt == "action_agent":
        return _action(worse, corpus.window_by_id()[key], rng)
    if prompt in ("personal_listener_gate", "personal_listener"):
        wid, _, persona = key.rpartition(":")
        w = corpus.window_by_id()[wid]
        if prompt == "personal_listener_gate":
            return _pl_gate(w, persona, rng)
        return _pl(worse, w, persona, rng)
    if prompt == "meeting_agent":
        return _meeting(worse, _golden_case("meeting", key), rng)
    if prompt == "proactive_conflict":
        return _conflict(worse, _golden_case("conflict", key), rng)
    if prompt == "proactive_stakeholder":
        return _stakeholder(_golden_case("stakeholder", key))
    if prompt in why.PROMPTS:
        return why.simulate(prompt, worse, key, rng)
    raise KeyError(f"no simulation for {ref}")
