"""Eval suites and the simulated model for "Decisions that know why" (`decision_why.py`).

Three suites over `tests/eval/corpus/decision_why_golden.json`, each through the real
`DecisionWhy` code (the router, the prompt, the label mapping, the verbatim-quote checks, the
confidence floors and the never-record guard); only core-api, the company brain and the model
answer are fakes:

    decision_assumptions  DecisionWhy.on_decision (watcher off, no goals or tasks: one call)
        precision / recall / f1 of the emitted assumptions against the expected ones (matched by
        keyword groups over statement + quote, highest confidence first); negative_fp_rate: the
        cases with no stated reason (hypotheticals, questions, no reason, never record) that
        still get an assumption; speaker_accuracy and watch_accuracy over the matched ones.
    decision_ripple       DecisionWhy.ripple_for (the ripple call alone)
        precision / recall / f1 over (goal or task, effect) pairs; false_positive_rate: the
        no-ripple cases that still get one.
    assumption_check      trigger A (DecisionWhy.on_decision, why off) or trigger B
                          (DecisionWhy.on_meeting_ended) by the case's evidence kind
        precision / recall / f1 over broken assumptions; false_positive_rate: the cases where
        nothing breaks (hypotheticals, questions, a different product, a fitting status update,
        a rumour) that still report a break.

`simulate(prompt, worse, key, rng)` is the simulated model for these prompts (`simulate.py`
dispatches here): the current prompts behave like a good, conservative model (the expected
items with calibrated confidence, an occasional miss on a `hard` item, a sub-floor stray on a
negative); the `@v0-worse` fixtures report every trap (hypotheticals, questions, past facts, the
decision restated, every goal and task, every passing mention) with high confidence and drop the
watch fields. Live recordings replace these (`EVAL_LIVE=1 EVAL_RECORD=1`, harness.py).
"""

from __future__ import annotations

import json
import random
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from libs.shared.org_context import CompanyContext, Goal
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.proactive_agent import ConfirmedDecision
from services.agent_runtime_service.decision_why import (
    DecisionWhy,
    OpenAssumption,
    WhyDeps,
    WhySettings,
    WhySource,
    parse_open,
    parse_why_source,
)
from services.agent_runtime_service.minutes import MinutesSource, parse_source
from tests.evalkit import corpus, scoring

if TYPE_CHECKING:  # models imports simulate, which imports this module
    from tests.evalkit.models import Model

FILE = "decision_why_golden.json"
PROMPTS = ("decision_assumptions", "decision_ripple", "assumption_check")
LINE_MS = 8_000
_NULL_WATCH = {"date": None, "value": None, "comparator": None, "unit": None, "issue_key": None}


def golden() -> dict[str, list[dict[str, Any]]]:
    data = corpus.load(FILE)
    return {k: list(data[k]) for k in ("assumptions", "ripple", "assumption_check")}


def case_by_id(section: str, key: str) -> dict[str, Any]:
    return next(c for c in golden()[section] if c["id"] == key)


# --- building the inputs ------------------------------------------------------------------------


def user(case_id: str, speaker: str) -> UUID:
    return uuid5(corpus.NS, f"why:{case_id}:{speaker}")


def _split(line: str) -> tuple[str, str]:
    speaker, _, text = line.partition(": ")
    return speaker, text


def speaker_labels(lines: list[str]) -> dict[str, str]:
    """Corpus speaker -> "Participant N", in order of first speaking (as `Labels.build`)."""
    out: dict[str, str] = {}
    for line in lines:
        s, _ = _split(line)
        out.setdefault(s, f"Participant {len(out) + 1}")
    return out


def _people(case_id: str, lines: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    order = list(speaker_labels(lines))
    participants = [
        {"user_id": str(user(case_id, s)), "name": f"{s.lower()}@eval.example"} for s in order
    ]
    return participants, {f"spk-{s}": str(user(case_id, s)) for s in order}


def _turns(lines: list[str], start_ms: int = 0) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(lines):
        s, text = _split(line)
        t = start_ms + i * LINE_MS
        out.append({"speaker_id": f"spk-{s}", "start_ms": t, "end_ms": t + 6_000, "text": text})
    return out


def decision_id(case_id: str) -> UUID:
    return uuid5(corpus.NS, f"why:{case_id}:decision")


def meeting_id(case_id: str) -> UUID:
    return uuid5(corpus.NS, f"why:{case_id}:meeting")


def why_body(
    case: dict[str, Any], *, actions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    lines = case.get("transcript") or []
    participants, speakers = _people(case["id"], lines)
    owner = user(case["id"], "owner")
    participants.append({"user_id": str(owner), "name": "owner@eval.example"})
    speaker = case.get("quote_speaker")
    return {
        "decision": {
            "id": str(decision_id(case["id"])),
            "meeting_id": str(meeting_id(case["id"])),
            "content": case["decision"],
            "rationale": case.get("rationale") or None,
            "quote": case.get("quote") or None,
            "quote_speaker_id": f"spk-{speaker}" if speaker else None,
            "quote_start_ms": None,
            "confirmed_by": str(owner),
            "confidential": False,
            "status": "confirmed",
        },
        "meeting_title": case["id"],
        "participants": participants,
        "speakers": speakers,
        "turns": _turns(lines),
        "candidate_actions": actions or [],
        "already_read": False,
    }


def confirmed(case: dict[str, Any]) -> ConfirmedDecision:
    return ConfirmedDecision(
        org_id=corpus.ORG,
        decision_id=decision_id(case["id"]),
        meeting_id=meeting_id(case["id"]),
        content=case["decision"],
        rationale=case.get("rationale") or "",
        confirmed_by=user(case["id"], "owner"),
    )


def assumption_id(case_id: str, n: int) -> UUID:
    return uuid5(corpus.NS, f"why:{case_id}:H{n}")


def open_items(case: dict[str, Any]) -> list[OpenAssumption]:
    return parse_open(
        {
            "items": [
                {
                    "id": str(assumption_id(case["id"], n)),
                    "decision_id": str(uuid5(corpus.NS, f"why:{case['id']}:old:{n}")),
                    "decision_content": a["decision"],
                    "kind": a["kind"],
                    "statement": a["statement"],
                    "quote": a.get("quote"),
                    "meeting_id": str(uuid5(corpus.NS, f"why:{case['id']}:old-meeting")),
                }
                for n, a in enumerate(case["assumptions"], start=1)
            ]
        }
    )


class StaticCore:
    def __init__(self, body: dict[str, Any] | None, items: list[OpenAssumption]) -> None:
        self.body, self.items = body, items

    async def why_source(self, org_id: UUID, decision_id: UUID) -> WhySource | None:
        return parse_why_source(self.body) if self.body else None

    async def open_assumptions(
        self,
        org_id: UUID,
        *,
        exclude_meeting_id: UUID | None = None,
        exclude_decision_id: UUID | None = None,
    ) -> list[OpenAssumption]:
        return list(self.items)


class StaticMinutes:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    async def source(self, org_id: UUID, meeting_id: UUID) -> MinutesSource | None:
        return parse_source(self.body)


def _brain(org: CompanyContext) -> Any:
    async def fetch(_org: UUID) -> CompanyContext:
        return org

    return fetch


def _why(
    loader: PromptLoader,
    router: LiteLLMRouter,
    *,
    core: Any = None,
    minutes: Any = None,
    org: CompanyContext | None = None,
    why_on: bool = True,
    watch_on: bool = False,
) -> DecisionWhy:
    return DecisionWhy(
        loader,
        router,
        deps=WhyDeps(core=core, minutes=minutes, org_context=_brain(org or CompanyContext())),
        settings=WhySettings(
            why_enabled=why_on, watch_enabled=watch_on, all_plans=True, settle_s=0
        ),
    )


# --- the suites ---------------------------------------------------------------------------------


async def run_assumptions(
    model: Model, loader: PromptLoader, router: LiteLLMRouter
) -> dict[str, Any]:
    tp = fp = fn = 0
    speaker_ok = speaker_n = watch_ok = watch_n = 0
    negatives = negative_hits = 0
    for case in golden()["assumptions"]:
        org = CompanyContext(never_record_topics=tuple(case.get("never_record", [])))
        why = _why(loader, router, core=StaticCore(why_body(case), []), org=org)
        model.case, model.persona = case["id"], ""
        records = await why.on_decision(confirmed(case), date.fromisoformat(case["meeting_date"]))
        emitted = records[0].result.payload["why"]["assumptions"] if records else []
        expected = case["expected"]
        if not expected:
            negatives += 1
            negative_hits += bool(emitted)
        free = list(range(len(expected)))
        for e in sorted(emitted, key=lambda a: -float(a["confidence"])):
            text = f"{e['statement']} {e['quote'] or ''}"
            hit = next(
                (i for i in free if scoring.keyword_match(text, expected[i]["keywords"])), None
            )
            if hit is None:
                fp += 1
                continue
            free.remove(hit)
            tp += 1
            want = expected[hit]
            speaker = want.get("speaker")
            speaker_n += 1
            speaker_ok += e["speaker_user_id"] == (
                str(user(case["id"], speaker)) if speaker else None
            )
            if want.get("watch"):
                watch_n += 1
                watch_ok += all(e["watch"].get(k) == v for k, v in want["watch"].items())
        fn += len(free)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {
        "cases": len(golden()["assumptions"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "negative_fp_rate": round(negative_hits / negatives, 4) if negatives else 0.0,
        "speaker_accuracy": round(speaker_ok / speaker_n, 4) if speaker_n else 1.0,
        "watch_accuracy": round(watch_ok / watch_n, 4) if watch_n else 1.0,
        "parse_failure_rate": round(model.parse_failure_rate("decision_assumptions"), 4),
    }


def ripple_inputs(case: dict[str, Any]) -> tuple[list[Goal], list[dict[str, Any]]]:
    goals = [
        Goal(id=str(uuid5(corpus.NS, f"why:{case['id']}:G{n}")), title=title)
        for n, title in enumerate(case["goals"], start=1)
    ]
    actions = [
        {
            "id": str(uuid5(corpus.NS, f"why:{case['id']}:A{n}")),
            "content": a["content"],
            "due_date": None,
            "meeting_id": str(uuid5(corpus.NS, f"why:{case['id']}:other")),
            "meeting_title": "Other meeting",
            "customer": a["customer"],
        }
        for n, a in enumerate(case["actions"], start=1)
    ]
    return goals, actions


async def run_ripple(model: Model, loader: PromptLoader, router: LiteLLMRouter) -> dict[str, Any]:
    got: list[set[tuple[str, int, str]]] = []
    want: list[set[tuple[str, int, str]]] = []
    for case in golden()["ripple"]:
        goals, actions = ripple_inputs(case)
        org = CompanyContext(goals=tuple(goals))
        why = _why(loader, router, org=org)
        source = parse_why_source(why_body(case, actions=actions))
        assert source is not None
        ctx = await why.context(corpus.ORG)
        model.case, model.persona = case["id"], ""
        goal_out, action_out = await why.ripple_for(
            corpus.ORG, source, case.get("rationale") or "", ctx
        )
        goal_index = {g.id: n for n, g in enumerate(goals, start=1)}
        action_index = {a["id"]: n for n, a in enumerate(actions, start=1)}
        got.append(
            {("goal", goal_index[g["goal_id"]], g["effect"]) for g in goal_out}
            | {("action", action_index[a["action_id"]], a["effect"]) for a in action_out}
        )
        want.append(
            {("goal", g["goal"], g["effect"]) for g in case["expected_goals"]}
            | {("action", a["action"], a["effect"]) for a in case["expected_actions"]}
        )
    p, r, f1, tp, fp, fn = scoring.sets(got, want)
    negatives = [i for i, w in enumerate(want) if not w]
    fpr = sum(1 for i in negatives if got[i]) / len(negatives) if negatives else 0.0
    return {
        "cases": len(want),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "parse_failure_rate": round(model.parse_failure_rate("decision_ripple"), 4),
    }


def evidence_minutes(case: dict[str, Any]) -> dict[str, Any]:
    lines = case["evidence"]["transcript"]
    participants, speakers = _people(case["id"], lines)
    return {
        "meeting_id": str(meeting_id(case["id"])),
        "title": case["id"],
        "confidential": False,
        "host_user_id": participants[0]["user_id"],
        "participants": participants,
        "speakers": speakers,
        "turns": _turns(lines),
    }


async def run_check(model: Model, loader: PromptLoader, router: LiteLLMRouter) -> dict[str, Any]:
    got: list[set[int]] = []
    want: list[set[int]] = []
    for case in golden()["assumption_check"]:
        items = open_items(case)
        index = {a.id: n for n, a in enumerate(items, start=1)}
        when = date.fromisoformat(case["evidence_date"])
        model.case, model.persona = case["id"], ""
        ev = case["evidence"]
        if ev["kind"] == "decision":
            body = why_body(
                {"id": case["id"], "decision": ev["decision"], "rationale": ev.get("rationale")}
            )
            why = _why(loader, router, core=StaticCore(body, items), why_on=False, watch_on=True)
            d = confirmed({"id": case["id"], "decision": ev["decision"], "rationale": ""})
            records = await why.on_decision(d, when)
        else:
            why = _why(
                loader,
                router,
                core=StaticCore(None, items),
                minutes=StaticMinutes(evidence_minutes(case)),
                watch_on=True,
            )
            records = await why.on_meeting_ended(corpus.ORG, meeting_id(case["id"]), "eval", when)
        breaks = [b for r in records for b in r.result.payload.get("breaks", [])]
        got.append({index[UUID(b["assumption_id"])] for b in breaks})
        want.append({e["assumption"] for e in case["expected"]})
    p, r, f1, *_ = scoring.sets(got, want)
    negatives = [i for i, w in enumerate(want) if not w]
    fpr = sum(1 for i in negatives if got[i]) / len(negatives) if negatives else 0.0
    return {
        "cases": len(want),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "parse_failure_rate": round(model.parse_failure_rate("assumption_check"), 4),
    }


# --- the simulated model ------------------------------------------------------------------------


def _r2(x: float) -> float:
    return round(x, 2)


def _assumptions(worse: bool, case: dict[str, Any], rng: random.Random) -> str:
    labels = speaker_labels(case.get("transcript") or [])

    def entry(x: dict[str, Any], lo: float, hi: float, watch: bool) -> dict[str, Any]:
        speaker = x.get("speaker")
        return {
            "kind": x["kind"],
            "statement": x["statement"],
            "quote": x.get("quote"),
            "speaker": labels.get(speaker) if speaker else None,
            "confidence": _r2(rng.uniform(lo, hi)),
            "watch": {**_NULL_WATCH, **(x.get("watch") or {})} if watch else dict(_NULL_WATCH),
        }

    if worse:
        out = [entry(x, 0.8, 0.95, False) for x in case["expected"]]
        out += [entry(t, 0.75, 0.9, False) for t in case.get("traps", [])]
        restated = {"kind": "other", "statement": case["decision"], "quote": None}
        out.append(entry(restated, 0.8, 0.9, False))
        return json.dumps({"assumptions": out})
    out = [
        entry(x, 0.75, 0.95, True)
        for x in case["expected"]
        if not (x.get("hard") and rng.random() < 0.1)
    ]
    if not case["expected"] and case.get("traps") and rng.random() < 0.25:
        out.append(entry(case["traps"][0], 0.3, 0.45, False))  # under the 0.5 floor
    return json.dumps({"assumptions": out})


def _ripple(worse: bool, case: dict[str, Any], rng: random.Random) -> str:
    if worse:
        goals = [
            {"goal": f"G{n}", "effect": "advances", "why": "Related.", "confidence": 0.8}
            for n in range(1, len(case["goals"]) + 1)
        ]
        actions = [
            {"action": f"A{n}", "effect": "changes", "why": "Related.", "confidence": 0.8}
            for n in range(1, len(case["actions"]) + 1)
        ]
        return json.dumps({"goals": goals, "actions": actions})
    goals = [
        {
            "goal": f"G{g['goal']}",
            "effect": g["effect"],
            "why": "The decision directly moves this goal.",
            "confidence": _r2(rng.uniform(0.72, 0.93)),
        }
        for g in case["expected_goals"]
    ]
    actions = [
        {
            "action": f"A{a['action']}",
            "effect": a["effect"],
            "why": "The decision directly affects this task.",
            "confidence": _r2(rng.uniform(0.72, 0.93)),
        }
        for a in case["expected_actions"]
    ]
    if not goals and not actions and rng.random() < 0.3:
        actions.append({"action": "A1", "effect": "changes", "why": "Weak.", "confidence": 0.42})
    return json.dumps({"goals": goals, "actions": actions})


def _check(worse: bool, case: dict[str, Any], rng: random.Random) -> str:
    ev = case["evidence"]
    labels = speaker_labels(ev.get("transcript") or [])

    def entry(x: dict[str, Any], lo: float, hi: float) -> dict[str, Any]:
        speaker = x.get("speaker")
        return {
            "assumption": f"H{x['assumption']}",
            "evidence_quote": x["quote"],
            "speaker": labels.get(speaker) if speaker else None,
            "rationale": "The new evidence contradicts this assumption.",
            "confidence": _r2(rng.uniform(lo, hi)),
        }

    if worse:
        out = [entry(x, 0.8, 0.95) for x in case["expected"]]
        out += [entry(t, 0.75, 0.9) for t in case.get("traps", [])]
        return json.dumps({"breaks": out})
    out = [entry(x, 0.75, 0.95) for x in case["expected"]]
    if not case["expected"] and case.get("traps") and rng.random() < 0.25:
        out.append(entry(case["traps"][0], 0.35, 0.55))  # under the 0.6 floor
    return json.dumps({"breaks": out})


def simulate(prompt: str, worse: bool, key: str, rng: random.Random) -> str:
    if prompt == "decision_assumptions":
        return _assumptions(worse, case_by_id("assumptions", key), rng)
    if prompt == "decision_ripple":
        return _ripple(worse, case_by_id("ripple", key), rng)
    if prompt == "assumption_check":
        return _check(worse, case_by_id("assumption_check", key), rng)
    raise KeyError(prompt)
