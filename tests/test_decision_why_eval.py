"""The golden set of "Decisions that know why" and its offline gate (`tests/evalkit/why.py`).

The three suites (`decision_assumptions`, `decision_ripple`, `assumption_check`) run inside the
harness's offline gate (`make eval`, `tests/test_eval_harness.py`) against the recordings in
`tests/eval/recordings`: simulated until the coordinator records them live. This file checks the
golden set itself (enough cases, the negative kinds, quotes that are really in their sources) and
that the gate is not a rubber stamp: the deliberately worse prompts fail every suite.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import pytest

from tests.evalkit import corpus, scoring, why
from tests.evalkit.harness import SUITES
from tests.evalkit.models import ReplayModel

SUITE_NAMES = ("decision_assumptions", "decision_ripple", "assumption_check")


def _norm(text: str) -> str:
    from services.agent_runtime_service.decision_why import _norm as norm

    return norm(text)


def test_every_prompt_has_eight_or_more_golden_cases_with_negatives():
    g = why.golden()
    for section in ("assumptions", "ripple", "assumption_check"):
        assert len(g[section]) >= 8, section
        ids = [c["id"] for c in g[section]]
        assert len(set(ids)) == len(ids), section
    kinds = Counter(c["category"] for c in g["assumptions"])
    assert {"hypothetical", "question", "no_reason", "never_record"} <= set(kinds)
    assert sum(1 for c in g["assumptions"] if not c["expected"]) >= 4
    assert sum(1 for c in g["ripple"] if not c["expected_goals"] and not c["expected_actions"]) >= 3
    checks = Counter(c["category"] for c in g["assumption_check"])
    assert {"hypothetical", "question", "rumour", "status_fits"} <= set(checks)
    assert sum(1 for c in g["assumption_check"] if not c["expected"]) >= 4
    assert {c["evidence"]["kind"] for c in g["assumption_check"]} == {"meeting", "decision"}


@pytest.mark.parametrize("case", why.golden()["assumptions"], ids=lambda c: c["id"])
def test_assumption_cases_quote_what_was_said(case: dict[str, Any]):
    said = " ".join(
        [case["decision"], case.get("rationale") or "", case.get("quote") or ""]
        + [line.partition(": ")[2] for line in case["transcript"]]
    )
    speakers = {line.partition(": ")[0] for line in case["transcript"]}
    for x in [*case["expected"], *case.get("traps", [])]:
        if x.get("quote"):
            assert _norm(x["quote"]) in _norm(said), (case["id"], x["quote"])
        assert x["kind"] in ("price", "date", "metric", "dependency", "person", "other")
    for x in case["expected"]:
        assert x["keywords"] and all(k == k.lower() for k in x["keywords"])
        assert scoring.keyword_match(f"{x['statement']} {x.get('quote') or ''}", x["keywords"])
        assert x.get("speaker") is None or x["speaker"] in speakers
    if case.get("quote_speaker"):
        assert case["quote_speaker"] in speakers


@pytest.mark.parametrize("case", why.golden()["ripple"], ids=lambda c: c["id"])
def test_ripple_cases_point_at_listed_items(case: dict[str, Any]):
    assert all(1 <= g["goal"] <= len(case["goals"]) for g in case["expected_goals"])
    assert all(1 <= a["action"] <= len(case["actions"]) for a in case["expected_actions"])
    assert all(g["effect"] in ("advances", "risks") for g in case["expected_goals"])
    assert all(a["effect"] in ("changes", "blocks", "unblocks") for a in case["expected_actions"])


@pytest.mark.parametrize("case", why.golden()["assumption_check"], ids=lambda c: c["id"])
def test_check_cases_quote_the_evidence(case: dict[str, Any]):
    ev = case["evidence"]
    if ev["kind"] == "meeting":
        text = " ".join(line.partition(": ")[2] for line in ev["transcript"])
        speakers = {line.partition(": ")[0] for line in ev["transcript"]}
    else:
        text, speakers = f"{ev['decision']} {ev.get('rationale') or ''}", set()
    n = len(case["assumptions"])
    for x in [*case["expected"], *case.get("traps", [])]:
        assert 1 <= x["assumption"] <= n
        assert _norm(x["quote"]) in _norm(text), (case["id"], x["quote"])
    for x in case["expected"]:
        assert x.get("speaker") is None or x["speaker"] in speakers
    broken = {x["assumption"] for x in case["expected"]}
    assert not broken & {t["assumption"] for t in case.get("traps", [])}


def test_the_recordings_cover_every_case_both_variants():
    for ref in (
        "decision_assumptions@v1",
        "decision_ripple@v1",
        "assumption_check@v1",
        "decision_assumptions@v0-worse",
        "decision_ripple@v0-worse",
        "assumption_check@v0-worse",
    ):
        data = json.loads((corpus.RECORDINGS_DIR / f"{ref}.json").read_text("utf-8"))
        assert data["prompt"] == ref and data["entries"], ref


@pytest.mark.eval
def test_offline_the_why_suites_pass_and_the_worse_prompts_fail():
    limits = json.loads(corpus.THRESHOLDS_FILE.read_text("utf-8"))
    for variant in ("current", "worse"):
        model = ReplayModel()
        report = {s: asyncio.run(SUITES[s](model, variant)) for s in SUITE_NAMES}
        assert not model.errors, model.errors
        failures = scoring.check(report, {s: limits[s] for s in SUITE_NAMES})
        if variant == "current":
            assert not failures, failures
            assert report["decision_assumptions"]["tp"] >= 6
        else:
            assert {f.split(".")[0] for f in failures} == set(SUITE_NAMES), failures
