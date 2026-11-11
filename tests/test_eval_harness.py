"""Evals that can fail (polish plan, Phase 3): the corpus, the gate, and the proof it can fail.

What runs where:
- plain `pytest`: everything below, including the `eval`-marked offline gate (it replays
  recordings: no network, no spend, about 10 s);
- `pytest -m eval`: only the gate tests (offline by default);
- `EVAL_LIVE=1 [EVAL_RECORD=1] uv run pytest -m eval tests/test_eval_harness.py -k live -s`:
  real model calls, the note-quality judge, and (with EVAL_RECORD=1) new recordings. The exact
  commands and costs are in tests/evalkit/harness.py.

The corpus: tests/eval/corpus/windows.json (24 hand-curated + 120 generated windows, seed
20261111), persona_pairs.json, meeting_golden.json, proactive_golden.json. Thresholds: one file,
tests/eval/thresholds.json. The deliberately worse prompts: tests/eval/prompts/*@v0-worse.md.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from libs.shared.never_record import NeverRecordPolicy
from tests.evalkit import corpus, generate, judge, scoring
from tests.evalkit.harness import SUITES, VARIANTS, WORSE, router_for, run_all, suites_from_env
from tests.evalkit.models import (
    LiveModel,
    ReplayModel,
    StaleRecording,
    prompt_index,
    write_recordings,
)
from tests.evalkit.report import export, registry_for

ROOT = Path(__file__).resolve().parents[1]
# The metric per suite that a worse prompt must lose on (higher is better unless listed in LOWER).
HEADLINE = {
    "decision": "f1",
    "action": "f1",
    "persona": "fact_overlap",
    "meeting": "update_f1",
    "proactive_conflict": "precision",
    "never_record": "leaks",
    "decision_assumptions": "precision",
    "decision_ripple": "precision",
    "assumption_check": "precision",
}
LOWER = {"leaks"}


def thresholds() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(corpus.THRESHOLDS_FILE.read_text("utf-8"))
    return data


# --- the corpus ----------------------------------------------------------------------------------


def test_the_corpus_has_100_plus_noisy_windows_of_every_kind():
    ws = corpus.windows()
    assert len(ws) >= 100 and len({w["id"] for w in ws}) == len(ws)
    sources = Counter(w["source"] for w in ws)
    assert sources["core"] >= 20 and sources["generated"] >= 100
    cats = Counter(w["category"] for w in ws)
    assert set(cats) == set(corpus.CATEGORIES)
    assert cats["decision"] + cats["decision_action"] >= 30
    assert cats["action"] + cats["decision_action"] >= 30
    assert cats["lookalike"] >= 20 and cats["never_record"] >= 15 and cats["question"] >= 10
    noise = Counter(k for w in ws for k in w["noise"])
    assert set(noise) == set(corpus.NOISE)
    assert all(noise[k] >= 15 for k in corpus.NOISE), noise
    assert sum(1 for w in ws if w["noise"]) / len(ws) >= 0.8
    modes = Counter(w["never_record"]["mode"] for w in ws if w["never_record"])
    assert set(modes) == {"outright", "alias", "ambiguous"} and modes["ambiguous"] >= 5
    looks = Counter(lk["kind"] for w in ws for lk in w["lookalikes"])
    assert looks["decision"] >= 10 and looks["action"] >= 10


@pytest.mark.parametrize("w", corpus.windows(), ids=lambda w: w["id"])
def test_every_window_is_well_formed(w: dict[str, Any]):
    spoken = {t["speaker"] for t in w["turns"]}
    assert spoken <= set(w["speakers"]), w["id"]
    for s, alias_of in w["speakers"].items():
        assert alias_of is None or alias_of in w["speakers"], (w["id"], s)
    assert all(t["text"].strip() for t in w["turns"])
    for item in [*w["decisions"], *w["actions"], *w["lookalikes"]]:
        assert item["keywords"] and all(k == k.lower() for k in item["keywords"])
        assert item["content"].strip()
    for a in w["actions"]:
        assert corpus.accepted_labels(w, a["assignee"]), w["id"]
        if a["due_date"]:
            date.fromisoformat(a["due_date"])
    if w["never_record"]:
        assert w["category"] == "never_record" and not w["decisions"] and not w["actions"]
        text = " ".join(t["text"] for t in w["turns"]).lower()
        assert w["never_record"]["canary"].lower() in text
        named = NeverRecordPolicy.from_org(corpus.NEVER_RECORD_ORG).hits(text)
        assert named == (w["never_record"]["mode"] != "ambiguous"), w["id"]


def test_label_churn_accepts_both_labels_for_one_person():
    w = corpus.window_by_id()["w-core-003"]
    assert corpus.accepted_labels(w, "S1") == {"Participant 1", "Participant 3"}
    assert corpus.accepted_labels(w, "S3") == {"Participant 1", "Participant 3"}
    assert corpus.accepted_labels(w, "S2") == {"Participant 2"}


def test_the_golden_sets_are_well_formed():
    pairs = corpus.persona_pairs()
    assert len(pairs) >= 20 and all(p["a"] != p["b"] for p in pairs)
    assert all(corpus.facts(corpus.window_by_id()[p["window"]]) for p in pairs)
    meeting = corpus.meeting_golden()
    assert len(meeting) >= 20
    kinds = Counter(c["kind"] for c in meeting)
    # false-positive cases: asides, passing mentions, host-set items, never-record
    assert kinds["aside"] >= 2 and kinds["passing_mention"] and kinds["host_set"]
    assert sum(1 for c in meeting if c["expected"]["off_agenda"]) >= 3
    pro = corpus.proactive_golden()
    conflict = Counter(c["category"] for c in pro["conflict"])
    assert len(pro["conflict"]) >= 20 and conflict["contradicting"] >= 5
    assert sum(conflict.values()) - conflict["contradicting"] >= 12  # the false-positive traps
    assert any(c["expected_alert"] for c in pro["workload"])
    assert any(not c["expected_alert"] for c in pro["workload"])
    assert sum(1 for c in pro["stakeholder"] if not c["expected"]) >= 5


def test_the_generator_reproduces_the_committed_fixtures():
    """Fixed seed, no clock: regenerating writes byte-identical files."""
    assert generate.main(["--check"]) == 0


def test_only_pinned_prompt_versions_may_be_simulated():
    simulated = set()
    for path in corpus.RECORDINGS_DIR.glob("*.json"):
        data = json.loads(path.read_text("utf-8"))
        assert data["prompt"] == path.stem
        if data["source"] == "simulated":
            simulated.add(data["prompt"])
    assert simulated <= generate.SIMULATED_REFS


# --- scoring -------------------------------------------------------------------------------------


def test_candidate_metrics_on_a_known_case():
    rows = [
        {
            "emitted": [
                {"content": "Use Postgres", "confidence": 0.9, "surfaced": True},
                {"content": "Maybe Vercel", "confidence": 0.8, "surfaced": True},
            ],
            "expected": [{"keywords": ["postgres"]}, {"keywords": ["stripe"]}],
            "lookalikes": [{"keywords": ["vercel"]}],
        },
        {"emitted": [], "expected": [], "lookalikes": [{"keywords": ["graphql"]}]},
    ]
    m = scoring.candidates(rows)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["surfaced_precision"] == 0.5
    assert m["brier"] == pytest.approx((0.1**2 + 0.8**2) / 2, abs=1e-4)
    assert m["lookalike_fp_rate"] == 0.5


def test_fact_overlap_on_a_known_case():
    facts = [{"keywords": ["postgres"]}, {"keywords": ["beta", "december"]}]
    m = scoring.fact_overlap(
        [
            ("- Use Postgres\n- Beta in December", "- Postgres it is", facts),
            ("- Beta ships in December", "- the beta, 2 December", facts),
        ]
    )
    assert m["fact_overlap"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["shared_coverage"] == pytest.approx((0.5 + 0.5) / 2)


def test_check_reports_every_missed_threshold():
    report = {"decision": {"f1": 0.7, "brier": 0.2}, "never_record": {"leaks": 0}}
    limits = {
        "decision": {"f1": {"min": 0.8}, "brier": {"max": 0.1}, "recall": {"min": 0.5}},
        "never_record": {"leaks": {"max": 0}},
        "_about": "x",
    }
    assert scoring.check(report, limits) == [
        "decision.f1 = 0.7 < min 0.8",
        "decision.brier = 0.2 > max 0.1",
        "decision.recall: missing",
    ]


def test_the_thresholds_file_covers_every_suite():
    limits = thresholds()
    assert set(SUITES) | {"judge"} <= set(limits)
    for suite, bounds in limits.items():
        if suite.startswith("_"):
            continue
        assert bounds, suite
        for metric, bound in bounds.items():
            assert set(bound) <= {"min", "max"} and bound, (suite, metric)


# --- replay and judge mechanics -----------------------------------------------------------------


def test_the_prompt_index_names_every_prompt_by_its_system_text():
    index = prompt_index()
    from services.agent_runtime_service.consumer import PROMPTS_DIR

    raw = (PROMPTS_DIR / "decision_gate@v2.md").read_text("utf-8")
    system = raw.split("\n---user---\n")[0]
    assert index.ref_for(system) == "decision_gate@v2"
    worse = (corpus.FIXTURE_PROMPTS_DIR / "decision_agent@v0-worse.md").read_text("utf-8")
    assert index.ref_for(worse.split("\n---user---\n")[0]) == "decision_agent@v0-worse"


async def test_a_recording_of_a_changed_prompt_is_refused(tmp_path: Path):
    body = json.loads((corpus.RECORDINGS_DIR / "decision_gate@v2.json").read_text("utf-8"))
    body["prompt_sha256"] = "0" * 64  # recorded against some other text of the prompt
    (tmp_path / "decision_gate@v2.json").write_text(json.dumps(body))
    model = ReplayModel(folder=tmp_path)
    model.case = "w-core-001"
    from services.agent_runtime_service.consumer import PROMPTS_DIR

    system = (PROMPTS_DIR / "decision_gate@v2.md").read_text("utf-8").split("\n---user---\n")[0]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": "Transcript (30 seconds):\nParticipant 1: hi"},
    ]
    with pytest.raises(StaleRecording, match="changed since it was recorded"):
        await model(model="x", messages=messages, max_tokens=20)
    model2 = ReplayModel(folder=tmp_path / "empty")
    model2.case = "w-core-001"
    with pytest.raises(StaleRecording, match="no recording"):
        await model2(model="x", messages=messages, max_tokens=20)
    assert model.errors and model2.errors


def test_the_judge_parses_and_aggregates():
    good = (
        '```json\n{"faithfulness": 5, "coverage": 4, "attribution": 4, "concision": 6, '
        '"ground_rules": 5, "overall": 4.5, "reason": "ok"}\n```'
    )
    assert judge.parse_judgement(good) == {
        "faithfulness": 5.0,
        "coverage": 4.0,
        "attribution": 4.0,
        "concision": 5.0,
        "ground_rules": 5.0,
        "overall": 4.5,
    }
    assert judge.parse_judgement('{"faithfulness": "high"}') is None
    assert judge.parse_judgement("no") is None
    agg = judge.aggregate([judge.parse_judgement(good), None])
    assert agg["judged"] == 1 and agg["unparsed"] == 1 and agg["overall"] == 4.5
    values = judge.render("w-core-001:eng", "- Use Postgres")
    assert (
        "Never record" in values["ground_rules"] and "Role: Engineering lead" in values["profile"]
    )
    assert "Postgres" in values["window"] and values["note"] == "- Use Postgres"
    prompt = judge.EvalLoader().load(judge.JUDGE_KEY, judge.JUDGE_VERSION)
    assert "{note}" in prompt.user_template and prompt.render_user(**values)


def test_the_eval_series_render_for_prometheus():
    reg = registry_for(
        {"never_record": {"leaks": 2}, "decision": {"f1": 0.9}},
        ["x"],
        mode="offline",
        variant="current",
    )
    from prometheus_client import generate_latest

    text = generate_latest(reg).decode()
    assert 'mytheus_eval_never_record_leaks{mode="offline",variant="current"} 2.0' in text
    assert 'mytheus_eval_threshold_failures{mode="offline",variant="current"} 1.0' in text
    assert 'metric="f1"' in text


# --- the offline gate ----------------------------------------------------------------------------


def _offline(variant: str) -> tuple[dict[str, dict[str, Any]], ReplayModel]:
    model = ReplayModel()
    report = asyncio.run(run_all(model, variant))
    assert not model.errors, model.errors
    return report, model


@pytest.fixture(scope="module")
def offline_reports() -> dict[str, dict[str, dict[str, Any]]]:
    return {v: _offline(v)[0] for v in VARIANTS}


def _gate_limits() -> dict[str, Any]:
    return {k: v for k, v in thresholds().items() if k != "judge"}


@pytest.mark.eval
def test_offline_the_current_prompts_meet_every_threshold(offline_reports):
    report = offline_reports["current"]
    failures = scoring.check(report, _gate_limits())
    export(report, failures, mode="offline", variant="current")
    assert not failures, "\n".join(failures)
    # measured, not skipped: every suite ran its cases. (It used to also require f1 < 1.0, a
    # property of the simulated answers; real recordings may score perfectly on these windows.
    # That the gate is not a rubber stamp is the next test: the worse prompts fail it.)
    assert report["decision"]["emitted"] > 0 and report["action"]["emitted"] > 0
    assert report["persona"]["pairs"] >= 20 and report["meeting"]["cases"] >= 20


@pytest.mark.eval
def test_offline_the_gate_fails_the_deliberately_worse_prompts(offline_reports):
    current, worse = offline_reports["current"], offline_reports["worse"]
    export(worse, scoring.check(worse, _gate_limits()), mode="offline", variant="worse")
    for suite, metric in HEADLINE.items():
        a, b = current[suite][metric], worse[suite][metric]
        if metric in LOWER:
            assert b > a, (suite, metric, a, b)
        else:
            assert b < a, (suite, metric, a, b)
    failures = scoring.check(worse, _gate_limits())
    failing_suites = {f.split(".")[0] for f in failures}
    assert set(HEADLINE) <= failing_suites, failures
    # every override in the worse variant is a real prompt file the harness ran
    assert all((corpus.FIXTURE_PROMPTS_DIR / f"{k}@{v}.md").exists() for k, v in WORSE.items())


@dataclass
class _FencedReplay(ReplayModel):
    """The recordings, with every decision extraction answered inside a ```json fence (a
    valid-but-variant answer real models give now and then)."""

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        text, usage = await super().answer(ref, key, kw)
        if ref.startswith("decision_agent") and not text.lstrip().startswith("```"):
            text = f"```json\n{text}\n```"
        return text, usage


@pytest.mark.eval
def test_offline_a_broken_parser_would_fail_the_gate(monkeypatch):
    """The gate also guards the code around the model: a candidate parser that loses its
    tolerance for code fences drops recall on fenced answers and the gate notices. (The first
    live recordings hold no fenced answer, so the fence is added here rather than relied on.)"""
    from services.agent_runtime_service.agents import candidates as mod
    from tests.evalkit.harness import run_candidates

    good = asyncio.run(run_candidates("decision", _FencedReplay(), "current"))
    real = mod.parse_candidates

    def strict(text: str, key: str = "candidates") -> list[Any]:
        return [] if text.lstrip().startswith("```") else real(text, key)

    monkeypatch.setattr(mod, "parse_candidates", strict)
    broken = asyncio.run(run_candidates("decision", _FencedReplay(), "current"))
    assert good["tp"] >= 1 and broken["tp"] < good["tp"]
    failures = scoring.check({"decision": broken}, _gate_limits())
    assert any(f.startswith("decision.") for f in failures), failures


# --- live (opt-in: EVAL_LIVE=1; the coordinator runs it) ----------------------------------------


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


@pytest.mark.eval
def test_live_eval_and_record():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (about 800 model calls, 3-5 USD)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    record = os.environ.get("EVAL_RECORD") == "1"
    suites = suites_from_env(os.environ.get("EVAL_SUITES"))
    partial = suites != tuple(SUITES)
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    shared: dict[str, dict[str, str]] = {}  # one answer per (prompt, case) across variants
    for variant in VARIANTS:
        model = LiveModel(shared=shared)
        reports[variant] = asyncio.run(run_all(model, variant, suites))
        if record:
            write_recordings(model.recorded, "recorded")
    if "persona" in suites:
        notes = reports["current"]["persona"]["notes"]
        reports["current"]["judge"] = asyncio.run(judge.judge_notes(router_for(LiveModel()), notes))
    limits = thresholds()
    failures = scoring.check(reports["current"], limits)
    mode = "live-partial" if partial else "live"
    export(reports["current"], failures, mode=mode, variant="current")
    export(reports["worse"], scoring.check(reports["worse"], limits), mode=mode, variant="worse")
    assert not failures, "\n".join(failures)
    for suite, metric in HEADLINE.items():
        if suite not in suites:
            continue
        a, b = reports["current"][suite][metric], reports["worse"][suite][metric]
        # a real model may happen not to leak under the worse prompt either: never better
        assert (b >= a) if metric in LOWER else (b < a), (suite, metric, a, b)
