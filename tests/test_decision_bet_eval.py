"""Golden-set eval for the bets proposer (`decision_bet@vN`, decisions as bets).

Scoring, over the proposals for each golden meeting (`score`):
- **recall**: expected expectations proposed on the right decision with the right kind and
  direction;
- **precision**: proposals that match an expected one (a proposal on a decision nobody stated an
  expectation for is a false positive);
- **date_accuracy**: matched proposals with the exact check date (relative dates resolved);
- **magnitude_accuracy**: matched proposals with the right size (or none when none was said);
- **no_invention**: cases that expect nothing propose nothing (must be 1.0);
- **leak_free**: no never-record text in any proposal (must be 1.0).

Offline (always, CI): the real `BetProposer` (never-record filter, labels, the prompt, the
parser and grounding) runs on each golden meeting with the model answer replayed from
`tests/eval/decision_bet_recording.json` (`source: simulated` until a live run records real
answers); the recording must match the prompt file (sha256) or the test fails as stale.

Live (`pytest -m eval -k live`, `EVAL_LIVE=1` and an `ANTHROPIC_API_KEY`): real Sonnet calls,
7 in all; writes `tests/eval/decision_bet_last_run.json`; with `EVAL_RECORD=1` the answers
replace the recording (`source: recorded`). Asserts the baseline's minimums.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.bets import BetProposer, BetSource, parse_bet_source
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.evalkit.models import prompt_index
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink, keyword_match

EVAL_DIR = Path(__file__).parent / "eval"
# Kept beside the golden set, not in tests/eval/recordings: that folder is the window harness's,
# which only lets its own generator write simulated answers there.
RECORDING = EVAL_DIR / "decision_bet_recording.json"
REF = "decision_bet@v1"
LINE_MS = 20_000


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "decision_bet_golden.json").read_text("utf-8"))
    return data


def baseline() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "decision_bet_baseline.json").read_text("utf-8"))
    return data


def build_source(case: dict[str, Any]) -> tuple[BetSource, list[str]]:
    """The golden meeting as core-api's minutes source; the recorded decisions' ids in order."""
    users = {label: uuid.uuid4() for label in case["participants"]}
    turns = []
    for i, line in enumerate(case["transcript"]):
        label, _, text = line.partition(": ")
        turns.append(
            {
                "speaker_id": str(users[label]),
                "start_ms": i * LINE_MS,
                "end_ms": i * LINE_MS + 5_000,
                "text": text,
            }
        )
    decisions = [{"id": str(uuid.uuid4()), **d} for d in case["recorded_decisions"]]
    body = {
        "meeting_id": str(uuid.uuid4()),
        "title": case["id"],
        "scheduled_start": f"{case['meeting_date']}T15:00:00Z",
        "participants": [
            {
                "user_id": str(u),
                "name": label.replace("Participant ", "Person "),
                "attendance": "live",
            }
            for label, u in users.items()
        ],
        "turns": turns,
        "decisions": decisions,
    }
    source = parse_bet_source(body, date(2026, 1, 1))
    assert source is not None
    return source, [d["id"] for d in decisions]


def score(runs: list[tuple[dict[str, Any], list[dict[str, Any]], list[str]]]) -> dict[str, float]:
    """`runs`: (golden case, the proposed expectations, the recorded decision ids)."""
    expected_n = matched = proposed_n = dates = sizes = 0
    invented = leaked = False
    for case, proposed, ids in runs:
        proposed_n += len(proposed)
        if not case["expected"] and proposed:
            invented = True
        text = json.dumps(proposed).lower()
        if any(f.lower() in text for f in case.get("forbidden", [])):
            leaked = True
        used: set[int] = set()
        for exp in case["expected"]:
            expected_n += 1
            want_id = ids[int(exp["ref"][1:]) - 1]
            for i, p in enumerate(proposed):
                if i in used or p["decision_id"] != want_id:
                    continue
                if p["kind"] != exp["kind"] or p["direction"] != exp["direction"]:
                    continue
                if not keyword_match(f"{p['statement']} {p['metric']}", exp["keywords"]):
                    continue
                used.add(i)
                matched += 1
                dates += p["check_date"] == exp["check_date"]
                want = exp["magnitude"]
                got = p["magnitude"]
                sizes += (want is None and got is None) or (
                    want is not None and got is not None and abs(float(got) - want) < 1e-6
                )
                break

    def rate(hit: int, total: int) -> float:
        return 1.0 if total == 0 else round(hit / total, 3)

    return {
        "recall": rate(matched, expected_n),
        "precision": rate(matched, proposed_n),
        "date_accuracy": rate(dates, matched),
        "magnitude_accuracy": rate(sizes, matched),
        "no_invention": 0.0 if invented else 1.0,
        "leak_free": 0.0 if leaked else 1.0,
    }


class _Source:
    def __init__(self, source: BetSource) -> None:
        self.value = source

    async def source(self, _org: uuid.UUID, _meeting: uuid.UUID) -> BetSource:
        return self.value


class _ReplayRouter:
    """The router's `complete` answering from the recording for the current case."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.case = ""
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> Any:
        self.calls.append(kw)

        class _Out:
            text = self.answers[self.case]

        return _Out()


async def _run(
    router: Any, case_hook: Any = None
) -> list[tuple[dict[str, Any], list[dict[str, Any]], list[str]]]:
    runs = []
    for case in golden()["cases"]:
        source, ids = build_source(case)
        ctx = CompanyContext(never_record_topics=tuple(case.get("never_record", [])))

        async def org_context(_org: uuid.UUID, c: CompanyContext = ctx) -> CompanyContext:
            return c

        if case_hook is not None:
            case_hook(case["id"])
        proposer = BetProposer(
            PromptLoader(PROMPTS_DIR), router, source=_Source(source), org_context=org_context
        )
        proposal = await proposer.propose(uuid.uuid4(), source.source.meeting_id, "eval")
        runs.append((case, proposal.payload["expectations"] if proposal else [], ids))
    return runs


# --- deterministic ---------------------------------------------------------------------------


def test_golden_set_is_well_formed():
    data = golden()
    assert len(data["cases"]) >= 6
    assert sum(1 for c in data["cases"] if not c["expected"]) >= 2  # restraint is tested
    for case in data["cases"]:
        labels = set(case["participants"])
        assert {ln.partition(": ")[0] for ln in case["transcript"]} <= labels
        for e in case["expected"]:
            assert 1 <= int(e["ref"][1:]) <= len(case["recorded_decisions"])
            date.fromisoformat(e["check_date"])


def test_scoring_rewards_the_right_bets_and_catches_inventions():
    case = golden()["cases"][0]
    _, ids = build_source(case)
    good = [
        {
            "decision_id": ids[0],
            "statement": "Annual billing cuts churn by 3 points",
            "metric": "Monthly churn",
            "kind": "metric",
            "direction": "decrease",
            "magnitude": 3,
            "check_date": "2027-03-31",
        }
    ]
    assert set(score([(case, good, ids)]).values()) == {1.0}
    wrong_date = [{**good[0], "check_date": "2027-03-01"}]
    assert score([(case, wrong_date, ids)])["date_accuracy"] == 0.0
    extra = [*good, {**good[0], "decision_id": ids[1], "statement": "Support lead lowers churn"}]
    assert score([(case, extra, ids)])["precision"] == 0.5
    quiet = golden()["cases"][1]
    _, qids = build_source(quiet)
    invented = [{**good[0], "decision_id": qids[0]}]
    assert score([(quiet, invented, qids)])["no_invention"] == 0.0


def test_recording_matches_the_prompt():
    data = json.loads(RECORDING.read_text("utf-8"))
    assert data["prompt"] == REF
    assert data["prompt_sha256"] == prompt_index().sha(REF), (
        "decision_bet@v1 changed since it was recorded: re-record it "
        "(EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_decision_bet_eval.py -k live)"
    )
    assert set(data["entries"]) == {c["id"] for c in golden()["cases"]}


async def test_offline_replay_meets_the_baseline():
    """The real proposer over every golden meeting, the model answer replayed."""
    answers = json.loads(RECORDING.read_text("utf-8"))["entries"]
    router = _ReplayRouter(answers)

    def hook(case_id: str) -> None:
        router.case = case_id

    runs = await _run(router, hook)
    m = score(runs)
    b = baseline()
    for key in (
        "recall",
        "precision",
        "date_accuracy",
        "magnitude_accuracy",
        "no_invention",
        "leak_free",
    ):
        assert m[key] >= b[f"min_{key}"], (key, m)
    # every call went through the extract tier with the versioned prompt
    assert {c["prompt_ref"] for c in router.calls} == {REF}
    assert {c["tier"] for c in router.calls} == {"extract"}
    # the never-record lines never reached the model (only the rule naming the topic did)
    sent = " ".join(c["user"] + c["cached_prefix"] for c in router.calls).lower()
    assert "2 million" not in sent and "between us" not in sent


# --- live -------------------------------------------------------------------------------------


@pytest.mark.eval
async def test_decision_bets_live():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends 7 Sonnet calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")

    async def unlimited(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": 0,
            "token_quota": 10**9,
            "hard_cap": 10**9,
            "enforcement": "enforce",
        }

    sink = _NoSpendSink()
    real = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    answers: dict[str, str] = {}
    current = {"case": ""}

    class _Recording:
        async def complete(self, **kw: Any) -> Any:
            out = await real.complete(**kw)
            answers[current["case"]] = out.text
            return out

    def hook(case_id: str) -> None:
        current["case"] = case_id

    runs = await _run(_Recording(), hook)
    m = score(runs)
    report = {
        "prompt": REF,
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        "metrics": m,
        "cases": {c["id"]: p for c, p, _ in runs},
    }
    (EVAL_DIR / "decision_bet_last_run.json").write_text(json.dumps(report, indent=2) + "\n")
    if os.environ.get("EVAL_RECORD") == "1":
        RECORDING.write_text(
            json.dumps(
                {
                    "prompt": REF,
                    "prompt_sha256": prompt_index().sha(REF),
                    "source": "recorded",
                    "about": "Real model answers recorded by the live eval (EVAL_RECORD=1).",
                    "entries": answers,
                },
                indent=2,
            )
            + "\n"
        )
    b = baseline()
    assert m["no_invention"] == 1.0 and m["leak_free"] == 1.0
    for key in ("recall", "precision", "date_accuracy", "magnitude_accuracy"):
        assert m[key] >= b[f"min_{key}"], (key, m)
