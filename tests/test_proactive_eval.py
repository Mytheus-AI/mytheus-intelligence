"""Golden-set eval for `ProactiveAgent`'s conflict judge (`proactive_conflict@v1`).

A false red alert is the costly failure (CLAUDE.md: "conflict false-positive rate is an eval
gate"), so precision is the headline. Over `tests/eval/proactive_conflict_golden.json`, one new
decision and one earlier candidate per case, run through the real agent (`check_decision`, the
real prompt, router and parser; only the candidate source is faked):

- **precision** / **recall** of "an alert is emitted" (confidence >= 0.4) against `contradicts`;
- the same at the surfaced band (> 0.7, the red banner);
- **false_positive_rate** = non-contradicting cases that raise an alert / non-contradicting cases.

Deterministic tests always run, including an offline scripted run through the real agent.
`test_conflict_judge_meets_the_baseline` (`pytest -m eval`) calls the real Sonnet judge once per
case, only with `EVAL_LIVE=1` and an `ANTHROPIC_API_KEY`; it writes
`tests/eval/proactive_conflict_last_run.json` and asserts against
`tests/eval/proactive_conflict_baseline.json` once that baseline is `measured`.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.proactive_agent import (
    ConfirmedDecision,
    ProactiveAgent,
    ProactiveDeps,
    ProactiveSettings,
    RagDecisionSearch,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.a2a_helpers import ScriptedRag, rag_passage
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink

EVAL_DIR = Path(__file__).parent / "eval"
CATEGORIES = {"contradicting", "superseding", "related", "unrelated"}


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (EVAL_DIR / "proactive_conflict_golden.json").read_text("utf-8")
    )
    return data


def score(cases: list[dict[str, Any]], confidences: list[float | None]) -> dict[str, float]:
    """`confidences[i]` is the emitted alert's confidence for case i, or None (no alert)."""

    def at(threshold: float, surfaced: bool) -> tuple[float, float]:
        def hit(c: float | None) -> bool:
            return c is not None and (c > threshold if surfaced else c >= threshold)

        tp = sum(1 for k, c in zip(cases, confidences, strict=True) if k["contradicts"] and hit(c))
        fp = sum(
            1 for k, c in zip(cases, confidences, strict=True) if not k["contradicts"] and hit(c)
        )
        pos = sum(1 for k in cases if k["contradicts"])
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / pos if pos else 0.0
        return precision, recall

    precision, recall = at(0.4, surfaced=False)
    s_precision, s_recall = at(0.7, surfaced=True)
    negatives = [c for c in cases if not c["contradicts"]]
    fps = sum(
        1
        for k, c in zip(cases, confidences, strict=True)
        if not k["contradicts"] and c is not None and c >= 0.4
    )
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "surfaced_precision": round(s_precision, 3),
        "surfaced_recall": round(s_recall, 3),
        "false_positive_rate": round(fps / len(negatives), 3) if negatives else 0.0,
    }


async def unlimited(_org: uuid.UUID) -> dict[str, Any]:
    return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9, "enforcement": "enforce"}


async def run_judge(router: LiteLLMRouter) -> tuple[list[dict[str, Any]], list[float | None]]:
    org, meeting, confirmer = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    cases = golden()["cases"]
    confidences: list[float | None] = []
    for case in cases:
        rag = ScriptedRag({confirmer: [rag_passage(case["candidate"], 0.8, kind="decision")]})
        agent = ProactiveAgent(
            PromptLoader(PROMPTS_DIR),
            router,
            deps=ProactiveDeps(decisions=RagDecisionSearch(rag)),
            settings=ProactiveSettings(all_plans=True),
        )
        alerts = await agent.check_decision(
            ConfirmedDecision(
                org_id=org,
                decision_id=uuid.uuid4(),
                meeting_id=meeting,
                content=case["decision"],
                rationale=case["rationale"],
                confirmed_by=confirmer,
            )
        )
        confidences.append(alerts[0].confidence if alerts else None)
    return cases, confidences


# --- deterministic -------------------------------------------------------------------------


def test_metrics_on_a_known_case():
    cases = [{"contradicts": True}, {"contradicts": True}, {"contradicts": False}] + [
        {"contradicts": False}
    ]
    m = score(cases, [0.9, 0.5, 0.6, None])
    assert m == {
        "precision": 0.667,
        "recall": 1.0,
        "surfaced_precision": 1.0,
        "surfaced_recall": 0.5,
        "false_positive_rate": 0.5,
    }
    assert score(cases, [None] * 4)["precision"] == 1.0  # no alert is never a false alarm


def test_the_golden_set_is_well_formed():
    cases = golden()["cases"]
    assert 8 <= len(cases) <= 12 and len({c["id"] for c in cases}) == len(cases)
    assert {c["category"] for c in cases} == CATEGORIES
    for c in cases:
        assert c["contradicts"] == (c["category"] == "contradicting"), c["id"]
        assert c["decision"].strip() and c["candidate"].strip()


def test_the_baseline_is_well_formed():
    b = json.loads((EVAL_DIR / "proactive_conflict_baseline.json").read_text("utf-8"))
    assert isinstance(b["measured"], bool) and b["golden_cases"] == len(golden()["cases"])
    for k in ("min_precision", "min_recall", "max_false_positive_rate"):
        assert 0 <= b[k] <= 1


class LabelJudge:
    """An offline judge that answers exactly as the golden labels say."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        user = kw["messages"][-1]["content"]
        case = next(c for c in golden()["cases"] if c["decision"] in user)
        entries = [{"candidate": 1, "confidence": 0.9, "rationale": "x"}]
        text = json.dumps({"conflicts": entries if case["contradicts"] else []})
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1}}


async def test_offline_scripted_run_through_the_real_agent():
    judge = LabelJudge()
    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), _NoSpendSink(), judge
    )
    cases, confidences = await run_judge(router)
    assert score(cases, confidences) == {
        "precision": 1.0,
        "recall": 1.0,
        "surfaced_precision": 1.0,
        "surfaced_recall": 1.0,
        "false_positive_rate": 0.0,
    }
    assert judge.calls == len(cases)


# --- live ----------------------------------------------------------------------------------


@pytest.mark.eval
async def test_conflict_judge_meets_the_baseline():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends up to 12 Sonnet calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    cases, confidences = await run_judge(router)
    m = score(cases, confidences)
    report = {
        "agent": "ProactiveAgent",
        "prompt": "proactive_conflict@v1",
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        **m,
        "cases": {
            c["id"]: {"category": c["category"], "contradicts": c["contradicts"], "alert": conf}
            for c, conf in zip(cases, confidences, strict=True)
        },
    }
    (EVAL_DIR / "proactive_conflict_last_run.json").write_text(json.dumps(report, indent=2))
    baseline = json.loads((EVAL_DIR / "proactive_conflict_baseline.json").read_text("utf-8"))
    if baseline["measured"]:
        assert m["precision"] >= baseline["min_precision"], report
        assert m["recall"] >= baseline["min_recall"], report
        assert m["false_positive_rate"] <= baseline["max_false_positive_rate"], report
