"""Golden-set eval for the `PersonalListenerAgent` relevance pre-gate (Plan 7-9 section 3.3 step 5,
doc 08 section 7: relevance precision/recall against role labels).

Two parts:
- `test_metrics_*`: the scoring arithmetic, deterministic, always run.
- `test_pregate_meets_the_baseline` (`pytest -m eval`): calls the REAL triage model once per
  golden case (14 Haiku calls, at most 20 output tokens each) and asserts precision and recall
  at the default threshold reach `tests/eval/personal_listener_baseline.json`. It runs only when
  `EVAL_LIVE=1`, because the suite never spends money by default (CLAUDE.md). It reads
  `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` from the environment or the repo's `.env`.

A prompt change is promoted only if this eval passes against the CURRENT default's measured
numbers (written into the baseline file after a recorded run).
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
from services.agent_runtime_service.agents.personal_listener_agent import (
    GATE_KEY,
    gate_values,
    parse_relevance,
    relevance_threshold,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR

EVAL_DIR = Path(__file__).parent / "eval"
ROOT = Path(__file__).resolve().parents[1]


def metrics(predicted: list[bool], labels: list[bool]) -> dict[str, float]:
    tp = sum(p and y for p, y in zip(predicted, labels, strict=True))
    fp = sum(p and not y for p, y in zip(predicted, labels, strict=True))
    fn = sum(y and not p for p, y in zip(predicted, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pass_rate": sum(predicted) / len(predicted),
    }


def golden() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(
        (EVAL_DIR / "personal_listener_golden.json").read_text(encoding="utf-8")
    )["cases"]
    return cases


def test_metrics_on_a_known_confusion_matrix():
    m = metrics([True, True, False, False, True], [True, False, True, False, True])
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["pass_rate"] == pytest.approx(0.6)
    assert metrics([False, False], [True, False])["precision"] == 0.0


def test_the_golden_set_is_balanced_and_well_formed():
    cases = golden()
    assert len({c["id"] for c in cases}) == len(cases) >= 12
    positives = sum(c["relevant"] for c in cases)
    assert 0.3 <= positives / len(cases) <= 0.7
    for c in cases:
        assert c["role"] and c["window"] and isinstance(c["relevant"], bool)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


class _NoSpendSink:
    def __init__(self) -> None:
        self.tokens = 0

    async def usage(self, **kw: Any) -> None:
        self.tokens += int(kw["tokens"])


@pytest.mark.eval
async def test_pregate_meets_the_baseline():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends ~14 Haiku calls)")
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

    org = uuid.uuid4()
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    loader = PromptLoader(PROMPTS_DIR)
    prompt = loader.load(GATE_KEY, loader.resolve(GATE_KEY, org))
    threshold = relevance_threshold()

    scores: list[float] = []
    for case in golden():
        out = await router.complete(
            tier="triage",
            system=prompt.system,
            user=prompt.render_user(
                **gate_values(
                    {
                        "role": case["role"],
                        "skills": case["skills"],
                        "listen_for": case.get("listen_for", []),
                        "ignore_topics": case.get("ignore_topics", []),
                    }
                ),
                window=case["window"],
            ),
            org_id=org,
            user_id=None,
            max_tokens=20,
        )
        scores.append(parse_relevance(out.text))

    labels = [c["relevant"] for c in golden()]
    m = metrics([s >= threshold for s in scores], labels)
    report = {
        "prompt": prompt.ref,
        "threshold": threshold,
        "tokens": sink.tokens,
        "scores": dict(zip([c["id"] for c in golden()], scores, strict=True)),
        **m,
    }
    (EVAL_DIR / "personal_listener_last_run.json").write_text(json.dumps(report, indent=2))
    baseline = json.loads((EVAL_DIR / "personal_listener_baseline.json").read_text())
    assert m["precision"] >= baseline["min_precision"], report
    assert m["recall"] >= baseline["min_recall"], report
