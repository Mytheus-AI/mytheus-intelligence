"""Golden-set eval for `ContextGapAgent`'s shared gate (P10; doc 08 section 6.3 style).

The gate decides, once per window for every participant, whether a reference to prior context
is a real gap and what to search for. Over `tests/eval/context_gap_golden.json`:

- **gap_recall** = gap windows the gate opens for / gap windows (the detector must hit first);
- **false_gap_rate** = no-gap windows it opens for / no-gap windows (each one costs retrieval for
  every participant and possibly A2A round-trips);
- **query_hit_rate** = opened gap windows whose standalone query contains every keyword group
  (the query names the topic, so retrieval can find it).

Deterministic tests always run, including an offline scripted run through the real detector,
gate prompt, router and parser. `test_context_gap_gate_meets_the_baseline` (`pytest -m eval`)
runs the real Haiku gate once per golden window, only with `EVAL_LIVE=1` and an
`ANTHROPIC_API_KEY`; it writes `tests/eval/context_gap_last_run.json` and asserts against
`tests/eval/context_gap_baseline.json` once that baseline is `measured`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.agents.context_gap_agent import (
    ContextGapAgent,
    ContextGapDeps,
    ContextGapSettings,
    GateVerdict,
    detect,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink, keyword_match

EVAL_DIR = Path(__file__).parent / "eval"
_LINE = re.compile(r"^(Participant \d+): (.+)$")
THRESHOLD = ContextGapSettings().gate_threshold


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "context_gap_golden.json").read_text("utf-8"))
    return data


def score(
    cases: list[dict[str, Any]], verdicts: list[GateVerdict | None], threshold: float = THRESHOLD
) -> dict[str, Any]:
    gaps = [(c, v) for c, v in zip(cases, verdicts, strict=True) if c["is_gap"]]
    clear = [(c, v) for c, v in zip(cases, verdicts, strict=True) if not c["is_gap"]]
    opened_gaps = [(c, v) for c, v in gaps if v is not None and v.passed(threshold)]
    false_gaps = [c for c, v in clear if v is not None and v.passed(threshold)]
    hits = [c for c, v in opened_gaps if v is not None and keyword_match(v.query, c["keywords"])]
    return {
        "gap_recall": len(opened_gaps) / len(gaps) if gaps else 0.0,
        "false_gap_rate": len(false_gaps) / len(clear) if clear else 0.0,
        "query_hit_rate": len(hits) / len(opened_gaps) if opened_gaps else 0.0,
    }


def window(org: uuid.UUID, meeting: uuid.UUID, index: int, text: str) -> TranscriptWindow:
    speakers: dict[str, str] = {}
    turns = []
    for n, line in enumerate(text.splitlines()):
        m = _LINE.match(line)
        assert m is not None, line
        sid = speakers.setdefault(m.group(1), str(uuid.uuid4()))
        turns.append(Turn(sid, m.group(2), n * 5_000, n * 5_000 + 4_000, [str(uuid.uuid4())]))
    start = index * 30_000
    return TranscriptWindow(org, meeting, f"{meeting}:{start}", start, start + 30_000, turns)


async def unlimited(_org: uuid.UUID) -> dict[str, Any]:
    return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9, "enforcement": "enforce"}


async def run_gate(router: LiteLLMRouter) -> tuple[list[dict[str, Any]], list[GateVerdict | None]]:
    org, meeting = uuid.uuid4(), uuid.uuid4()
    agent = ContextGapAgent(
        PromptLoader(PROMPTS_DIR), router, deps=ContextGapDeps(), settings=ContextGapSettings()
    )
    await agent.spawn(org_id=org, meeting_id=meeting, user_id=meeting)
    cases = golden()["cases"]
    verdicts = [
        await agent.assess(window(org, meeting, i, c["window"])) for i, c in enumerate(cases)
    ]
    return cases, verdicts


# --- deterministic ---------------------------------------------------------------------------


def test_metrics_on_a_known_case():
    cases = [
        {"is_gap": True, "keywords": ["pricing"]},
        {"is_gap": True, "keywords": ["auth"]},
        {"is_gap": False, "keywords": []},
        {"is_gap": False, "keywords": []},
    ]
    verdicts: list[GateVerdict | None] = [
        GateVerdict(True, "Q3 pricing decision", 0.9, None),  # opened, query hits
        GateVerdict(True, "the thing from before", 0.8, None),  # opened, query misses
        GateVerdict(True, "build times", 0.6, None),  # a false gap
        GateVerdict(True, "agenda", 0.3, None),  # below the threshold: closed
    ]
    m = score(cases, verdicts)
    assert m == {"gap_recall": 1.0, "false_gap_rate": 0.5, "query_hit_rate": 0.5}
    assert score(cases, [None] * 4) == {
        "gap_recall": 0.0,
        "false_gap_rate": 0.0,
        "query_hit_rate": 0.0,
    }


def test_the_golden_set_is_well_formed():
    cases = golden()["cases"]
    assert len({c["id"] for c in cases}) == len(cases) >= 16
    gaps = [c for c in cases if c["is_gap"]]
    assert 0.4 <= len(gaps) / len(cases) <= 0.6
    org, meeting = uuid.uuid4(), uuid.uuid4()
    for i, c in enumerate(cases):
        labels: list[str] = []
        for line in c["window"].splitlines():
            m = _LINE.match(line)
            assert m, (c["id"], line)
            if m.group(1) not in labels:
                labels.append(m.group(1))
        assert labels == [f"Participant {n + 1}" for n in range(len(labels))], c["id"]
        assert bool(c["keywords"]) == c["is_gap"], c["id"]
        assert all(k == k.lower() for k in c["keywords"]), c["id"]
        if c["is_gap"]:  # every real gap must reach the gate
            assert detect(window(org, meeting, i, c["window"])) is not None, c["id"]
    # most no-gap windows are hard ones: they trip the detector and only the gate can say no
    tripped = [
        c
        for i, c in enumerate(cases)
        if not c["is_gap"] and detect(window(org, meeting, i, c["window"]))
    ]
    assert len(tripped) >= 5


def test_the_baseline_is_well_formed():
    b = json.loads((EVAL_DIR / "context_gap_baseline.json").read_text("utf-8"))
    assert isinstance(b["measured"], bool)
    assert b["golden_windows"] == len(golden()["cases"])
    for k in ("min_gap_recall", "max_false_gap_rate", "min_query_hit_rate"):
        assert 0 <= b[k] <= 1


class LabelGate:
    """An offline model that answers the gate exactly as the golden labels say."""

    def __init__(self, always_gap: bool = False) -> None:
        self.always_gap = always_gap
        self.calls = 0

    async def __call__(self, **kw: Any) -> Any:
        self.calls += 1
        user = kw["messages"][-1]["content"]
        case = next(c for c in golden()["cases"] if c["window"].splitlines()[0] in user)
        gap = self.always_gap or case["is_gap"]
        query = " ".join(g.split("|")[0] for g in case["keywords"]) + " earlier decision"
        text = json.dumps(
            {"is_gap": gap, "query": query if gap else "", "score": 0.9 if gap else 0.1}
        )
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1}}


async def test_offline_scripted_run_through_the_real_gate():
    llm = LabelGate()
    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), _NoSpendSink(), llm
    )
    cases, verdicts = await run_gate(router)
    assert score(cases, verdicts) == {
        "gap_recall": 1.0,
        "false_gap_rate": 0.0,
        "query_hit_rate": 1.0,
    }
    reached = sum(v is not None for v in verdicts)
    assert llm.calls == reached < len(cases)  # windows without a reference never reach the model

    noisy = LabelGate(always_gap=True)
    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), _NoSpendSink(), noisy
    )
    cases, verdicts = await run_gate(router)
    m = score(cases, verdicts)
    clear = [c for c in cases if not c["is_gap"]]
    assert m["false_gap_rate"] == pytest.approx(
        sum(v is not None for c, v in zip(cases, verdicts, strict=True) if not c["is_gap"])
        / len(clear)
    )


# --- live ------------------------------------------------------------------------------------


@pytest.mark.eval
async def test_context_gap_gate_meets_the_baseline():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends up to 18 Haiku calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    cases, verdicts = await run_gate(router)
    m = score(cases, verdicts)
    report = {
        "agent": "ContextGapAgent",
        "gate_prompt": "context_gap_gate@v1",
        "threshold": THRESHOLD,
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        **m,
        "cases": {
            c["id"]: {
                "is_gap": c["is_gap"],
                "reached_gate": v is not None,
                "gate": (
                    None if v is None else {"is_gap": v.is_gap, "query": v.query, "score": v.score}
                ),
            }
            for c, v in zip(cases, verdicts, strict=True)
        },
    }
    (EVAL_DIR / "context_gap_last_run.json").write_text(json.dumps(report, indent=2))
    baseline = json.loads((EVAL_DIR / "context_gap_baseline.json").read_text("utf-8"))
    if baseline["measured"]:
        assert m["gap_recall"] >= baseline["min_gap_recall"], report
        assert m["false_gap_rate"] <= baseline["max_false_gap_rate"], report
        assert m["query_hit_rate"] >= baseline["min_query_hit_rate"], report
