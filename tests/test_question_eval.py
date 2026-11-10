"""Golden-set eval for `QuestionAgent` (Plan 7-9 section 3.10 step 3, doc 08 section 6.3:
question usefulness, "Ask-rate proxy >= baseline").

Scoring, over SURFACED suggestions only (the ones a user actually sees):
- a suggestion is **useful** if it addresses a still-unaddressed gap of its window (its question
  contains every keyword group of the gap; greedy, highest confidence first);
- **ask_rate** = useful / surfaced: the proxy for how often a user would press Ask;
- **gap_recall** = gaps addressed / gaps: how many real gaps get a question at all;
- **noise_rate** = no-gap windows with any surfaced suggestion / no-gap windows: interruptions
  where nothing needed asking;
- **priority_valid**: every emitted suggestion has priority in {1,2,3} (always 1.0 by
  construction, asserted so a parser change cannot regress it).

Deterministic tests always run. `test_question_agent_meets_the_baseline` (`pytest -m eval`) runs
the real graph (Haiku gate, Sonnet proposal) once per golden window, only with `EVAL_LIVE=1` and
an `ANTHROPIC_API_KEY`; it writes `tests/eval/question_last_run.json` and asserts against
`tests/eval/question_baseline.json` once that baseline is `measured`. The cooldown is disabled
for the eval: every golden window is independent.
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
from libs.shared.state import AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents.question_agent import QuestionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink, keyword_match

EVAL_DIR = Path(__file__).parent / "eval"
_LINE = re.compile(r"^(You|Participant \d+): (.+)$")


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "question_golden.json").read_text("utf-8"))
    return data


def score(windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> dict[str, Any]:
    """`windows` is (emitted suggestions, expected gaps) per golden case."""
    surfaced_total = useful = gaps = addressed = 0
    empty_windows = noisy = 0
    priorities_ok = True
    for emitted, expected in windows:
        priorities_ok &= all(s.get("priority") in (1, 2, 3) for s in emitted)
        shown = sorted((s for s in emitted if s["surfaced"]), key=lambda s: -float(s["confidence"]))
        free = list(range(len(expected)))
        for s in shown:
            hit = next(
                (i for i in free if keyword_match(s["question"], expected[i]["keywords"])), None
            )
            if hit is not None:
                free.remove(hit)
                useful += 1
        surfaced_total += len(shown)
        gaps += len(expected)
        addressed += len(expected) - len(free)
        if not expected:
            empty_windows += 1
            noisy += bool(shown)
    return {
        "surfaced": surfaced_total,
        "useful": useful,
        "ask_rate": useful / surfaced_total if surfaced_total else 0.0,
        "gap_recall": addressed / gaps if gaps else 0.0,
        "noise_rate": noisy / empty_windows if empty_windows else 0.0,
        "priority_valid": priorities_ok,
    }


# --- deterministic ---------------------------------------------------------------------------


def _s(
    question: str, confidence: float = 0.9, surfaced: bool = True, priority: int = 2
) -> dict[str, Any]:
    return {
        "question": question,
        "confidence": confidence,
        "surfaced": surfaced,
        "priority": priority,
    }


def test_metrics_on_a_known_case():
    windows = [
        # one gap; a useful surfaced question and a useless surfaced one
        ([_s("What does it cost?"), _s("Is everyone happy?", 0.8)], [{"keywords": ["cost"]}]),
        # one gap; only a silent suggestion addresses it: not counted
        ([_s("Who owns it?", 0.6, surfaced=False)], [{"keywords": ["who"]}]),
        # no gap, a surfaced suggestion: noise
        ([_s("Any risks?")], []),
        # no gap, nothing surfaced
        ([_s("Any risks?", 0.5, surfaced=False)], []),
    ]
    m = score(windows)
    assert (m["surfaced"], m["useful"]) == (3, 1)
    assert m["ask_rate"] == pytest.approx(1 / 3)
    assert m["gap_recall"] == pytest.approx(1 / 2)
    assert m["noise_rate"] == pytest.approx(1 / 2)
    assert m["priority_valid"] is True
    assert score([([_s("x", priority=4)], [])])["priority_valid"] is False
    assert score([([], [])]) == {
        "surfaced": 0,
        "useful": 0,
        "ask_rate": 0.0,
        "gap_recall": 0.0,
        "noise_rate": 0.0,
        "priority_valid": True,
    }


def test_the_golden_set_is_well_formed():
    cases = golden()["cases"]
    assert len({c["id"] for c in cases}) == len(cases) >= 12
    empty = sum(1 for c in cases if not c["gaps"])
    assert 0.35 <= empty / len(cases) <= 0.65, "about half the windows need no question"
    for c in cases:
        assert c["profile"]["role"], c["id"]
        others = []
        for line in c["window"].splitlines():
            m = _LINE.match(line)
            assert m, (c["id"], line)
            if m.group(1) != "You" and m.group(1) not in others:
                others.append(m.group(1))
        assert others == [f"Participant {i + 1}" for i in range(len(others))], c["id"]
        for gap in c["gaps"]:
            assert gap["keywords"] and all(k == k.lower() for k in gap["keywords"])


def test_the_baseline_is_well_formed():
    b = json.loads((EVAL_DIR / "question_baseline.json").read_text("utf-8"))
    assert isinstance(b["measured"], bool)
    assert 0 <= b["min_ask_rate"] <= 1 and 0 <= b["min_gap_recall"] <= 1
    assert 0 <= b["max_noise_rate"] <= 1
    assert b["golden_windows"] == len(golden()["cases"])


# --- live ------------------------------------------------------------------------------------


def _window(
    org: uuid.UUID, meeting: uuid.UUID, me: uuid.UUID, index: int, text: str
) -> TranscriptWindow:
    speakers: dict[str, str] = {"You": str(me)}
    turns = []
    for n, line in enumerate(text.splitlines()):
        m = _LINE.match(line)
        assert m is not None
        sid = speakers.setdefault(m.group(1), str(uuid.uuid4()))
        turns.append(Turn(sid, m.group(2), n * 5_000, n * 5_000 + 4_000, [f"c{index}-{n}"]))
    start = index * 30_000
    return TranscriptWindow(org, meeting, f"{meeting}:{start}", start, start + 30_000, turns)


@pytest.mark.eval
async def test_question_agent_meets_the_baseline(monkeypatch):
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends up to 14 Haiku + 14 Sonnet calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    monkeypatch.setenv("QUESTION_COOLDOWN_WINDOWS", "0")

    async def unlimited(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": 0,
            "token_quota": 10**9,
            "hard_cap": 10**9,
            "enforcement": "enforce",
        }

    data = golden()
    org, meeting, me = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    agent = QuestionAgent(PromptLoader(PROMPTS_DIR), router)
    await agent.spawn(org_id=org, meeting_id=meeting, user_id=me)

    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    per_case: dict[str, Any] = {}
    for i, case in enumerate(data["cases"]):
        w = _window(org, meeting, me, i, case["window"])
        state = AgentState(
            meeting_id=meeting, user_id=me, chunk=w, context=case["profile"], org_id=org
        )
        before = len(sink.tiers)
        result = await agent.process(state)
        emitted = list(result.payload["suggestions"]) if result is not None else []
        windows.append((emitted, case["gaps"]))
        per_case[case["id"]] = {
            "gate": result.relevance_score if result is not None else None,
            "tiers": sink.tiers[before:],
            "gaps": len(case["gaps"]),
            "emitted": [
                {k: s[k] for k in ("question", "priority", "confidence", "surfaced")}
                for s in emitted
            ],
        }

    m = score(windows)
    report = {
        "agent": "QuestionAgent",
        "prompt": agent.prompt.ref if agent.prompt else "",
        "gate_prompt": agent.gate_prompt.ref if agent.gate_prompt else "",
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        **m,
        "cases": per_case,
    }
    (EVAL_DIR / "question_last_run.json").write_text(json.dumps(report, indent=2))
    assert m["priority_valid"], report
    baseline = json.loads((EVAL_DIR / "question_baseline.json").read_text("utf-8"))
    if baseline["measured"]:
        assert m["ask_rate"] >= baseline["min_ask_rate"], report
        assert m["gap_recall"] >= baseline["min_gap_recall"], report
        assert m["noise_rate"] <= baseline["max_noise_rate"], report
