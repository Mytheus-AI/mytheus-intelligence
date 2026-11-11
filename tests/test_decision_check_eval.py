"""Offline eval for the decision-conflict judge (`decision_conflict_check@v1`, rag-service).

Over `tests/eval/decision_conflict_check_golden.json` (one piece of writing per case, the
confirmed decisions retrieval would show the judge, and which one it contradicts or reopens),
each case runs through the REAL `DecisionCheckService` (real prompt, real `LiteLLMRouter`,
real parser, bands, quote check); only retrieval and the model's answer are scripted.

Where the answers come from, the harness's recorded / simulated pattern
(`tests/evalkit/harness.py`):
- offline (default, CI): `tests/eval/recordings/decision_conflict_check@v1.json`. Until a live
  run records real answers its `source` is `simulated`: `simulated_answer` below writes what a
  good but imperfect model plausibly returns (one reopening judged under the surface band, one
  reopening called a contradiction, two related cases answered in the silent band), and the
  recording must equal it, so a hand edit cannot drift. `prompt_sha256` must match the prompt
  file: a prompt change without a new recording fails here.
- live: `EVAL_LIVE=1` (and an `ANTHROPIC_API_KEY`) calls the real Sonnet judge once per case
  (15 calls) and writes `tests/eval/decision_conflict_check_last_run.json`; with
  `EVAL_RECORD=1` as well, the answers replace the recording (`source: recorded`). The
  coordinator runs it; builders never do.

Metrics: `precision` (surfaced conflicts that name the expected decision / all surfaced),
`recall` (positive cases whose expected decision is surfaced), `false_positive_rate` (negative
cases with any surfaced conflict; the headline gate: a false alarm interrupts someone at work),
`relation_accuracy` (among the hits, contradicts vs reopens), `quote_verbatim` (surfaced quotes
found in the writing). `decision_conflict_check_baseline.json` holds the gate; `measured` stays
false until the first live run sets the bounds from real numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.org_context import EMPTY, CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.rag_service.contracts import AnswerPassage, Citation, RagQueryRequest, RagResponse
from services.rag_service.decision_check import (
    CheckContext,
    CheckRequest,
    CheckResponse,
    DecisionCheckService,
)
from services.rag_service.domain import Caller
from services.rag_service.main import PROMPTS_DIR
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink

EVAL_DIR = Path(__file__).parent / "eval"
GOLDEN = EVAL_DIR / "decision_conflict_check_golden.json"
RECORDING = EVAL_DIR / "recordings" / "decision_conflict_check@v1.json"
BASELINE = EVAL_DIR / "decision_conflict_check_baseline.json"
PROMPT_REF = "decision_conflict_check@v1"
CATEGORIES = {"contradicts", "reopens", "supports", "related", "unrelated"}
NS = uuid.UUID("6f1d2c1e-6a51-4c4b-9e0e-0a1d0c15c0a1")
SEED = 20261111
ORG, USER = uuid.uuid5(NS, "org"), uuid.uuid5(NS, "user")


def golden() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(GOLDEN.read_text("utf-8"))["cases"]
    return cases


def prompt_sha() -> str:
    return hashlib.sha256((PROMPTS_DIR / f"{PROMPT_REF}.md").read_bytes()).hexdigest()


def decision_id(case: dict[str, Any], index: int) -> uuid.UUID:
    return uuid.uuid5(NS, f"{case['id']}:{index}")


# --- the simulated model -------------------------------------------------------------------


def _first_sentence(text: str, which: int = 0) -> str:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts[min(which, len(parts) - 1)]


def simulated_answer(case: dict[str, Any]) -> str:
    """Deterministic per case: what a good but imperfect judge returns."""
    rng = random.Random(f"{SEED}:{PROMPT_REF}:{case['id']}")
    entries: list[dict[str, Any]] = []
    category, expected = case["category"], case["expected"]
    if category == "contradicts":
        entries.append(
            {
                "candidate": expected,
                "relation": "contradicts",
                "confidence": round(rng.uniform(0.88, 0.96), 2),
                "quote": _first_sentence(case["text"], 1 if case["id"] == "c01-discount" else 0),
                "why": f"This goes against the decision: {case['candidates'][expected - 1][0]}",
            }
        )
    elif category == "reopens":
        if case["id"] == "r02-hiring":
            confidence = 0.66  # hedged: judged under the surface band (a miss)
        else:
            confidence = round(rng.uniform(0.78, 0.86), 2)
        entries.append(
            {
                # One reopening read as a contradiction: still a hit, a relation miss.
                "relation": "contradicts" if case["id"] == "r03-sso" else "reopens",
                "candidate": expected,
                "confidence": confidence,
                "quote": _first_sentence(case["text"], 1 if case["id"] == "r01-annual" else 0),
                "why": f"This was settled: {case['candidates'][expected - 1][0]}",
            }
        )
    elif category == "related" and case["id"] in ("x01-related-pricing", "x02-other-scope"):
        entries.append(
            {
                "candidate": 1,
                "relation": "reopens" if case["id"] == "x01-related-pricing" else "contradicts",
                "confidence": 0.45 if case["id"] == "x01-related-pricing" else 0.62,
                "quote": _first_sentence(case["text"]),
                "why": "Same topic as a confirmed decision.",
            }
        )
    text = json.dumps({"conflicts": entries})
    return f"```json\n{text}\n```" if rng.random() < 0.1 else text


def simulated_recording() -> dict[str, Any]:
    return {
        "prompt": PROMPT_REF,
        "prompt_sha256": prompt_sha(),
        "source": "simulated",
        "about": (
            "Simulated answers (test_decision_check_eval.simulated_answer) until a live run "
            "records real ones (EVAL_LIVE=1 EVAL_RECORD=1)."
        ),
        "entries": {c["id"]: simulated_answer(c) for c in golden()},
    }


def recording() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(RECORDING.read_text("utf-8"))
    return data


# --- running a case through the real service -----------------------------------------------


class GoldenRetriever:
    """Retrieval is not under test here: the case's decisions, as rag-service cites them."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse:
        out = []
        for i, (content, title, day) in enumerate(self.case["candidates"], start=1):
            out.append(
                AnswerPassage(
                    n=i,
                    text=content,
                    rerank_score=1.0 - i / 100,
                    citation=Citation(
                        kind="decision",
                        decision_id=decision_id(self.case, i),
                        meeting_id=uuid.uuid5(NS, f"meeting:{title}"),
                    ),
                    meeting_title=title,
                    occurred_at=datetime.fromisoformat(day).replace(tzinfo=UTC),
                )
            )
        return RagResponse(answer_passages=out)


class NoOrgRules:
    async def get_strict(self, org_id: uuid.UUID) -> CompanyContext:
        return EMPTY


async def unlimited(_org: uuid.UUID) -> dict[str, Any]:
    return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9, "enforcement": "enforce"}


def replay(answer: str) -> Any:
    async def completion(**_kw: Any) -> Any:
        return {
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    return completion


async def run_case(case: dict[str, Any], router: LiteLLMRouter) -> CheckResponse:
    svc = DecisionCheckService(
        retriever=GoldenRetriever(case),
        lexical=None,
        org_context=NoOrgRules(),
        judge=router,
        prompts=PromptLoader(PROMPTS_DIR),
    )
    req = CheckRequest(
        text=case["text"], source=case["source"], context=CheckContext(place_label=case["place"])
    )
    return await svc.check(Caller(org_id=ORG, user_id=USER), req)


async def run_offline() -> list[CheckResponse]:
    entries = recording()["entries"]
    out = []
    for case in golden():
        router = LiteLLMRouter(
            BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited),
            _NoSpendSink(),
            replay(entries[case["id"]]),
        )
        out.append(await run_case(case, router))
    return out


# --- scoring -------------------------------------------------------------------------------


def score(cases: list[dict[str, Any]], results: list[CheckResponse]) -> dict[str, float]:
    flags = hits = relation_ok = positives = hit_cases = negatives = false_cases = 0
    quotes = verbatim = 0
    for case, res in zip(cases, results, strict=True):
        expected = decision_id(case, case["expected"]) if case["expected"] else None
        flagged = {c.decision_id: c for c in res.conflicts}
        flags += len(flagged)
        for c in res.conflicts:
            quotes += 1
            verbatim += int(
                " ".join(c.quote.lower().split()) in " ".join(case["text"].lower().split())
            )
        if expected is None:
            negatives += 1
            false_cases += int(bool(flagged))
            continue
        positives += 1
        if expected in flagged:
            hits += 1
            hit_cases += 1
            relation_ok += int(flagged[expected].relation == case["relation"])
    return {
        "precision": round(hits / flags, 3) if flags else 1.0,
        "recall": round(hit_cases / positives, 3) if positives else 0.0,
        "false_positive_rate": round(false_cases / negatives, 3) if negatives else 0.0,
        "relation_accuracy": round(relation_ok / hit_cases, 3) if hit_cases else 0.0,
        "quote_verbatim": round(verbatim / quotes, 3) if quotes else 1.0,
    }


def gate(m: dict[str, float], b: dict[str, Any]) -> list[str]:
    fails = []
    if m["precision"] < b["min_precision"]:
        fails.append("precision")
    if m["recall"] < b["min_recall"]:
        fails.append("recall")
    if m["false_positive_rate"] > b["max_false_positive_rate"]:
        fails.append("false_positive_rate")
    if m["relation_accuracy"] < b["min_relation_accuracy"]:
        fails.append("relation_accuracy")
    if m["quote_verbatim"] < b["min_quote_verbatim"]:
        fails.append("quote_verbatim")
    return fails


# --- deterministic -------------------------------------------------------------------------


def test_the_golden_set_is_well_formed():
    cases = golden()
    assert 12 <= len(cases) <= 16 and len({c["id"] for c in cases}) == len(cases)
    assert {c["category"] for c in cases} == CATEGORIES
    for c in cases:
        positive = c["category"] in ("contradicts", "reopens")
        assert (c["expected"] is not None) == positive, c["id"]
        assert (c["relation"] == c["category"]) if positive else c["relation"] is None
        assert c["candidates"] and all(len(x) == 3 for x in c["candidates"])
        if positive:
            assert 1 <= c["expected"] <= len(c["candidates"])
        assert c["source"] in ("slack", "extension", "mcp", "app")


def test_the_recording_matches_the_prompt_and_the_simulation():
    rec = recording()
    assert rec["prompt"] == PROMPT_REF
    assert rec["prompt_sha256"] == prompt_sha(), "the prompt changed: re-record or re-simulate"
    assert set(rec["entries"]) == {c["id"] for c in golden()}
    if rec["source"] == "simulated":
        assert rec == simulated_recording(), "regenerate with simulated_recording()"
    else:
        assert rec["source"] == "recorded"


def test_the_baseline_is_well_formed():
    b = json.loads(BASELINE.read_text("utf-8"))
    assert isinstance(b["measured"], bool) and b["prompt"] == PROMPT_REF
    assert b["golden_cases"] == len(golden())
    for k in (
        "min_precision",
        "min_recall",
        "max_false_positive_rate",
        "min_relation_accuracy",
        "min_quote_verbatim",
    ):
        assert 0 <= b[k] <= 1


def test_metrics_on_a_known_case():
    case_pos = {"id": "p", "text": "We use MySQL.", "expected": 1, "relation": "contradicts"}
    case_neg = {"id": "n", "text": "Lunch on Thursday.", "expected": None, "relation": None}
    from services.rag_service.decision_check import Conflict

    def conflict(case: dict[str, Any], rel: str, quote: str) -> Conflict:
        return Conflict(
            decision_id=decision_id(case, 1),
            decision_content="x",
            quote=quote,
            relation=rel,  # type: ignore[arg-type]
            confidence=0.9,
            why="x",
        )

    results = [
        CheckResponse(status="ok", conflicts=[conflict(case_pos, "reopens", "we use mysql")]),
        CheckResponse(status="ok", conflicts=[conflict(case_neg, "contradicts", "invented")]),
    ]
    assert score([case_pos, case_neg], results) == {
        "precision": 0.5,
        "recall": 1.0,
        "false_positive_rate": 1.0,
        "relation_accuracy": 0.0,
        "quote_verbatim": 0.5,
    }


@pytest.mark.eval
async def test_offline_replay_through_the_real_service_meets_the_gate():
    cases = golden()
    results = await run_offline()
    m = score(cases, results)
    if recording()["source"] == "simulated":
        # The simulation's known imperfections, exactly: r02 missed (0.66), r03 called a
        # contradiction, the two related cases answered below the surface band.
        assert m == {
            "precision": 1.0,
            "recall": 0.857,
            "false_positive_rate": 0.0,
            "relation_accuracy": 0.833,
            "quote_verbatim": 1.0,
        }
    assert all(r.status == "ok" and r.prompt == PROMPT_REF for r in results)
    assert gate(m, json.loads(BASELINE.read_text("utf-8"))) == []


# --- live ----------------------------------------------------------------------------------


@pytest.mark.eval
async def test_the_judge_meets_the_baseline_live():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends 15 Sonnet calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    import litellm

    answers: dict[str, str] = {}
    current: dict[str, str] = {}

    async def recorded_completion(**kw: Any) -> Any:
        res = await litellm.acompletion(**kw)
        answers[current["id"]] = res["choices"][0]["message"]["content"] or ""
        return res

    sink = _NoSpendSink()
    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink, recorded_completion
    )
    cases, results = golden(), []
    for case in cases:
        current["id"] = case["id"]
        results.append(await run_case(case, router))
    m = score(cases, results)
    report = {
        "prompt": PROMPT_REF,
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        **m,
        "cases": {
            c["id"]: {
                "category": c["category"],
                "expected": c["expected"],
                "surfaced": [
                    {"relation": x.relation, "confidence": x.confidence, "quote": x.quote}
                    for x in r.conflicts
                ],
            }
            for c, r in zip(cases, results, strict=True)
        },
    }
    (EVAL_DIR / "decision_conflict_check_last_run.json").write_text(json.dumps(report, indent=2))
    if os.environ.get("EVAL_RECORD") == "1":
        rec = {
            "prompt": PROMPT_REF,
            "prompt_sha256": prompt_sha(),
            "source": "recorded",
            "about": "Real model answers recorded by the live eval (EVAL_LIVE=1 EVAL_RECORD=1).",
            "entries": answers,
        }
        RECORDING.write_text(json.dumps(rec, indent=2) + "\n")
    baseline = json.loads(BASELINE.read_text("utf-8"))
    if baseline["measured"]:
        assert gate(m, baseline) == [], report
