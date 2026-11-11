"""Golden-set eval for the in-meeting memory of `DecisionAgent` / `ActionAgent` (polish program,
"Memory inside the meeting"; decision_agent@v3, action_agent@v3).

Each case is a short stretch plus the meeting so far: the earlier stretches and the cards
already raised, with their status. A case passes when the emitted cards, after
`memory.reconcile`, have the expected outcome: `update` (the labelled card is updated, its content
matching the keywords), `new` (a new card matching the keywords, nothing updated),
`nothing_new` (no new card), `none` (nothing at all).

- deterministic (always run): the golden set is well formed, the judge on known answers, and
  the code guarantee: even a model that ignores the memory and repeats every card verbatim
  never re-proposes a dismissed or settled card and never duplicates an open one.
- `test_memory_meets_the_baseline` (`pytest -m eval`, `EVAL_LIVE=1` + `ANTHROPIC_API_KEY`):
  the real gate + extraction per case; writes `tests/eval/meeting_memory_last_run.json` and
  asserts `min_accuracy` once the baseline is `measured`.
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
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentState
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.candidates import CandidateAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.memory import (
    CONTEXT_KEY,
    LedgerEntry,
    MeetingMemory,
    reconcile,
    render_memory,
)
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink, _window, keyword_match

EVAL_DIR = Path(__file__).parent / "eval"
OUTCOMES = {"update", "new", "nothing_new", "none"}
_STATUS = {"open": "open", "settled": "confirmed", "dismissed": "dismissed"}


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "meeting_memory_golden.json").read_text("utf-8"))
    return data


def case_memory(case: dict[str, Any]) -> tuple[MeetingMemory, dict[str, str]]:
    """The case's memory and its label -> card id map (ids are fresh per run)."""
    kind = case["agent"]
    entries, ids = [], {}
    for n, item in enumerate(case["ledger"]):
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{case['id']}:{item['label']}"))
        ids[item["label"]] = cid
        entries.append(
            LedgerEntry(cid, kind, item["text"], status=_STATUS[item["status"]], seq=float(n))
        )
    return MeetingMemory(previous=list(case["earlier"]), entries=entries), ids


def judge(emitted: list[dict[str, Any]], expected: dict[str, Any], ids: dict[str, str]) -> bool:
    updates = [c for c in emitted if c.get("updates_candidate_id")]
    new = [c for c in emitted if not c.get("updates_candidate_id")]
    outcome = expected["outcome"]
    if outcome == "none":
        return not emitted
    if outcome == "nothing_new":
        return not new
    if outcome == "new":
        return any(keyword_match(c["content"], expected["keywords"]) for c in new)
    target = ids[expected["label"]]
    return any(
        c["updates_candidate_id"] == target and keyword_match(c["content"], expected["keywords"])
        for c in updates
    ) and not any(keyword_match(c["content"], expected["keywords"]) for c in new)


# --- deterministic ---------------------------------------------------------------------------


def test_the_golden_set_is_well_formed():
    cases = golden()["cases"]
    assert len({c["id"] for c in cases}) == len(cases) >= 12
    assert {c["agent"] for c in cases} == {"decision", "action"}
    assert {c["expected"]["outcome"] for c in cases} == OUTCOMES
    for c in cases:
        labels = {i["label"] for i in c["ledger"]}
        assert all(i["status"] in _STATUS for i in c["ledger"]), c["id"]
        if c["expected"]["outcome"] == "update":
            assert c["expected"]["label"] in labels, c["id"]
        if c["expected"]["outcome"] in ("update", "new"):
            assert all(k == k.lower() for k in c["expected"]["keywords"]), c["id"]
        block, mapped = render_memory(case_memory(c)[0], c["agent"])
        assert set(mapped) == labels and (block or not c["ledger"] and not c["earlier"])
    b = json.loads((EVAL_DIR / "meeting_memory_baseline.json").read_text("utf-8"))
    assert b["golden_cases"] == len(cases) and 0 < b["min_accuracy"] <= 1


def test_the_judge_on_known_answers():
    ids = {"D1": "x"}
    upd = {"outcome": "update", "label": "D1", "keywords": ["beta"]}
    assert judge([{"content": "Ship the beta", "updates_candidate_id": "x"}], upd, ids)
    assert not judge([{"content": "Ship the beta"}], upd, ids)
    assert judge([], {"outcome": "none"}, ids)
    assert judge([{"content": "x", "updates_candidate_id": "x"}], {"outcome": "nothing_new"}, ids)
    assert not judge([{"content": "x"}], {"outcome": "nothing_new"}, ids)


def test_a_model_that_ignores_the_memory_still_never_repeats_a_closed_card():
    """The worst model: it re-proposes every card of the ledger verbatim, as new. The code
    guarantee alone (`reconcile`) must turn that into nothing for dismissed and settled cards
    and into no duplicate for open ones."""
    for case in golden()["cases"]:
        mem, ids = case_memory(case)
        raw = [
            {"id": f"new-{n}", "content": item["text"], "confidence": 0.9}
            for n, item in enumerate(case["ledger"])
        ]
        out = reconcile(raw, case["agent"], mem, {}, [None] * len(raw))
        assert not [c for c in out if not c.get("updates_candidate_id")], case["id"]
        closed = {ids[i["label"]] for i in case["ledger"] if i["status"] != "open"}
        assert not [c for c in out if c["id"] in closed], case["id"]


# --- live ------------------------------------------------------------------------------------


@pytest.mark.eval
async def test_memory_meets_the_baseline():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends up to 12 Haiku + 12 Sonnet calls)")
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

    data = golden()
    org, meeting, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    loader = PromptLoader(PROMPTS_DIR)
    reference = date.fromisoformat(data["reference_date"])
    agents: dict[str, CandidateAgent] = {
        "decision": DecisionAgent(loader, router),
        "action": ActionAgent(loader, router, today=lambda: reference),
    }
    for agent in agents.values():
        await agent.spawn(org_id=org, meeting_id=meeting, user_id=meeting)
    passed, per_case = 0, {}
    for i, case in enumerate(data["cases"]):
        mem, ids = case_memory(case)
        w, _ = _window(org, meeting, i, case["window"])
        state = AgentState(
            meeting_id=meeting, user_id=user, chunk=w, context={CONTEXT_KEY: mem}, org_id=org
        )
        result = await agents[case["agent"]].process(state)
        emitted = list(result.payload["candidates"]) if result is not None else []
        ok = judge(emitted, case["expected"], ids)
        passed += ok
        per_case[case["id"]] = {
            "ok": ok,
            "expected": case["expected"],
            "emitted": [
                {k: c.get(k) for k in ("content", "updates_candidate_id", "quote", "confidence")}
                for c in emitted
            ],
        }
    accuracy = passed / len(data["cases"])
    report = {
        "prompts": [f"{k}@{loader.resolve(k, org)}" for k in ("decision_agent", "action_agent")],
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        "accuracy": accuracy,
        "cases": per_case,
    }
    (EVAL_DIR / "meeting_memory_last_run.json").write_text(json.dumps(report, indent=2))
    baseline = json.loads((EVAL_DIR / "meeting_memory_baseline.json").read_text("utf-8"))
    if baseline["measured"]:
        assert accuracy >= baseline["min_accuracy"], report
