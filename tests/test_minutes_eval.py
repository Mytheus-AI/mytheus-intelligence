"""Golden-set eval for the minutes writer (`minutes@vN`, minutes for everyone).

Scoring, over the minutes one ended meeting produces (shared + one personal set per person):
- **decision_recall**: expected decisions found in the shared minutes with the expected status;
- **item_recall**: expected open items found in the shared minutes with the expected owner;
- **question_recall**: expected unanswered questions found in the shared minutes;
- **promise_recall**: each person's expected promises found in THEIR personal minutes;
- **shared_promise_free**: no shared minutes carries a promise (must be 1.0);
- **leak_free**: no forbidden (never-record) text anywhere in any minutes (must be 1.0).

Deterministic tests always run. `test_minutes_meet_the_baseline` (`pytest -m eval`) runs the real
writer (Sonnet, one call per participant plus the shared one) per golden meeting, only with
`EVAL_LIVE=1` and an `ANTHROPIC_API_KEY`; it writes `tests/eval/minutes_last_run.json` and
asserts against `tests/eval/minutes_baseline.json` once that baseline is `measured`.
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
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.minutes import MinutesSource, MinutesWriter, parse_source
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink, keyword_match

EVAL_DIR = Path(__file__).parent / "eval"
LINE_MS = 20_000


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "minutes_golden.json").read_text("utf-8"))
    return data


def build_source(case: dict[str, Any]) -> tuple[MinutesSource, dict[str, uuid.UUID]]:
    """The golden meeting as core-api's minutes source; label -> synthetic user id."""
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
    body = {
        "meeting_id": str(uuid.uuid4()),
        "title": case["id"],
        "participants": [
            {
                "user_id": str(u),
                "name": label.replace("Participant ", "Person "),
                "attendance": "live",
            }
            for label, u in users.items()
        ],
        "turns": turns,
        "decisions": [{"id": str(uuid.uuid4()), **d} for d in case.get("recorded_decisions", [])],
    }
    source = parse_source(body)
    assert source is not None
    return source, users


def _all_text(minutes: dict[str, Any]) -> str:
    parts = [minutes.get("summary", "")]
    for key in ("decisions", "open_items", "unanswered_questions", "promises"):
        parts.extend(str(e.get("text", "")) for e in minutes.get(key, []))
    return " ".join(parts).lower()


def score(
    runs: list[tuple[dict[str, Any], dict[str, Any], dict[str, uuid.UUID]]],
) -> dict[str, Any]:
    """`runs`: (golden case, {"shared": payload, user id str: payload}, label -> user id)."""
    got = {"decision": [0, 0], "item": [0, 0], "question": [0, 0], "promise": [0, 0]}
    shared_free = leak_free = True
    for case, out, users in runs:
        shared = out.get("shared") or {}
        if shared.get("promises"):
            shared_free = False
        texts = " ".join(_all_text(m) for m in out.values())
        if any(f.lower() in texts for f in case.get("forbidden", [])):
            leak_free = False
        for d in case["decisions"]:
            got["decision"][1] += 1
            got["decision"][0] += any(
                keyword_match(e["text"], d["keywords"]) and e["status"] == d["status"]
                for e in shared.get("decisions", [])
            )
        for o in case["open_items"]:
            got["item"][1] += 1
            owner = str(users[o["owner"]]) if o.get("owner") else None
            got["item"][0] += any(
                keyword_match(e["text"], o["keywords"]) and e.get("owner_user_id") == owner
                for e in shared.get("open_items", [])
            )
        for q in case["questions"]:
            got["question"][1] += 1
            got["question"][0] += any(
                keyword_match(e["text"], q["keywords"])
                for e in shared.get("unanswered_questions", [])
            )
        for label, promises in case["promises"].items():
            mine = out.get(str(users[label])) or {}
            for p in promises:
                got["promise"][1] += 1
                got["promise"][0] += any(
                    keyword_match(e["text"], p["keywords"]) for e in mine.get("promises", [])
                )

    def rate(hit: int, total: int) -> float:
        return 1.0 if total == 0 else hit / total

    return {
        "decision_recall": rate(*got["decision"]),
        "item_recall": rate(*got["item"]),
        "question_recall": rate(*got["question"]),
        "promise_recall": rate(*got["promise"]),
        "shared_promise_free": 1.0 if shared_free else 0.0,
        "leak_free": 1.0 if leak_free else 0.0,
    }


# --- deterministic ----------------------------------------------------------------------------


def test_golden_set_is_well_formed():
    data = golden()
    assert len(data["cases"]) >= 3
    for case in data["cases"]:
        labels = set(case["participants"])
        assert {ln.partition(": ")[0] for ln in case["transcript"]} <= labels
        assert all(o["owner"] in labels for o in case["open_items"] if o.get("owner"))
        assert set(case["promises"]) <= labels
        assert all(
            d["status"] in ("agreed", "tentative", "deferred", "reversed")
            for d in case["decisions"]
        )


def test_scoring_rewards_the_right_minutes_and_catches_leaks():
    case = golden()["cases"][0]
    _, users = build_source(case)
    perfect_shared = {
        "summary": "Price up 8%.",
        "decisions": [
            {"text": "Raise the price by 8%", "status": "agreed"},
            {"text": "Reseller discount waits until October", "status": "deferred"},
        ],
        "open_items": [
            {"text": "Confirm the vendor split", "owner_user_id": str(users["Participant 2"])},
            {"text": "Draft the announcement", "owner_user_id": str(users["Participant 3"])},
        ],
        "unanswered_questions": [{"text": "Does legal review the terms?"}],
        "promises": [],
    }
    out = {
        "shared": perfect_shared,
        str(users["Participant 2"]): {"promises": [{"text": "Confirm the vendor split"}]},
        str(users["Participant 3"]): {"promises": [{"text": "Draft the announcement"}]},
    }
    assert set(score([(case, out, users)]).values()) == {1.0}
    worse = json.loads(json.dumps(out))
    worse["shared"]["decisions"][1]["status"] = "agreed"
    worse["shared"]["promises"] = [{"text": "x"}]
    m = score([(case, worse, users)])
    assert m["decision_recall"] == 0.5 and m["shared_promise_free"] == 0.0
    leaky_case = golden()["cases"][1]
    _, u2 = build_source(leaky_case)
    leak = {"shared": {"summary": "Layoffs are planned for March."}}
    assert score([(leaky_case, leak, u2)])["leak_free"] == 0.0


def test_never_record_spacing_keeps_the_hiring_decision():
    """The golden lines are 20 s apart: the window naming the topic and the next are dropped,
    the salary-band decision two windows later survives."""
    from libs.shared.never_record import NeverRecordPolicy
    from services.agent_runtime_service.minutes import quiet_turns

    case = golden()["cases"][1]
    source, _ = build_source(case)
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("layoffs",)))
    kept = " ".join(t.text for t in quiet_turns(source.turns, policy)).lower()
    assert "layoffs" not in kept and "reductions" not in kept
    assert "salary bands" in kept and "interviews" in kept


# --- live -------------------------------------------------------------------------------------


class _Source:
    def __init__(self, source: MinutesSource) -> None:
        self.value = source

    async def source(self, _org: uuid.UUID, _meeting: uuid.UUID) -> MinutesSource:
        return self.value


@pytest.mark.eval
async def test_minutes_meet_the_baseline():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends about 10 Sonnet calls)")
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
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    runs = []
    per_case: dict[str, Any] = {}
    for case in golden()["cases"]:
        source, users = build_source(case)
        org_ctx = CompanyContext(never_record_topics=tuple(case.get("never_record", [])))

        async def org_context(_org: uuid.UUID, ctx: CompanyContext = org_ctx) -> CompanyContext:
            return ctx

        writer = MinutesWriter(
            PromptLoader(PROMPTS_DIR), router, source=_Source(source), org_context=org_context
        )
        written = await writer.write(uuid.uuid4(), source.meeting_id, "eval")
        out = {("shared" if w.user_id is None else str(w.user_id)): w.payload for w in written}
        runs.append((case, out, users))
        per_case[case["id"]] = out
    m = score(runs)
    baseline = json.loads((EVAL_DIR / "minutes_baseline.json").read_text("utf-8"))
    report = {
        "prompt": baseline["prompt"],
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        "metrics": m,
        "cases": per_case,
    }
    (EVAL_DIR / "minutes_last_run.json").write_text(json.dumps(report, indent=2) + "\n")
    assert m["shared_promise_free"] == 1.0 and m["leak_free"] == 1.0
    if baseline.get("measured"):
        for key in ("decision_recall", "item_recall", "question_recall", "promise_recall"):
            assert m[key] >= baseline[f"min_{key}"], (key, m[key])
