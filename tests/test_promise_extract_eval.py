"""Promise extraction (`promise_extract@vN`, promise ledger): unit checks of what the extractor
trusts, and its golden-set eval.

Scoring over `tests/eval/promise_extract_golden.json` (16 messages sent on Wed 2026-11-11):
- precision: extracted promises that match an expected one (keyword groups);
- recall: expected promises found;
- false_promise_rate: negative messages (requests, other people's promises, hedges, past work,
  questions, an injection) that produced any promise;
- due_date_accuracy / recipient_accuracy: over matched promises;
- never_record_leak: a message naming a never-record topic that reached the model or produced a
  promise (must be 0);
- parse_failure_rate: answers that are not the expected JSON.
Thresholds: `tests/eval/thresholds.json` (`promise_extract`).

Modes, as the shared harness (`tests/evalkit/harness.py`):
- offline (`pytest -m eval`, CI): replays `tests/eval/recordings/promise_extract@v1.json` through
  the real extractor, parser and router; no network, no spend. Until the first live run records
  it, the simulated answers in `tests/eval/promise_extract_simulated.json` are replayed instead
  (they are kept out of `recordings/`: the harness lets no new prompt pass on simulated answers).
- live: `EVAL_LIVE=1 uv run pytest -m eval tests/test_promise_extract_eval.py -k live -s`
  (15 Haiku calls); `EVAL_RECORD=1` as well writes the answers as the recording.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import EMPTY, CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.promises import (
    ExtractRequest,
    Person,
    PromiseExtractor,
    parse,
)
from tests.evalkit import scoring
from tests.evalkit.harness import TokenSink, router_for
from tests.evalkit.models import LiveModel, Model, ReplayModel, prompt_index, write_recordings

EVAL_DIR = Path(__file__).parent / "eval"
ORG = UUID("0e7a1000-0000-4000-8000-0000000000aa")
REF = "promise_extract@v1"


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "promise_extract_golden.json").read_text("utf-8"))
    return data


def org_context() -> CompanyContext:
    return CompanyContext(never_record_topics=tuple(golden()["never_record_topics"]))


class StaticOrgs:
    def __init__(self, ctx: CompanyContext) -> None:
        self.ctx = ctx
        self.asked = 0

    async def get_strict(self, _org: UUID) -> CompanyContext:
        self.asked += 1
        return self.ctx


def request_for(case: dict[str, Any], ref_date: date) -> ExtractRequest:
    return ExtractRequest(
        channel=case["channel"],
        text=case["text"],
        reference_date=ref_date,
        author_name=case["author"],
        people=tuple(
            Person(label, f"{label.lower()}@example.com") for label in case.get("people", [])
        ),
    )


def _matches(content: str, keywords: list[str]) -> bool:
    low = content.lower()
    return all(any(alt in low for alt in group.split("|")) for group in keywords)


def score(rows: list[dict[str, Any]], model: Model | None = None) -> dict[str, Any]:
    """rows: {case, extracted: [ExtractedPromise as dict], status, reached_model}."""
    extracted = matched = expected_n = 0
    due_ok = due_n = rec_ok = rec_n = 0
    negatives = negatives_hit = leaks = 0
    for row in rows:
        case, found = row["case"], row["extracted"]
        exp = case["expected"]
        expected_n += len(exp)
        extracted += len(found)
        if case.get("never_record"):
            leaks += int(bool(found) or row["reached_model"])
        if not exp:
            negatives += 1
            negatives_hit += int(bool(found))
        used: set[int] = set()
        for e in exp:
            hit = next(
                (
                    i
                    for i, f in enumerate(found)
                    if i not in used and _matches(f["content"], e["keywords"])
                ),
                None,
            )
            if hit is None:
                continue
            used.add(hit)
            matched += 1
            f = found[hit]
            due_n += 1
            due_ok += int(f["due_date"] == e["due_date"])
            rec_n += 1
            people = case.get("people", [])
            if e.get("recipient"):
                want = f"{people[e['recipient'] - 1].lower()}@example.com"
                rec_ok += int(f["recipient_email"] == want)
            elif e.get("recipient_label"):
                rec_ok += int(e["recipient_label"] in (f["recipient_label"] or "").lower())
            else:
                rec_ok += int(f["recipient_email"] is None)

    def rate(a: int, b: int, empty: float = 1.0) -> float:
        return round(a / b, 4) if b else empty

    return {
        "precision": rate(matched, extracted),
        "recall": rate(matched, expected_n),
        "false_promise_rate": rate(negatives_hit, negatives, 0.0),
        "due_date_accuracy": rate(due_ok, due_n),
        "recipient_accuracy": rate(rec_ok, rec_n),
        "never_record_leak": float(leaks),
        "parse_failure_rate": (
            round(model.parse_failure_rate("promise_extract"), 4) if model else 0.0
        ),
        "cases": len(rows),
    }


async def run(model: Model) -> dict[str, Any]:
    data = golden()
    ref_date = date.fromisoformat(data["reference_date"])
    orgs = StaticOrgs(org_context())
    extractor = PromiseExtractor(router_for(model, TokenSink()), PromptLoader(PROMPTS_DIR), orgs)
    rows = []
    for case in data["cases"]:
        model.case, model.persona = case["id"], ""
        before = len(model.calls)
        result = await extractor.extract(ORG, request_for(case, ref_date))
        rows.append(
            {
                "case": case,
                "status": result.status,
                "reached_model": len(model.calls) > before,
                "extracted": [p.as_dict() for p in result.promises],
            }
        )
    return score(rows, model)


# --- deterministic ---------------------------------------------------------------------------


def test_the_golden_set_is_well_formed() -> None:
    data = golden()
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids)) >= 15
    negatives = [c for c in data["cases"] if not c["expected"]]
    assert len(negatives) >= 6
    for case in data["cases"]:
        for e in case["expected"]:
            assert e["keywords"] and set(e) >= {"keywords", "due_date", "recipient"}
            if e["recipient"]:
                assert 1 <= e["recipient"] <= len(case["people"])


def test_the_prompt_is_registered_and_cacheable() -> None:
    loader = PromptLoader(PROMPTS_DIR)
    prompt = loader.load("promise_extract", loader.resolve("promise_extract", ORG))
    assert prompt.ref == REF and len(prompt.system) > 2400
    user = prompt.render_user(
        author="Maya",
        channel="Slack",
        weekday="Wednesday",
        reference_date="2026-11-11",
        people="1. Sam",
        never_record="(none)",
        context="",
        text='I\'ll send {it} by "Friday"',
    )
    assert 'I\'ll send {it} by "Friday"' in user and "Wednesday 2026-11-11" in user


REQ = ExtractRequest(
    channel="slack",
    text="@Sam I'll send you the revised deck by Friday. Also the layoffs plan.",
    reference_date=date(2026, 11, 11),
    people=(Person("Sam", "sam@example.com"),),
)


def _answer(*promises: dict[str, Any]) -> str:
    return json.dumps({"promises": list(promises)})


def test_the_answer_is_checked_not_trusted() -> None:
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("layoffs",)))
    good = {
        "content": "Send Sam the revised deck",
        "due_hint": "by Friday",
        "due_date": "2026-11-13",
        "recipient": 1,
        "quote": "I'll send you the revised deck by Friday",
        "confidence": 0.9,
    }
    [p] = parse(_answer(good), REQ, policy) or ()
    assert p.recipient_email == "sam@example.com" and p.recipient_label == "Sam"
    assert p.due_date == date(2026, 11, 13) and p.quote == good["quote"]
    # An invented quote is dropped, a far-off date is nulled, a bad recipient number is ignored.
    [q] = (
        parse(
            _answer(
                {**good, "quote": "I promise everything", "due_date": "2031-01-01", "recipient": 7}
            ),
            REQ,
            policy,
        )
        or ()
    )
    assert q.quote is None and q.due_date is None and q.recipient_email is None
    # Below the gate, or naming a never-record topic: dropped.
    assert parse(_answer({**good, "confidence": 0.65}), REQ, policy) == ()
    assert parse(_answer({**good, "content": "Draft the layoffs plan"}), REQ, policy) == ()
    # Not JSON: a parse failure (None), never a crash.
    assert parse("I think there is a promise here", REQ, policy) is None
    assert parse("```json\n" + _answer(good) + "\n```", REQ, EMPTY_POLICY) is not None


EMPTY_POLICY = NeverRecordPolicy.from_org(EMPTY)


class Scripted(Model):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.users: list[str] = []

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        content = kw["messages"][1]["content"]
        self.users.append(content if isinstance(content, str) else json.dumps(content))
        return self.text, {}


async def test_never_record_messages_never_reach_a_model() -> None:
    model = Scripted(_answer())
    orgs = StaticOrgs(CompanyContext(never_record_topics=("layoffs",)))
    extractor = PromiseExtractor(router_for(model), PromptLoader(PROMPTS_DIR), orgs)
    result = await extractor.extract(ORG, REQ)
    assert result.status == "never_record" and model.calls == [] and orgs.asked == 1


async def test_a_spent_budget_answers_budget_denied() -> None:
    class Blocked:
        async def complete(self, **_kw: Any) -> Any:
            raise BudgetBlocked(str(ORG))

    extractor = PromiseExtractor(Blocked(), PromptLoader(PROMPTS_DIR), None)  # type: ignore[arg-type]
    clean = ExtractRequest(channel="slack", text="I'll do it", reference_date=date(2026, 11, 11))
    result = await extractor.extract(ORG, clean)
    assert result.status == "budget_denied" and result.promises == ()


async def test_the_extractor_renders_people_by_number_and_never_their_email() -> None:
    model = Scripted(_answer())
    extractor = PromiseExtractor(router_for(model), PromptLoader(PROMPTS_DIR), None)
    clean = ExtractRequest(
        channel="email",
        text="I'll send it",
        reference_date=date(2026, 11, 11),
        people=(Person("Dana", "dana@example.com"),),
    )
    await extractor.extract(ORG, clean)
    assert len(model.calls) == 1 and model.calls[0].ref == REF
    assert "1. Dana" in model.users[0] and "dana@example.com" not in model.users[0]


def test_scoring_rewards_the_right_answers_and_catches_mistakes() -> None:
    data = golden()
    perfect = []
    for case in data["cases"]:
        found = []
        for e in case["expected"]:
            people = case.get("people", [])
            found.append(
                {
                    "content": " ".join(g.split("|")[0] for g in e["keywords"]),
                    "due_date": e["due_date"],
                    "recipient_email": (
                        f"{people[e['recipient'] - 1].lower()}@example.com"
                        if e.get("recipient")
                        else None
                    ),
                    "recipient_label": e.get("recipient_label"),
                }
            )
        perfect.append({"case": case, "extracted": found, "reached_model": False})
    report = score(perfect)
    assert report["precision"] == report["recall"] == 1.0
    assert report["false_promise_rate"] == 0.0 and report["never_record_leak"] == 0.0
    assert scoring.check({"promise_extract": report}, _thresholds()) == []
    wrong = [dict(r) for r in perfect]
    stray = {
        "content": "Send the contract",
        "due_date": None,
        "recipient_email": None,
        "recipient_label": None,
    }
    wrong[2] = {**wrong[2], "extracted": [stray]}
    wrong[11] = {**wrong[11], "reached_model": True}
    bad = score(wrong)
    assert bad["false_promise_rate"] > 0 and bad["never_record_leak"] == 1.0
    assert scoring.check({"promise_extract": bad}, _thresholds())


def _thresholds() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "thresholds.json").read_text("utf-8"))
    return {"promise_extract": data["promise_extract"]}


# --- the eval ----------------------------------------------------------------------------------


SIMULATED = EVAL_DIR / "promise_extract_simulated.json"


class SimulatedAnswers(Model):
    """Replays `promise_extract_simulated.json` with the replay model's prompt-sha check."""

    def __init__(self) -> None:
        super().__init__()
        self.data = json.loads(SIMULATED.read_text("utf-8"))

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        if ref != REF or self.data["prompt_sha256"] != prompt_index().sha(REF):
            self.errors.append(f"{ref}: the simulated answers were written for another prompt")
            return "", {}
        return str(self.data["entries"][key]), {}


@pytest.mark.eval
async def test_promise_extraction_meets_the_thresholds_offline() -> None:
    """The recorded answers once a live run has recorded them; until then the simulated ones
    (which exercise the extractor, parser, gates and scoring end to end, not the model)."""
    recorded = (EVAL_DIR / "recordings" / f"{REF}.json").exists()
    model: Model = ReplayModel() if recorded else SimulatedAnswers()
    report = await run(model)
    assert model.errors == [], model.errors
    failures = scoring.check({"promise_extract": report}, _thresholds())
    assert failures == [], (failures, report)


@pytest.mark.eval
async def test_promise_extraction_live() -> None:
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (15 Haiku calls)")
    from tests.test_decision_action_eval import _load_dotenv

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    model = LiveModel()
    report = await run(model)
    out = EVAL_DIR / "reports"
    out.mkdir(exist_ok=True)
    (out / "live-promise_extract.json").write_text(json.dumps(report, indent=2) + "\n")
    if os.environ.get("EVAL_RECORD") == "1":
        write_recordings(model.recorded, "recorded")
    failures = scoring.check({"promise_extract": report}, _thresholds())
    assert failures == [], (failures, report)
