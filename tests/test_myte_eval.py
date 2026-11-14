"""Ask Myte eval set and harness (mytheus-docs ADR-0047, "an 80-question eval set across the ten
areas ... gates prompt changes in CI").

`tests/eval/myte_golden.json`: 80 questions, eight in each of the plan's ten areas (meetings,
schedule, decisions, my work, people, inbox, company brain, your Myte, help, actions), including
abstention, refusal-to-act-without-confirmation and cross-user privacy cases; plus the ADR-0049
(v2) cases: asking a teammate's Myte and drafting peer answers (`peer_myte`), memories
(`memory`, including a one-off detail that must not become one), the new actions
(`actions_v2`), attachments (`attachments`) and person scope with its label (`person_scope`,
including production's "what are his work"). A case may carry the v2 request fields
(`memories`, `attachments`, `mode`, `peer`) and `expect.answer_must_not_contain`. Each case carries
the tools a good answer uses, what must happen (abstain, a proposal of a kind, a card, a degraded
answer, arguments a tool must receive, text the model must never be given) and plain-language
criteria. `world` is the fake bff's default data; a case overrides it in `fixture`.

Offline (always, in CI): every case runs through the real route, loop and router against the
fake bff and a scripted model that replays the case's `script`. It checks the plumbing, not the
model: the D8 event contract, steps paired, sources numbered, citations resolved through the
real source lines, `done.abstained` / `degraded`, proposals and cards, the person's token on
every bff call, no write tool offered to the model or reaching bff, thinking blocks kept across
rounds, scope arguments, and redaction / hidden lineage never reaching the model.

Live (`EVAL_LIVE=1` and `ANTHROPIC_API_KEY`, never in CI): the same cases against the real
model (Sonnet 5.5 and Haiku 5.5 via LiteLLM) and the same fake bff; it scores each case on the
checkable expectations and fails below `LIVE_PASS_RATE`. The criteria are for reading the
printed answers.

    EVAL_LIVE=1 uv run pytest -m eval tests/test_myte_eval.py -k live -s
    EVAL_LIVE=1 EVAL_CASES=memory-03,peer-myte-04 uv run pytest ... -k live -s   # a subset
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from services.rag_service import main as rag_main
from services.rag_service.myte.tools import LOCAL_TOOLS
from tests.conftest import sign_test_token
from tests.myte_kit import WRITE_TOOLS, FakeBff, ScriptedModel, check_stream, make_loop, parse_sse

pytestmark = pytest.mark.eval

GOLDEN = json.loads((Path(__file__).parent / "eval" / "myte_golden.json").read_text())
CASES = GOLDEN["cases"]
AREAS = {
    "meetings",
    "schedule",
    "decisions",
    "my_work",
    "people",
    "inbox",
    "company_brain",
    "your_myte",
    "help",
    "actions",
}
V2_AREAS = {"peer_myte", "memory", "actions_v2", "attachments", "person_scope"}
LIVE_PASS_RATE = 0.85
ORG = uuid.uuid4()


def test_the_set_has_80_questions_eight_per_area_with_the_hard_cases():
    assert len({c["id"] for c in CASES}) == len(CASES)
    v1 = [c for c in CASES if c["area"] in AREAS]
    assert len(v1) == 80
    assert Counter(c["area"] for c in v1) == {a: 8 for a in AREAS}
    assert {c["area"] for c in CASES} == AREAS | V2_AREAS
    expects = [c["expect"] for c in v1]
    assert sum(e["abstain"] for e in expects) >= 10
    assert sum(bool(e["proposal"]) for e in expects) >= 6
    privacy = [c for c in CASES if "privacy" in c["expect"]["criteria"].lower()]
    assert len(privacy) >= 5
    refusals = [c for c in CASES if "refus" in c["expect"]["criteria"].lower()]
    assert len(refusals) >= 2


def test_the_v2_cases_cover_every_new_kind_and_mode():
    v2 = [c for c in CASES if c["area"] in V2_AREAS]
    kinds = {c["expect"]["proposal"] for c in v2}
    assert {"ask_myte", "memory", "schedule_meeting"} <= kinds
    assert sum(c.get("mode") == "peer_answer" for c in v2) >= 3
    assert sum(bool(c.get("attachments")) for c in v2) >= 2
    labelled = [c for c in v2 if (c.get("scope") or {}).get("label")]
    assert any(c["question"] == "what are his work" for c in labelled)
    # a one-off detail that must not become a memory
    assert any(c["area"] == "memory" and not c["expect"]["proposal"] for c in v2)
    for c in CASES:
        assert c["question"].strip() or c.get("mode") == "peer_answer", c["id"]
        assert c["expect"]["criteria"].strip()
        scripted = {t["tool"] for turn in c["script"] for t in turn.get("tools", [])}
        assert scripted == set(c["expect"]["tools"]), c["id"]


def _body(case: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"question": case["question"], "want_title": True}
    for key in (
        "scope",
        "page",
        "persona",
        "timezone",
        "history",
        "memories",
        "attachments",
        "mode",
        "peer",
    ):
        if key in case:
            body[key] = case[key]
    return body


def _fixtures(case: dict[str, Any]) -> dict[str, Any]:
    return {**GOLDEN["world"], **case["fixture"]}


def _run(case: dict[str, Any], model: Any) -> tuple[list[tuple[str, dict[str, Any]]], FakeBff, str]:
    bff = FakeBff(fixtures=_fixtures(case))
    client = TestClient(rag_main.build_app(myte=make_loop(model, bff)))
    token = sign_test_token(ORG)
    res = client.post(
        "/api/rag/myte", json=_body(case), headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    return parse_sse(res.text), bff, f"Bearer {token}"


def _score(
    case: dict[str, Any],
    events: list[tuple[str, dict[str, Any]]],
    bff: FakeBff,
    auth: str,
    model_calls: list[dict[str, Any]],
    *,
    live: bool,
) -> list[str]:
    """Failed expectations for one case; [] is a pass."""
    exp = case["expect"]
    fails = list(check_stream(events))
    done = [d for n, d in events if n == "done"]
    if not done:
        return fails + [f"no done event: {[d for n, d in events if n == 'error']}"]
    text = "".join(d["text"] for n, d in events if n == "delta")
    if done[0]["abstained"] != exp["abstain"]:
        fails.append(f"abstained={done[0]['abstained']}, expected {exp['abstain']}")
    if not live and done[0]["degraded"] != exp["degraded"]:
        fails.append(f"degraded={done[0]['degraded']}, expected {exp['degraded']}")
    proposals = [d["kind"] for n, d in events if n == "proposal"]
    if exp["proposal"] and exp["proposal"] not in proposals:
        fails.append(f"expected a {exp['proposal']} proposal, got {proposals}")
    if not exp["proposal"] and proposals:
        fails.append(f"unexpected proposals {proposals}")
    cards = [d["kind"] for n, d in events if n == "card"]
    if not live and exp["card"] and exp["card"] not in cards:
        fails.append(f"expected a {exp['card']} card, got {cards}")
    if "[?]" in text:
        fails.append("a citation did not resolve to a source the model was given")
    if not exp["abstain"] and not done[0]["cited"] and not exp["proposal"]:
        fails.append("an answer that does not abstain must cite")
    if any(c.authorization != auth for c in bff.calls):
        fails.append("a bff call did not carry the person's own token")
    if {c.name for c in bff.calls} & WRITE_TOOLS:
        fails.append("a write tool reached bff")
    called = {c.name for c in bff.calls}
    expected_mcp = set(exp["tools"]) - LOCAL_TOOLS - WRITE_TOOLS
    if live:
        if expected_mcp and not exp["abstain"] and not (called & expected_mcp):
            fails.append(f"called {sorted(called)}, expected one of {sorted(expected_mcp)}")
    else:
        if called != expected_mcp:
            fails.append(f"bff got {sorted(called)}, expected {sorted(expected_mcp)}")
    for tool, args in exp["args"].items():
        got = [c.arguments for c in bff.calls if c.name == tool]
        if got and not any(all(a.get(k) == v for k, v in args.items()) for a in got):
            fails.append(f"{tool} args {got} lack {args}")
        if not got and not live:
            fails.append(f"{tool} was not called")
    for phrase in exp.get("answer_must_not_contain", []):
        if phrase.lower() in text.lower():
            fails.append(f"the answer says {phrase!r}")
    if case.get("mode") == "peer_answer":
        extra = {n for n, _ in events} & {"proposal", "card", "followups", "title"}
        if extra:
            fails.append(f"peer answers must not emit {sorted(extra)}")
    sent = json.dumps([c.get("messages") for c in model_calls], default=str)
    for secret in exp["model_must_not_see"]:
        if secret in sent:
            fails.append(f"the model was given {secret!r}")
        if secret in text:
            fails.append(f"the answer contains {secret!r}")
    return fails


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_offline_plumbing(case: dict[str, Any]):
    model = ScriptedModel([dict(t) for t in case["script"]])
    events, bff, auth = _run(case, model)
    fails = _score(case, events, bff, auth, model.calls, live=False)
    fails += model.problems
    assert fails == [], f"{case['id']}: {fails}"


@pytest.mark.skipif(
    os.environ.get("EVAL_LIVE") != "1", reason="live eval: set EVAL_LIVE=1 and ANTHROPIC_API_KEY"
)
def test_live_answers_meet_the_expectations():  # pragma: no cover - spends real tokens
    import litellm

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")

    calls: list[dict[str, Any]] = []

    async def recording(**kw: Any) -> Any:
        calls.append(kw)
        return await litellm.acompletion(**kw)

    by_area: dict[str, list[bool]] = defaultdict(list)
    failures: dict[str, list[str]] = {}
    # EVAL_CASES=id1,id2 re-runs only those cases (to check a fix without paying for the set);
    # the gate is only meaningful on the full set.
    only = {c for c in os.environ.get("EVAL_CASES", "").split(",") if c}
    cases = [c for c in CASES if not only or c["id"] in only]
    for case in cases:
        calls.clear()
        events, bff, auth = _run(case, recording)
        fails = _score(case, events, bff, auth, list(calls), live=True)
        by_area[case["area"]].append(not fails)
        text = "".join(d["text"] for n, d in events if n == "delta")
        print(f"\n[{case['id']}] {case['question']}\n  tools: {[c.name for c in bff.calls]}")
        print(f"  answer: {text}\n  criteria: {case['expect']['criteria']}")
        if fails:
            failures[case["id"]] = fails
            print(f"  FAIL: {fails}")
    total = sum(sum(v) for v in by_area.values()) / len(cases)
    print(f"failing: {sorted(failures)}")
    for area, results in sorted(by_area.items()):
        print(f"{area:15s} {sum(results)}/{len(results)}")
    print(f"pass rate {total:.0%}")
    assert total >= LIVE_PASS_RATE, failures
