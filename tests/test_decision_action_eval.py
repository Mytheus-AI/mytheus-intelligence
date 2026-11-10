"""Golden-set eval for `DecisionAgent` and `ActionAgent` (Plan 7-9 section 3.5 step 4, doc 08
section 6.3: extraction F1 + confidence calibration via Brier).

Scoring. Per window, the agent's emitted candidates (silent and surfaced; dropped ones are never
emitted) are matched greedily, highest confidence first, to the expected items: a candidate
matches the first still-unmatched expected item whose every keyword group appears in its
lowercased content (a group "a|b" is satisfied by either). Matched candidates are true
positives, unmatched candidates false positives, unmatched expected items false negatives.
F1 is over the whole set. Brier is the mean over emitted candidates of
`(confidence - matched)^2`, matched being 1 or 0: lower is better calibrated.

Two parts:
- deterministic tests (always run): the metric arithmetic on a known case, and that the golden
  sets are well-formed.
- `test_<agent>_meets_the_baseline` (`pytest -m eval`): runs the real agent graph (Haiku gate,
  Sonnet extraction) once per golden window. It runs only with `EVAL_LIVE=1` and an
  `ANTHROPIC_API_KEY` (env or the repo's `.env`), because the suite never spends money by
  default. It writes `tests/eval/<agent>_last_run.json` and asserts against
  `tests/eval/<agent>_baseline.json` only once that baseline is `measured`.
"""

from __future__ import annotations

import json
import os
import re
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
from libs.shared.state import AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.candidates import CandidateAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.consumer import PROMPTS_DIR

EVAL_DIR = Path(__file__).parent / "eval"
ROOT = Path(__file__).resolve().parents[1]
_LINE = re.compile(r"^(Participant \d+): (.+)$")


# --- scoring ---------------------------------------------------------------------------------


def keyword_match(content: str, keywords: list[str]) -> bool:
    text = content.lower()
    return all(any(alt in text for alt in group.split("|")) for group in keywords)


def match_window(
    emitted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Each emitted candidate with the expected item it matched (or None), greedy by
    descending confidence."""
    free = list(range(len(expected)))
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for cand in sorted(emitted, key=lambda c: -float(c["confidence"])):
        hit = next(
            (i for i in free if keyword_match(cand["content"], expected[i]["keywords"])), None
        )
        if hit is not None:
            free.remove(hit)
        pairs.append((cand, expected[hit] if hit is not None else None))
    return pairs


def score(
    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, Any]:
    """`windows` is (emitted, expected) per golden case."""
    tp = fp = fn = 0
    squared: list[float] = []
    for emitted, expected in windows:
        pairs = match_window(emitted, expected)
        hits = sum(1 for _, e in pairs if e is not None)
        tp += hits
        fp += len(pairs) - hits
        fn += len(expected) - hits
        squared += [(float(c["confidence"]) - (e is not None)) ** 2 for c, e in pairs]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": sum(squared) / len(squared) if squared else None,
        "emitted": len(squared),
    }


def golden(agent: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / f"{agent}_golden.json").read_text("utf-8"))
    return data


# --- deterministic ---------------------------------------------------------------------------


def test_keyword_groups_need_every_group_and_any_alternative():
    assert keyword_match("Ship the Beta on 2 December", ["beta", "ship|release"])
    assert keyword_match("Release the beta", ["beta", "ship|release"])
    assert not keyword_match("Ship the GA build", ["beta", "ship|release"])


def test_metrics_on_a_known_case():
    windows = [
        # two expected; one matched (0.9), one false positive (0.6), one missed
        (
            [
                {"content": "Go with Postgres", "confidence": 0.9},
                {"content": "Rewrite onboarding", "confidence": 0.6},
            ],
            [{"keywords": ["postgres"]}, {"keywords": ["stripe"]}],
        ),
        # nothing expected, nothing emitted
        ([], []),
        # nothing expected, one false positive
        ([{"content": "Maybe Vercel", "confidence": 0.5}], []),
        # greedy: the higher-confidence candidate takes the single expected item first
        (
            [
                {"content": "Drop the widget", "confidence": 0.45},
                {"content": "Cut the iOS widget from v1", "confidence": 0.8},
            ],
            [{"keywords": ["widget"]}],
        ),
    ]
    m = score(windows)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 3, 1)
    assert m["precision"] == pytest.approx(2 / 5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 * 0.4 * (2 / 3) / (0.4 + 2 / 3))
    expected_brier = (0.1**2 + 0.6**2 + 0.5**2 + 0.2**2 + 0.45**2) / 5
    assert m["brier"] == pytest.approx(expected_brier)
    assert m["emitted"] == 5

    assert score([([], [{"keywords": ["x"]}])]) == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "brier": None,
        "emitted": 0,
    }
    perfect = score([([{"content": "x", "confidence": 1.0}], [{"keywords": ["x"]}])])
    assert perfect["f1"] == 1.0 and perfect["brier"] == 0.0


@pytest.mark.parametrize("agent", ["decision", "action"])
def test_the_golden_sets_are_well_formed(agent: str):
    cases = golden(agent)["cases"]
    assert len({c["id"] for c in cases}) == len(cases) >= 12
    empty = sum(1 for c in cases if not c["expected"])
    assert 0.35 <= empty / len(cases) <= 0.65, "about half the windows hold no item"
    assert sum(1 for c in cases if len(c["expected"]) >= 2) >= 2
    for c in cases:
        labels = []
        for line in c["window"].splitlines():
            m = _LINE.match(line)
            assert m, (c["id"], line)
            if m.group(1) not in labels:
                labels.append(m.group(1))
        # numbered in order of first speaking, exactly as the agent renders a window
        assert labels == [f"Participant {i + 1}" for i in range(len(labels))], c["id"]
        for item in c["expected"]:
            assert item["keywords"] and all(k == k.lower() for k in item["keywords"])
            if "assignee" in item:
                assert item["assignee"] in labels, c["id"]
            if "due_date" in item:
                date.fromisoformat(item["due_date"])


@pytest.mark.parametrize("agent", ["decision", "action"])
def test_the_baselines_are_well_formed(agent: str):
    b = json.loads((EVAL_DIR / f"{agent}_baseline.json").read_text("utf-8"))
    assert isinstance(b["measured"], bool)
    assert 0 < b["min_f1"] <= 1 and 0 <= b["max_brier"] <= 1
    assert b["golden_windows"] == len(golden(agent)["cases"])


# --- live ------------------------------------------------------------------------------------


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
        self.tiers: list[str] = []

    async def usage(self, **kw: Any) -> None:
        self.tokens += int(kw["tokens"])
        self.tiers.append(str(kw["model_tier"]))


def _window(
    org: uuid.UUID, meeting: uuid.UUID, index: int, text: str
) -> tuple[TranscriptWindow, dict[str, str]]:
    """The golden text as a window, one synthetic UUID speaker per label. Returns the window and
    the speaker id -> label map, to read the agent's assignee back as a label."""
    speakers: dict[str, str] = {}
    turns = []
    for n, line in enumerate(text.splitlines()):
        m = _LINE.match(line)
        assert m is not None
        sid = speakers.setdefault(m.group(1), str(uuid.uuid4()))
        turns.append(
            Turn(
                speaker_id=sid,
                text=m.group(2),
                start_ms=n * 5_000,
                end_ms=n * 5_000 + 4_000,
                chunk_ids=[f"c{index}-{n}"],
            )
        )
    start = index * 30_000
    w = TranscriptWindow(
        org_id=org,
        meeting_id=meeting,
        window_id=f"{meeting}:{start}",
        start_ms=start,
        end_ms=start + 30_000,
        turns=turns,
    )
    return w, {sid: label for label, sid in speakers.items()}


def _readable(cand: dict[str, Any], label_of: dict[str, str]) -> dict[str, Any]:
    """A candidate for the report, with speaker ids shown as the golden labels."""
    out = {k: v for k, v in cand.items() if k not in ("id", "assignee_id", "stakeholder_ids")}
    if "assignee_id" in cand:
        out["assignee"] = label_of.get(cand["assignee_id"] or "")
    if "stakeholder_ids" in cand:
        out["stakeholders"] = [label_of.get(s, s) for s in cand["stakeholder_ids"]]
    return out


async def _live_eval(agent_name: str, cls: type[CandidateAgent]) -> None:
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends up to 14 Haiku + 14 Sonnet calls)")
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

    data = golden(agent_name)
    org, meeting, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    sink = _NoSpendSink()
    router = LiteLLMRouter(BudgetGate(fakeredis.aioredis.FakeRedis(), unlimited), sink)
    loader = PromptLoader(PROMPTS_DIR)
    if cls is ActionAgent:
        reference = date.fromisoformat(data["reference_date"])
        agent: CandidateAgent = ActionAgent(loader, router, today=lambda: reference)
    else:
        agent = cls(loader, router)
    await agent.spawn(org_id=org, meeting_id=meeting, user_id=meeting)

    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    per_case: dict[str, Any] = {}
    field_checks = {"assignee": [0, 0], "due_date": [0, 0]}  # [correct, labelled matches]
    for i, case in enumerate(data["cases"]):
        w, label_of = _window(org, meeting, i, case["window"])
        context: dict[str, Any] = {}
        if case.get("never_record_topics"):  # ADR-0039 C2: the company ground rule
            context["org"] = CompanyContext(never_record_topics=tuple(case["never_record_topics"]))
        state = AgentState(meeting_id=meeting, user_id=user, chunk=w, context=context, org_id=org)
        before = len(sink.tiers)
        result = await agent.process(state)
        emitted = list(result.payload["candidates"]) if result is not None else []
        windows.append((emitted, case["expected"]))
        for cand, exp in match_window(emitted, case["expected"]):
            if exp is None:
                continue
            if "assignee" in exp:
                got = label_of.get(cand.get("assignee_id") or "")
                field_checks["assignee"][1] += 1
                field_checks["assignee"][0] += got == exp["assignee"]
            if "due_date" in exp:
                field_checks["due_date"][1] += 1
                field_checks["due_date"][0] += cand.get("due_date") == exp["due_date"]
        per_case[case["id"]] = {
            # the gate score travels only on a result; a gated-out window made one triage call
            "gate": result.relevance_score if result is not None else None,
            "tiers": sink.tiers[before:],
            "expected": len(case["expected"]),
            "emitted": [_readable(c, label_of) for c in emitted],
        }

    m = score(windows)
    report = {
        "agent": cls.name,
        "prompt": f"{cls.prompt_key}@{loader.resolve(cls.prompt_key, org)}",
        "gate_prompt": f"{cls.gate_key}@{loader.resolve(cls.gate_key, org)}",
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        **m,
        "field_accuracy": {
            k: (c / n if n else None) for k, (c, n) in field_checks.items() if cls is ActionAgent
        },
        "cases": per_case,
    }
    (EVAL_DIR / f"{agent_name}_last_run.json").write_text(json.dumps(report, indent=2))
    baseline = json.loads((EVAL_DIR / f"{agent_name}_baseline.json").read_text("utf-8"))
    if baseline["measured"]:
        assert m["f1"] >= baseline["min_f1"], report
        assert m["brier"] is not None and m["brier"] <= baseline["max_brier"], report


@pytest.mark.eval
async def test_decision_agent_meets_the_baseline():
    await _live_eval("decision", DecisionAgent)


@pytest.mark.eval
async def test_action_agent_meets_the_baseline():
    await _live_eval("action", ActionAgent)
