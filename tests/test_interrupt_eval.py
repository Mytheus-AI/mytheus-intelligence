"""Golden-set eval for the live watch's interrupt gate (`interrupt_gate@vN`).

A wrong interruption pulls a person out of their focus, so precision is the headline. Over
`tests/eval/interrupt_gate_golden.json`, one live-meeting window per case, run through the real
watch (`ListenWatch.observe`: the pre-filter, name redaction, the router, the parser and the
confidence threshold; only core-api is faked in process):

- **precision** / **recall** of "an interruption is posted";
- **reason_accuracy**: the posted reason (decision, question, promise) among true positives;
- **false_positive_rate**: negatives that were posted / negatives;
- **parse_failure_rate**: model answers the parser could not read (`agent_metrics.output_ok`).

Modes, as the eval harness (tests/evalkit/harness.py):
- offline (always, and under `pytest -m eval`): with a live recording at
  `tests/eval/recordings/interrupt_gate@vN.json`, `ReplayModel` replays it (a changed prompt fails
  until it is recorded again). Until one exists, `LabelModel` answers as the golden labels say
  (`simulated()`), the label-judge pattern of `test_proactive_eval.py`: it proves the watch's
  plumbing and scoring, never the prompt. A simulated answer is never written to recordings/
  (`generate.SIMULATED_REFS` keeps new prompts out of it): a new prompt needs a live recording.
- live: `EVAL_LIVE=1` (one Haiku call per case that passes the pre-filter, about 13 calls) writes
  `tests/eval/interrupt_gate_last_run.json`; with `EVAL_RECORD=1` the answers become the
  recording (`source: recorded`). Asserts against the baseline once it is `measured`:
      EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_interrupt_eval.py -k live -s
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.watch import (
    Interruption,
    Listener,
    Listeners,
    ListenWatch,
    PostAnswer,
    WatchSettings,
    parse_listener,
    prefilter,
)
from tests.a2a_helpers import budget
from tests.evalkit.models import LiveModel, Model, ReplayModel, prompt_index, write_recordings
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink

EVAL_DIR = Path(__file__).parent / "eval"
REF = "interrupt_gate@v1"
RECORDING = EVAL_DIR / "recordings" / f"{REF}.json"
REASONS = {"decision", "question", "promise"}


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "interrupt_gate_golden.json").read_text("utf-8"))
    return data


def baseline() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (EVAL_DIR / "interrupt_gate_baseline.json").read_text("utf-8")
    )
    return data


def build(case: dict[str, Any]) -> tuple[TranscriptWindow, Listener]:
    org, meeting = uuid.uuid4(), uuid.uuid4()
    speakers: dict[str, str] = {}
    turns = []
    for i, line in enumerate(case["transcript"]):
        label, _, text = line.partition(": ")
        sid = speakers.setdefault(label, str(uuid.uuid4()))
        turns.append(Turn(speaker_id=sid, text=text, start_ms=i * 5000, end_ms=i * 5000 + 4000))
    window = TranscriptWindow(
        org_id=org,
        meeting_id=meeting,
        window_id=f"{meeting}:0",
        start_ms=0,
        end_ms=turns[-1].end_ms,
        turns=turns,
        labels={sid: label for label, sid in speakers.items()},
    )
    person = parse_listener({"user_id": str(uuid.uuid4()), "interruptible": True, **case["person"]})
    assert person is not None
    return window, person


class FakeCore:
    """core-api in process: the case's one listener, every interruption recorded."""

    def __init__(self, window: TranscriptWindow, person: Listener, title: str) -> None:
        self.found = Listeners(
            meeting_id=window.meeting_id,
            title=title,
            confidential=False,
            status="live",
            listeners=(person,),
        )
        self.posts: list[Interruption] = []

    async def listeners(self, _org: uuid.UUID, _meeting: uuid.UUID) -> Listeners:
        return self.found

    async def interrupt(
        self, _org: uuid.UUID, _meeting: uuid.UUID, interruption: Interruption
    ) -> PostAnswer:
        self.posts.append(interruption)
        return PostAnswer(id=uuid.uuid4(), status="delivered")


async def run(model: Model) -> list[Interruption | None]:
    """Every golden case through the real watch; the posted interruption per case (or None)."""
    router = LiteLLMRouter(budget(), _NoSpendSink(), model)
    out: list[Interruption | None] = []
    for case in golden()["cases"]:
        model.case = case["id"]
        window, person = build(case)
        core = FakeCore(window, person, case["title"])
        watch = ListenWatch(
            PromptLoader(PROMPTS_DIR),
            router,
            source=core,
            redis=fakeredis.aioredis.FakeRedis(),
            settings=WatchSettings(),
        )
        await watch.observe(window)
        out.append(core.posts[0] if core.posts else None)
    return out


def score(
    cases: list[dict[str, Any]], posted: list[Interruption | None], parse_failure_rate: float
) -> dict[str, float]:
    pairs = list(zip(cases, posted, strict=True))
    tp = [(c, p) for c, p in pairs if c["interrupt"] and p is not None]
    fp = sum(1 for c, p in pairs if not c["interrupt"] and p is not None)
    positives = sum(1 for c in cases if c["interrupt"])
    negatives = len(cases) - positives
    return {
        "precision": round(len(tp) / (len(tp) + fp), 3) if tp or fp else 1.0,
        "recall": round(len(tp) / positives, 3) if positives else 0.0,
        "reason_accuracy": (
            round(sum(1 for c, p in tp if p and p.reason == c["reason"]) / len(tp), 3)
            if tp
            else 0.0
        ),
        "false_positive_rate": round(fp / negatives, 3) if negatives else 0.0,
        "parse_failure_rate": round(parse_failure_rate, 3),
    }


def meets(m: dict[str, float], b: dict[str, Any]) -> bool:
    return (
        m["precision"] >= b["min_precision"]
        and m["recall"] >= b["min_recall"]
        and m["reason_accuracy"] >= b["min_reason_accuracy"]
        and m["false_positive_rate"] <= b["max_false_positive_rate"]
        and m["parse_failure_rate"] <= b["max_parse_failure_rate"]
    )


def simulated() -> dict[str, str]:
    """The answers a model that agrees with every label gives, for the cases that reach it."""
    entries: dict[str, str] = {}
    for case in golden()["cases"]:
        if not case["prefilter"]:
            continue
        if not case["interrupt"]:
            entries[case["id"]] = json.dumps({"interrupt": False})
            continue
        quote = case["transcript"][0].partition(": ")[2]
        entries[case["id"]] = json.dumps(
            {
                "interrupt": True,
                "reason": case["reason"],
                "confidence": 0.86,
                "headline": f"{case['topic']} needs you",
                "detail": f"The meeting is on {case['topic'].lower()} and it needs you.",
                "topic": case["topic"],
                "quote": quote,
            }
        )
    return dict(sorted(entries.items()))


class LabelModel(Model):
    """Answers as the golden labels say (no recording yet)."""

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        return simulated()[key], {}


# --- deterministic -------------------------------------------------------------------------


def test_the_golden_set_is_well_formed() -> None:
    cases = golden()["cases"]
    assert 12 <= len(cases) <= 20 and len({c["id"] for c in cases}) == len(cases)
    assert {c["reason"] for c in cases if c["interrupt"]} == REASONS
    assert sum(1 for c in cases if not c["interrupt"]) >= 5
    for c in cases:
        assert (c["reason"] in REASONS) == c["interrupt"], c["id"]
        window, person = build(c)
        hit = bool(prefilter("\n".join(t.text for t in window.turns), person))
        assert hit is c["prefilter"], c["id"]
        assert c["interrupt"] is False or c["prefilter"], c["id"]


def test_the_baseline_is_well_formed() -> None:
    b = baseline()
    assert isinstance(b["measured"], bool) and b["golden_cases"] == len(golden()["cases"])
    assert b["prompt"] == REF
    for k in ("min_precision", "min_recall", "min_reason_accuracy", "max_false_positive_rate"):
        assert 0 <= b[k] <= 1


def test_metrics_on_a_known_case() -> None:
    def post(reason: str) -> Interruption:
        return Interruption(uuid.uuid4(), "k", reason, "h", 0.9)

    cases = [
        {"interrupt": True, "reason": "decision"},
        {"interrupt": True, "reason": "promise"},
        {"interrupt": False, "reason": None},
        {"interrupt": False, "reason": None},
    ]
    m = score(cases, [post("decision"), post("question"), post("decision"), None], 0.25)
    assert m == {
        "precision": 0.667,
        "recall": 1.0,
        "reason_accuracy": 0.5,
        "false_positive_rate": 0.5,
        "parse_failure_rate": 0.25,
    }
    assert score(cases, [None] * 4, 0.0)["precision"] == 1.0


def test_a_recording_if_any_is_current() -> None:
    if not RECORDING.exists():
        pytest.skip("interrupt_gate@v1 has no live recording yet (the coordinator records it)")
    data = json.loads(RECORDING.read_text("utf-8"))
    assert data["source"] == "recorded"
    assert data["prompt_sha256"] == prompt_index().sha(REF), "prompt changed: re-record"


async def _offline() -> tuple[dict[str, float], Model]:
    model: Model = ReplayModel() if RECORDING.exists() else LabelModel()
    posted = await run(model)
    assert model.errors == []
    return score(golden()["cases"], posted, model.parse_failure_rate("interrupt_gate")), model


async def test_offline_replay_through_the_real_watch() -> None:
    m, model = await _offline()
    reached = sum(1 for c in golden()["cases"] if c["prefilter"])
    assert len(model.calls) == reached  # the pre-filter stopped the rest before any model call
    assert meets(m, baseline()), m


@pytest.mark.eval
async def test_offline_gate() -> None:
    m, _ = await _offline()
    assert meets(m, baseline()), m


# --- live ----------------------------------------------------------------------------------


@pytest.mark.eval
async def test_interrupt_gate_live() -> None:
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (spends about 13 Haiku calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    model = LiveModel()
    posted = await run(model)
    cases = golden()["cases"]
    m = score(cases, posted, model.parse_failure_rate("interrupt_gate"))
    report = {
        "prompt": REF,
        "calls": len(model.calls),
        **m,
        "cases": {
            c["id"]: {
                "expected": c["reason"] if c["interrupt"] else None,
                "posted": None if p is None else {"reason": p.reason, "confidence": p.confidence},
            }
            for c, p in zip(cases, posted, strict=True)
        },
    }
    (EVAL_DIR / "interrupt_gate_last_run.json").write_text(json.dumps(report, indent=2) + "\n")
    if os.environ.get("EVAL_RECORD") == "1":
        write_recordings(model.recorded, "recorded")
    if baseline()["measured"]:
        assert meets(m, baseline()), report
