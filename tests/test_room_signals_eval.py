"""Golden-set eval for the room signals (a3): `last_word@vN`, `consensus_read@vN`,
`resume_point@vN`, each through the real writer and router (`tests/evalkit/room_signals.py` has
the cases, the source builder and the scoring).

Modes:

    # offline (default, CI): replays tests/eval/recordings/<ref>.json; no network, no spend
    uv run pytest -m eval tests/test_room_signals_eval.py

    # live (the coordinator's): real Haiku + Sonnet calls (about 7 calls); writes
    # tests/eval/room_signals_last_run.json and asserts the baselines once they are `measured`
    EVAL_LIVE=1 uv run pytest -m eval tests/test_room_signals_eval.py -k live -s

    # live + record: the answers replace the simulated recordings (`source: recorded`)
    EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_room_signals_eval.py -k live -s

The recordings start `simulated` (the golden cases' ideal answers), so the offline gate proves
the pipeline (rules, parsing, quote checks, never-record filter, the contract) turns right
answers into the right payload; the live run measures the prompts themselves.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import fakeredis
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.room_signals import RoomSignalsWriter
from services.agent_runtime_service.room_signals_rules import RoomSignalsSource
from tests.evalkit import room_signals as kit
from tests.evalkit.corpus import EVAL_DIR, RECORDINGS_DIR
from tests.evalkit.models import LiveModel, Model, ReplayModel, prompt_index, write_recordings
from tests.test_decision_action_eval import _load_dotenv, _NoSpendSink
from tests.test_room_signals import validate

BASELINES = ("last_word", "consensus_read", "resume_point")


class _Source:
    def __init__(self, source: RoomSignalsSource) -> None:
        self.value = source

    async def source(self, _org: uuid.UUID, _meeting: uuid.UUID) -> RoomSignalsSource:
        return self.value


async def _unlimited(_org: uuid.UUID) -> dict[str, Any]:
    return {"tokens_consumed": 0, "token_quota": 10**9, "hard_cap": 10**9, "enforcement": "enforce"}


async def run_cases(
    model: Model,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], dict[str, uuid.UUID]]], _NoSpendSink]:
    sink = _NoSpendSink()
    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), _unlimited), sink, completion_fn=model
    )
    runs = []
    for case in kit.golden()["cases"]:
        source, users = kit.build_source(case)
        ctx = kit.org_context(case)

        async def org_context(_org: uuid.UUID, c: CompanyContext = ctx) -> CompanyContext:
            return c

        model.case = case["id"]
        writer = RoomSignalsWriter(
            PromptLoader(PROMPTS_DIR), router, source=_Source(source), org_context=org_context
        )
        analysis = await writer.write(uuid.uuid4(), source.meeting_id, "eval")
        assert analysis is not None
        runs.append((case, analysis.payload, users))
    return runs, sink


# --- deterministic ----------------------------------------------------------------------------


def test_golden_set_is_well_formed() -> None:
    cases = kit.golden()["cases"]
    assert len(cases) >= 5
    for case in cases:
        labels = [p["label"] for p in case["participants"]]
        spoken: list[str] = []
        for who, start, end, text in case["transcript"]:
            assert who in labels and end >= start and text.strip()
            if who not in spoken:
                spoken.append(who)
        # labels follow the order of first speaking, as the writer assigns them
        assert spoken == labels[: len(spoken)], case["id"]
        exp = case["expect"]
        assert {n["kind"] for n in exp["nudges"]} <= {
            "cut_off",
            "spoken_over",
            "hand_not_called",
            "topic_silent",
        }
        assert all(n["who"] in labels for n in exp["nudges"])
        for c in exp["consensus"]:
            assert c["strength"] in ("strong", "mixed", "weak", "unclear")
            assert set(c["ask"]) <= set(labels)
        assert set(case.get("simulated", {})) <= set(kit.REFS)
    # every feature is covered both ways
    assert any(c["expect"]["resume"]["open"] for c in cases)
    assert any(not c["expect"]["resume"]["open"] for c in cases)
    assert {c["expect"]["consensus"][0]["strength"] for c in cases if c["expect"]["consensus"]} >= {
        "strong",
        "mixed",
        "weak",
    }
    assert {n["kind"] for c in cases for n in c["expect"]["nudges"]} >= {
        "cut_off",
        "hand_not_called",
        "topic_silent",
    }
    assert any(c.get("never_record") for c in cases)


def test_recordings_match_the_golden_simulated_answers() -> None:
    """While a recording is simulated it is exactly what the golden set says (regenerate with
    `uv run python -m tests.evalkit.room_signals`); a live recording is left alone."""
    index = prompt_index()
    for ref, body in kit.simulated_recordings().items():
        on_disk = json.loads((RECORDINGS_DIR / f"{ref}.json").read_text("utf-8"))
        assert on_disk["prompt_sha256"] == index.sha(ref), f"{ref}: prompt changed, re-record"
        if on_disk["source"] == "simulated":
            assert on_disk == body, f"{ref}: regenerate the simulated recording"


def test_scoring_rewards_the_right_reading_and_catches_leaks() -> None:
    case = kit.golden()["cases"][0]
    _, users = kit.build_source(case)
    u = {lab: str(v) for lab, v in users.items()}
    good = {
        "last_word": [
            {
                "user_id": u["Participant 3"],
                "kind": "cut_off",
                "topic": "enterprise pricing",
                "quote": "I just want to say",
            },
            {
                "user_id": u["Participant 4"],
                "kind": "topic_silent",
                "topic": "pricing",
                "quote": None,
            },
        ],
        "consensus": [
            {
                "strength": "weak",
                "yes": 2,
                "hedged": 0,
                "silent": 1,
                "objected": 0,
                "talked_over": 1,
                "ask": [
                    {"user_id": u["Participant 3"], "reason": "talked_over"},
                    {"user_id": u["Participant 4"], "reason": "silent"},
                ],
            }
        ],
        "resume_point": {
            "reason": "cut_off",
            "topic": "the migration risk",
            "positions": [],
            "cut_off": {
                "user_id": u["Participant 3"],
                "quote": "The migration risk is that the old billing tables",
            },
        },
    }
    assert set(kit.score([(case, good, users)]).values()) == {1.0}
    bad = json.loads(json.dumps(good))
    bad["last_word"][0]["topic"] = "the weather"
    bad["consensus"][0]["strength"] = "strong"
    bad["resume_point"]["cut_off"]["quote"] = "words nobody said"
    bad["last_word"].append(
        {"user_id": u["Participant 1"], "kind": "cut_off", "topic": "x", "quote": None}
    )
    m = kit.score([(case, bad, users)])
    assert m["topic_hit_rate"] == 0.5 and m["strength_accuracy"] == 0.0
    assert m["nudge_precision"] == 2 / 3 and m["quote_verbatim_rate"] == 0.5
    leaky_case = kit.golden()["cases"][3]
    _, u2 = kit.build_source(leaky_case)
    leak = {
        "last_word": [],
        "consensus": [],
        "resume_point": {
            "reason": "time_ran_out",
            "topic": "the layoffs",
            "positions": [],
            "cut_off": None,
        },
    }
    assert kit.score([(leaky_case, leak, u2)])["leak_free"] == 0.0


# --- offline: the recordings, through the real writer -----------------------------------------


@pytest.mark.eval
async def test_offline_replay_meets_the_gate() -> None:
    model = ReplayModel()
    runs, sink = await run_cases(model)
    assert model.errors == []
    for _case, payload, _users in runs:
        validate(payload)
        assert payload["status"] == "ready"
    m = kit.score(runs)
    # The simulated recordings are the ideal answers: the pipeline must turn them into exactly
    # the expected reading. (Live recordings are gated by the measured baselines instead.)
    sources = {
        json.loads((RECORDINGS_DIR / f"{r}.json").read_text("utf-8"))["source"]
        for r in kit.REFS.values()
    }
    if sources == {"simulated"}:
        assert set(m.values()) == {1.0}, m
    else:
        _assert_baselines(m)
    assert m["leak_free"] == 1.0 and m["quote_verbatim_rate"] == 1.0
    # one call per prompt each case needs, nothing else
    assert {c.ref for c in model.calls} == set(kit.REFS.values())
    assert len(model.calls) == sum(len(c.get("simulated", {})) for c in kit.golden()["cases"])
    assert sink.tokens > 0  # every call metered


def _assert_baselines(m: dict[str, float]) -> None:
    for name in BASELINES:
        baseline = json.loads((EVAL_DIR / f"{name}_baseline.json").read_text("utf-8"))
        if not baseline.get("measured"):
            continue
        for key, floor in baseline.items():
            if key.startswith("min_"):
                assert m[key[4:]] >= floor, (name, key, m[key[4:]])


# --- live (the coordinator runs it; never in CI) ----------------------------------------------


@pytest.mark.eval
async def test_live_room_signals_meet_the_baselines() -> None:
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (about 7 Haiku/Sonnet calls)")
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    model = LiveModel()
    runs, sink = await run_cases(model)
    m = kit.score(runs)
    report = {
        "prompts": dict(kit.REFS),
        "tokens": sink.tokens,
        "calls": len(sink.tiers),
        "metrics": m,
        "cases": {case["id"]: payload for case, payload, _ in runs},
    }
    (EVAL_DIR / "room_signals_last_run.json").write_text(json.dumps(report, indent=2) + "\n")
    if os.environ.get("EVAL_RECORD") == "1":
        write_recordings(model.recorded, "recorded")
    assert m["leak_free"] == 1.0
    _assert_baselines(m)
