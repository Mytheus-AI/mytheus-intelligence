"""Golden-set eval for the briefing writer (`briefing@vN`, a briefing from your agent).

Each golden case is the source core-api's claim answers for one person. The real
`BriefingWriter` runs on it (parsing, never-record filtering, ref -> link mapping); only the
model's answer is replayed. Metrics, over the cases:

- **coverage**: the items that must be spoken (keywords) are spoken;
- **link_accuracy**: each of them carries the right "open the moment" link (type and record);
- **length_ok**: the script's words fall inside the case's window (about the target, never more
  than a third over; a quiet case stays short);
- **greets_by_name**: the opening says the person's first name;
- **order_ok**: segments follow opening, changed, waiting, promise, question, closing;
- **spoken_style**: no bullets, markdown, labels such as "C1" or URLs in what is said;
- **leak_free**: no never-record text anywhere (must be 1.0);
- **parse_failure_rate**: answers that were not the JSON asked for.

Modes (the evalkit pattern, `tests/evalkit/harness.py`):

    # offline (CI, default): replays tests/eval/recordings/briefing@*.json; no network
    uv run pytest -m eval tests/test_briefing_eval.py

    # live: real Sonnet calls for both variants (about 8 calls); fails on the thresholds
    EVAL_LIVE=1 uv run pytest -m eval tests/test_briefing_eval.py -k live -s

    # live + record: the answers replace the recordings (`source: recorded`)
    EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_briefing_eval.py -k live -s

The first recordings are `source: simulated` (hand-written answers to the golden set) until the
coordinator records a live run; the worse fixture (`tests/eval/prompts/briefing@v0-worse.md`)
must lose on the headline metrics, so the gate can fail.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from libs.shared.org_context import CompanyContext
from services.agent_runtime_service.briefing import KINDS, BriefingWriter
from tests.evalkit.harness import EvalLoader, TokenSink, router_for
from tests.evalkit.models import LiveModel, Model, ReplayModel, prompt_index, write_recordings

EVAL_DIR = Path(__file__).parent / "eval"
SUITE = "briefing"
WORSE = {"briefing": "v0-worse"}
# What separates a briefing from a generic recap. Coverage is gated by thresholds but is not a
# headline: with real answers (live 2026-11-11) the worse prompt covers the points too; it fails
# on links, greeting, length and spoken style.
HEADLINE = ("link_accuracy", "greets_by_name", "spoken_style")
ORG = UUID(int=900)
_STYLE = re.compile(r"(^\s*[-*•]\s)|(\*\*)|(#)|(\b[CWPQ]\d+\b)|(https?://)")


def golden() -> list[dict[str, Any]]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "briefing_golden.json").read_text("utf-8"))
    return list(data["cases"])


def thresholds() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EVAL_DIR / "thresholds.json").read_text("utf-8"))
    return dict(data[SUITE])


def _words(segments: list[dict[str, Any]]) -> int:
    return sum(len(s["text"].split()) for s in segments)


def score(runs: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> dict[str, float]:
    """`runs`: (golden case, the result body the writer produced, None when unparseable)."""
    cover = [0, 0]
    linked = [0, 0]
    length = greet = order = style = leak = parsed = 0
    for case, out in runs:
        exp = case["expect"]
        segments = (out or {}).get("segments") or []
        if out is not None and out.get("status") == "ready":
            parsed += 1
        text = " ".join(s["text"] for s in segments).lower()
        for item in exp["cover"]:
            cover[1] += 1
            linked[1] += 1
            hits = [
                s for s in segments if all(k.lower() in s["text"].lower() for k in item["keywords"])
            ]
            cover[0] += bool(hits)
            want = item["link"]
            linked[0] += any(
                s.get("link")
                and s["link"]["type"] == want["type"]
                and s["link"].get("id") == want["id"]
                for s in hits
            )
        n = _words(segments)
        length += exp["min_words"] <= n <= exp["max_words"]
        greet += bool(
            segments
            and segments[0]["kind"] == "opening"
            and exp["first_name"].lower() in segments[0]["text"].lower()
        )
        ranks = [KINDS.index(s["kind"]) for s in segments]
        order += ranks == sorted(ranks)
        style += not any(_STYLE.search(s["text"]) for s in segments)
        leak += not any(f.lower() in text for f in exp["forbidden"])
    total = len(runs) or 1

    def rate(hit: int, of: int) -> float:
        return 1.0 if of == 0 else round(hit / of, 3)

    return {
        "coverage": rate(*cover),
        "link_accuracy": rate(*linked),
        "length_ok": round(length / total, 3),
        "greets_by_name": round(greet / total, 3),
        "order_ok": round(order / total, 3),
        "spoken_style": round(style / total, 3),
        "leak_free": round(leak / total, 3),
        "parse_failure_rate": round(1 - parsed / total, 3),
    }


def check(metrics: dict[str, float], limits: dict[str, Any]) -> list[str]:
    failures = []
    for metric, bound in limits.items():
        value = metrics[metric]
        if "min" in bound and value < bound["min"]:
            failures.append(f"{metric} = {value} < min {bound['min']}")
        if "max" in bound and value > bound["max"]:
            failures.append(f"{metric} = {value} > max {bound['max']}")
    return failures


async def run(model: Model, variant: str) -> tuple[dict[str, float], dict[str, Any]]:
    loader = EvalLoader(WORSE if variant == "worse" else None)
    router = router_for(model, TokenSink())
    runs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    outputs: dict[str, Any] = {}
    for case in golden():
        org = CompanyContext(never_record_topics=tuple(case.get("never_record", [])))

        async def org_context(_org: UUID, ctx: CompanyContext = org) -> CompanyContext:
            return ctx

        model.case = case["id"]
        writer = BriefingWriter(loader, router, org_context=org_context)
        out = await writer.write(ORG, case["source"])
        outputs[case["id"]] = out
        runs.append((case, out if out.get("status") == "ready" else None))
    return score(runs), outputs


# --- deterministic ----------------------------------------------------------------------------


def test_the_golden_set_is_well_formed():
    cases = golden()
    assert len(cases) >= 4 and len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        src, exp = case["source"], case["expect"]
        assert exp["first_name"] == src["person"]["first_name"]
        assert exp["min_words"] < exp["max_words"] <= round(src["target_words"] * 4 / 3)
        pools = {"C": "changed", "W": "waiting", "P": "promises", "Q": "questions"}
        for item in exp["cover"]:
            pool = src[pools[item["ref"][0]]]
            assert int(item["ref"][1:]) <= len(pool), (case["id"], item["ref"])
            assert item["keywords"]
    assert any(c["never_record"] for c in golden())
    assert any(c["source"]["length"] == "short" for c in golden())


def test_scoring_rewards_a_good_briefing_and_catches_a_bad_one():
    case = golden()[0]
    good = {
        "status": "ready",
        "segments": [
            {"kind": "opening", "text": "Hi Sam, it's Juniper.", "link": None},
            *[
                {
                    "kind": "changed",
                    "text": " ".join(item["keywords"]) + " " + "word " * 15,
                    "link": item["link"],
                }
                for item in case["expect"]["cover"]
            ],
        ],
    }
    m = score([(case, good)])
    assert m["coverage"] == 1.0 and m["link_accuracy"] == 1.0 and m["greets_by_name"] == 1.0
    bad = {
        "status": "ready",
        "segments": [{"kind": "changed", "text": "- C1: **x**", "link": None}],
    }
    m = score([(case, bad)])
    assert m["coverage"] == 0.0 and m["spoken_style"] == 0.0 and m["greets_by_name"] == 0.0
    assert score([(case, None)])["parse_failure_rate"] == 1.0


def test_the_recordings_match_the_prompts_they_were_made_for():
    for ref in ("briefing@v1", "briefing@v0-worse"):
        data = json.loads((EVAL_DIR / "recordings" / f"{ref}.json").read_text("utf-8"))
        assert data["prompt_sha256"] == prompt_index().sha(ref), f"re-record {ref}"
        assert set(data["entries"]) == {c["id"] for c in golden()}


def test_the_thresholds_exist_and_are_well_formed():
    limits = thresholds()
    assert set(limits) == {
        "coverage",
        "link_accuracy",
        "length_ok",
        "greets_by_name",
        "order_ok",
        "spoken_style",
        "leak_free",
        "parse_failure_rate",
    }
    assert limits["leak_free"] == {"min": 1.0}


# --- offline (recorded / simulated) -----------------------------------------------------------


@pytest.mark.eval
async def test_offline_the_current_prompt_meets_the_thresholds():
    model = ReplayModel()
    metrics, _ = await run(model, "current")
    assert not model.errors, model.errors
    assert not check(metrics, thresholds()), metrics


@pytest.mark.eval
async def test_offline_the_worse_prompt_loses():
    current, _ = await run(ReplayModel(), "current")
    worse_model = ReplayModel()
    worse, _ = await run(worse_model, "worse")
    assert not worse_model.errors, worse_model.errors
    assert check(worse, thresholds()), worse
    for metric in HEADLINE:
        assert worse[metric] < current[metric], (metric, worse[metric], current[metric])
    assert worse["leak_free"] == 1.0  # the writer drops never-record text whatever the prompt


# --- live -------------------------------------------------------------------------------------


@pytest.mark.eval
async def test_live_briefings_meet_the_thresholds():
    if os.environ.get("EVAL_LIVE") != "1":
        pytest.skip("live eval: set EVAL_LIVE=1 (about 8 Sonnet calls)")
    from tests.test_decision_action_eval import _load_dotenv

    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    shared: dict[str, dict[str, str]] = {}
    reports = {}
    recorded: dict[str, dict[str, str]] = {}
    for variant in ("current", "worse"):
        model = LiveModel(shared=shared)
        reports[variant], outputs = await run(model, variant)
        recorded.update(model.recorded)
        (EVAL_DIR / f"briefing_last_run_{variant}.json").write_text(
            json.dumps({"metrics": reports[variant], "cases": outputs}, indent=2) + "\n"
        )
    if os.environ.get("EVAL_RECORD") == "1":
        write_recordings(recorded, "recorded")
    assert not check(reports["current"], thresholds()), reports["current"]
