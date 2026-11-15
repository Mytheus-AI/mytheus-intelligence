"""The red-team driver.

    uv run python -m research.redteam.run corpus     # paraphrase (Haiku batch) + instances
    uv run python -m research.redteam.run baseline --runs 3
    uv run python -m research.redteam.run fixes --runs 3
    uv run python -m research.redteam.run adaptive --rounds 3
    uv run python -m research.redteam.run report

`--metrics` (default `../research/metrics/redteam` next to the repos) receives the corpus, the
response store (`cache/responses.jsonl`), raw records, verdicts and the summary. The API key is
read from `--env-file` (the intelligence `.env`); spend goes through the shared ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from research.redteam import analyze, corpus, org
from research.redteam.corpus import ConflictCase, Instance
from research.redteam.harness import BASELINES, Condition, Harness, drive, write_jsonl
from research.redteam.llm import BatchLLM, load_key

ROOT = Path(__file__).resolve().parents[3]  # .../Mytheus when run from a worktree under wt/
DEFAULT_METRICS = Path("/Users/sushan/Projects/Mytheus/research/metrics/redteam")
DEFAULT_ENV = Path("/Users/sushan/Projects/Mytheus/mytheus-intelligence/.env")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_corpus(metrics: Path) -> tuple[list[Instance], list[ConflictCase]]:
    rows = [
        json.loads(line)
        for line in (metrics / "corpus/instances.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    conflicts = [
        ConflictCase(**json.loads(line))
        for line in (metrics / "corpus/conflicts.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    return [Instance.from_json(r) for r in rows], conflicts


def make_llm(metrics: Path, env_file: Path) -> BatchLLM:
    load_key(env_file)
    return BatchLLM(store=metrics / "cache/responses.jsonl", metrics_root=metrics.parent)


def cmd_corpus(args: argparse.Namespace) -> None:
    from research.redteam.paraphrase import run as paraphrase

    org.check_canaries()
    out = args.metrics / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    para_file = out / "paraphrases.json"
    if para_file.exists():
        para = json.loads(para_file.read_text("utf-8"))
    else:
        para = paraphrase(make_llm(args.metrics, args.env_file), para_file)
    built = corpus.build(para)
    corpus.save(built, out)
    print(
        f"benign {len(built['benign'])}, attacks {len(built['attacks'])}, "
        f"conflict cases {len(built['conflicts'])}"
    )


def run_conditions(
    args: argparse.Namespace, phase: str, conditions: list[Condition], with_extras: bool
) -> None:
    instances, conflicts = load_corpus(args.metrics)
    gap = [i for i in instances if i.pipeline == "context_gap"]
    cands = [i for i in instances if i.pipeline == "candidates"]
    harness = Harness(make_llm(args.metrics, args.env_file))
    runs = list(range(1, args.runs + 1))

    async def jobs() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for run in runs:
            for cond in conditions:
                out += await harness.gap_all(gap, cond, run)
            if with_extras:
                out += [await harness.candidates_one(i, run) for i in cands]
                out += [await harness.conflict_one(c, run) for c in conflicts]
        return out

    start = time.time()
    records = asyncio.run(drive(harness, jobs, f"redteam {phase}"))
    raw = args.metrics / "raw"
    write_jsonl(raw / f"{phase}_gap.jsonl", [r for r in records if "fills" in r])
    if with_extras:
        write_jsonl(raw / f"{phase}_candidates.jsonl", [r for r in records if "candidates" in r])
        write_jsonl(raw / f"{phase}_conflicts.jsonl", [r for r in records if "raw" in r])
    meta = {
        "phase": phase,
        "conditions": [
            {
                "name": c.name,
                "note": c.note,
                "env": dict(c.env),
                "a2a": c.a2a,
                "class_filter": c.class_filter,
                "as_requester": c.as_requester,
                "grants": len(c.grants),
            }
            for c in conditions
        ],
        "runs": runs,
        "git_sha": git_sha(),
        "started": start,
        "finished": time.time(),
        "models": {"triage": "claude-haiku-5-5", "extract": "claude-sonnet-5-5 (effort low)"},
        "n_gap_instances": len(gap),
        "n_candidate_windows": len(cands) if with_extras else 0,
        "n_conflict_cases": len(conflicts) if with_extras else 0,
    }
    (raw / f"{phase}_meta.json").write_text(json.dumps(meta, indent=1), "utf-8")
    print(f"{phase}: {len(records)} records in {time.time() - start:.0f}s")


def cmd_baseline(args: argparse.Namespace) -> None:
    run_conditions(args, "baseline", list(BASELINES.values()), with_extras=True)


def cmd_fixes(args: argparse.Namespace) -> None:
    from research.redteam.fixes import FIX_CONDITIONS

    run_conditions(args, "fixes", list(FIX_CONDITIONS.values()), with_extras=False)


def cmd_report(args: argparse.Namespace) -> None:
    from research.redteam.report import report

    report(args.metrics)


def cmd_adaptive(args: argparse.Namespace) -> None:
    from research.redteam.adaptive import run_adaptive

    run_adaptive(args.metrics, make_llm(args.metrics, args.env_file), args.rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus", "baseline", "fixes", "adaptive", "report"])
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    {
        "corpus": cmd_corpus,
        "baseline": cmd_baseline,
        "fixes": cmd_fixes,
        "adaptive": cmd_adaptive,
        "report": cmd_report,
    }[args.cmd](args)
    _ = analyze  # verdicts are computed by `report`


if __name__ == "__main__":
    main()
