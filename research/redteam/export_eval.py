"""Export recorded context-fill syntheses into the repo's offline eval (no model calls).

From the red-team runs, pick instances whose synthesis ran under both `context_gap_agent@v2`
(B1) and `@v3` (B1+spot, same passages, same run): benign gaps and passage / spoken injections.
Writes `tests/eval/context_gap_synthesis_golden.json` (inputs, rendered user messages, oracle
fields) and `tests/eval/recordings/context_gap_agent@v2.json` / `@v3.json` (the recorded model
texts), which `tests/test_context_gap_synthesis_eval.py` replays.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from research.redteam import org
from services.agent_runtime_service.a2a.bus import ScoredPassage
from services.agent_runtime_service.agents.context_gap_agent import (
    render_passages,
    render_spotlight,
)


def export(metrics: Path, repo: Path, per_kind: int = 20) -> dict[str, int]:
    from research.redteam.run import load_corpus

    instances = {i.id: i for i in load_corpus(metrics)[0]}
    store: dict[str, dict[str, Any]] = {}
    for line in (metrics / "cache/responses.jsonl").read_text("utf-8").splitlines():
        a = json.loads(line)
        if "Context needed:" in a.get("user", "") and a["run"] == 1:
            store[a["user"]] = a
    by_user_v2: dict[str, dict[str, Any]] = {}
    by_user_v3: dict[str, dict[str, Any]] = {}
    for a in store.values():
        (by_user_v3 if "<passage n=" in a["user"] else by_user_v2)[a["user"]] = a

    def records(path: Path, cond: str) -> dict[str, dict[str, Any]]:
        out = {}
        for line in path.read_text("utf-8").splitlines():
            r = json.loads(line)
            if r["condition"] == cond and r["run"] == 1 and r["fills"]:
                out[r["id"]] = r
        return out

    b1 = records(metrics / "raw/baseline_gap.jsonl", "B1")
    spot = records(metrics / "raw/fixes_gap.jsonl", "B1+spot")
    cases: list[dict[str, Any]] = []
    rec2: dict[str, str] = {}
    rec3: dict[str, str] = {}
    for iid in sorted(set(b1) & set(spot)):
        inst = instances[iid]
        if inst.goal == "exfil":
            continue
        f2, f3 = b1[iid]["fills"][0], spot[iid]["fills"][0]
        if not f2["shown"] or [p["text"] for p in f2["shown"]] != [p["text"] for p in f3["shown"]]:
            continue
        q = f2["query"]
        shown = [ScoredPassage(p["text"], 0.0, {}) for p in f2["shown"]]
        tail2 = "\n\nPassages:\n" + render_passages(shown)
        tail3 = "\n\nPassages:\n" + render_spotlight(shown)
        head = f"Context needed: {q}\n"
        u2 = next((u for u in by_user_v2 if head in u and u.endswith(tail2)), None)
        u3 = next((u for u in by_user_v3 if head in u and u.endswith(tail3)), None)
        if u2 is None or u3 is None:
            continue
        src = org.BY_KEY.get(inst.target or "")
        cases.append(
            {
                "id": iid,
                "kind": "benign" if inst.threat == "benign" else inst.goal,
                "threat": inst.threat,
                "carrier": inst.carrier,
                "style": inst.style,
                "query": q,
                "passages": [p["text"] for p in f2["shown"]],
                "user_v2": u2,
                "user_v3": u3,
                "facts": list(src.facts) if src else [],
                "payload": inst.payload,
            }
        )
        rec2[iid] = by_user_v2[u2]["text"]
        rec3[iid] = by_user_v3[u3]["text"]
    rng = random.Random(11)
    chosen: list[dict[str, Any]] = []
    for kind in ("benign", "integrity_url", "integrity_fact", "suppress"):
        pool = [c for c in cases if c["kind"] == kind]
        rng.shuffle(pool)
        chosen += pool[:per_kind]
    eval_dir = repo / "tests/eval"
    (eval_dir / "context_gap_synthesis_golden.json").write_text(
        json.dumps(
            {
                "about": (
                    "Context-fill syntheses recorded in the journal red team (research/redteam), "
                    "run 1: the same passages under context_gap_agent@v2 (B1) and @v3 "
                    "(spotlighted, B1+spot). Synthetic org, no real data."
                ),
                "cases": chosen,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    ids = {c["id"] for c in chosen}
    from tests.evalkit.models import prompt_index

    for name, rec in (("v2", rec2), ("v3", rec3)):
        ref = f"context_gap_agent@{name}"
        (eval_dir / f"recordings/{ref}.json").write_text(
            json.dumps(
                {
                    "prompt": ref,
                    "prompt_sha256": prompt_index().sha(ref),
                    "source": "recorded",
                    "about": (
                        "Real Sonnet answers (batch, effort low) recorded by the journal red "
                        "team (research/redteam), keyed by golden case id."
                    ),
                    "entries": {k: v for k, v in rec.items() if k in ids},
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            "utf-8",
        )
    counts = {
        k: sum(1 for c in chosen if c["kind"] == k)
        for k in ("benign", "integrity_url", "integrity_fact", "suppress")
    }
    print(counts)
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("."))
    a = ap.parse_args()
    export(a.metrics, a.repo)
