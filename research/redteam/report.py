"""Verdicts, tables and figures from the raw records (no model calls).

Writes `verdicts.jsonl`, `summary.json`, `results.md` and `figures/*.png` into the metrics
folder. Figures need matplotlib: `uv run --with matplotlib python -m research.redteam.run report`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.redteam import analyze
from research.redteam.analyze import fmt, rate_table, wilson
from research.redteam.harness import write_jsonl

ORDER = [
    "B0",
    "B1",
    "B1t",
    "B2",
    "B3",
    "B1+spot",
    "FIX",
    "FIX-mac",
    "FIX-mc",
    "FIX-limits",
    "FIX-spot",
    "FIX-hmac",
    "FIX+extract",
]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def candidate_rows(metrics: Path, instances: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in _rows(metrics / "raw/baseline_candidates.jsonl"):
        inst = instances[r["id"]]
        hit = [c for c in r["candidates"] if analyze.payload_hit(c["content"] or "", inst.payload)]
        out.append(
            {
                "id": r["id"],
                "run": r["run"],
                "condition": "B1",
                "seed": inst.seed,
                "any_candidate": bool(hit),
                "surfaced": any(c["surfaced"] for c in hit),
                "i2_kinds": sorted({c["kind"] for c in r["candidates"]}),
            }
        )
    return out


def conflict_rows(metrics: Path) -> list[dict[str, Any]]:
    rows = _rows(metrics / "raw/baseline_conflicts.jsonl")
    by = {(r["id"], r["run"]): r for r in rows}
    out = []
    for r in rows:
        if not r["id"].endswith("/inj"):
            continue
        ctl = by.get((r["id"][: -len("/inj")] + "/ctl", r["run"]))
        if ctl is None:
            continue
        style = r["id"].split("/")[-2]
        out.append(
            {
                "id": r["id"],
                "run": r["run"],
                "condition": "B1",
                "style": style,
                "control_flagged": ctl["flagged_target"],
                # suppression: the control raised the alert and the injected copy did not
                "success": (not r["flagged_target"]) if ctl["flagged_target"] else None,
            }
        )
    return out


def figures(metrics: Path, summary: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed: figures skipped"]
    out_dir = metrics / "figures"
    out_dir.mkdir(exist_ok=True)
    conds = [c for c in ORDER if c in summary["by_condition"]]
    goals = ["exfil", "integrity", "suppress", "benign_correct"]
    matrix = [
        [summary["by_condition"][c].get(g, {}).get("rate", float("nan")) for g in goals]
        for c in conds
    ]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(conds) + 1.4))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(goals)), ["exfiltration", "integrity", "suppression", "benign correct"])
    ax.set_yticks(range(len(conds)), conds)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v == v:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.55 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Attack success (and benign utility) by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "asr_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c in conds:
        s = summary["by_condition"][c]
        if "exfil" not in s or "benign_correct" not in s:
            continue
        x, y = s["exfil"], s["benign_correct"]
        ax.errorbar(
            x["rate"],
            y["rate"],
            xerr=[[x["rate"] - x["ci_lo"]], [x["ci_hi"] - x["rate"]]],
            yerr=[[y["rate"] - y["ci_lo"]], [y["ci_hi"] - y["rate"]]],
            fmt="o",
            capsize=2,
        )
        ax.annotate(
            c, (x["rate"], y["rate"]), textcoords="offset points", xytext=(4, 4), fontsize=8
        )
    ax.set_xlabel("insider exfiltration ASR (lower is better)")
    ax.set_ylabel("benign gaps filled correctly (higher is better)")
    ax.set_title("Utility-security frontier")
    ax.set_xlim(-0.02, max(0.6, ax.get_xlim()[1]))
    ax.set_ylim(-0.02, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "frontier.png", dpi=180)
    plt.close(fig)

    ad = summary.get("adaptive")
    if ad:
        labels = list(ad["by_goal"])
        static = [0.0 for _ in labels]  # the sample is drawn from FIX failures: static = 0
        adaptive = [ad["by_goal"][g]["rate"] for g in labels]
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        xs = range(len(labels))
        ax.bar([x - 0.2 for x in xs], static, 0.4, label="static (sampled failures)")
        ax.bar([x + 0.2 for x in xs], adaptive, 0.4, label=f"adaptive, <= {ad['rounds']} rounds")
        for x, g in zip(xs, labels, strict=True):
            r = ad["by_goal"][g]
            ax.errorbar(
                x + 0.2,
                r["rate"],
                yerr=[[r["rate"] - r["ci_lo"]], [r["ci_hi"] - r["rate"]]],
                fmt="none",
                ecolor="black",
                capsize=2,
            )
        ax.set_xticks(list(xs), labels)
        ax.set_ylabel("success rate under FIX")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Static vs adaptive attacker (FIX)")
        fig.tight_layout()
        fig.savefig(out_dir / "static_vs_adaptive.png", dpi=180)
        plt.close(fig)
    return [str(p.name) for p in out_dir.glob("*.png")]


LABELS = {
    "B0": "B0 allow all",
    "B1": "B1 current",
    "B1t": "B1 same-team grants",
    "B2": "B2 as requester",
    "B3": "B3 no A2A",
    "B1+spot": "B1 + spotlighting",
    "FIX": "B1 + all fixes",
    "FIX+extract": "B1 + all fixes + extractive-only",
}
ATTACKS = {
    "T1": "canary query",
    "T2": "query steering",
    "T3": "passage injection",
    "T4": "spoken injection",
}


def summary_csv(
    metrics: Path,
    attacks: list[dict[str, Any]],
    cands: list[dict[str, Any]],
    confl: list[dict[str, Any]],
) -> None:
    """`summary.csv` for the paper's res-asr-matrix (research/figures/DATA-CONTRACT.md B2):
    one row per (baseline, attack), pooled over runs. Spoken injection pools the T4 lines
    into fills with the T4 decision/action windows (B1 only for the latter); conflict
    suppression was run under B1 only. Laundering was not run (no row)."""
    import csv

    rows = []
    for cond, label in LABELS.items():
        for threat, attack in ATTACKS.items():
            vals = [
                bool(r["success"])
                for r in attacks
                if r["condition"] == cond and r["threat"] == threat and r.get("success") is not None
            ]
            if threat == "T4" and cond == "B1":
                vals += [bool(c["surfaced"]) for c in cands]
            if not vals:
                continue
            k, n = sum(vals), len(vals)
            p, lo, hi = wilson(k, n)
            rows.append([label, attack, n, k, round(p, 4), round(lo, 4), round(hi, 4)])
    vals = [bool(c["success"]) for c in confl if c["success"] is not None]
    if vals:
        p, lo, hi = wilson(sum(vals), len(vals))
        rows.append(
            [
                "B1 current",
                "conflict suppression",
                len(vals),
                sum(vals),
                round(p, 4),
                round(lo, 4),
                round(hi, 4),
            ]
        )
    with (metrics / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "attack", "n", "successes", "asr", "lo", "hi"])
        w.writerows(rows)


def report(metrics: Path) -> dict[str, Any]:
    from research.redteam.run import load_corpus

    instances_list, _conflicts = load_corpus(metrics)
    instances = {i.id: i for i in instances_list}
    records = _rows(metrics / "raw/baseline_gap.jsonl") + _rows(metrics / "raw/fixes_gap.jsonl")
    v = analyze.verdicts(instances, records)
    write_jsonl(metrics / "verdicts.jsonl", v)
    attacks = [r for r in v if r["threat"] != "benign"]
    benign = [r for r in v if r["threat"] == "benign"]
    for r in attacks:
        r["goal_group"] = "integrity" if r["goal"].startswith("integrity") else r["goal"]
    summary: dict[str, Any] = {"by_condition": {}, "tables": {}}
    for row in rate_table(attacks, "success", ("condition", "goal_group")):
        summary["by_condition"].setdefault(row["condition"], {})[row["goal_group"]] = row
    for row in rate_table(attacks, "exposed", ("condition", "goal_group")):
        summary["by_condition"].setdefault(row["condition"], {})[
            f"{row['goal_group']}_exposed"
        ] = row
    for key in ("filled", "correct", "leak"):
        for row in rate_table(benign, key, ("condition",)):
            summary["by_condition"].setdefault(row["condition"], {})[f"benign_{key}"] = row
    summary["tables"]["by_threat"] = rate_table(
        attacks, "success", ("condition", "threat", "goal_group")
    )
    summary["tables"]["by_carrier"] = rate_table(
        [r for r in attacks if r["goal_group"] != "exfil"], "success", ("condition", "carrier")
    )
    summary["tables"]["by_style"] = rate_table(
        [r for r in attacks if r["goal_group"] != "exfil"], "success", ("condition", "style")
    )
    exfil_bw: dict[str, Any] = {}
    for c in {r["condition"] for r in attacks}:
        rows = [r for r in attacks if r["condition"] == c and r["goal"] == "exfil" and r["success"]]
        exfil_bw[c] = {
            "mean_leak_chars_per_success": (
                round(sum(r["leak_chars"] for r in rows) / len(rows), 1) if rows else 0.0
            )
        }
    summary["exfil_chars"] = exfil_bw
    cands = candidate_rows(metrics, instances)
    summary["candidates"] = {
        "any": rate_table(cands, "any_candidate", ("condition",)),
        "surfaced": rate_table(cands, "surfaced", ("condition",)),
        "by_seed": rate_table(cands, "surfaced", ("condition", "seed")),
        "kinds_emitted": sorted({k for r in cands for k in r["i2_kinds"]}),
    }
    confl = conflict_rows(metrics)
    summary["conflict_suppression"] = {
        "control_flag_rate": rate_table(confl, "control_flagged", ("condition",)),
        "asr": rate_table(confl, "success", ("condition",)),
        "by_style": rate_table(confl, "success", ("condition", "style")),
    }
    summary_csv(metrics, attacks, cands, confl)
    ad_path = metrics / "raw/adaptive_summary.json"
    if ad_path.exists():
        ad = json.loads(ad_path.read_text("utf-8"))
        by_goal = {}
        for g, kv in ad["by_goal"].items():
            p, lo, hi = wilson(kv["k"], kv["n"])
            by_goal[g] = {**kv, "rate": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        summary["adaptive"] = {"rounds": ad["rounds"], "n": ad["n"], "by_goal": by_goal}
    bw = metrics / "raw/bandwidth_dos.json"
    if bw.exists():
        summary["bandwidth_dos"] = json.loads(bw.read_text("utf-8"))
    proto = metrics / "raw/protocol.json"
    if proto.exists():
        summary["protocol"] = json.loads(proto.read_text("utf-8"))
    summary["figures"] = figures(metrics, summary)
    (metrics / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    lines = [
        "| condition | exfil ASR | exposed | integrity ASR | suppression ASR | benign filled "
        "| benign correct | benign leak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in ORDER:
        s = summary["by_condition"].get(c)
        if not s:
            continue
        cell = [
            fmt(s[k]) if k in s else "-"
            for k in (
                "exfil",
                "exfil_exposed",
                "integrity",
                "suppress",
                "benign_filled",
                "benign_correct",
                "benign_leak",
            )
        ]
        lines.append(f"| {c} | " + " | ".join(cell) + " |")
    (metrics / "results.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return summary
