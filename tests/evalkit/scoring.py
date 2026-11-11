"""Metric arithmetic for the eval harness. Pure functions, no agents, no I/O.

Candidates (DecisionAgent / ActionAgent), per agent over the whole corpus. Emitted candidates
(surfaced and silent; dropped ones never leave the agent) are matched greedily, highest confidence
first, to the window's expected items: a candidate matches the first unmatched item whose every
keyword group appears in its content. Then:
- precision / recall / f1 over all emitted candidates;
- surfaced_precision: of the candidates a user is shown (> 0.7), the share that are real;
- brier: mean over emitted candidates of (confidence - matched)^2, lower is better calibrated;
- lookalike_fp_rate: of the lookalike statements of this agent's kind (proposals, wishes, past
  decisions), the share that came back as a SURFACED candidate;
- assignee_accuracy (actions): matched candidates whose assignee is one of the expected
  speaker's labels (churned labels included).

Persona pairs: per pair, the facts (the window's decisions and actions) each note captures, by
the same keyword groups. fact_overlap is |A and B| / |A or B| (Jaccard; 0 when both missed all),
shared_coverage is |A and B| / |facts|, both averaged over pairs.

The rest are precision / recall / false-positive rates over sets, documented where used.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def keyword_match(content: str, keywords: list[str]) -> bool:
    text = content.lower()
    return all(any(alt in text for alt in group.split("|")) for group in keywords)


def match(
    emitted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    free = list(range(len(expected)))
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for cand in sorted(emitted, key=lambda c: -float(c["confidence"])):
        hit = next(
            (i for i in free if keyword_match(str(cand["content"]), expected[i]["keywords"])), None
        )
        if hit is not None:
            free.remove(hit)
        pairs.append((cand, expected[hit] if hit is not None else None))
    return pairs


def _ratio(n: float, d: float, empty: float = 0.0) -> float:
    return n / d if d else empty


def candidates(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Each row: `emitted` (candidates with content, confidence, surfaced, optional `assignee`
    label), `expected` items, `lookalikes` of this kind, and `assignee_ok(item, label)`."""
    tp = fp = fn = 0
    shown = shown_ok = 0
    squared: list[float] = []
    looks = looks_shown = 0
    assigned = assigned_ok = 0
    for row in rows:
        pairs = match(row["emitted"], row["expected"])
        hits = sum(1 for _, e in pairs if e is not None)
        tp += hits
        fp += len(pairs) - hits
        fn += len(row["expected"]) - hits
        for cand, exp in pairs:
            squared.append((float(cand["confidence"]) - (exp is not None)) ** 2)
            if cand.get("surfaced"):
                shown += 1
                shown_ok += exp is not None
            if exp is not None and "assignee" in exp and "assignee_ok" in row:
                assigned += 1
                assigned_ok += bool(row["assignee_ok"](exp, cand.get("assignee")))
        stray = [c for c, e in pairs if e is None and c.get("surfaced")]
        for lk in row["lookalikes"]:
            looks += 1
            looks_shown += any(keyword_match(str(c["content"]), lk["keywords"]) for c in stray)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    out: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_ratio(2 * precision * recall, precision + recall), 4),
        "surfaced_precision": round(_ratio(shown_ok, shown, 1.0), 4),
        "brier": round(_ratio(sum(squared), len(squared)), 4),
        "lookalike_fp_rate": round(_ratio(looks_shown, looks), 4),
        "emitted": len(squared),
    }
    if assigned:
        out["assignee_accuracy"] = round(assigned_ok / assigned, 4)
    return out


def captured(note: str, facts: list[dict[str, Any]]) -> set[int]:
    return {i for i, f in enumerate(facts) if keyword_match(note, f["keywords"])}


def fact_overlap(rows: Iterable[tuple[str, str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Each row: (note A, note B, facts)."""
    overlaps: list[float] = []
    coverage: list[float] = []
    for a, b, facts in rows:
        ca, cb = captured(a, facts), captured(b, facts)
        both, either = ca & cb, ca | cb
        overlaps.append(_ratio(len(both), len(either)))
        coverage.append(_ratio(len(both), len(facts)))
    return {
        "pairs": len(overlaps),
        "fact_overlap": round(_ratio(sum(overlaps), len(overlaps)), 4),
        "shared_coverage": round(_ratio(sum(coverage), len(coverage)), 4),
    }


def sets(got: list[set[Any]], want: list[set[Any]]) -> tuple[float, float, float, int, int, int]:
    """Micro precision / recall / f1 over per-case sets, plus tp / fp / fn."""
    tp = sum(len(g & w) for g, w in zip(got, want, strict=True))
    fp = sum(len(g - w) for g, w in zip(got, want, strict=True))
    fn = sum(len(w - g) for g, w in zip(got, want, strict=True))
    p, r = _ratio(tp, tp + fp, 1.0), _ratio(tp, tp + fn, 1.0)
    return p, r, _ratio(2 * p * r, p + r), tp, fp, fn


def binary(got: list[bool], want: list[bool]) -> dict[str, float]:
    """Precision / recall of a yes-no outcome and its false-positive rate on the negatives."""
    tp = sum(1 for g, w in zip(got, want, strict=True) if g and w)
    fp = sum(1 for g, w in zip(got, want, strict=True) if g and not w)
    pos, neg = sum(want), len(want) - sum(want)
    return {
        "precision": round(_ratio(tp, tp + fp, 1.0), 4),
        "recall": round(_ratio(tp, pos, 1.0), 4),
        "false_positive_rate": round(_ratio(fp, neg), 4),
        "accuracy": round(
            _ratio(sum(1 for g, w in zip(got, want, strict=True) if g == w), len(want)), 4
        ),
    }


def check(report: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> list[str]:
    """Every threshold the report misses, as readable lines. A metric the report lacks fails."""
    failures: list[str] = []
    for suite, limits in thresholds.items():
        if suite.startswith("_") or suite not in report:
            continue
        for metric, bound in limits.items():
            value = report[suite].get(metric)
            if not isinstance(value, int | float):
                failures.append(f"{suite}.{metric}: missing")
                continue
            if "min" in bound and value < bound["min"]:
                failures.append(f"{suite}.{metric} = {value} < min {bound['min']}")
            if "max" in bound and value > bound["max"]:
                failures.append(f"{suite}.{metric} = {value} > max {bound['max']}")
    return failures
