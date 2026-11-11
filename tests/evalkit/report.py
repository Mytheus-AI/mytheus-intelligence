"""Eval results as files and, optionally, as Prometheus series.

`export(report, failures, mode=..., variant=...)`:
- with `EVAL_REPORT_DIR` set (or in live mode, default `tests/eval/reports`), writes
  `<mode>-<variant>.json` (the full report and the threshold failures) and
  `<mode>-<variant>.prom` (Prometheus text format, for a node-exporter textfile collector);
- with `EVAL_PUSHGATEWAY_URL` set, pushes the same series to a Pushgateway under job
  `mytheus-eval`.

Series (gauges): `mytheus_eval_metric{suite,metric,mode,variant}`,
`mytheus_eval_threshold_failures{mode,variant}` and `mytheus_eval_never_record_leaks{mode,variant}`
(the alert rule `NeverRecordLeakEval` in mytheus-platform fires on any leak of the current
prompts). Nothing is exported when neither variable is set and the mode is offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, generate_latest, push_to_gateway

from tests.evalkit.corpus import EVAL_DIR


def registry_for(
    report: dict[str, dict[str, Any]], failures: list[str], *, mode: str, variant: str
) -> CollectorRegistry:
    reg = CollectorRegistry()
    metric = Gauge(
        "mytheus_eval_metric",
        "Eval harness metric value.",
        ["suite", "metric", "mode", "variant"],
        registry=reg,
    )
    for suite, values in sorted(report.items()):
        for name, value in sorted(values.items()):
            if isinstance(value, int | float) and not isinstance(value, bool):
                metric.labels(suite, name, mode, variant).set(float(value))
    Gauge(
        "mytheus_eval_threshold_failures",
        "Eval metrics that missed their threshold in thresholds.json.",
        ["mode", "variant"],
        registry=reg,
    ).labels(mode, variant).set(len(failures))
    leaks = report.get("never_record", {}).get("leaks", 0)
    Gauge(
        "mytheus_eval_never_record_leaks",
        "Never-record canaries or terms that left the agent tier in the red-team eval.",
        ["mode", "variant"],
        registry=reg,
    ).labels(mode, variant).set(float(leaks))
    return reg


def export(
    report: dict[str, dict[str, Any]], failures: list[str], *, mode: str, variant: str
) -> Path | None:
    folder = os.environ.get("EVAL_REPORT_DIR", "").strip()
    target = Path(folder) if folder else (EVAL_DIR / "reports" if mode.startswith("live") else None)
    reg = registry_for(report, failures, mode=mode, variant=variant)
    written = None
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        body = {"mode": mode, "variant": variant, "failures": failures, "report": report}
        written = target / f"{mode}-{variant}.json"
        written.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", "utf-8")
        (target / f"{mode}-{variant}.prom").write_bytes(generate_latest(reg))
    gateway = os.environ.get("EVAL_PUSHGATEWAY_URL", "").strip()
    if gateway:
        push_to_gateway(gateway, job="mytheus-eval", registry=reg, grouping_key={"mode": mode})
    return written
