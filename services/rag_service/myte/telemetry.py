"""Ask Myte latency (ADR-0049 R4): three histograms and one log line per answer.

- `myte_first_step_seconds{mode}`: request accepted -> the first `step` event (the first live
  sign of work). Not observed for an answer that looked nothing up.
- `myte_first_text_seconds{mode}`: request accepted -> the first `delta` (answer text).
- `myte_answer_seconds{mode,outcome}`: request accepted -> `done` (outcome `done`) or `error`
  (outcome = the error reason: budget, session, not_connected, unavailable, invalid).

Targets (ADR-0049): first step < 1 s and first text < 2.5 s at p50.

Labels are bounded enums only (`mode` ask|peer_answer, `outcome` six values): no org, person or
thread id is ever a label (`libs/shared/metrics.py` rules). Registered on the default registry at
import, like `libs/shared/agent_metrics.py`, so rag-service's `/metrics` serves them. The
per-answer log line carries the org for correlation; telemetry never breaks an answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from prometheus_client import Histogram

logger = logging.getLogger("services.rag_service.myte.latency")

_BUCKETS = (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 45, 60, 90)

FIRST_STEP = Histogram(
    "myte_first_step_seconds",
    "Ask Myte: seconds from the request to the first live step.",
    ["mode"],
    buckets=_BUCKETS,
)
FIRST_TEXT = Histogram(
    "myte_first_text_seconds",
    "Ask Myte: seconds from the request to the first answer text.",
    ["mode"],
    buckets=_BUCKETS,
)
ANSWER = Histogram(
    "myte_answer_seconds",
    "Ask Myte: seconds from the request to done or error.",
    ["mode", "outcome"],
    buckets=_BUCKETS,
)


@dataclass
class AnswerTimer:
    """One answer's timings. `seen(event)` on every emitted event; `finish(...)` once."""

    mode: str
    org_id: UUID
    started: float = field(default_factory=time.monotonic)
    first_step: float | None = None
    first_text: float | None = None
    rounds: int = 0
    tools: int = 0
    finished: bool = False

    def seen(self, name: str) -> None:
        try:
            now = time.monotonic() - self.started
            if name == "step" and self.first_step is None:
                self.first_step = now
                FIRST_STEP.labels(self.mode).observe(now)
            elif name == "delta" and self.first_text is None:
                self.first_text = now
                FIRST_TEXT.labels(self.mode).observe(now)
        except Exception:  # noqa: BLE001 - telemetry never breaks an answer
            logger.debug("myte timer failed", exc_info=True)

    def finish(self, outcome: str) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            total = time.monotonic() - self.started
            ANSWER.labels(self.mode, outcome).observe(total)
            logger.info(
                "myte answer org=%s mode=%s outcome=%s first_step_s=%s first_text_s=%s "
                "answer_s=%.3f rounds=%d tools=%d",
                self.org_id,
                self.mode,
                outcome,
                _fmt(self.first_step),
                _fmt(self.first_text),
                total,
                self.rounds,
                self.tools,
                extra={
                    "myte_mode": self.mode,
                    "myte_outcome": outcome,
                    "myte_first_step_s": self.first_step,
                    "myte_first_text_s": self.first_text,
                    "myte_answer_s": round(total, 3),
                    "myte_rounds": self.rounds,
                    "myte_tools": self.tools,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("myte timer failed", exc_info=True)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
