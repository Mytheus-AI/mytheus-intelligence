"""Agent quality telemetry: per agent and per workspace (polish plan, "Quality telemetry").

The questions the dashboard answers (mytheus-platform `observability/grafana/dashboards/
agents.json`), each from the place in the code that already decides it:

- **Time to card**: `mytheus_agent_time_to_card_seconds{agent,org_id}` (histogram, defined in
  `libs/shared/metrics.py` and observed by the emitter where a windowed result leaves).
- **Gate pass rate**: `mytheus_agent_gate_total{agent,gate,outcome,org_id}`.
  `gate="relevance"`, `outcome` pass|fail: the Haiku pre-gate in front of an extraction
  (`BaseAgent.process` reads it off the final state: a pre-gate ran when `relevance_score` is
  set, and it passed when the extraction node ran, i.e. `confidence` is set).
  `gate="confidence"`, `outcome` surfaced|silent|dropped: the doc 08 band of what the extraction
  produced. surfaced and silent are counted where a result is emitted (`Emitter.agent_result`,
  every agent including ProactiveAgent's event-driven alerts); dropped where an extraction ran and
  produced nothing (`BaseAgent.process`). "Share shown" is surfaced / (surfaced+silent+dropped).
- **Parse failures**: `mytheus_agent_parsed_total{agent,org_id}` (model answers checked against
  the prompt's answer format) and `mytheus_agent_parse_failures_total{agent,org_id}` (those the
  agent's own parser could not use). Checked in the router on every completed call whose
  `prompt_ref` has a known format (`OUTPUT_FORMATS`); a prompt with no known format is not
  counted in either.
- **Tokens and spend**: `mytheus_agent_tokens_total{agent,tier,kind,org_id}` (kind
  input|output|cache_read|cache_write; input is the uncached part) and
  `mytheus_agent_spend_usd_total{agent,tier,org_id}` (list price, `libs/shared/cost.py`), per
  real model call (a response-cache hit costs nothing and is not counted).
- **Meeting time**: `mytheus_agent_meeting_seconds_total{org_id}`: transcript seconds the runtime
  processed (each closed window once). Spend per meeting-hour is
  `3600 * rate(spend) / rate(meeting_seconds)`.
- **Never record**: `mytheus_agent_never_record_skips_total{stage,org_id}`. `stage="window"`: a
  window (or the one after it) that named a never-record topic, so no agent ran; `stage="output"`:
  an agent result the never-record filter removed on its way out (the model wrote about a topic it
  was told not to: the guard caught it, but the prompt did not hold).

Labels. `agent` is the frozen class name (from the prompt key when the router is the caller,
`agent_for_prompt`), `org_id` the tenant UUID. This is a deliberate, documented exception to
`metrics.py`'s "no tenant label" rule, required by the per-workspace dashboard: the label set is
bounded by the number of workspaces (tens at this stage), and nothing finer (no user, meeting or
window id) is ever a label. A value that is not a UUID is recorded as `unknown`.

Telemetry never breaks an agent: every helper swallows its own errors.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

from prometheus_client import Counter

from libs.shared.cost import usd
from libs.shared.json_answer import json_object

logger = logging.getLogger(__name__)

AGENT_GATE = Counter(
    "mytheus_agent_gate_total",
    "Agent gate outcomes: relevance pre-gate pass|fail and confidence band "
    "surfaced|silent|dropped, per agent and workspace.",
    ["agent", "gate", "outcome", "org_id"],
)
AGENT_PARSED = Counter(
    "mytheus_agent_parsed_total",
    "Model answers checked against their prompt's answer format.",
    ["agent", "org_id"],
)
AGENT_PARSE_FAILURES = Counter(
    "mytheus_agent_parse_failures_total",
    "Model answers that failed their prompt's answer format (the agent could not use them).",
    ["agent", "org_id"],
)
AGENT_TOKENS = Counter(
    "mytheus_agent_tokens_total",
    "Model tokens per agent, tier and kind (input is the uncached part).",
    ["agent", "tier", "kind", "org_id"],
)
AGENT_SPEND = Counter(
    "mytheus_agent_spend_usd_total",
    "Model spend at list price (USD) per agent and tier.",
    ["agent", "tier", "org_id"],
)
AGENT_MEETING_SECONDS = Counter(
    "mytheus_agent_meeting_seconds_total",
    "Transcript seconds the agent runtime processed (each closed window once).",
    ["org_id"],
)
AGENT_NEVER_RECORD = Counter(
    "mytheus_agent_never_record_skips_total",
    "Never-record skips: stage=window (no agent ran) or stage=output (a result was removed).",
    ["stage", "org_id"],
)

UNKNOWN = "unknown"

# Prompt key -> the frozen agent class that sends it. Longest prefix wins, so a new version or a
# `_gate` sibling needs no entry. A key no entry covers is labelled by its own key.
PROMPT_AGENTS: dict[str, str] = {
    "personal_listener": "PersonalListenerAgent",
    "decision": "DecisionAgent",
    "action": "ActionAgent",
    "question": "QuestionAgent",
    "context_gap": "ContextGapAgent",
    "proactive": "ProactiveAgent",
    "meeting_agent": "MeetingAgent",
    "minutes": "MinutesAgent",
    # a3 room signals: the MeetingAgent's post-meeting reading (room_signals.py)
    "last_word": "MeetingAgent",
    "consensus_read": "MeetingAgent",
    "resume_point": "MeetingAgent",
    "never_record_check": "NeverRecordGuard",
}

# How each prompt answers, for the parse check. ("number", key): a JSON object with a number
# under `key` (the gates). ("list", key): a JSON object with a list under `key`. ("object", key):
# a JSON object holding `key`. ("header", "CONFIDENCE"): a `CONFIDENCE: x` first line.
OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    "personal_listener_gate": ("number", "relevance"),
    "personal_listener": ("header", "CONFIDENCE"),
    "decision_gate": ("number", "score"),
    "decision_agent": ("list", "candidates"),
    "action_gate": ("number", "score"),
    "action_agent": ("list", "candidates"),
    "question_gate": ("number", "score"),
    "question_agent": ("list", "suggestions"),
    "context_gap_gate": ("number", "score"),
    "context_gap_agent": ("object", "content"),
    "meeting_agent": ("list", "updates"),
    "proactive_conflict": ("list", "conflicts"),
    "proactive_stakeholder": ("number", "score"),
    "proactive_already_decided_gate": ("number", "score"),
    "proactive_already_decided": ("list", "matches"),
    "proactive_topic_watch": ("list", "topics"),
    "last_word": ("list", "signals"),
    "consensus_read": ("list", "stances"),
    "resume_point": ("object", "open"),
}

_CONFIDENCE = re.compile(r"^\s*CONFIDENCE\s*:\s*[0-9]*\.?[0-9]+\s*$", re.IGNORECASE)


def org_label(org_id: object) -> str:
    if isinstance(org_id, UUID):
        return str(org_id)
    try:
        return str(UUID(str(org_id)))
    except (TypeError, ValueError):
        return UNKNOWN


def prompt_key(prompt_ref: str | None) -> str:
    return (prompt_ref or "").partition("@")[0]


def agent_for_prompt(prompt_ref: str | None) -> str:
    key = prompt_key(prompt_ref)
    if not key:
        return UNKNOWN
    best = max((p for p in PROMPT_AGENTS if key.startswith(p)), key=len, default=None)
    return PROMPT_AGENTS[best] if best else key


def output_ok(prompt_ref: str | None, text: str) -> bool | None:
    """Whether `text` is a usable answer for the prompt's format: the same tolerance as the
    agents' parsers (code fences and prose around one JSON object are fine). None when the prompt
    has no known format (not counted)."""
    fmt = OUTPUT_FORMATS.get(prompt_key(prompt_ref))
    if fmt is None:
        return None
    kind, key = fmt
    if kind == "header":
        return bool(_CONFIDENCE.match(text.strip().partition("\n")[0]))
    obj = json_object(text, key)  # the same reading as the agents' parsers (json_answer.py)
    if obj is None:
        return False
    value = obj.get(key)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "list":
        return isinstance(value, list)
    return key in obj


# --- observation helpers (each a single call at the point that decides) -----------------------


def observe_call(prompt_ref: str | None, org_id: object, tier: str, usage: Any, text: str) -> None:
    """One completed model call (router). `usage` is the router's `Usage`: `uncached_input`,
    `output_tokens`, `cache_read_tokens`, `cache_write_tokens`."""
    try:
        agent, org = agent_for_prompt(prompt_ref), org_label(org_id)
        uncached = int(usage.uncached_input)
        out = int(usage.output_tokens)
        read, write = int(usage.cache_read_tokens), int(usage.cache_write_tokens)
        for kind, n in (
            ("input", uncached),
            ("output", out),
            ("cache_read", read),
            ("cache_write", write),
        ):
            if n > 0:
                AGENT_TOKENS.labels(agent, tier, kind, org).inc(n)
        if tier in ("triage", "extract", "high_stakes"):
            spend = usd(
                tier,
                input_tokens=uncached,
                output_tokens=out,
                cache_read_tokens=read,
                cache_write_tokens=write,
            )
            if spend > 0:
                AGENT_SPEND.labels(agent, tier, org).inc(spend)
        ok = output_ok(prompt_ref, text)
        if ok is not None:
            AGENT_PARSED.labels(agent, org).inc()
            if not ok:
                AGENT_PARSE_FAILURES.labels(agent, org).inc()
    except Exception:  # noqa: BLE001 - telemetry never breaks a call
        logger.debug("observe_call failed", exc_info=True)


def observe_gate(agent: str, org_id: object, gate: str, outcome: str) -> None:
    try:
        AGENT_GATE.labels(agent, gate, outcome, org_label(org_id)).inc()
    except Exception:  # noqa: BLE001
        logger.debug("observe_gate failed", exc_info=True)


def observe_window(agent: str, org_id: object, final: Any) -> None:
    """After `BaseAgent.process`: the relevance pre-gate outcome (when the agent has one) and a
    `dropped` band when the extraction ran and produced nothing. A produced result is counted
    surfaced|silent where it is emitted (`observe_result`), after the never-record filter."""
    try:
        relevance = getattr(final, "relevance_score", None)
        extracted = getattr(final, "confidence", None) is not None
        if relevance is not None:
            observe_gate(agent, org_id, "relevance", "pass" if extracted else "fail")
        if extracted and getattr(final, "result", None) is None:
            observe_gate(agent, org_id, "confidence", "dropped")
    except Exception:  # noqa: BLE001
        logger.debug("observe_window failed", exc_info=True)


def surfaced_of(result: Any) -> bool:
    """Whether an emitted result is shown: its payload's `surfaced` (a note, a meeting update),
    an alert's `alert.surfaced`, any surfaced candidate, else confidence > 0.7."""
    payload = getattr(result, "payload", None) or {}
    if isinstance(payload.get("surfaced"), bool):
        return bool(payload["surfaced"])
    alert = payload.get("alert")
    if isinstance(alert, dict) and isinstance(alert.get("surfaced"), bool):
        return bool(alert["surfaced"])
    for key in ("candidates", "suggestions"):
        items = payload.get(key)
        if isinstance(items, list) and items:
            return any(isinstance(i, dict) and i.get("surfaced") is True for i in items)
    return float(getattr(result, "confidence", 0.0) or 0.0) > 0.7


def observe_result(agent: str, org_id: object, result: Any) -> None:
    """An emitted `agent-results` record: its confidence band, surfaced or silent."""
    try:
        observe_gate(agent, org_id, "confidence", "surfaced" if surfaced_of(result) else "silent")
    except Exception:  # noqa: BLE001
        logger.debug("observe_result failed", exc_info=True)


def observe_meeting_window(window: Any) -> None:
    """One closed window the runtime processed: its transcript seconds."""
    try:
        seconds = max(0, int(window.end_ms) - int(window.start_ms)) / 1000
        if seconds > 0:
            AGENT_MEETING_SECONDS.labels(org_label(window.org_id)).inc(seconds)
    except Exception:  # noqa: BLE001
        logger.debug("observe_meeting_window failed", exc_info=True)


def never_record_skip(stage: str, org_id: object) -> None:
    try:
        AGENT_NEVER_RECORD.labels(stage, org_label(org_id)).inc()
    except Exception:  # noqa: BLE001
        logger.debug("never_record_skip failed", exc_info=True)
