"""`DecisionAgent`: one per meeting, proposes decision candidates. Never commits anything.

Plan 7-9 section 3.5 step 1; doc 08 section 2.1; Phase 8 Part A contract section 1. The graph
(Haiku pre-gate `decision_gate@vN`, Sonnet extraction `decision_agent@vN`, per-candidate bands,
one `decision_candidate` result per window) lives in `candidates.CandidateAgent`.

Per candidate the model returns `content`, `rationale` (string or null), `stakeholders`
("Participant N" labels) and `confidence`; the payload carries `rationale` and
`stakeholder_ids` (the labels mapped back to the window's UUID speaker ids). The `decisions`
event and the AGE `PRODUCED` edge happen only when a user confirms in core-api.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from services.agent_runtime_service.agents.candidates import (
    CandidateAgent,
    resolve_labels,
    text_of,
)


class DecisionAgent(CandidateAgent):
    name: ClassVar[str] = "DecisionAgent"
    prompt_key: ClassVar[str] = "decision_agent"
    scope: ClassVar[Literal["participant", "meeting"]] = "meeting"
    gate_key: ClassVar[str] = "decision_gate"
    threshold_env: ClassVar[str] = "DECISION_GATE_THRESHOLD"
    kind: ClassVar[str] = "decision_candidate"

    def shape(self, raw: dict[str, Any], labels: dict[str, str]) -> dict[str, Any] | None:
        return {
            "rationale": text_of(raw, "rationale"),
            "stakeholder_ids": resolve_labels(raw.get("stakeholders"), labels),
        }
