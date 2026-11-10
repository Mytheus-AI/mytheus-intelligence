"""`ActionAgent`: one per meeting, proposes action-item candidates. Never commits anything.

Plan 7-9 section 3.5 step 2; doc 08 section 2.1; Phase 8 Part A contract section 1. The graph
(Haiku pre-gate `action_gate@vN`, Sonnet extraction `action_agent@vN`, per-candidate bands,
one `action_candidate` result per window) lives in `candidates.CandidateAgent`.

Per candidate the model returns `content`, `assignee` (a "Participant N" label or null),
`due_date` (ISO date or null), `depends_on` (free text) and `confidence`. Relative dates
("by Friday") need a reference: the user message carries today's UTC date and weekday from an
injectable clock, so tests are deterministic. A `due_date` that is not a valid ISO date is
null; an `assignee` that is not a UUID speaker of the window is null. The `actions` event and
the AGE `ASSIGNED_TO` edge happen only when a user confirms in core-api.

ADR-0039 C4 (`action_agent@v2`): the user message lists the org's `topic_owners` as `T1. billing`
(never the owner's id or name). A candidate may answer `"topic": "T2"`; when that names a listed
topic, the candidate gains `suggested_owner_user_id` (the topic's owner). core-api stores it as
the candidate's assignee suggestion, or passes it through to the UI. Nothing else changes: the
model's `assignee` (a speaker in the window) stays as it was.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, ClassVar, Literal

from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.candidates import CandidateAgent, resolve_label
from services.agent_runtime_service.agents.company import (
    company_of,
    company_values,
    owner_for,
    owner_topics,
    render_owners,
)

MAX_DEPENDS_ON = 5
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def utc_today() -> date:
    return datetime.now(UTC).date()


def parse_due_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None


def parse_depends_on(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return items[:MAX_DEPENDS_ON]


class ActionAgent(CandidateAgent):
    name: ClassVar[str] = "ActionAgent"
    prompt_key: ClassVar[str] = "action_agent"
    scope: ClassVar[Literal["participant", "meeting"]] = "meeting"
    gate_key: ClassVar[str] = "action_gate"
    threshold_env: ClassVar[str] = "ACTION_GATE_THRESHOLD"
    kind: ClassVar[str] = "action_candidate"

    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        today: Callable[[], date] = utc_today,
    ) -> None:
        super().__init__(loader, router)
        self._today = today

    def user_values(self, context: dict[str, Any] | None = None) -> dict[str, str]:
        today = self._today()
        return {
            "today": f"{_WEEKDAYS[today.weekday()]} {today.isoformat()}",
            "owners": render_owners(owner_topics(company_of(context))),
            **company_values(context),
        }

    def extra(self, raw: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
        owner = owner_for(raw.get("topic"), owner_topics(company_of(context)))
        return {"suggested_owner_user_id": str(owner)} if owner is not None else {}

    def shape(self, raw: dict[str, Any], labels: dict[str, str]) -> dict[str, Any] | None:
        return {
            "assignee_id": resolve_label(raw.get("assignee"), labels),
            "due_date": parse_due_date(raw.get("due_date")),
            "depends_on": parse_depends_on(raw.get("depends_on")),
        }
