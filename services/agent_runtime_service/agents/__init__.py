"""The agents this runtime spawns. Each entry is a factory, so the spawner can create one
instance per participant (or per meeting) with its own lifecycle.

Plan 7-9: `PersonalListenerAgent` (P7, one per participant), `DecisionAgent` and `ActionAgent`
(P8, one per meeting, candidates only), `QuestionAgent` (P9, one per participant, ADR-0034),
`ContextGapAgent` (P10, one per meeting, one result per requester, ADR-0013). Proactive is P11.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.grounding import Grounder, grounder_from_env
from services.agent_runtime_service.agents.personal_listener_agent import PersonalListenerAgent
from services.agent_runtime_service.agents.question_agent import QuestionAgent

if TYPE_CHECKING:
    from services.agent_runtime_service.agents.context_gap_agent import ContextGapDeps


def registered_agents(
    loader: PromptLoader,
    router: LiteLLMRouter,
    grounder: Grounder | None = None,
    context_gap: ContextGapDeps | None = None,
) -> list[Callable[[], BaseAgent]]:
    """`grounder` defaults to the env-configured one (None unless QUESTION_RAG_GROUNDING=1).
    `context_gap` defaults to RAG only from env (silent without `RAG_CALLER_TOKEN`); the
    consumer passes the full A2A wiring."""
    # Imported here, not at the top: context_gap_agent imports the a2a package, which imports
    # helpers from this package (grounding, candidates), and a top-level import would cycle.
    from services.agent_runtime_service.agents.context_gap_agent import (
        ContextGapAgent,
        deps_from_env,
    )

    grounding = grounder if grounder is not None else grounder_from_env()
    gap = context_gap if context_gap is not None else deps_from_env()
    return [
        lambda: PersonalListenerAgent(loader, router),
        lambda: DecisionAgent(loader, router),
        lambda: ActionAgent(loader, router),
        lambda: QuestionAgent(loader, router, grounder=grounding),
        lambda: ContextGapAgent(loader, router, deps=gap),
    ]
