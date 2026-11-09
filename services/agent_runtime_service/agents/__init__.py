"""The agents this runtime spawns. Each entry is a factory, so the spawner can create one
instance per participant (or per meeting) with its own lifecycle.

Plan 7-9: `PersonalListenerAgent` (P7, one per participant), `DecisionAgent` and `ActionAgent`
(P8, one per meeting, candidates only), `QuestionAgent` (P9, one per participant, ADR-0034).
ContextGap and Proactive are Plans 10-11.
"""

from __future__ import annotations

from collections.abc import Callable

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.grounding import Grounder, grounder_from_env
from services.agent_runtime_service.agents.personal_listener_agent import PersonalListenerAgent
from services.agent_runtime_service.agents.question_agent import QuestionAgent


def registered_agents(
    loader: PromptLoader, router: LiteLLMRouter, grounder: Grounder | None = None
) -> list[Callable[[], BaseAgent]]:
    """`grounder` defaults to the env-configured one (None unless QUESTION_RAG_GROUNDING=1)."""
    grounding = grounder if grounder is not None else grounder_from_env()
    return [
        lambda: PersonalListenerAgent(loader, router),
        lambda: DecisionAgent(loader, router),
        lambda: ActionAgent(loader, router),
        lambda: QuestionAgent(loader, router, grounder=grounding),
    ]
