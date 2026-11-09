"""The agents this runtime spawns. Each entry is a factory, so the spawner can create one
instance per participant (or per meeting) with its own lifecycle.

Plan 7-9: `PersonalListenerAgent` (P7, one per participant), `DecisionAgent` and `ActionAgent`
(P8, one per meeting, candidates only). Question (P9) registers here when it lands;
ContextGap and Proactive are Plans 10-11.
"""

from __future__ import annotations

from collections.abc import Callable

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.personal_listener_agent import PersonalListenerAgent


def registered_agents(loader: PromptLoader, router: LiteLLMRouter) -> list[Callable[[], BaseAgent]]:
    return [
        lambda: PersonalListenerAgent(loader, router),
        lambda: DecisionAgent(loader, router),
        lambda: ActionAgent(loader, router),
    ]
