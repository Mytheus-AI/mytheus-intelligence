"""The agents this runtime spawns. Each entry is a factory, so the spawner can create one
instance per participant (or per meeting) with its own lifecycle.

Plan 7-9 ships `PersonalListenerAgent` in P7 (work item 3). Decision/Action (P8) and
Question (P9) register here when they land; ContextGap and Proactive are Plans 10-11.
"""

from __future__ import annotations

from collections.abc import Callable

from libs.shared.base_agent import BaseAgent
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service.agents.personal_listener_agent import PersonalListenerAgent


def registered_agents(loader: PromptLoader, router: LiteLLMRouter) -> list[Callable[[], BaseAgent]]:
    return [lambda: PersonalListenerAgent(loader, router)]
