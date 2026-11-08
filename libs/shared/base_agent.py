"""`BaseAgent`: the shared LangGraph lifecycle, `spawn` / `process` / `terminate`.

Plan 7-9 section 3.2 step 3, doc 08 section 3.2. A subclass declares its frozen class name,
its prompt key and its scope, and builds a LangGraph `StateGraph(AgentState)` whose nodes
return partial `AgentState` updates; the last node sets `result`.

Fail closed, in the base class so no agent can forget:
- no `org_id` at spawn: `spawn` raises (no tenant, no agent);
- the router raises `BudgetBlocked`: `process` returns None and records it on the agent, so
  the runtime can tell the participant once, and nothing else is called for that window.

`scope` decides how the spawner instantiates the agent: `participant` (one per participant per
meeting, `PersonalListenerAgent`) or `meeting` (one per meeting: Decision/Action/Question).
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal
from uuid import UUID

from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import AgentResult, AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: ClassVar[str]  # frozen canon class name, e.g. "PersonalListenerAgent"
    prompt_key: ClassVar[str]
    scope: ClassVar[Literal["participant", "meeting"]]

    def __init__(self, loader: PromptLoader, router: LiteLLMRouter) -> None:
        self._loader = loader
        self.router = router
        self.org_id: UUID | None = None
        self.meeting_id: UUID | None = None
        self.user_id: UUID | None = None
        self.prompt: Prompt | None = None
        self.blocked = False
        self._graph: Any = None

    @property
    def agent_id(self) -> str:
        return f"{self.name}:{self.meeting_id}:{self.user_id}"

    async def spawn(self, *, org_id: UUID | None, meeting_id: UUID, user_id: UUID) -> None:
        if org_id is None:
            raise ValueError(f"{self.name}: org_id is required (fail closed)")
        self.org_id, self.meeting_id, self.user_id = org_id, meeting_id, user_id
        version = self._loader.resolve(self.prompt_key, org_id)
        self.prompt = self._loader.load(self.prompt_key, version)
        self._graph = self.build_graph()

    @abstractmethod
    def build_graph(self) -> Any:
        """Return a compiled LangGraph over `AgentState`."""

    async def process(self, state: AgentState) -> AgentResult | None:
        if self._graph is None or self.prompt is None:
            raise RuntimeError(f"{self.name}.process before spawn")
        if state.org_id != self.org_id:  # one agent instance never serves two tenants
            raise ValueError(f"{self.name}: window org_id does not match the spawned agent")
        state = dataclasses.replace(
            state, agent_id=self.agent_id, prompt_version=self.prompt.version
        )
        try:
            out = await self._graph.ainvoke(state)
        except BudgetBlocked:
            self.blocked = True
            return None
        final = out if isinstance(out, AgentState) else dataclasses.replace(state, **dict(out))
        return final.result

    async def terminate(self) -> None:
        """Release per-meeting resources. Agents hold no durable state, so the base does
        nothing; a subclass with buffers flushes them here."""
        self._graph = None
