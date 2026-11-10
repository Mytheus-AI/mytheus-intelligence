"""What the agents learned from people's feedback, put into each participant's context (polish
program wave 2, "Agents that learn and wrap up").

Three things per participant, all from core-api (the AI tier never reads Postgres):

1. **Their thresholds.** `AgentPolicyClient.policy` (`GET /internal/agent-policy`) carries the
   workspace's calibrated bands and each person's own for the user-scoped agents;
   `CONTEXT_KEY` becomes the person's merged map (`AgentPolicy.merged_for`), so the agents band
   `surfaced` exactly as core-api's store and reload do.
2. **Guidance examples** (`GET /internal/agent-guidance?user_id=`): per agent, recent items
   people confirmed, corrected and dismissed. For the meeting-wide DecisionAgent and ActionAgent
   they are the workspace's (never from a confidential meeting); for the QuestionAgent and the
   PersonalListenerAgent they are only this person's own (their suggestions and notes are
   private). Rendered by `guidance.py` into the versioned `{guidance}` prompt block.
3. **Their memory across meetings** (`GET /internal/person-memory/{user_id}`): top topics,
   recurring questions, open actions and promises. Rendered by `person_memory.py` into the
   `{person_memory}` block (PersonalListenerAgent notes, QuestionAgent suggestions) and the
   `{memory_focus}` gate line. Empty when the person switched learning off.

Same caller scheme as the policy (`X-Mytheus-Service: agent-runtime-service`,
`CORE_CALLER_TOKEN`, `X-Mytheus-Org`), cached per (org, user) for `LEARNING_TTL_S` (default
60 s); a failure is "nothing learned", cached 15 s, so an outage costs one call per user per
15 s and the prompts render exactly as without learning. `invalidate` drops the caches when an
`agent-feedback` event arrives (`feedback.py`), so a confirm or dismiss shapes the next window.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx

from services.agent_runtime_service.policy import CONTEXT_KEY, SERVICE_NAME, AgentPolicyClient

logger = logging.getLogger(__name__)

GUIDANCE_KEY = "guidance"  # context key: {agent: Guidance}
MEMORY_KEY = "person_memory"  # context key: PersonMemory | None
DEFAULT_TTL_S = 60.0
FAILURE_TTL_S = 15.0
MAX_TEXT = 280


@dataclass(frozen=True)
class Example:
    action: str  # confirm | dismiss | edit
    text: str
    before: str | None = None


@dataclass(frozen=True)
class Guidance:
    scope: str  # workspace | person
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class OpenItem:
    content: str
    due: str | None = None


@dataclass(frozen=True)
class PersonMemory:
    topics: tuple[str, ...] = ()
    recurring_questions: tuple[str, ...] = ()
    open_actions: tuple[OpenItem, ...] = ()
    open_promises: tuple[OpenItem, ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.topics or self.recurring_questions or self.open_actions or self.open_promises
        )


def _text(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    flat = " ".join(raw.split())
    return flat[:MAX_TEXT] or None


def parse_guidance(body: object) -> dict[str, Guidance]:
    agents = body.get("agents") if isinstance(body, Mapping) else None
    out: dict[str, Guidance] = {}
    for agent, block in (agents.items() if isinstance(agents, Mapping) else []):
        if not isinstance(block, Mapping):
            continue
        examples: list[Example] = []
        for raw in block.get("examples") or []:
            if not isinstance(raw, Mapping):
                continue
            action, text = raw.get("action"), _text(raw.get("text"))
            if action not in {"confirm", "dismiss", "edit"} or text is None:
                continue
            before = _text(raw.get("before")) if action == "edit" else None
            if action == "edit" and before is None:
                continue
            examples.append(Example(action=str(action), text=text, before=before))
        if examples:
            scope = "person" if block.get("scope") == "person" else "workspace"
            out[str(agent)] = Guidance(scope=scope, examples=tuple(examples))
    return out


def _items(raw: object) -> tuple[OpenItem, ...]:
    out: list[OpenItem] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, Mapping) and (content := _text(item.get("content"))):
            out.append(OpenItem(content=content, due=_text(item.get("due"))))
    return tuple(out)


def _strings(raw: object) -> tuple[str, ...]:
    return tuple(t for t in (_text(v) for v in (raw if isinstance(raw, list) else [])) if t)


def parse_memory(body: object) -> PersonMemory | None:
    if not isinstance(body, Mapping) or body.get("learning_enabled") is False:
        return None
    memory = PersonMemory(
        topics=_strings(body.get("topics")),
        recurring_questions=_strings(body.get("recurring_questions")),
        open_actions=_items(body.get("open_actions")),
        open_promises=_items(body.get("open_promises")),
    )
    return None if memory.empty else memory


@dataclass
class _Cache:
    entries: dict[tuple[UUID, UUID | None], tuple[float, Any]] = field(default_factory=dict)


class LearningClient:
    """Guidance examples and person memory from core-api, cached; never raises."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        policy: AgentPolicyClient | None = None,
        ttl_s: float | None = None,
        timeout_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = (base_url or os.environ.get("CORE_URL", "")).strip().rstrip("/")
        self._token = (
            token if token is not None else os.environ.get("CORE_CALLER_TOKEN", "")
        ).strip()
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get("LEARNING_TTL_S", "") or DEFAULT_TTL_S)
            except ValueError:
                ttl_s = DEFAULT_TTL_S
        self._ttl = max(0.0, ttl_s)
        self._timeout = timeout_s
        self._clock = clock
        self._transport = transport
        self._policy = policy
        self._guidance = _Cache()
        self._memory = _Cache()
        self.fetches = 0

    @property
    def configured(self) -> bool:
        return bool(self._url and self._token)

    def invalidate(self, org_id: UUID, user_id: UUID | None = None) -> None:
        """Feedback arrived: drop this person's guidance and memory, and every guidance entry
        of the org (the workspace examples the meeting agents see changed for everyone)."""
        for key in [k for k in self._guidance.entries if k[0] == org_id]:
            self._guidance.entries.pop(key, None)
        if user_id is None:
            for key in [k for k in self._memory.entries if k[0] == org_id]:
                self._memory.entries.pop(key, None)
        else:
            self._memory.entries.pop((org_id, user_id), None)
        if self._policy is not None:
            self._policy.invalidate(org_id)

    async def guidance(self, org_id: UUID, user_id: UUID | None) -> dict[str, Guidance]:
        path = "/internal/agent-guidance" + (f"?user_id={user_id}" if user_id else "")
        found = await self._cached(self._guidance, org_id, user_id, path, parse_guidance, {})
        return found if isinstance(found, dict) else {}

    async def memory(self, org_id: UUID, user_id: UUID) -> PersonMemory | None:
        found = await self._cached(
            self._memory, org_id, user_id, f"/internal/person-memory/{user_id}", parse_memory, None
        )
        return found if isinstance(found, PersonMemory) else None

    async def enrich(
        self, org_id: UUID, contexts: dict[UUID, dict[str, Any] | None]
    ) -> dict[UUID, dict[str, Any] | None]:
        """Each participant's context with their thresholds, guidance and memory. A context
        that is None (no user context: the agents skip that user) stays None."""
        policy = await self._policy.policy(org_id) if self._policy is not None else None
        users = [u for u, c in contexts.items() if c is not None]
        found = await asyncio.gather(
            *(self.guidance(org_id, u) for u in users),
            *(self.memory(org_id, u) for u in users),
        )
        guidance = dict(zip(users, found[: len(users)], strict=True))
        memory = dict(zip(users, found[len(users) :], strict=True))
        out: dict[UUID, dict[str, Any] | None] = {}
        for user, context in contexts.items():
            if context is None:
                out[user] = None
                continue
            extra: dict[str, Any] = {GUIDANCE_KEY: guidance[user], MEMORY_KEY: memory[user]}
            if policy is not None:
                extra[CONTEXT_KEY] = policy.merged_for(user)
            out[user] = {**context, **extra}
        return out

    async def _cached(
        self,
        cache: _Cache,
        org_id: UUID,
        user_id: UUID | None,
        path: str,
        parse: Callable[[object], Any],
        empty: Any,
    ) -> Any:
        key = (org_id, user_id)
        now = self._clock()
        hit = cache.entries.get(key)
        if hit is not None and now < hit[0]:
            return hit[1]
        value, ok = await self._fetch(org_id, path, parse, empty)
        ttl = self._ttl if ok else min(self._ttl, FAILURE_TTL_S)
        if ttl > 0:
            cache.entries[key] = (now + ttl, value)
        return value

    async def _fetch(
        self, org_id: UUID, path: str, parse: Callable[[object], Any], empty: Any
    ) -> tuple[Any, bool]:
        if not self.configured:
            return empty, True  # unconfigured: nothing learned, prompts render as before
        self.fetches += 1
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("learning read %s unreachable org=%s: %s", path, org_id, exc)
            return empty, False
        if res.status_code != 200:
            logger.warning("learning read %s org=%s answered %s", path, org_id, res.status_code)
            return empty, False
        try:
            return parse(res.json()), True
        except ValueError:
            return empty, False
