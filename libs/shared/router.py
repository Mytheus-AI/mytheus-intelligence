"""`LiteLLMRouter`: tiered model routing with the budget gate, prompt caching and metering on
every call. Plan 7-9 section 3.2 steps 7, 8, 10 and 11.

Tiers (mytheus-docs ADR-0027 D2, ADR-0029 D2). Agents name a tier, never a model:

    triage      -> Haiku 5.5  (`claude-haiku-5-5`)   relevance pre-gate, scoring
    extract     -> Sonnet 5.5 (`claude-sonnet-5-5`)  notes, decisions, actions, questions
    high_stakes -> unmapped until Plan 10; asking for it raises (no silent Opus)

Every call, in order:
0. The user message is passed through the egress redactor (`libs/shared/pii.py`): emails,
   phone numbers, SSN- and card-shaped numbers are replaced before anything leaves for
   Anthropic, unless the tenant turned `pii_redaction_enabled` off. The system prompt is a
   reviewed template and carries no meeting text, so it is not redacted (and stays cacheable).
1. `BudgetGate.check(org_id, est_tokens)`. `block` raises `BudgetBlocked` and no model is
   called; `throttle` downgrades the call to `triage`.
2. The system prompt is sent as a content block with `cache_control: ephemeral`, so
   Anthropic caches the stable prefix (template, taxonomy) across windows and participants.
3. Optional streaming: each text delta is handed to `on_delta(seq, text)` (the runtime puts
   them on `agent-stream`).
Anthropic API keys that are not scoped to a workspace must name one on every request
(`anthropic-workspace-id`); set `ANTHROPIC_WORKSPACE_ID` and it is sent as that header.

4. Exactly one `usage-events` `agent_token` record per call, with the tier name, keyed
   `usage:{org_id}:agent_token:{occurred_at}:{call_id}`. It is emitted as soon as the
   provider answers, before the caller parses anything, so a parse failure is still metered.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from libs.shared.budget import BudgetGate
from libs.shared.pii import redact_always

logger = logging.getLogger(__name__)

Tier = Literal["triage", "extract", "high_stakes"]

TIER_MODELS: dict[str, str] = {
    "triage": "anthropic/claude-haiku-5-5",
    "extract": "anthropic/claude-sonnet-5-5",
}
TIER_NAMES: dict[str, str] = {"triage": "haiku", "extract": "sonnet", "high_stakes": "opus"}


class BudgetBlocked(Exception):
    """The tenant is at its hard cap (with enforcement on): no model was called."""


class UsageSink(Protocol):
    async def usage(
        self, *, org_id: UUID, user_id: UUID | None, tokens: int, model_tier: str, call_id: str
    ) -> None: ...


@dataclass(frozen=True)
class Completion:
    text: str
    tier: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


OnDelta = Callable[[int, str], Awaitable[None]]
Redactor = Callable[[UUID, str], Awaitable[str]]
CompletionFn = Callable[..., Awaitable[Any]]


def _default_completion() -> CompletionFn:
    import litellm

    return litellm.acompletion  # type: ignore[no-any-return]


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


def _text_of(resp: Any) -> str:
    choice = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
    message = choice["message"] if isinstance(choice, dict) else choice.message
    content = message["content"] if isinstance(message, dict) else message.content
    return content or ""


class LiteLLMRouter:
    def __init__(
        self,
        budget: BudgetGate,
        sink: UsageSink,
        completion_fn: CompletionFn | None = None,
        models: dict[str, str] | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._budget = budget
        self._redact = redactor or redact_always
        self._sink = sink
        self._complete = completion_fn or _default_completion()
        self._models = models or TIER_MODELS
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self._extra: dict[str, Any] = (
            {"extra_headers": {"anthropic-workspace-id": workspace}} if workspace else {}
        )

    async def complete(
        self,
        *,
        tier: Tier,
        system: str,
        user: str,
        org_id: UUID,
        user_id: UUID | None,
        max_tokens: int,
        est_tokens: int | None = None,
        on_delta: OnDelta | None = None,
    ) -> Completion:
        if tier not in self._models:
            raise ValueError(f"tier {tier!r} has no model mapped (ADR-0029 D2)")
        user = await self._redact(org_id, user)
        estimate = est_tokens if est_tokens is not None else max_tokens + len(user) // 3
        decision = await self._budget.check(org_id, estimate)
        if decision == "block":
            raise BudgetBlocked(str(org_id))
        if decision == "throttle":
            tier = "triage"
        model = self._models[tier]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": user},
        ]
        call_id = str(uuid.uuid4())
        if on_delta is None:
            resp = await self._complete(
                model=model, messages=messages, max_tokens=max_tokens, **self._extra
            )
            usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
            in_tok, out_tok = _usage_tokens(usage)
            await self._meter(org_id, user_id, in_tok + out_tok, tier, call_id)
            text = _text_of(resp)
        else:
            text, in_tok, out_tok = await self._stream(model, messages, max_tokens, on_delta)
            await self._meter(org_id, user_id, in_tok + out_tok, tier, call_id)
        return Completion(
            text=text, tier=tier, model=model, input_tokens=in_tok, output_tokens=out_tok
        )

    async def _meter(
        self, org_id: UUID, user_id: UUID | None, tokens: int, tier: str, call_id: str
    ) -> None:
        await self._sink.usage(
            org_id=org_id,
            user_id=user_id,
            tokens=tokens,
            model_tier=TIER_NAMES[tier],
            call_id=call_id,
        )

    async def _stream(
        self, model: str, messages: list[dict[str, Any]], max_tokens: int, on_delta: OnDelta
    ) -> tuple[str, int, int]:
        stream = await self._complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **self._extra,
        )
        parts: list[str] = []
        in_tok = out_tok = 0
        seq = 0
        async for chunk in stream:
            usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
            if usage:
                in_tok, out_tok = _usage_tokens(usage)
            choices = chunk.get("choices") if isinstance(chunk, dict) else chunk.choices
            if not choices:
                continue
            delta = choices[0]["delta"] if isinstance(choices[0], dict) else choices[0].delta
            piece = (delta.get("content") if isinstance(delta, dict) else delta.content) or ""
            if piece:
                parts.append(piece)
                await on_delta(seq, piece)
                seq += 1
        return "".join(parts), in_tok, out_tok
