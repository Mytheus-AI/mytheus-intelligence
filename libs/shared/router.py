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
1b. When the caller names its `prompt_ref` and a `ResponseCache` is wired, an exact match on
   (tier, model, system, normalized user) is answered from Redis (ADR-0030 D1): no model call,
   no metering, no deltas. A miss is stored after the call.
2. The system prompt is sent as a content block with `cache_control: ephemeral`, so
   Anthropic caches the stable prefix (template, taxonomy) across windows and participants.
3. Optional streaming: each text delta is handed to `on_delta(seq, text)` (the runtime puts
   them on `agent-stream`).
Anthropic API keys that are not scoped to a workspace must name one on every request
(`anthropic-workspace-id`); set `ANTHROPIC_WORKSPACE_ID` and it is sent as that header.

4. Exactly one `usage-events` `agent_token` record per call, with the tier name, keyed
   `usage:{org_id}:agent_token:{occurred_at}:{call_id}`. It is emitted as soon as the
   provider answers, before the caller parses anything, so a parse failure is still metered.

Model settings (polish plan, Phase 0):
- Effort. Sonnet 5.5 thinks adaptively at effort `high` by default, and thinking tokens are
  output tokens: they count against `max_tokens` and are billed at the output price. Every
  extract-tier call names its `effort` deliberately (`low` for extraction: notes, decisions,
  actions; `medium` for synthesis and judgement: questions, context fills, conflict and
  already-decided checks); the router sends `thinking: {type: adaptive}` with
  `output_config: {effort}`. Triage (Haiku) keeps thinking off. A throttled call downgraded to
  triage drops the effort with the thinking.
- `max_tokens` is a ceiling, not a cost, so the agents ask for room for thinking plus the
  parsed lines (`EXTRACT_MAX_TOKENS`); the budget estimate no longer uses it (it would block a
  tenant near its cap for headroom it never spends). A response that still stops at
  `max_tokens` is logged with its prompt, never silently parsed as "nothing to say".
- Caching. The system prompt is breakpoint 1. `cached_prefix` (the org-stable head of the user
  message: the company block, the never-record line, topic owners) is sent as its own content
  block with breakpoint 2, so a company block of up to ~7k tokens is read from the cache at
  0.1x on every window of every participant instead of being paid in full.
- Metering is in budget units (`libs/shared/cost.py`): tokens weighed by the tier's list price,
  cache writes at 1.25x and reads at 0.1x the input price, one unit = one uncached Sonnet 5.5
  input token. The `tokens` field and the budget API keep their names.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from libs.shared.budget import BudgetGate, Decision
from libs.shared.cost import budget_units
from libs.shared.llm_cache import CachedCompletion, ResponseCache, cache_key
from libs.shared.pii import redact_always

logger = logging.getLogger(__name__)

Tier = Literal["triage", "extract", "high_stakes"]
Effort = Literal["low", "medium", "high"]
# Per call, deliberately: extraction (notes, decision and action candidates) reads what was said
# and writes it down, `low`; synthesis and judgement (questions to ask, context fills, conflict
# and already-decided checks) weighs several inputs, `medium`. Neither needs `high`.
EXTRACTION_EFFORT: Effort = "low"
SYNTHESIS_EFFORT: Effort = "medium"

# Room for adaptive thinking plus the parsed lines on an extract-tier call. A ceiling only: what
# is billed is what is generated. 8192 keeps non-streaming requests far from HTTP timeouts.
EXTRACT_MAX_TOKENS = 8192
EST_OUTPUT_TOKENS = 1024  # the budget estimate's output allowance per call

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
    input_tokens: int  # every input token, cached or not (as the provider reports prompt tokens)
    output_tokens: int  # thinking included
    cached: bool = False  # answered from the response cache: no model call
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    units: int = 0  # what this call cost the tenant's budget (`libs/shared/cost.py`)
    truncated: bool = False  # stopped at max_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


OnDelta = Callable[[int, str], Awaitable[None]]
Redactor = Callable[[UUID, str], Awaitable[str]]
CompletionFn = Callable[..., Awaitable[Any]]


def _default_completion() -> CompletionFn:
    import litellm

    return litellm.acompletion  # type: ignore[no-any-return]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # all prompt tokens (litellm counts cache reads and writes in them)
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def uncached_input(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens - self.cache_write_tokens)

    def units(self, tier: str) -> int:
        if not (self.input_tokens or self.output_tokens):
            return 0
        return budget_units(
            tier,
            input_tokens=self.uncached_input,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )


def _int(get: Callable[..., Any], key: str) -> int:
    try:
        return int(get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    read = _int(get, "cache_read_input_tokens")
    if not read:
        details = get("prompt_tokens_details", None)
        if details is not None:
            dget = (
                details.get if isinstance(details, dict) else lambda k, d=0: getattr(details, k, d)
            )
            read = _int(dget, "cached_tokens")
    return Usage(
        input_tokens=_int(get, "prompt_tokens"),
        output_tokens=_int(get, "completion_tokens"),
        cache_read_tokens=read,
        cache_write_tokens=_int(get, "cache_creation_input_tokens"),
    )


def _finish_reason(resp: Any) -> str:
    try:
        choice = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
        reason = choice.get("finish_reason") if isinstance(choice, dict) else choice.finish_reason
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""
    return str(reason or "")


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
        cache: ResponseCache | None = None,
    ) -> None:
        self._cache = cache
        self._budget = budget
        self._redact = redactor or redact_always
        self._sink = sink
        self._complete = completion_fn or _default_completion()
        self._models = models or TIER_MODELS
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self._extra: dict[str, Any] = (
            {"extra_headers": {"anthropic-workspace-id": workspace}} if workspace else {}
        )

    async def budget_state(self, org_id: UUID) -> Decision:
        """The tenant's budget state without spending: `ok`, `throttle` or `block` (a zero-token
        check of the same gate every call goes through). ContextGapAgent reads it to take the
        RAG-only path when throttled (plan 10-13 3.3, the R6 descope as a runtime degradation)."""
        return await self._budget.check(org_id, 0)

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
        prompt_ref: str | None = None,
        effort: Effort | None = None,
        cached_prefix: str = "",
    ) -> Completion:
        """`user` is the per-call part of the user message; `cached_prefix`, when given, is its
        org-stable head (sent first, behind its own cache breakpoint). `est_tokens` is in budget
        units; by default it is estimated from the text lengths and `EST_OUTPUT_TOKENS`."""
        if tier not in self._models:
            raise ValueError(f"tier {tier!r} has no model mapped (ADR-0029 D2)")
        user = await self._redact(org_id, user)
        prefix = await self._redact(org_id, cached_prefix) if cached_prefix else ""
        estimate = (
            est_tokens
            if est_tokens is not None
            else Usage(
                input_tokens=(len(system) + len(prefix) + len(user)) // 3,
                output_tokens=min(max_tokens, EST_OUTPUT_TOKENS),
            ).units(tier)
        )
        decision = await self._budget.check(org_id, estimate)
        if decision == "block":
            raise BudgetBlocked(str(org_id))
        if decision == "throttle":
            tier = "triage"
        model = self._models[tier]
        key = None
        if self._cache is not None and prompt_ref:
            key = cache_key(
                org_id, prompt_ref, tier=tier, model=model, system=system, user=prefix + user
            )
            hit = await self._cache.get(key)
            if hit is not None:
                return Completion(
                    text=hit.text,
                    tier=tier,
                    model=model,
                    input_tokens=hit.input_tokens,
                    output_tokens=hit.output_tokens,
                    cached=True,
                )
        user_content: str | list[dict[str, Any]] = user
        if prefix:
            # Breakpoint 2: system + this org's company block are one cached prefix.
            user_content = [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user},
            ]
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": user_content},
        ]
        call_id = str(uuid.uuid4())
        extra = dict(self._extra)
        if tier == "triage":
            # Haiku 5.5 thinks by default (effort `medium`), and a gate's few output tokens
            # were spent on thinking blocks: the call stopped at `length` with empty text, which
            # parses as score 0 and silently closed the gate. Triage calls are short
            # classifications; thinking stays off for them (allowed at effort <= high).
            extra["thinking"] = {"type": "disabled"}
        elif effort is not None:
            # Sonnet 5.5: adaptive thinking at the effort the agent chose (its default is high).
            extra["thinking"] = {"type": "adaptive"}
            extra["output_config"] = {"effort": effort}
        if on_delta is None:
            resp = await self._complete(
                model=model, messages=messages, max_tokens=max_tokens, **extra
            )
            raw = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
            usage = _usage(raw)
            units = usage.units(tier)
            await self._meter(org_id, user_id, units, tier, call_id)
            text = _text_of(resp)
            truncated = _finish_reason(resp) == "length"
        else:
            text, usage, truncated = await self._stream(
                model, messages, max_tokens, on_delta, extra
            )
            units = usage.units(tier)
            await self._meter(org_id, user_id, units, tier, call_id)
        if truncated:
            logger.warning(
                "%s stopped at max_tokens=%s (tier %s, %s output tokens): the answer may be cut",
                prompt_ref or "a call",
                max_tokens,
                tier,
                usage.output_tokens,
            )
        in_tok, out_tok = usage.input_tokens, usage.output_tokens
        if key is not None and self._cache is not None and text and not truncated:
            await self._cache.set(key, CachedCompletion(text, in_tok, out_tok))
        return Completion(
            text=text,
            tier=tier,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            units=units,
            truncated=truncated,
        )

    async def _meter(
        self, org_id: UUID, user_id: UUID | None, units: int, tier: str, call_id: str
    ) -> None:
        """One `usage-events` record; `tokens` carries budget units (`libs/shared/cost.py`)."""
        await self._sink.usage(
            org_id=org_id,
            user_id=user_id,
            tokens=units,
            model_tier=TIER_NAMES[tier],
            call_id=call_id,
        )

    async def _stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        on_delta: OnDelta,
        extra: dict[str, Any],
    ) -> tuple[str, Usage, bool]:
        stream = await self._complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )
        parts: list[str] = []
        usage = Usage()
        truncated = False
        seq = 0
        async for chunk in stream:
            raw = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
            if raw:
                usage = _usage(raw)
            choices = chunk.get("choices") if isinstance(chunk, dict) else chunk.choices
            if not choices:
                continue
            reason = (
                choices[0].get("finish_reason")
                if isinstance(choices[0], dict)
                else getattr(choices[0], "finish_reason", None)
            )
            truncated = truncated or reason == "length"
            delta = choices[0]["delta"] if isinstance(choices[0], dict) else choices[0].delta
            piece = (delta.get("content") if isinstance(delta, dict) else delta.content) or ""
            if piece:
                parts.append(piece)
                await on_delta(seq, piece)
                seq += 1
        return "".join(parts), usage, truncated
