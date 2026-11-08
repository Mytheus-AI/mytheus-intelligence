"""Exact-match LLM response cache, `nexus:{org_id}:llmcache:{prompt_ref}:{input_hash}`.

mytheus-docs ADR-0030 D1: the P7 form of ADR-0004 stage 4's "semantic cache". The hash covers
the tier, the model, the system prompt and the (already redacted) user message, after
lowercasing and collapsing whitespace, so trivially different renderings of the same window
share an entry. TTL 1 hour.

A hit calls no model, so it spends no tokens, emits no `usage-events` record and charges no
budget. The router still runs the budget check before looking here: a `blocked` tenant gets
nothing, cached or not.

The cache is an optimisation, never a dependency: every Redis error is a miss on read and a
no-op on write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

TTL_S = 3600
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def cache_key(
    org_id: UUID, prompt_ref: str, *, tier: str, model: str, system: str, user: str
) -> str:
    digest = hashlib.sha256(
        "\x1f".join((tier, model, normalize(system), normalize(user))).encode()
    ).hexdigest()
    return f"nexus:{org_id}:llmcache:{prompt_ref}:{digest}"


@dataclass(frozen=True)
class CachedCompletion:
    text: str
    input_tokens: int
    output_tokens: int


class ResponseCache:
    def __init__(self, redis: Any, ttl_s: int = TTL_S) -> None:
        self._redis = redis
        self._ttl = ttl_s

    async def get(self, key: str) -> CachedCompletion | None:
        try:
            raw = await self._redis.get(key)
        except Exception:
            logger.warning("llm cache read failed; treating as a miss")
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return CachedCompletion(
                text=str(data["text"]),
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
            )
        except (ValueError, KeyError, TypeError):
            return None

    async def set(self, key: str, value: CachedCompletion) -> None:
        payload = json.dumps(
            {
                "text": value.text,
                "input_tokens": value.input_tokens,
                "output_tokens": value.output_tokens,
            }
        )
        try:
            await self._redis.set(key, payload, ex=self._ttl)
        except Exception:
            logger.warning("llm cache write failed; continuing uncached")
