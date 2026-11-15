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

Integrity (journal red team, `LLM_CACHE_HMAC=1`, the default): Redis has no per-service ACL
today, and the key is predictable from the prompt, so anyone who can write Redis could plant an
"answer" that every agent would then surface as model output. Each entry therefore carries an
HMAC-SHA256 over its key and its fields under `LLM_CACHE_SECRET` (or, without one, a key
derived from `AGENT_IDENTITY_SECRET`); an entry that does not verify is a miss. With the HMAC on
and no secret at all the cache is off (fail closed): set `LLM_CACHE_SECRET` on every service
that wires a `ResponseCache` (agent-runtime-service, rag-service).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

TTL_S = 3600
_WS = re.compile(r"\s+")


HMAC_ENV = "LLM_CACHE_HMAC"
SECRET_ENV = "LLM_CACHE_SECRET"


def hmac_enabled() -> bool:
    return os.environ.get(HMAC_ENV, "1").strip() != "0"


def mac_key() -> bytes | None:
    own = os.environ.get(SECRET_ENV, "").strip()
    if len(own) >= 32:
        return own.encode()
    identity = os.environ.get("AGENT_IDENTITY_SECRET", "").strip()
    if len(identity) >= 32:
        return hmac.new(identity.encode(), b"llm-response-cache-v1", hashlib.sha256).digest()
    return None


def _mac(secret: bytes, key: str, text: str, input_tokens: int, output_tokens: int) -> str:
    material = json.dumps([key, text, input_tokens, output_tokens], ensure_ascii=False)
    return hmac.new(secret, material.encode(), hashlib.sha256).hexdigest()


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
    def __init__(self, redis: Any, ttl_s: int = TTL_S, signed: bool | None = None) -> None:
        self._redis = redis
        self._ttl = ttl_s
        self._signed = hmac_enabled() if signed is None else signed
        self._key = mac_key() if self._signed else None
        if self._signed and self._key is None:
            logger.warning("%s is on but no %s is set: the LLM cache is off", HMAC_ENV, SECRET_ENV)

    @property
    def enabled(self) -> bool:
        return not self._signed or self._key is not None

    async def get(self, key: str) -> CachedCompletion | None:
        if not self.enabled:
            return None
        try:
            raw = await self._redis.get(key)
        except Exception:
            logger.warning("llm cache read failed; treating as a miss")
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            found = CachedCompletion(
                text=str(data["text"]),
                input_tokens=int(data["input_tokens"]),
                output_tokens=int(data["output_tokens"]),
            )
        except (ValueError, KeyError, TypeError):
            return None
        if self._signed:
            assert self._key is not None
            want = _mac(self._key, key, found.text, found.input_tokens, found.output_tokens)
            if not hmac.compare_digest(want, str(data.get("mac", ""))):
                logger.warning("llm cache entry failed its MAC; treating as a miss")
                return None
        return found

    async def set(self, key: str, value: CachedCompletion) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "text": value.text,
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
        }
        if self._signed:
            assert self._key is not None
            entry["mac"] = _mac(self._key, key, value.text, value.input_tokens, value.output_tokens)
        payload = json.dumps(entry)
        try:
            await self._redis.set(key, payload, ex=self._ttl)
        except Exception:
            logger.warning("llm cache write failed; continuing uncached")
