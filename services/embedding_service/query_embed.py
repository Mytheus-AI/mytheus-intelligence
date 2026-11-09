"""The query-embed path — plan 3.7 step 5.

`POST /embed`: 1-3 query variants, cached at `nexus:{org_id}:qembed:{query_hash}` (~10-min
TTL). Consumed later by `rag-service`'s `expand` -> `retrieve` fan-out; built here as the
contract per the plan's own scope note. This repo's internal gRPC mesh (`proto/`) is still a
placeholder (`RagQueryPlaceholder`, reserved but not yet defined) — this task builds the REST
contract for real (real Redis cache, real model), and the gRPC transport wrapper is left for
whichever task actually stands up the mesh, per CLAUDE.md's "Add an inference endpoint"
recipe step 2 ("Expose it over gRPC only for internal callers"). Routing this over gRPC instead
of REST is a transport change only; the cache key, TTL, and batching stay identical.

Real BGE-small-en query embeds (the SAME model as the ingest lane, via the SEPARATE
`QueryEmbedder` Ray Serve deployment when Ray is running — `embed_batch` directly otherwise,
so this endpoint works even without a Ray cluster in dev, matching this repo's "GPU optional
locally" convention).

Security fix (whole-band review, Important finding): `org_id` used to come from a
client-supplied `X-Org-Id` header, fail-closed only on a malformed UUID — nothing verified the
caller held a legitimate identity for that org, so a forged header could poison another
tenant's `nexus:{org_id}:qembed:` cache. `org_id` now comes ONLY from
`libs.shared.auth.verified_org_id` — see that module's docstring for the full verification
chain.
"""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis

from libs.shared.auth import verified_or_service_org_id
from services.embedding_service.model import embed_batch

# rag-service calls this as itself (mytheus-docs ADR-0033); a user's token still works.
CALLER_TOKENS_ENV = "MYTHEUS_EMBED_CALLER_TOKENS"
_caller_org_id = verified_or_service_org_id(CALLER_TOKENS_ENV, frozenset({"rag-service"}))

_QEMBED_TTL = 600  # ~10 minutes, per plan 3.7 step 5


class EmbedRequest(BaseModel):
    queries: list[str]  # 1-3 query variants


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int = 384
    cached: list[bool]


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:32]


def _cache_key(org_id: UUID, query: str) -> str:
    return f"nexus:{org_id}:qembed:{_query_hash(query)}"


def create_query_embed_app(redis: Redis) -> FastAPI:
    # Not `Redis[bytes]`: see libs/shared/consumer.py's note on this redis-py version's stubs.
    app = FastAPI(title="embedding-service-query-embed")

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(
        req: EmbedRequest,
        org_id: UUID = Depends(_caller_org_id),  # noqa: B008 - FastAPI's own required pattern
    ) -> EmbedResponse:
        if not (1 <= len(req.queries) <= 3):
            raise HTTPException(status_code=422, detail="queries must contain 1-3 variants")

        vectors: list[list[float]] = [None] * len(req.queries)  # type: ignore[list-item]
        cached_flags = [False] * len(req.queries)
        to_embed: list[tuple[int, str]] = []

        for i, q in enumerate(req.queries):
            cached = await redis.get(_cache_key(org_id, q))
            if cached is not None:
                vectors[i] = json.loads(cached)
                cached_flags[i] = True
            else:
                to_embed.append((i, q))

        if to_embed:
            fresh_vectors, _elapsed = embed_batch([q for _, q in to_embed])
            for (i, q), vec in zip(to_embed, fresh_vectors, strict=True):
                vectors[i] = vec
                await redis.set(_cache_key(org_id, q), json.dumps(vec), ex=_QEMBED_TTL)

        return EmbedResponse(vectors=vectors, cached=cached_flags)

    return app
