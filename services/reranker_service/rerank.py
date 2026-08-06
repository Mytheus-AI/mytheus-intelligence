"""The rerank path — plan 3.8 steps 2-3.

`POST /rerank`: top-50 candidates in, joint (query, passage) cross-encoder scoring, top-10 out.
`org_id` rides on every request for audit/rate accounting ONLY — reranker-service owns no data
and makes no tenant-scoped storage decision from it — but it is threaded through to every log
line regardless, per the task brief ("a request carries its org_id through to every log and
metric").

Security fix (whole-band review, Important finding): `org_id` used to come from a
client-supplied `X-Org-Id` header, same convention as `embedding-service`'s `POST /embed`,
fail-closed only on a malformed UUID — with nothing verifying the caller held a legitimate
identity for that org. Audit-log data is still sensitive even when it gates no storage
decision: a forged `org_id` here pollutes another tenant's log/metric trail. `org_id` now comes
ONLY from `libs.shared.auth.verified_org_id` — see that module's docstring for the full
verification chain.

Built here as REST for the same reason `embedding-service`'s `query_embed.py` was: this repo's
internal gRPC mesh (`proto/rag.proto`) is still a placeholder (`RagQueryPlaceholder`, reserved
but not defined), so the plan's "internal gRPC only" framing is implemented as a real REST
contract now (real model, real scoring), and the gRPC transport wrapper is left for whichever
task stands up the mesh — per `CLAUDE.md`'s "Add an inference endpoint" recipe step 2. Routing
this over gRPC later is a transport change only; the request/response shape and the no-cross-
request-batching guarantee (see `model.py`) stay identical.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from libs.shared.auth import verified_org_id
from services.reranker_service.model import rerank_batch

logger = logging.getLogger("reranker_service")

_MAX_PASSAGES = 50  # plan 3.8: "top-50 -> top-10"


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=_MAX_PASSAGES)
    top_k: int = 10


class RerankedPassage(BaseModel):
    passage: str
    score: float


class RerankResponse(BaseModel):
    results: list[RerankedPassage]
    model: str = "BAAI/bge-reranker-base"


def create_rerank_app() -> FastAPI:
    app = FastAPI(title="reranker-service-rerank")

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(
        req: RerankRequest,
        org_id: UUID = Depends(verified_org_id),  # noqa: B008 - FastAPI's own required pattern
    ) -> RerankResponse:
        top_k = max(1, min(req.top_k, len(req.passages)))
        ranked, elapsed = rerank_batch(req.query, req.passages, top_k=top_k)

        # org_id on every log line for this request, per the task brief — audit/rate
        # accounting only; it never gates which passages get scored (that's this request's
        # own req.passages, always — see model.py).
        logger.info(
            "rerank request",
            extra={
                "org_id": str(org_id),
                "num_passages": len(req.passages),
                "top_k": len(ranked),
                "latency_ms": round(elapsed * 1000, 2),
            },
        )

        return RerankResponse(results=[RerankedPassage(passage=p, score=s) for p, s in ranked])

    return app
