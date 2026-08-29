"""The detect path - `POST /ner`, internal-only (canon `_canon/01-service-catalog.md`).

Text in, typed spans out. `org_id` rides every request for log and rate accounting ONLY -
`ner-service` owns no data and makes no tenant-scoped storage decision from it - and is threaded
into every log line regardless, the same convention `reranker_service/rerank.py` follows.

WHY THIS ROUTE IS GUARDED DIFFERENTLY FROM `POST /rerank`

`/rerank` takes `org_id` from `libs.shared.auth.verified_org_id`, a verified Clerk session
token. This route cannot: its caller is `transcription-service`'s redaction pass, running from a
Deepgram callback with no user and no session token in scope. It uses the workload credential
instead (`libs/shared/internal_auth.py`, which explains what that is and is not worth).

Note what that means for `org_id` here, stated plainly: it is CALLER-ASSERTED, not verified. It
is safe in a way it would not be on `/rerank`, because nothing on this path is keyed by it -
no storage, no retrieval, no cache. The exposure is a caller with a valid workload credential
writing another tenant's id into this service's logs and rate counters. That is real and it is
bounded there.

THE REQUEST BODY IS UNREDACTED PII, SO IT IS NEVER LOGGED

This is the one endpoint in the program whose input is, by construction, exactly the personal
data the whole redaction pipeline exists to remove. `text` is not logged, not echoed in an
error, and not included in any metric label. The response carries `text` per span because the
caller needs it to splice - it already holds the input, so this adds no exposure - but nothing
here writes it anywhere.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from libs.shared.internal_auth import require_internal_service
from services.ner_service.model import MIN_SCORE, MODEL_NAME, detect

logger = logging.getLogger("ner_service")

# One transcript final, one document chunk. A caller sending a whole meeting in one request
# would hold a model process for seconds inside somebody's p95 < 2 s caption budget.
_MAX_CHARS = 8000


class NerRequest(BaseModel):
    org_id: UUID = Field(
        description="Log and rate accounting only. Caller-asserted - see the module docstring."
    )
    text: str = Field(min_length=1, max_length=_MAX_CHARS)
    min_score: float = Field(
        default=MIN_SCORE,
        ge=0.0,
        le=1.0,
        description="Floor for a span to be returned. Raising it trades recall for a cleaner "
        "transcript and a smaller token space; lowering it does the reverse.",
    )


class NerSpan(BaseModel):
    entity_type: str = Field(description="PERSON, ORG or LOC (ADR-0020 D3).")
    start: int = Field(description="Character offset into the request text, inclusive.")
    end: int = Field(description="Character offset into the request text, exclusive.")
    text: str
    score: float


class NerResponse(BaseModel):
    spans: list[NerSpan] = Field(
        description="Sorted by `start` and guaranteed non-overlapping, so a caller can splice "
        "right-to-left without corrupting the text."
    )
    model: str = MODEL_NAME


def create_ner_app() -> FastAPI:
    app = FastAPI(title="ner-service-ner")

    @app.post(
        "/ner",
        response_model=NerResponse,
        dependencies=[Depends(require_internal_service)],
    )
    async def ner(req: NerRequest) -> NerResponse:
        spans, elapsed = detect(req.text, min_score=req.min_score)

        # org_id on every log line, per the convention. NOTHING derived from `req.text` is
        # logged: not the text, not a span's matched substring, not a sample. `num_spans` and
        # the type histogram are the most that can be said without leaking the input.
        logger.info(
            "ner request",
            extra={
                "org_id": str(req.org_id),
                "chars": len(req.text),
                "num_spans": len(spans),
                "types": sorted({str(s["entity_type"]) for s in spans}),
                "latency_ms": round(elapsed * 1000, 2),
            },
        )

        return NerResponse(spans=[NerSpan(**s) for s in spans])

    return app
