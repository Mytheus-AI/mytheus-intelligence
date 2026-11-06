"""`POST /ner/detect`, `/readyz`. See the service README for the full contract.

Design (ADR-0020): ner-service DETECTS, it never redacts, never stores, and returns offsets and
types only, never the matched text. Logs carry org_id and counts/lengths, never text.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from libs.shared.auth import verified_org_id
from services.ner_service.model import ModelHolder, NerModel, SpacyNerModel
from services.ner_service.regex_detector import detect_regex
from services.ner_service.spans import Span, resolve_spans

logger = logging.getLogger("ner_service")

DEFAULT_MAX_CHARS = 100_000
# JSON can inflate text (escapes, 4-byte chars); allow headroom over chars before parsing.
_BODY_BYTES_PER_CHAR = 6


def max_chars() -> int:
    return int(os.environ.get("NER_MAX_CHARS", DEFAULT_MAX_CHARS))


class DetectRequest(BaseModel):
    text: str
    # Opt-in degraded mode: if the model is unavailable, return regex-only spans (200,
    # `partial=true`) instead of 503. Default False => callers fail closed.
    allow_regex_fallback: bool = False


class EntitySpan(BaseModel):
    type: str
    start: int
    end: int
    score: float
    source: str


class DetectResponse(BaseModel):
    spans: list[EntitySpan]
    model: str
    mode: Literal["model", "regex_only"]
    partial: bool  # True => PERSON/ORG/LOC were NOT checked (regex_only)
    offset_unit: Literal["codepoint"] = "codepoint"


def create_ner_app(model: NerModel | None = None) -> FastAPI:
    app = FastAPI(title="ner-service-ner")
    holder = ModelHolder(model or SpacyNerModel())
    app.state.holder = holder

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        ok = await run_in_threadpool(holder.ensure_loaded)
        if not ok:
            raise HTTPException(status_code=503, detail="NER model not loaded")
        return {"status": "ready", "model": holder.model.name}

    async def _size_guard(request: Request) -> None:
        cl = request.headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > max_chars() * _BODY_BYTES_PER_CHAR + 1024:
            raise HTTPException(status_code=413, detail="Request body too large")

    @app.post("/ner/detect", response_model=DetectResponse)
    async def detect(
        req: DetectRequest,
        org_id: UUID = Depends(verified_org_id),  # noqa: B008
        _: None = Depends(_size_guard),  # noqa: B008
    ) -> DetectResponse:
        limit = max_chars()
        if len(req.text) > limit:
            raise HTTPException(status_code=413, detail=f"text exceeds {limit} characters")

        t0 = time.perf_counter()
        spans: list[Span] = detect_regex(req.text)
        mode: Literal["model", "regex_only"] = "regex_only"
        name = "regex"

        if await run_in_threadpool(holder.ensure_loaded):
            try:
                if req.text:
                    spans += await run_in_threadpool(holder.model.detect, req.text)
                mode, name = "model", holder.model.name
            except Exception as exc:  # noqa: BLE001
                logger.error("ner inference failed", extra={"error_type": type(exc).__name__})

        if mode == "regex_only" and not req.allow_regex_fallback:
            logger.warning("ner unavailable", extra={"org_id": str(org_id)})
            raise HTTPException(status_code=503, detail="NER model unavailable")

        resolved = resolve_spans(spans, len(req.text))
        logger.info(
            "ner detect",
            extra={
                "org_id": str(org_id),
                "text_len": len(req.text),
                "num_spans": len(resolved),
                "mode": mode,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )
        return DetectResponse(
            spans=[
                EntitySpan(type=s.type, start=s.start, end=s.end, score=s.score, source=s.source)
                for s in resolved
            ],
            model=name,
            mode=mode,
            partial=mode == "regex_only",
        )

    return app
