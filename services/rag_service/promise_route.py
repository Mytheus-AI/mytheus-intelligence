"""`POST /internal/promises/extract` (promise ledger): the person's agent reads one of their own
messages and returns the promises they made in it. Served by rag-service because it is the
intelligence tier's synchronous HTTP surface with a wired, budget-gated router; the extractor
itself is `services/agent_runtime_service/promises.py` (`promise_extract@vN`).

Caller: integration-service only, with its entry of `MYTHEUS_RAG_CALLER_TOKENS`
(`X-Mytheus-Service: integration-service`, its secret, `X-Mytheus-Org`). A user token is never
accepted here. Writes nothing: the answer goes back to the caller, which hands it to core-api.

503 when rag-service is not wired (no `RAG_SERVICE_TOKEN`) or the company brain cannot be read
(the never-record topics must be known before a message is sent to a model); the caller
retries.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from libs.shared.org_context import OrgContextUnavailable
from libs.shared.service_auth import ServiceAuthError, authenticate
from services.agent_runtime_service.promises import (
    ExtractRequest,
    Person,
    PromiseExtractor,
)

logger = logging.getLogger(__name__)

CALLER_TOKENS_ENV = "MYTHEUS_RAG_CALLER_TOKENS"
PROMISE_CALLERS = frozenset({"integration-service"})


class PersonIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)


class PromiseExtractIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    channel: Literal["slack", "email"]
    text: str = Field(min_length=1, max_length=8000)
    reference_date: date
    author_name: str | None = Field(default=None, max_length=200)
    occurred_at: str | None = Field(default=None, max_length=64)
    people: list[PersonIn] = Field(default_factory=list, max_length=20)
    context: str | None = Field(default=None, max_length=4000)


class PromiseOut(BaseModel):
    content: str
    due_hint: str | None
    due_date: date | None
    recipient_label: str | None
    recipient_email: str | None
    quote: str | None
    confidence: float


class PromiseExtractOut(BaseModel):
    status: Literal["ok", "budget_denied", "never_record"]
    prompt: str | None
    promises: list[PromiseOut]


def promise_caller(request: Request) -> UUID:
    """The tenant of an authenticated integration-service call; 401 otherwise."""
    headers = {k.lower(): v for k, v in request.headers.items()}
    try:
        caller = authenticate(
            headers,
            env_name=CALLER_TOKENS_ENV,
            allowed=PROMISE_CALLERS | {"agent-runtime-service", "bff"},
        )
    except ServiceAuthError:
        raise HTTPException(status_code=401, detail="Unauthenticated") from None
    if caller.service not in PROMISE_CALLERS:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    return caller.org_id


def get_extractor(request: Request) -> PromiseExtractor:
    extractor: PromiseExtractor | None = getattr(request.app.state, "promises", None)
    if extractor is None:
        raise HTTPException(status_code=503, detail="Promise extraction is not configured")
    return extractor


async def extract_promises(
    body: PromiseExtractIn,
    org_id: Annotated[UUID, Depends(promise_caller)],
    extractor: Annotated[PromiseExtractor, Depends(get_extractor)],
) -> PromiseExtractOut:
    req = ExtractRequest(
        channel=body.channel,
        text=body.text,
        reference_date=body.reference_date,
        author_name=body.author_name,
        people=tuple(Person(p.label, p.email) for p in body.people),
        context=body.context,
    )
    try:
        result = await extractor.extract(org_id, req)
    except OrgContextUnavailable:
        logger.warning("promise extraction: company context unavailable for %s", org_id)
        raise HTTPException(status_code=503, detail="Company context unavailable") from None
    return PromiseExtractOut(
        status=result.status,
        prompt=result.prompt,
        promises=[PromiseOut(**p.as_dict()) for p in result.promises],
    )
