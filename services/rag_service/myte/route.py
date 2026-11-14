"""`POST /api/rag/myte`: Ask Myte's answer stream (mytheus-docs ADR-0047 D8).

Auth: the person's Clerk session, verified exactly as `/api/rag/query` verifies it (RS256
against Clerk's JWKS, the org from the verified token's internal org claim). Service callers
are refused here: the loop calls bff's MCP tools AS THE PERSON, by forwarding this request's
`Authorization` header, so only a person's own token makes sense. No or a bad token is a plain
HTTP 401 before any stream starts.

Everything else is `200 text/event-stream`, one `event: <name>` + `data: <json>` per event, and
ends after `done` or `error`. A body that does not parse or breaks the limits is one
`error {reason: "invalid"}` event; rag-service without its secrets is
`error {reason: "not_connected"}`. While a long lookup runs, a `: keepalive` comment goes out
every 15 s so no proxy closes an idle stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from libs.shared.auth import _bearer, org_id_from, verify_token
from libs.shared.service_auth import SERVICE_HEADER
from services.rag_service.domain import Caller
from services.rag_service.myte.contracts import MyteRequest, invalid_message
from services.rag_service.myte.loop import ERROR_MESSAGES, Event, MyteLoop

logger = logging.getLogger(__name__)

KEEPALIVE_S = 15.0
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class MyteCaller:
    def __init__(self, caller: Caller, authorization: str) -> None:
        self.caller = caller
        self.authorization = authorization


def resolve_myte_caller(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
) -> MyteCaller:
    if SERVICE_HEADER in {k.lower() for k in request.headers.keys()}:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    claims = verify_token(credentials)
    assert credentials is not None  # verify_token raised otherwise
    return MyteCaller(
        Caller(org_id=org_id_from(claims), clerk_user_id=str(claims["sub"])),
        f"Bearer {credentials.credentials}",
    )


def format_event(event: Event) -> str:
    data = json.dumps(event.data, separators=(",", ":"), ensure_ascii=False, default=str)
    return f"event: {event.name}\ndata: {data}\n\n"


async def _one(name: str, data: dict[str, Any]) -> AsyncIterator[Event]:
    yield Event(name, data)


async def sse(events: AsyncIterator[Event], keepalive_s: float = KEEPALIVE_S) -> AsyncIterator[str]:
    it = aiter(events)
    pending: asyncio.Future[Event] = asyncio.ensure_future(anext(it))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield ": keepalive\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            yield format_event(event)
            pending = asyncio.ensure_future(anext(it))
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            await aclose()


async def ask_myte(
    request: Request, who: Annotated[MyteCaller, Depends(resolve_myte_caller)]
) -> StreamingResponse:
    events: AsyncIterator[Event]
    try:
        body = json.loads(await request.body())
        req = MyteRequest.model_validate(body)
    except ValidationError as exc:
        errors: list[dict[str, Any]] = [dict(e) for e in exc.errors()]
        events = _one("error", {"reason": "invalid", "message": invalid_message(errors)})
    except ValueError:
        events = _one("error", {"reason": "invalid", "message": ERROR_MESSAGES["invalid"]})
    else:
        loop: MyteLoop | None = getattr(request.app.state, "myte", None)
        if loop is None:
            events = _one(
                "error", {"reason": "not_connected", "message": ERROR_MESSAGES["not_connected"]}
            )
        else:
            events = loop.stream(req, who.caller, who.authorization)
    return StreamingResponse(sse(events), media_type="text/event-stream", headers=SSE_HEADERS)
