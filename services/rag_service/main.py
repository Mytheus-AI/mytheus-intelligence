"""rag-service: `POST /api/rag/query`, the frozen five-stage `RAGService` (mytheus-docs doc 09,
ADR-0033).

Two callers:
- a user, through `api-gateway`, with their Clerk session token. rag-service re-verifies it
  (the gateway is a shield, not a trust delegation); the org comes from the verified token and
  the user from its `sub`, which core-api resolves to `users.id` in the scope call;
- an agent (`agent-runtime-service`, for `QuestionAgent` grounding) with its service
  credentials (`MYTHEUS_RAG_CALLER_TOKENS`) and `X-Mytheus-User`: the participant it asks for.
  Scope is per user, so an agent sees exactly that participant's meetings;
- `bff`, asking on a signed-in user's behalf (`POST /api/v1/ask` and the MCP ask tool), with
  the same service-credential shape: `X-Mytheus-Service: bff`, its secret from
  `MYTHEUS_RAG_CALLER_TOKENS` (`bff=<secret>`), and `X-Mytheus-User` = that user's `users.id`.
  Scope is still that one user's.

Downstream, rag-service calls core-api, search-service, embedding-service, reranker-service and
knowledge-graph-service as itself (`RAG_SERVICE_TOKEN`), and reads Qdrant with the read-only key.

Env: RAG_SERVICE_TOKEN, MYTHEUS_RAG_CALLER_TOKENS, CORE_API_URL, EMBEDDING_URL and RERANKER_URL (the
service base including its `/internal` mount), SEARCH_GRPC_URL, KNOWLEDGE_GRAPH_URL (optional),
BILLING_URL, REDIS_URL, REDPANDA_BROKERS, SCHEMA_REGISTRY_URL, QDRANT_URL,
QDRANT_READ_ONLY_API_KEY, ANTHROPIC_API_KEY.

Run: uvicorn services.rag_service.main:app --port 8201
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials

from libs.shared.app import create_app
from libs.shared.auth import _bearer, org_id_from, verify_token
from libs.shared.service_auth import SERVICE_HEADER, ServiceAuthError, authenticate
from services.rag_service.clients import ScopeUnavailable
from services.rag_service.contracts import RagQueryRequest, RagResponse
from services.rag_service.domain import Caller
from services.rag_service.pipeline import AnswerCache, RAGService

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
CALLER_TOKENS_ENV = "MYTHEUS_RAG_CALLER_TOKENS"
ALLOWED_CALLERS = frozenset({"agent-runtime-service", "bff"})
USER_HEADER = "x-mytheus-user"


class NotConfigured(RuntimeError):
    """rag-service's own secrets are absent: serve 503, do not crash."""


def resolve_caller(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
) -> Caller:
    headers = {k.lower(): v for k, v in request.headers.items()}
    if SERVICE_HEADER in headers:
        try:
            caller = authenticate(headers, env_name=CALLER_TOKENS_ENV, allowed=ALLOWED_CALLERS)
            user_id = UUID(headers.get(USER_HEADER, ""))
        except (ServiceAuthError, ValueError):
            raise HTTPException(status_code=401, detail="Unauthenticated") from None
        return Caller(org_id=caller.org_id, user_id=user_id)
    claims = verify_token(credentials)
    return Caller(org_id=org_id_from(claims), clerk_user_id=str(claims["sub"]))


def get_rag(request: Request) -> RAGService:
    svc: RAGService | None = getattr(request.app.state, "rag", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="rag-service is starting")
    return svc


async def query(
    body: RagQueryRequest,
    caller: Annotated[Caller, Depends(resolve_caller)],
    svc: Annotated[RAGService, Depends(get_rag)],
) -> RagResponse:
    try:
        return await svc.query(caller, body)
    except ScopeUnavailable:
        logger.exception("rag: scope unavailable")
        raise HTTPException(status_code=503, detail="Retrieval is unavailable") from None


def build_app(service: RAGService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        closers: list[Any] = []
        if getattr(app.state, "rag", None) is None and not getattr(app.state, "wired", False):
            # Not configured (no RAG_SERVICE_TOKEN yet, before enable-tiers.sh has run) is a
            # 503 on the route, not a crash loop: one tier's missing secret must not fail the
            # whole stack's deploy health gate. /healthz stays up; the reason is logged.
            app.state.wired = True
            try:
                app.state.rag, closers = _wire_from_env()
            except NotConfigured as exc:
                logger.error("rag-service not configured: %s; /api/rag/query answers 503", exc)
        yield
        for close in closers:  # pragma: no cover
            close()

    app = create_app("rag-service")
    app.router.lifespan_context = lifespan
    app.state.rag = service
    app.add_api_route("/api/rag/query", query, methods=["POST"], response_model=RagResponse)
    return app


def _wire_from_env() -> tuple[RAGService, list[Any]]:  # pragma: no cover - production wiring
    from confluent_kafka import Producer
    from redis.asyncio import from_url

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.llm_cache import ResponseCache
    from libs.shared.pii import EgressRedactor
    from libs.shared.prompts import PromptLoader
    from libs.shared.qdrant import read_client
    from libs.shared.router import LiteLLMRouter
    from services.knowledge_graph_service.taxonomy import distinct_topics
    from services.rag_service.answer import AnswerWriter
    from services.rag_service.clients import (
        CoreScopeClient,
        GrpcLexicalSearch,
        HttpEmbedder,
        HttpGraphSearch,
        HttpReranker,
        QdrantVectorSearch,
    )
    from services.rag_service.expand import Expander
    from services.rag_service.metering import GateProbe, UsageEmitter

    token = os.environ.get("RAG_SERVICE_TOKEN", "")
    if len(token) < 32:
        raise NotConfigured("RAG_SERVICE_TOKEN must be set (32+ characters)")
    redis = from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
    gate = BudgetGate(redis, BillingRefresher(service="rag-service", token=token))
    producer = Producer(
        {
            "bootstrap.servers": os.environ.get("REDPANDA_BROKERS", "localhost:9092"),
            "acks": "all",
            "enable.idempotence": True,
            "linger.ms": 5,
        }
    )
    usage = UsageEmitter(
        producer, AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    )
    router = LiteLLMRouter(gate, usage, redactor=EgressRedactor(redis), cache=ResponseCache(redis))
    prompts = PromptLoader(PROMPTS_DIR)
    service = RAGService(
        scope=CoreScopeClient(os.environ.get("CORE_API_URL", "http://core-api:8000"), token),
        embedder=HttpEmbedder(
            os.environ.get("EMBEDDING_URL", "http://localhost:8101/internal"), token
        ),
        vector=QdrantVectorSearch(read_client()),
        lexical=GrpcLexicalSearch(os.environ.get("SEARCH_GRPC_URL", "localhost:50103"), token),
        graph=HttpGraphSearch(os.environ.get("KNOWLEDGE_GRAPH_URL"), token),
        reranker=HttpReranker(
            os.environ.get("RERANKER_URL", "http://localhost:8102/internal"), token
        ),
        expander=Expander(router, prompts, distinct_topics()),
        answerer=AnswerWriter(router, prompts),
        budget=GateProbe(gate),
        cache=cast(AnswerCache, redis),
    )
    return service, [usage.flush]


app = build_app()
