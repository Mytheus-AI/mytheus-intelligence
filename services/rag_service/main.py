"""rag-service: `POST /api/rag/query`, the frozen five-stage `RAGService` (mytheus-docs doc 09,
ADR-0033).

Four callers:
- a user, through `api-gateway`, with their Clerk session token. rag-service re-verifies it
  (the gateway is a shield, not a trust delegation); the org comes from the verified token and
  the user from its `sub`, which core-api resolves to `users.id` in the scope call;
- an agent (`agent-runtime-service`, for `QuestionAgent` grounding) with its service
  credentials (`MYTHEUS_RAG_CALLER_TOKENS`) and `X-Mytheus-User`: the participant it asks for.
  Scope is per user, so an agent sees exactly that participant's meetings;
- `bff`, asking on a signed-in user's behalf (`POST /api/v1/ask` and the MCP ask tool), with
  the same service-credential shape: `X-Mytheus-Service: bff`, its secret from
  `MYTHEUS_RAG_CALLER_TOKENS` (`bff=<secret>`), and `X-Mytheus-User` = that user's `users.id`.
  Scope is still that one user's;
- `core-api`, ONLY on `POST /internal/decisions/check` (A1 "Decisions that follow you",
  `decision_check.py`): is this text against a confirmed decision? Same service-credential shape
  (`core-api=<secret>` in `MYTHEUS_RAG_CALLER_TOKENS`, `X-Mytheus-User` = the person whose text
  it is). core-api may not call `/api/rag/query`, and the query callers may not call the check.
  Answers 503 when not wired, when retrieval scope is unavailable, or when the org's
  never-record topics cannot be read (identity-tenant-service's GetOrgContext,
  `IDENTITY_GRPC_ADDR`): text that cannot be screened is never checked.

Downstream, rag-service calls core-api, search-service, embedding-service, reranker-service and
knowledge-graph-service as itself (`RAG_SERVICE_TOKEN`), and reads Qdrant with the read-only key.

Env: RAG_SERVICE_TOKEN, MYTHEUS_RAG_CALLER_TOKENS, CORE_API_URL, EMBEDDING_URL and RERANKER_URL (the
service base including its `/internal` mount), SEARCH_GRPC_URL, KNOWLEDGE_GRAPH_URL (optional),
IDENTITY_GRPC_ADDR (the decision check's never-record list), BILLING_URL, REDIS_URL,
REDPANDA_BROKERS, SCHEMA_REGISTRY_URL, QDRANT_URL, QDRANT_READ_ONLY_API_KEY, ANTHROPIC_API_KEY.

Run: uvicorn services.rag_service.main:app --port 8201
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials

from libs.shared.app import create_app
from libs.shared.auth import _bearer, org_id_from, verify_token
from libs.shared.service_auth import SERVICE_HEADER, ServiceAuthError, authenticate
from services.agent_runtime_service.promises import PromiseExtractor
from services.rag_service.clients import ScopeUnavailable
from services.rag_service.contracts import RagQueryRequest, RagResponse
from services.rag_service.decision_check import (
    CheckRequest,
    CheckResponse,
    DecisionCheckService,
    NeverRecordUnavailable,
)
from services.rag_service.domain import Caller
from services.rag_service.pipeline import AnswerCache, RAGService
from services.rag_service.promise_route import PromiseExtractOut, extract_promises

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
# The agents' prompt registry: `promise_extract@vN` is the person's agent's prompt.
AGENT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent_runtime_service" / "prompts"
CALLER_TOKENS_ENV = "MYTHEUS_RAG_CALLER_TOKENS"
# Who may call `/api/rag/query`, and who may call `/internal/decisions/check`. Every name in
# `MYTHEUS_RAG_CALLER_TOKENS` must be one of `KNOWN_CALLERS` (the env is parsed against it); each
# route then admits only its own set.
ALLOWED_CALLERS = frozenset({"agent-runtime-service", "bff"})
CHECK_CALLERS = frozenset({"core-api"})
# integration-service is admitted to `POST /internal/promises/extract` only (promise_route.py),
# never to `/api/rag/query`; core-api only to `/internal/decisions/check`.
KNOWN_CALLERS = ALLOWED_CALLERS | CHECK_CALLERS | {"integration-service"}
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
            return _service_caller(headers, ALLOWED_CALLERS)
        except (ServiceAuthError, ValueError):
            raise HTTPException(status_code=401, detail="Unauthenticated") from None
    claims = verify_token(credentials)
    return Caller(org_id=org_id_from(claims), clerk_user_id=str(claims["sub"]))


def _service_caller(headers: dict[str, str], admitted: frozenset[str]) -> Caller:
    caller = authenticate(headers, env_name=CALLER_TOKENS_ENV, allowed=KNOWN_CALLERS)
    if caller.service not in admitted:
        raise ServiceAuthError("not an allowed caller for this route")
    return Caller(org_id=caller.org_id, user_id=UUID(headers.get(USER_HEADER, "")))


def resolve_check_caller(request: Request) -> Caller:
    """Service credentials only (no Clerk token): core-api, asking as one person."""
    headers = {k.lower(): v for k, v in request.headers.items()}
    try:
        return _service_caller(headers, CHECK_CALLERS)
    except (ServiceAuthError, ValueError):
        raise HTTPException(status_code=401, detail="Unauthenticated") from None


def get_check(request: Request) -> DecisionCheckService:
    svc: DecisionCheckService | None = getattr(request.app.state, "check", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="The decision check is not configured")
    return svc


async def check_decisions(
    body: CheckRequest,
    caller: Annotated[Caller, Depends(resolve_check_caller)],
    svc: Annotated[DecisionCheckService, Depends(get_check)],
) -> CheckResponse:
    try:
        return await svc.check(caller, body)
    except ScopeUnavailable:
        logger.exception("decision check: scope unavailable")
        raise HTTPException(status_code=503, detail="Retrieval is unavailable") from None
    except NeverRecordUnavailable:
        logger.warning("decision check: never-record topics unavailable; refusing to check")
        raise HTTPException(status_code=503, detail="The check is unavailable") from None


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


def build_app(
    service: RAGService | None = None,
    check: DecisionCheckService | None = None,
    promises: PromiseExtractor | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        closers: list[Any] = []
        if getattr(app.state, "rag", None) is None and not getattr(app.state, "wired", False):
            # Not configured (no RAG_SERVICE_TOKEN yet, before enable-tiers.sh has run) is a
            # 503 on the route, not a crash loop: one tier's missing secret must not fail the
            # whole stack's deploy health gate. /healthz stays up; the reason is logged.
            app.state.wired = True
            try:
                app.state.rag, app.state.check, app.state.promises, closers = _wire_from_env()
            except NotConfigured as exc:
                logger.error("rag-service not configured: %s; /api/rag/query answers 503", exc)
        yield
        for close in closers:  # pragma: no cover
            close()

    app = create_app("rag-service")
    app.router.lifespan_context = lifespan
    app.state.rag = service
    app.state.check = check
    app.add_api_route("/api/rag/query", query, methods=["POST"], response_model=RagResponse)
    app.add_api_route(
        "/internal/decisions/check",
        check_decisions,
        methods=["POST"],
        response_model=CheckResponse,
    )
    app.state.promises = promises
    # Promise ledger: the person's agent reads one of their own messages (integration-service).
    app.add_api_route(
        "/internal/promises/extract",
        extract_promises,
        methods=["POST"],
        response_model=PromiseExtractOut,
    )
    return app


def _wire_from_env() -> (
    tuple[RAGService, DecisionCheckService, PromiseExtractor, list[Any]]
):  # pragma: no cover - production wiring
    from confluent_kafka import Producer
    from redis.asyncio import from_url

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.llm_cache import ResponseCache
    from libs.shared.org_context import OrgContextClient
    from libs.shared.pii import EgressRedactor
    from libs.shared.prompts import PromptLoader
    from libs.shared.qdrant import read_client
    from libs.shared.router import LiteLLMRouter
    from services.knowledge_graph_service.taxonomy import distinct_topics
    from services.rag_service.answer import AnswerWriter
    from services.rag_service.clients import (
        CoreDecisionLexical,
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
    core_url = os.environ.get("CORE_API_URL", "http://core-api:8000")
    check = DecisionCheckService(
        # Expansion off: the text is the query, and a Haiku rewrite per keystroke-debounced
        # draft would double the cost of every check for no recall the judge needs.
        retriever=service.with_config(replace(service.config, expand=False)),
        lexical=CoreDecisionLexical(core_url, token),
        org_context=OrgContextClient(),
        judge=router,
        prompts=prompts,
    )
    extractor = PromiseExtractor(
        router,
        PromptLoader(AGENT_PROMPTS_DIR),
        # Strict: the never-record topics must be known before a message reaches a model.
        # Needs IDENTITY_GRPC_ADDR (the same GetOrgContext address agent-runtime uses).
        OrgContextClient(),
    )
    return service, check, extractor, [usage.flush]


app = build_app()
