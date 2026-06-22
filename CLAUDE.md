# CLAUDE.md — `reprise-intelligence`

Working guide for AI coding agents in this repo. Read this before editing. Keep `system.md` and the canon docs open for detail.

## Purpose

`reprise-intelligence` is the **AI / inference tier** of **Reprise AI** (internal codename NEXUS), an intelligent meeting-orchestration multi-tenant SaaS where every participant gets a personal AI agent. This repo owns the six LangGraph agents, the hybrid RAG pipeline, GPU embedding/reranker serving, the Apache AGE knowledge graph, lexical search, and document ingestion. It is a distinct scaling family and cost center (autoscales on queue depth / GPU / tokens) and is **stateless against the relational core** — it reads context and emits events, and never writes Postgres directly.

## Stack & versions

- **Python 3.11+**, **FastAPI**
- **LangGraph** (agents, `BaseAgent`), **LiteLLM** (Haiku → Sonnet → Opus tiered routing)
- **Ray Serve** + **vLLM** (GPU model serving)
- **Qdrant client**; **Apache AGE / Cypher**; **sentence-transformers** (BGE-small-en, BGE-reranker-base)
- **gRPC** (internal inference mesh); **Redpanda** (Kafka API) + Avro Schema Registry; **Redis**
- Tooling: `ruff`, `black`, `mypy`, `pytest`

## Service layout

```
services/agent-runtime-service/    # six LangGraph agents, BaseAgent, tiered routing, budgets
services/rag-service/              # RAGService expand→retrieve→rerank→assemble→cite (gRPC orchestrator)
services/embedding-service/        # BGE-small-en 384-dim (GPU); ONLY writer of Qdrant
services/reranker-service/         # BGE-reranker-base cross-encoder (GPU); top-50→top-10
services/knowledge-graph-service/  # Apache AGE nexus_knowledge; ONLY writer; writes on confirmation
services/search-service/           # Postgres FTS arm of hybrid retrieval
services/ingestion-service/        # unstructured.io parsing; produces document-chunks + document_elements
libs/shared/                       # tenancy, authz, tracing, LiteLLM router, prompt loader, envelopes
proto/                             # gRPC service + message definitions
```

## Cross-cutting rules (enforce in code, not by convention)

1. **`org_id` on every hop.** Propagate `org_id` in gRPC metadata; filter every Qdrant query by `org_id` payload (even on per-tenant collections); parameterize every AGE Cypher template by `$org_id`; prefix every Redis key with `nexus:{org_id}:`. Fail closed if `org_id` is missing.
2. **The AI tier NEVER writes Postgres directly.** Emit `agent-results` and rely on `core-api`'s transactional outbox. (Exception: `ingestion-service` owns `document_chunks` / `document_elements`; `search-service` builds FTS indexes — these are the only relational writers in this repo, and they write only their owned tables.)
3. **Knowledge-graph writes ONLY on confirmation.** `knowledge-graph-service` creates `PRODUCED` / `ASSIGNED_TO` edges only by consuming confirmed `decisions` / `actions` from `core-api`. Agents emit candidates only — never write the graph from an unconfirmed claim.
4. **Deny-by-default agent-to-agent authz.** Every `agent-requests` message is same-tenant-only (org from `requester_agent_id`, not the payload), role-scope-checked (no grant ⇒ deny), and audited to `audit_logs`.
5. **Per-tenant LLM budgets enforced.** Every LLM call checks the `nexus:{org_id}:budget:{period}` Redis mirror of `agent_budgets`; alert at 80%/95%, deny at `hard_cap`. Emit `usage-events` for metered tokens.
6. **Confidence thresholds (verbatim).** `> 0.7` surface · `0.4–0.7` silent log to `worklogs` · `< 0.4` drop. Overridable per tenant via `tenant_settings.agent_confidence_overrides`.

## Naming authority

Frozen canon identifiers come from `reprise-docs` `_canon/` and doc 24 (Glossary & Naming). **Never invent or rename.** When in doubt, look it up; do not guess.

- **Agent class names (verbatim):** `PersonalListenerAgent`, `DecisionAgent`, `ActionAgent`, `QuestionAgent`, `ContextGapAgent`, `ProactiveAgent`.
- **Topics (verbatim, kebab-case):** `transcript-chunks`, `document-chunks`, `agent-results`, `agent-stream`, `agent-requests`, `agent-responses`, `decisions`, `actions`, `notifications`, `usage-events`. Keep `agent-requests` / `agent-responses` exactly.
- **AGE labels (verbatim):** nodes `Person`, `Meeting`, `Decision`, `Action`, `Topic`; edges `ATTENDED`, `PRODUCED`, `ASSIGNED_TO`, `RELATES_TO`, `FOLLOWS_FROM`. Graph name is `nexus_knowledge`.
- **Qdrant collections (verbatim):** `transcript_chunks__{org_id}`, `document_chunks__{org_id}`, `decisions__{org_id}`.
- **Postgres-origin names (verbatim):** `transcript_chunks`, `worklogs`, `agent_suggestions`, `agent_collaborations`, `agent_budgets`, `document_chunks`, `document_elements`.
- **Redis prefix (verbatim):** `nexus:{org_id}:…`.

A rename is never silent: it lands in the Glossary (and a superseding ADR) **first**, then propagates here.

## Do

- Run agents on **30-second windows**, not raw chunks.
- Route through LiteLLM tiers; reserve Opus for confirmed high-stakes reasoning only.
- Embed-on-ingest; query-embed only on the read path; cache query embeddings at `nexus:{org_id}:qembed:{query_hash}`.
- Keep `rag-service` an orchestrator (no model in-process); reach model servers over gRPC.
- Drop any RAG passage that cannot be bound to a citation.
- Add type hints; write tests first (TDD); run the eval harness on any prompt change.

## Don't

- Don't write Postgres from `agent-runtime-service` / `rag-service` / `embedding-service` / `reranker-service` / `knowledge-graph-service`.
- Don't write Qdrant from anything other than `embedding-service`, or the AGE graph from anything other than `knowledge-graph-service`.
- Don't let an agent cross an org boundary or query without an authz grant.
- Don't surface low-confidence output, or RAG text without a resolvable citation.
- Don't invent agent / topic / label / collection names — look them up in `_canon/` and doc 24.
- Don't add a `Co-Authored-By` footer to commits in this project.

## Run / test / eval

```bash
docker compose up -d                                   # Qdrant + Postgres/AGE + Redpanda + Redis
uvicorn rag_service.app:app --reload --port 8081       # run the RAG orchestrator
python -m agent_runtime_service.main                   # run the agent runtime
pytest                                                  # unit + integration tests (mock GPU/LLM in CI)
pytest -m eval                                          # golden-set eval harness (agents + RAG quality gates)
ruff check . && black --check . && mypy .              # lint + format + types
```

## PR conventions

- **Conventional commits**, trunk-based with PRs.
- A prompt change must pass the eval harness and (for production) a 48h ~1% canary via tenant prompt pins before the global default advances.
- Commits in this project must **NOT** include a `Co-Authored-By` footer.
