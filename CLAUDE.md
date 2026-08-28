# CLAUDE.md

This is `mytheus-intelligence`, the AI / inference tier of Mytheus (internal codename NEXUS): six LangGraph agents, a hybrid RAG pipeline, GPU embedding/reranker serving, the Apache AGE knowledge graph, lexical search, and document ingestion. The single most important thing to internalize: this tier NEVER writes Postgres directly. It emits `agent-results` and `core-api` commits them through its transactional outbox; the knowledge graph is written only on user confirmation. The agent tier proposes; `core-api` commits.

## Repo at a glance

Mytheus is a multi-tenant meeting-orchestration SaaS where every participant gets a personal AI agent. The platform is split into three scaling families: the CRUD/control plane (`mytheus-core`), the real-time tier (`mytheus-realtime`), and this AI/inference tier. This repo is its own cost center: it autoscales on queue depth, GPU utilization, and token throughput, independently of CRUD req/s and real-time connection count. Seven services live here:

- `agent-runtime-service` - the six LangGraph agents and the cost-controlled LLM pipeline.
- `rag-service` - the sync orchestrator of hybrid retrieval.
- `embedding-service` - GPU embeddings; the only writer of Qdrant.
- `reranker-service` - GPU cross-encoder rerank.
- `knowledge-graph-service` - the only writer of the AGE `nexus_knowledge` graph.
- `search-service` - the Postgres FTS arm of hybrid retrieval.
- `ingestion-service` - document parsing into typed, chunked elements.

## Architecture orientation

There are three pipelines and one cross-cutting collaboration mechanism. Hold these models in your head before touching code.

Agent pipeline. `transcript-chunks` arrive as raw Deepgram utterances. `agent-runtime-service` merges them by speaker turn into 30-second `TranscriptWindow` objects (about 90 windows per meeting versus 360 raw chunks). Agents run per window, never per raw chunk. Each window flows through the cost pipeline (relevance pre-gate, tiered routing, caching, budget check) and agents emit to `agent-results` (durable output, consumed by `core-api`) and `agent-stream` (token deltas to the UI via `realtime-gateway`).

RAG pipeline. `rag-service` owns the frozen five-stage `RAGService`: expand, retrieve, rerank, assemble, cite. Retrieval is hybrid: three arms run in parallel and are fused with Reciprocal Rank Fusion (RRF, k=60). The arms are Qdrant vector search (via `embedding-service`), AGE graph traversal `RELATES_TO` plus `FOLLOWS_FROM` lineage (via `knowledge-graph-service`), and Postgres FTS over `tsvector` (via `search-service`). `rag-service` fans out to those servers over the internal gRPC mesh; `org_id` rides in gRPC metadata on every hop. Every assembled passage must bind to a citation or it is dropped. Warm p95 budget is < 500 ms; the cross-encoder rerank is the cold-path long pole, so `reranker-service` keeps a warm-replica floor.

Inference serving. `embedding-service` (BGE-small-en, 384-dim) and `reranker-service` (BGE-reranker-base cross-encoder) are dedicated GPU model servers on Ray Serve / vLLM, deliberately off the API tier with their own autoscalers. Embedding is bi-encoder (precomputed at ingest, cheap query-embed); reranking is a full cross-encoder forward pass per (query, passage) pair, about 50x the work, only affordable over the 50 RRF survivors. Opposite cost profiles, separate deployments.

Agent-to-agent collaboration. `ContextGapAgent` asks other users' agents for context over `agent-requests` / `agent-responses`, carrying `correlation_id` and a `deadline`. The fallback chain is peer-agents, then related-meeting agents, then RAG, then silent. Authz is deny-by-default and same-tenant-only. p95 round-trip budget is < 1.5 s so a fill lands inside the live conversation.

## Service map

| Service | Responsibility | Owns / calls | Events in | Events out |
| --- | --- | --- | --- | --- |
| `agent-runtime-service` | The six agents, `BaseAgent` lifecycle, tiered routing, per-tenant budgets | Calls `get_user_context` over gRPC; reads Redis budget mirror. Writes no Postgres | `transcript-chunks`, `document-chunks`, `agent-responses` | `agent-results`, `agent-stream`, `agent-requests`, `usage-events` |
| `rag-service` | `expand -> retrieve -> rerank -> assemble -> cite`; RRF fusion; Redis answer cache | Calls embedding / reranker / knowledge-graph / search over gRPC | (sync `POST /api/rag/query`) | `usage-events` |
| `embedding-service` | BGE-small-en 384-dim; embed-on-ingest + query-embed | The only writer of Qdrant per-tenant collections | `transcript-chunks`, `document-chunks` | (upserts Qdrant) |
| `reranker-service` | BGE-reranker-base cross-encoder; top-50 -> top-10; warm-replica floor | Owns no data | (sync gRPC from rag-service) | none |
| `knowledge-graph-service` | The only writer of AGE `nexus_knowledge`; Cypher parameterized by `$org_id` | Reads served to rag-service and ProactiveAgent | `decisions`, `actions` (confirmed, from core-api) | (writes AGE edges) |
| `search-service` | Postgres FTS arm; `tsvector` over `transcript_chunks` + `chat_messages`, RLS by `org_id` | Builds and owns its FTS indexes | `transcript-chunks` | (sync gRPC from rag-service) |
| `ingestion-service` | `unstructured.io` parsing into typed chunked elements; blobs to object storage (**Cloudflare R2**, S3-compatible; MinIO locally) | Owns `document_chunks` / `document_elements` + object key prefix `{org_id}/` | `integration-events` | `document-chunks` |

## The six agents

One `PersonalListenerAgent` is spawned per participant per meeting; the others are spawned per meeting. All confidence handling follows the same gate: `> 0.7` surface to the user, `0.4-0.7` silent log to `worklogs`, `< 0.4` drop.

| Agent | Trigger | Role | Output |
| --- | --- | --- | --- |
| `PersonalListenerAgent` | A 30s `transcript-chunks` window for a meeting the user is in | `get_user_context(user_id)` then score window relevance 0-1 to that user (a pre-gate distinct from confidence) | Worklog content on `agent-results` (persisted to `worklogs` by core-api) |
| `DecisionAgent` | A `transcript-chunks` window | Detect decisions; extract rationale, stakeholders, confidence | Decision candidate on `agent-results`; confirm/edit/dismiss card. Never silently commits |
| `ActionAgent` | A `transcript-chunks` window | Detect action items; extract `assignee_id`, `due_date`, dependencies, confidence | Action candidate on `agent-results` (new-task toast). On confirm, AGE `ASSIGNED_TO` edge |
| `QuestionAgent` | A window plus a context-gap heuristic | Propose role-specific questions, `priority` 1-3, status Ask/Dismiss/Save-for-later | `agent_suggestions` on `agent-results` (suggestion card) |
| `ContextGapAgent` | A reference to context a participant lacks | Query other users' agents via `agent-requests` with `correlation_id` + `deadline`; chain peer-agents -> related-meeting -> RAG -> silent | Context-fill card; `agent-requests` to peers. Silent on empty (no fabrication) |
| `ProactiveAgent` | Confirmed `decisions` / `actions`; AGE graph queries | Conflict (AGE query + semantic over `decisions__{org_id}`), workload (open actions per assignee, 80%/95%), missing-stakeholder (org-graph traversal) | `notifications` on `agent-results`. Opus-tier; conflict false-positive rate is an eval gate |

## Where things live

```
services/agent-runtime-service/    # six LangGraph agents, BaseAgent, AgentState, tiered routing, budgets
  agents/                          # one module per agent class; the BaseAgent lifecycle
services/rag-service/              # RAGService five stages; gRPC orchestrator; RRF; Redis answer cache
services/embedding-service/        # BGE-small-en 384-dim (GPU Ray Serve); ONLY writer of Qdrant
services/reranker-service/         # BGE-reranker-base cross-encoder (GPU Ray Serve); top-50 -> top-10
services/knowledge-graph-service/  # Apache AGE nexus_knowledge; ONLY writer; on-confirm Cypher
services/search-service/           # Postgres FTS arm of hybrid retrieval
services/ingestion-service/        # unstructured.io parsing; produces document-chunks + document_elements
libs/shared/                       # tenancy, authz, tracing, LiteLLM router, PromptLoader, event envelopes
proto/                             # gRPC service + message definitions for the internal mesh
```

When adding cross-service behavior, prefer extending `libs/shared/` (tenancy, authz, router, prompt loader, envelopes) over duplicating it per service. Inference belongs in the GPU servers behind gRPC, not in-process in `rag-service`.

## Invariants you must not break

Enforce these in code, fail closed. These are the rules that keep the tier multi-tenant-safe and cost-bounded.

| Rule | Why | If you break it |
| --- | --- | --- |
| `org_id` on every gRPC hop, every Qdrant payload filter, every AGE `$org_id`, every Redis key prefix `nexus:{org_id}:` | Multi-tenant isolation is defense-in-depth across all layers | Cross-tenant data leak; the worst-class incident in a multi-tenant SaaS |
| The AI tier NEVER writes Postgres directly | The relational core is the system of record; `core-api`'s outbox is the only durable write path | Split-brain state, lost or duplicated rows, no confirmation semantics |
| Knowledge-graph writes ONLY on confirmation | Agents propose candidates; users commit. Edges must reflect confirmed truth | The graph fills with unconfirmed/hallucinated decisions; `ProactiveAgent` conflict detection corrupts |
| Deny-by-default agent-to-agent authz | An agent must not read another user's context without a same-tenant, role-scoped grant | One user's agent exfiltrates another user's or another org's context |
| Per-tenant LLM budgets enforced inline | This tier is a cost center; the R4 cost spiral is the headline risk | A runaway tenant burns unbounded tokens; the budget mirror exists to gate every call fast |
| Confidence thresholds (`>0.7` surface, `0.4-0.7` silent log, `<0.4` drop) | Hallucination control; low-confidence output must not reach the user | Users lose trust in surfaced notes/decisions; eval calibration breaks |
| Tiered LLM routing Haiku -> Sonnet -> Opus via LiteLLM | The router picks the cheapest model that can do the job (Haiku ~45%, Sonnet ~55%, Opus <1%) | Cost blows up; Opus on every window defeats the whole cost design |

Exception to the Postgres rule: `ingestion-service` owns `document_chunks` / `document_elements`, and `search-service` owns its FTS indexes. These two write only their own tables and are the only relational writers in this repo.

## Naming authority

Agent class names, topic names, AGE node/edge labels, and Qdrant collection names are frozen in `mytheus-docs` `_canon/` and doc 24 (Glossary & Naming). Never invent or rename. A rename lands in the Glossary and a superseding ADR first, then propagates here.

- Agent classes: `PersonalListenerAgent`, `DecisionAgent`, `ActionAgent`, `QuestionAgent`, `ContextGapAgent`, `ProactiveAgent`.
- Topics (kebab-case): `transcript-chunks`, `document-chunks`, `agent-requests`, `agent-responses`, `agent-results`, `agent-stream`, plus `decisions`, `actions`, `notifications`, `usage-events`, `integration-events`.
- AGE graph: `nexus_knowledge`. Nodes: `Person`, `Meeting`, `Decision`, `Action`, `Topic`. Edges: `ATTENDED`, `PRODUCED`, `ASSIGNED_TO`, `RELATES_TO`, `FOLLOWS_FROM`.
- Qdrant collections: `transcript_chunks__{org_id}`, `document_chunks__{org_id}`, `decisions__{org_id}`.
- Postgres-origin names: `transcript_chunks`, `worklogs`, `agent_suggestions`, `agent_collaborations`, `agent_budgets`, `document_chunks`, `document_elements`.
- Redis prefix: `nexus:{org_id}:`.

When in doubt, look it up. Do not guess.

## Common tasks (recipes)

### Add a new LangGraph agent

1. Create `services/agent-runtime-service/agents/<your>_agent.py` subclassing `BaseAgent`. Implement the lifecycle: `spawn(meeting_id, user_id, org_id)`, `process` (per 30s window), `terminate`. `spawn` must assert `org_id` is present and fail closed.
2. Populate `AgentState`: carry `org_id`, `agent_id`, `relevance_score` (if pre-gated), `confidence`, `prompt_version`, `token_budget_remaining`, and `correlation_id` (if collaborating).
3. In `process`, call `check_budget(org_id, est_tokens)` before any LLM call (deny-by-default when `token_budget_remaining <= 0`). Route the call through the LiteLLM tier helper; do not hardcode a model.
4. Apply the confidence gate: `>0.7` emit to `agent-results`; `0.4-0.7` silent log to `worklogs`; `<0.4` drop. Never surface uncited or low-confidence output.
5. Load prompts through `PromptLoader` as `{prompt_key}@{version}` so tenant pins and staged rollout work; do not inline prompt strings.
6. Register the agent in the runtime spawner (per-participant vs per-meeting) and add it to the golden-set eval harness before merge.
7. Use only the frozen class name pattern; the name is canon if this is a recognized agent, otherwise it must be added to the Glossary first.

### Add a retrieval arm to rag-service

1. Define the gRPC service/message in `proto/` and regenerate stubs. The request must carry `org_id` in gRPC metadata.
2. Stand up (or extend) the backing server. The arm must apply tenant isolation in its own store (Qdrant payload filter / AGE `$org_id` / Postgres RLS).
3. In `rag-service`, add the arm to the parallel fan-out in the `retrieve` stage and feed its ranked results into the RRF fusion (k=60) alongside the existing vector/graph/FTS arms.
4. Keep the contract intact: results flow through `rerank` (top-50 -> top-10), `assemble` (token-budget packing, default 3,000 tokens, dedupe, diversity), and `cite`. A passage that cannot bind a citation is dropped.
5. Respect the warm p95 < 500 ms budget; the arm runs in parallel, so do not add a serial dependency.

### Add an inference endpoint to embedding/reranker

1. Add the Ray Serve deployment in the relevant GPU service (`embedding-service` or `reranker-service`); keep it off the API tier.
2. Expose it over gRPC only for internal callers (`embedding-service` query-embed is `POST /embed`, internal-only). `embedding-service` stays the only writer of Qdrant; reranker owns no data.
3. Honor the SLOs: embedding batch-of-32 < 100 ms; reranker warm p95 < 500 ms with a warm-replica floor >= 1 per active region.
4. Cache where the design says to: query embeddings at `nexus:{org_id}:qembed:{query_hash}`.
5. Provide a CI mock of the model server so tests stay deterministic and GPU-free.

### Add a Cypher template to knowledge-graph-service

1. Add the template in `knowledge-graph-service`, parameterized by `$org_id` so traversal cannot leave the org subgraph. Use only frozen node/edge labels.
2. If it writes, it must be driven by a confirmed `decisions` / `actions` event consumed from `core-api`. Never write the graph from an unconfirmed agent claim.
3. If it reads (for `rag-service` or `ProactiveAgent`), keep it side-effect-free.
4. Add idempotency: the consumer dedupes on `idempotency_key` via the inbox table + `nexus:{org_id}:dedupe:{topic}:{idempotency_key}`.

### Tune the LLM cost pipeline

The five levers, in order: relevance pre-gating (cheap Haiku triage, ~35% pass), windowed batching (30s windows), tiered routing (Haiku triage -> Sonnet extract -> Opus high-stakes), prompt caching (Anthropic prompt cache on stable prefixes + semantic cache, applied after tiering), and per-tenant budgets (`nexus:{org_id}:budget:{period}` mirror of `agent_budgets`, alert 80%/95%, deny at `hard_cap`). Change one lever at a time, re-run the eval harness, and emit `usage-events` for any metered tokens.

## Testing

- `pytest` for unit and integration. Mock the GPU model servers and all LLM calls in CI; never hit a real GPU or paid LLM in the test suite.
- Eval / quality harness (run as `pytest -m eval`): golden-set gates for agents (extraction F1, confidence calibration via Brier, relevance precision/recall, question usefulness, conflict false-positive rate) and for RAG (citation validity, retrieval quality). A prompt change must beat the current default before it can roll out.
- Keep evals deterministic: pin seeds, freeze golden inputs, mock model outputs. Flaky non-determinism in the eval suite blocks the prompt-change gate.
- Quality gates: `ruff check .`, `black --check .`, `mypy .` must be clean. Full type hints are required.

## Local development

```bash
# install (per service)
python -m venv .venv && source .venv/bin/activate
pip install -e "services/<name>[dev]"

# backing services: Qdrant + Postgres/AGE + Redpanda + Redis
docker compose up -d

# run a service
uvicorn rag_service.app:app --reload --port 8081     # RAG orchestrator
python -m agent_runtime_service.main                  # agent runtime (consumes transcript-chunks)

# test, eval, lint
pytest
pytest -m eval
ruff check . && black --check . && mypy .
```

Environment variables: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (LiteLLM), `LITELLM_CONFIG` (tier mapping + budget hooks), `QDRANT_URL` / `QDRANT_API_KEY`, `POSTGRES_DSN` (Postgres + AGE `nexus_knowledge`), `REDPANDA_BROKERS` / `SCHEMA_REGISTRY_URL`, `REDIS_URL` (keys prefixed `nexus:{org_id}:`), `RAY_ADDRESS` / `EMBEDDING_GPU` / `RERANKER_GPU`, `OTEL_EXPORTER_OTLP_ENDPOINT` / `SENTRY_DSN`. A GPU is optional locally; CPU inference works at dev scale and CI mocks the model servers.

## Pitfalls

- Writing Postgres from this tier. Emit `agent-results` and let `core-api`'s outbox commit. Only `ingestion-service` and `search-service` write their own owned tables.
- Writing the graph without confirmation. Edges come only from confirmed `decisions` / `actions` events, never from a raw agent candidate.
- Missing `org_id` in gRPC metadata or a Qdrant payload filter. Per-tenant collections are not enough; the payload filter and the metadata propagation are mandatory belt-and-suspenders.
- Blowing the per-tenant budget. Check the Redis budget mirror inline before every LLM call; do not defer the check.
- Exceeding the latency budget. Keep RAG arms parallel, keep `reranker-service` warm, keep `rag-service` an orchestrator with no in-process model.
- Non-deterministic eval flakiness. Mocked model outputs and pinned seeds, or the prompt-change gate becomes meaningless.

## Working with other repos

- Consumes event schemas from `mytheus-events` (frozen Avro envelopes in `07-event-architecture.md` / `_canon/`).
- Receives `transcript-chunks` from `mytheus-realtime` (transcription) and `document-chunks` work driven from `mytheus-backend` integrations.
- Emits `agent-results` consumed by `mytheus-backend` `core-api`, which is the producer of confirmed `decisions` / `actions`.
- Fans out `agent-stream` token deltas to `mytheus-realtime` (`realtime-gateway`) for the live UI.

## Commit and PR conventions

- Trunk-based development with PRs and review. Conventional commits.
- A prompt change must pass the eval harness and, for production, a 48h ~1% tenant-pin canary before the global default advances. A regression rolls the pin back instantly without redeploying.
- Commits in this project must NOT include a `Co-Authored-By` footer.

## Progress documentation - single home

Every record of work done - after a session, a feature, a fix, or a band step - lands in ONE place: `mytheus-docs/work-done-till-now/`. Not in this repo.

- One file per plan band, covering every repo that band touches: `work-done-till-now/01-plan-0-scaffold-and-contracts.md`, then `02-plan-1-2-...`, and so on. Add to the current band's file; start a new file only when a new band starts.
- Write it comprehensively enough that another person, or their coding agent, can pick the work up cold: what was built, what was verified and by which exact command, and **why** each decision was taken. Record every deviation from the plan together with its rationale. A deviation with no recorded reason is a bug, not a decision.
- Read `mytheus-docs/work-done-till-now/README.md` first. It holds the required structure and the writing conventions.
- Do NOT create or extend a per-repo progress log. `PREVIOUS.md` in this repo is historical: leave it in place, do not add to it, and fold anything new into the band file.
- That folder **records**; it never **defines**. A new or changed name, number, or decision still goes canon-first (`_canon/`), then an ADR, then the reader docs. Cite them from the band file.
- Do not overcomplicate. Build the optimal, professional, simple working version first, and add complexity only when something concrete requires it.
