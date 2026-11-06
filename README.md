# Mytheus - Intelligence / Inference Tier (`mytheus-intelligence`)

> The AI/inference tier of **Mytheus** - the six LangGraph agents, the hybrid RAG pipeline, GPU embedding/reranker serving, the Apache AGE knowledge graph, lexical search, and document ingestion. Internal architecture codename: **NEXUS**.

## What this is

`mytheus-intelligence` is the **AI / inference tier** of Mytheus - an intelligent meeting-orchestration multi-tenant SaaS where every participant gets a personal AI agent. This repo owns the intelligence and retrieval surfaces of the platform:

- **Agents** - six LangGraph agents that turn a live meeting into personal notes, confirmed decisions and actions, real-time questions, cross-meeting context fills, and proactive alerts, on a cost-controlled LLM pipeline.
- **RAG** - request-time hybrid retrieval (`expand → retrieve → rerank → assemble → cite`) that fuses vector, graph, and full-text-search signals and returns cited passages.
- **GPU inference** - dedicated embedding and reranker model servers on Ray Serve / vLLM, off the API tier.
- **Knowledge graph** - the Apache AGE `nexus_knowledge` graph, recording confirmed decisions and actions.
- **Search** - the Postgres FTS arm of hybrid retrieval.
- **Ingestion** - document parsing into typed, chunked elements.

This tier is its own **scaling family and cost center**: it autoscales on queue depth, GPU utilization, and token throughput - independently of the CRUD/control plane (req/s) and the real-time tier (concurrent connections). It is **stateless against the relational core**: it reads context and emits events, and durable agent output reaches Postgres only via `core-api`'s transactional outbox. It never writes Postgres directly.

See the canon architecture docs for the full picture: `08-ai-agent-architecture.md` (agents), `09-rag-and-inference-serving.md` (RAG + inference serving), `03-container-architecture.md` (containers + gRPC mesh + ownership), `05-data-architecture.md` (stores), `07-event-architecture.md` (topics).

## Services in this repo

| Service | Responsibility | Runtime | Scales on |
|---|---|---|---|
| `agent-runtime-service` | The six LangGraph agents; `BaseAgent` lifecycle; confidence thresholds; cost-controlled tiered LLM routing; per-tenant budgets. Consumes `transcript-chunks` / `document-chunks` / `agent-responses`; produces `agent-results` / `agent-stream` / `agent-requests` / `usage-events`. Never writes Postgres. | FastAPI + LangGraph (Python 3.11+) | token throughput + chunk backlog |
| `rag-service` | `RAGService` `expand → retrieve → rerank → assemble → cite`; sync orchestrator of hybrid retrieval over gRPC; RRF fusion; Redis answer cache. | FastAPI (Python 3.11+) | query req/s |
| `embedding-service` | BGE-small-en, 384-dim. Embed-on-ingest + query-embed; **only writer of Qdrant** (per-tenant collections). | Ray Serve / vLLM (GPU) | queue depth + GPU util |
| `reranker-service` | BGE-reranker-base cross-encoder; top-50 → top-10; warm-replica floor. | Ray Serve / vLLM (GPU) | rerank req/s + GPU util |
| `ner-service` | Lightweight NER (PERSON/ORG/LOC + regex-validated EMAIL/PHONE/IBAN/CARD/GOVID) for PII redaction; detects only, owns no data. See `services/ner_service/README.md`. | FastAPI (CPU spaCy, `ner` extra) | detect req/s |
| `knowledge-graph-service` | Apache AGE `nexus_knowledge`; **only writer**; Cypher templates parameterized by `$org_id`; writes `decisions`/`actions` edges **only on confirmation**. | FastAPI (gRPC) | Cypher query req/s |
| `search-service` | Postgres FTS arm of hybrid retrieval (`tsvector` over `transcript_chunks`; `chat_messages` pending a canon topic decision, see design note). | FastAPI + internal gRPC | search req/s + index size |
| `ingestion-service` | `unstructured.io` document parsing; produces `document-chunks` and `document_elements`; writes original blobs to object storage. | FastAPI worker | ingestion backlog |

## The six agents

- **`PersonalListenerAgent`** - per participant per meeting; scores each `transcript-chunks` window's relevance to the user's role and emits role-relevant worklog notes.
- **`DecisionAgent`** - detects decisions, extracts rationale/stakeholders/confidence; emits a *candidate* only (no silent commit).
- **`ActionAgent`** - detects action items, extracts assignee/due-date/dependencies/confidence; emits a *candidate* only.
- **`QuestionAgent`** - detects context gaps in live conversation and proposes role-specific questions with priority 1–3.
- **`ContextGapAgent`** - the cross-meeting collaborator: asks other users' agents over `agent-requests`/`agent-responses`, falling through peer-agents → related-meeting agents → RAG → silent.
- **`ProactiveAgent`** - conflict / workload / missing-stakeholder detection over confirmed `decisions`/`actions` and the AGE graph.

Confidence surfacing policy (verbatim canon): `> 0.7` surface · `0.4–0.7` silent log to `worklogs` · `< 0.4` drop.

## Tech stack

- **FastAPI** + **Python 3.11+**
- **LangGraph** - the agent graphs and shared `BaseAgent` lifecycle
- **LiteLLM** - tiered LLM routing (Haiku → Sonnet → Opus)
- **Ray Serve** + **vLLM** - GPU model serving
- **Qdrant client** - per-tenant vector collections
- **Apache AGE / Cypher** - the `nexus_knowledge` graph
- **sentence-transformers** - BGE-small-en (embeddings) + BGE-reranker-base (reranker)
- **gRPC** - internal inference mesh

## Repo structure

```
mytheus-intelligence/
├── services/
│   ├── agent-runtime-service/
│   ├── rag-service/
│   ├── embedding-service/
│   ├── reranker-service/
│   ├── knowledge-graph-service/
│   ├── search-service/
│   └── ingestion-service/
├── libs/
│   └── shared/                # tenancy, authz, tracing, LiteLLM router, prompt loader, event envelopes
├── proto/                     # gRPC service + message definitions for the internal mesh
└── README.md
```

## Quick start

**Prerequisites**

- Python 3.12 (`.python-version`; `requires-python >= 3.11`) and [`uv`](https://docs.astral.sh/uv/)
- A GPU is optional for local development (CPU inference works for `embedding-service`/`reranker-service` at dev scale).
- Docker, for the backing services (see below).
- The Anthropic API key for any path that exercises LiteLLM.

**Install**

```bash
uv sync                 # one workspace-wide environment; add --extra gpu for Ray Serve
cp .env.example .env
```

**Backing services - what this repo's compose does and does not run**

`docker-compose.yml` here defines **only Qdrant (:6333, with write + read-only API keys) and MinIO (:9002 API / :9003 console)**. It does **not** run Postgres, Apache AGE, Redis or Redpanda, and it is not a Postgres+AGE+Redpanda stack. Those come from the `mytheus-backend` stack, which this tier shares:

| Dependency | Port | Provided by |
|---|---|---|
| Postgres (schema `intelligence`; `mytheus_ai_app` / `mytheus_ai_relay` roles) | 5443 | `mytheus-backend` compose |
| Redis | 6390 | `mytheus-backend` compose |
| Redpanda (Kafka API) | 9092 | `mytheus-backend` compose |
| Schema Registry | 8081 | `mytheus-backend` compose |
| Qdrant | 6333 | this repo (`docker compose up -d`) |
| MinIO | 9002 / 9003 | this repo (`docker compose up -d`) |

```bash
docker compose up -d                    # Qdrant + MinIO (this repo)
# start the mytheus-backend stack for Postgres / Redis / Redpanda / Schema Registry
uv run alembic upgrade head             # intelligence schema (needs BOOTSTRAP_DATABASE_URL)
```

**Run a service** (from the repo root; the dev port for each is below)

```bash
uv run uvicorn services.<svc>.main:app --reload --port <port>
# e.g. the search service REST API:
uv run uvicorn services.search_service.main:app --reload --port 8203

# event consumers / relay / gRPC are separate processes (see each module's docstring):
uv run python -m services.embedding_service.worker       # embed-on-ingest consumer
uv run python -m services.embedding_service.relay_main   # the ONE outbox relay for this repo
uv run python -m services.search_service.worker          # transcript-chunks -> FTS index
uv run python -m services.search_service.grpc_main       # internal gRPC FTS arm (GRPC_PORT, :50103)
uv run python -m services.ingestion_service.worker       # integration-events consumer
```

**Dev ports** (8200+; 8081 is Redpanda's in-network Schema Registry port; its host port is 8085)

| Service | Module | HTTP dev port |
|---|---|---|
| `agent-runtime-service` | `services.agent_runtime_service.main:app` | 8200 |
| `rag-service` | `services.rag_service.main:app` | 8201 |
| `embedding-service` | `services.embedding_service.main:app` | 8202 |
| `search-service` | `services.search_service.main:app` | 8203 (gRPC FTS: 50103) |
| `reranker-service` | `services.reranker_service.main:app` | 8204 |
| `ingestion-service` | `services.ingestion_service.main:app` | 8205 |
| `knowledge-graph-service` | `services.knowledge_graph_service.main:app` | 8206 |

Ray Serve (GPU/CPU model servers) listens on 8000 inside its container; see "GPU serving".

**Test**

```bash
uv run pytest                           # infra tests skip (or fail loud with MYTHEUS_REQUIRE_*=1) when the stack is down
uv run ruff check . && uv run black --check . && uv run mypy .
```

## Internal gRPC: search-service FTS arm

`proto/rag.proto` defines `FtsService.FtsSearch` (query, optional `meeting_id`, `limit` 1..50). `services/search_service/grpc_server.py` serves it for `rag-service`; REST `GET /search` is unchanged and both run the same query (`services/search_service/fts.py`), so RLS + the explicit `org_id` filter apply identically. `org_id` is never in the message: the caller sends `authorization: Bearer <Clerk JWT>` in gRPC metadata, verified with the same chain as REST. Status codes: `UNAUTHENTICATED` (missing/invalid token), `UNAVAILABLE` (IdP outage), `INVALID_ARGUMENT` (empty query, bad limit/`meeting_id`). There is no `protoc` in the toolchain yet, so `libs/shared/rag_proto.py` builds the message classes at runtime from a descriptor; a test asserts it matches `rag.proto` field-for-field. Swap in generated stubs when protoc lands.

Covers `transcript_chunks` only. **`chat_messages` is not covered** - see the design note below.

### `chat_messages` FTS: what is needed (design note, nothing built)

`system.md` and `CLAUDE.md` say the FTS arm spans `transcript_chunks` + `chat_messages`. Only the first is possible today. `chat_messages` is owned by `core-api` (`mytheus-backend`, schema `public`), is written directly with no outbox, and the registered topic catalog has no chat topic. This tier does not read another repo's tables, so a search read-model cannot be built here until a **canon topic decision** is made. Options, for the canon owners (`mytheus-events` / `mytheus-docs`) to pick between:

1. **Recommended: a new `chat-messages` topic**, produced by `core-api` through its outbox, carrying `org_id`/`meeting_id`/`message_id`/`sender_id`/`text`/`created_at` and an edit/delete tombstone flag. `search-service` then adds a second consumer group + `intelligence.chat_message_search` (same shape as `transcript_chunk_search`: generated `tsvector`, RLS by `org_id`, redacted/deleted rows never indexed) and a `source` field on `FtsHit` so `rag-service` can cite chat vs transcript.
2. A cross-repo `SELECT` grant on `public.chat_messages` for `mytheus_ai_app`: rejected by the "never join into another repo's tables" rule and it couples schema changes.

Until then the arm is transcript-only and `rag-service` must not assume chat recall.

## GPU serving (embedding-service, reranker-service)

Both are Ray Serve apps (`serve.py`). `EMBEDDING_GPU=1` / `RERANKER_GPU=1` request `num_gpus` and the CUDA device; `0` (default) runs on CPU.

- **Warm floor (reranker):** `autoscaling_config.min_replicas` = `RERANKER_MIN_REPLICAS` (default 1, clamped >= 1, so it never scales to zero), `RERANKER_MAX_REPLICAS` (default 4), 10-minute downscale delay. Each replica loads the model and runs a real forward pass in `__init__`, so Ray Serve does not route to it until it is warm; the FastAPI process (`main.py`) does the same at startup (`RERANKER_PRELOAD=0` / `EMBEDDING_PRELOAD=0` to opt out, e.g. tests). "Never zero during business hours, per region" scheduling is an ops concern above one Serve config.
- **Images:** `services/<svc>/Dockerfile.gpu` (CUDA 12.4.1 runtime base, pinned tag, uv 0.5.11 pinned, non-root uid 10001, weights baked in). Build from the repo root: `docker build -f services/reranker_service/Dockerfile.gpu -t mytheus/reranker-service:gpu .`
- **Ray Serve configs:** `services/<svc>/serve_config.gpu.yaml` (GPU) and `serve_config.yaml` (CPU), run with `serve run <file>`.

**UNMEASURED / UNVERIFIED.** There is no CUDA device in the authoring environment: the Dockerfiles were not built, the GPU Serve configs were not run, and none of the GPU SLOs have a measurement - embedding batch-of-32 < 100ms warm, reranker warm p95 < 500ms and cold < 1.5s. Pin the base image by digest and measure on a GPU host before treating any of these as met.

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic key for LiteLLM tiered routing + query expansion |
| `LITELLM_CONFIG` | Router config (model → tier mapping, per-tenant budget hooks) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant endpoint (per-tenant collections) |
| `DATABASE_URL` / `RELAY_DATABASE_URL` / `BOOTSTRAP_DATABASE_URL` | Postgres (schema `intelligence`) as `mytheus_ai_app` / `mytheus_ai_relay` / the migration owner. The AGE graph connection is `knowledge-graph-service`'s own setting |
| `REDPANDA_BROKERS` / `SCHEMA_REGISTRY_URL` | Event backbone (Kafka API) + Avro Schema Registry |
| `REDIS_URL` | (dev default `redis://localhost:6390/0`) Answer cache, query-embed cache, per-tenant budget mirror, dedupe inbox (keys prefixed `nexus:{org_id}:`) |
| `RAY_ADDRESS` / `EMBEDDING_GPU` / `RERANKER_GPU` | Ray Serve cluster + GPU placement for the model servers |
| `RERANKER_MIN_REPLICAS` / `RERANKER_MAX_REPLICAS` / `RERANKER_PRELOAD` / `EMBEDDING_PRELOAD` | Reranker warm floor and startup model preload |
| `GRPC_PORT` | search-service gRPC FTS port (default 50103) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `SENTRY_DSN` | OpenTelemetry traces + error reporting |

## Testing

- **`pytest`** - unit and integration tests; infra-backed tests (Postgres/Redis/Redpanda/Qdrant/MinIO) skip when the stack is down; mock the GPU model servers and LLM calls in CI.
- **Eval harness** - golden-set quality gates for the agents (extraction F1, confidence calibration, relevance precision/recall, conflict false-positive rate) and for RAG (citation validity, retrieval quality). Prompt changes must pass the eval suite before rollout.

## Related repos

- `mytheus-intelligence` (this repo) - AI / inference tier (`agent-runtime-service`, `rag-service`, `embedding-service`, `reranker-service`, `knowledge-graph-service`, `search-service`, `ingestion-service`).
- `mytheus-backend` - CRUD / control plane plus async orchestration: `api-gateway`, `bff`, `identity-tenant-service`, `core-api`, `meeting-service`, `integration-service`, `billing-metering-service`, `admin-ops-service`, `notification-service`, `workflow-service`.
- `mytheus-realtime` - real-time tier (`realtime-gateway`, `transcription-service`).
- `mytheus-events` - Avro schemas, the topic taxonomy, and Schema Registry contracts.
- `mytheus-platform` - deploy substrate: Docker Compose, the deploy scripts, CI templates.
- `mytheus-ui` - the Next.js 14 web app (`web-app`) and agent sidebar.
- `mytheus-mobile` - the Expo / React Native client.
- `mytheus-docs` - the canon architecture set, `_canon/` source of truth, and the Glossary (doc 24).

## License

Proprietary. © Mytheus. All rights reserved.
