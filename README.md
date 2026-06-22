# Reprise AI - Intelligence / Inference Tier (`reprise-intelligence`)

> The AI/inference tier of **Reprise AI** - the six LangGraph agents, the hybrid RAG pipeline, GPU embedding/reranker serving, the Apache AGE knowledge graph, lexical search, and document ingestion. Internal architecture codename: **NEXUS**.

## What this is

`reprise-intelligence` is the **AI / inference tier** of Reprise AI - an intelligent meeting-orchestration multi-tenant SaaS where every participant gets a personal AI agent. This repo owns the intelligence and retrieval surfaces of the platform:

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
| `knowledge-graph-service` | Apache AGE `nexus_knowledge`; **only writer**; Cypher templates parameterized by `$org_id`; writes `decisions`/`actions` edges **only on confirmation**. | FastAPI (gRPC) | Cypher query req/s |
| `search-service` | Postgres FTS arm of hybrid retrieval (`tsvector` over `transcript_chunks` + `chat_messages`). | FastAPI (gRPC) | search req/s + index size |
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
reprise-intelligence/
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

- Python 3.11+
- A GPU is optional for local development (CPU inference works for `embedding-service`/`reranker-service` at dev scale; mock the model servers in tests).
- Docker Compose for the local backing services: **Qdrant**, **Postgres + Apache AGE**, **Redpanda**, **Redis**.
- LLM provider API keys (Anthropic / OpenAI) for any path that exercises LiteLLM.

**Install**

```bash
# from the repo root, per service or via the workspace tooling
python -m venv .venv && source .venv/bin/activate
pip install -e "services/<name>[dev]"
```

**Run the backing services**

```bash
docker compose up -d   # Qdrant + Postgres/AGE + Redpanda + Redis
```

**Run a service**

```bash
# example: the RAG orchestrator
uvicorn rag_service.app:app --reload --port 8081

# example: the agent runtime (consumes transcript-chunks)
python -m agent_runtime_service.main
```

**Test**

```bash
pytest
```

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | LLM provider keys for LiteLLM tiered routing + query expansion |
| `LITELLM_CONFIG` | Router config (model → tier mapping, per-tenant budget hooks) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant endpoint (per-tenant collections) |
| `POSTGRES_DSN` | Postgres + Apache AGE (`nexus_knowledge`) connection string |
| `REDPANDA_BROKERS` / `SCHEMA_REGISTRY_URL` | Event backbone (Kafka API) + Avro Schema Registry |
| `REDIS_URL` | Answer cache, query-embed cache, per-tenant budget mirror, dedupe inbox (keys prefixed `nexus:{org_id}:`) |
| `RAY_ADDRESS` / `EMBEDDING_GPU` / `RERANKER_GPU` | Ray Serve cluster + GPU placement for the model servers |
| `OTEL_EXPORTER_OTLP_ENDPOINT` / `SENTRY_DSN` | OpenTelemetry traces + error reporting |

## Testing

- **`pytest`** - unit and integration tests; mock the GPU model servers and LLM calls in CI.
- **Eval harness** - golden-set quality gates for the agents (extraction F1, confidence calibration, relevance precision/recall, conflict false-positive rate) and for RAG (citation validity, retrieval quality). Prompt changes must pass the eval suite before rollout.

## Related repos

- `reprise-intelligence` (this repo) - AI / inference tier.
- `reprise-core` - CRUD / control plane (identity-tenant, core-api, meeting, billing-metering, integration, admin-ops, notification).
- `reprise-realtime` - real-time tier (realtime-gateway, transcription-service).
- `reprise-workflows` - async orchestration (Temporal workers).
- `reprise-web` - the Next.js 14 web app and agent sidebar.
- `reprise-gateway` - api-gateway and bff edge.
- `reprise-infra` - Kubernetes (EKS/GKE) + Terraform IaC, CI/CD.
- `reprise-docs` - the canon architecture set, `_canon/` source of truth, and the Glossary (doc 24).

## License

Proprietary. © Reprise AI. All rights reserved.
