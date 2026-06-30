# `mytheus-intelligence` - System Documentation

Deep architecture reference for the **Mytheus** intelligence / inference tier (internal codename **NEXUS**). This document is derived from the canon set; where a name, number, or rule is frozen, it is reproduced verbatim. Authoritative sources: `08-ai-agent-architecture.md`, `09-rag-and-inference-serving.md`, `03-container-architecture.md`, `05-data-architecture.md`, `07-event-architecture.md`, and the `_canon/` files they cite.

---

## 1. Purpose & bounded context

Mytheus is an intelligent meeting-orchestration multi-tenant SaaS. Every participant gets a **personal AI agent** that attends their meetings, captures role-relevant notes, suggests questions in real time, detects decisions / conflicts / workload issues, and manages follow-ups. The differentiator is twofold: **personalized per-participant intelligence** (not one shared note-taker) plus **agent-to-agent collaboration across meetings** to fill context gaps.

`mytheus-intelligence` is the **AI / inference tier** - one of three scaling families (CRUD/control plane, real-time, AI/inference). Its bounded context is intelligence and retrieval: agents, RAG, GPU model serving, the knowledge graph, lexical search, and document ingestion. It is a distinct **cost center** that autoscales on queue depth, GPU utilization, and token throughput, independently of CRUD req/s and real-time connections.

**Out of scope** (lives in other repos / tiers): meeting transport and transcription (LiveKit → Deepgram → `transcript-chunks`); the relational system of record and the confirmation write path (`core-api`); billing/metering aggregation and Stripe; the realtime WebSocket fan-out; Temporal workflows.

---

## 2. The six LangGraph agents

All six are LangGraph graphs on a shared `BaseAgent` (§3). One `PersonalListenerAgent` is spawned **per participant per meeting**; the others are spawned per meeting and fan their reasoning across the roster.

### `PersonalListenerAgent`
- **Trigger:** a 30-second `transcript-chunks` window for a meeting the user is in.
- **Role:** calls `get_user_context(user_id)` (role/skills/team/recent topics; Redis-cached 5-min TTL) and scores each window's **relevance 0–1** to that user. The relevance score is the *pre-gate* (does this matter to this user) - distinct from the confidence surfacing gate.
- **Output:** worklog content via `agent-results` (persisted to `worklogs` by `core-api`: `content`, `relevance_score`, `created_by_agent = true`, `source_chunk_ids`). Surfaced in the My Notes sidebar live.
- **Confidence handling:** above the per-tenant relevance threshold it produces; surfacing then obeys the §6 confidence thresholds.

### `DecisionAgent`
- **Trigger:** a `transcript-chunks` window.
- **Role:** detects decisions; extracts `rationale`, stakeholders, and a `confidence` score.
- **Output:** a **decision candidate** on `agent-results`; surfaces a confirm/edit/dismiss card. **Never silently commits** - the `decisions` event and the AGE `PRODUCED` edge happen only on user confirmation (§9).
- **Confidence handling:** confidence-scored candidate; surfaced per §6.

### `ActionAgent`
- **Trigger:** a `transcript-chunks` window.
- **Role:** detects action items; extracts `assignee_id`, `due_date`, dependencies, and `confidence`.
- **Output:** an **action candidate** on `agent-results` (new-task toast). On confirmation the `actions` event and the AGE `ASSIGNED_TO` edge are created.
- **Confidence handling:** same candidate-then-confirm rule; surfaced per §6.

### `QuestionAgent`
- **Trigger:** a `transcript-chunks` window + a context-gap heuristic.
- **Role:** detects context gaps in live conversation and proposes role-specific questions with `priority` 1–3 and `status` in {Ask, Dismiss, Save-for-later}. Save-for-later persists into the next meeting (`agent_suggestions`).
- **Output:** `agent_suggestions` via `agent-results` (suggestion card).
- **Confidence handling:** surfaced per §6.

### `ContextGapAgent`
- **Trigger:** a reference to context a participant lacks.
- **Role:** the novel multi-agent piece. Queries **other users' agents** over `agent-requests`/`agent-responses` with `correlation_id` and a deadline, falling through the chain **peer-agents → related-meeting agents → RAG → silent** (§8). Every round-trip is logged to `agent_collaborations`.
- **Output:** a context-fill card; `agent-requests` to peers.
- **Confidence handling:** stays silent rather than fabricating when the chain exhausts (silent-on-empty).

### `ProactiveAgent`
- **Trigger:** confirmed `decisions`/`actions`; AGE graph queries.
- **Role:** three detectors - **conflict** (a new confirmed decision contradicting a past one, via an AGE `nexus_knowledge` query through `knowledge-graph-service`, complemented by semantic conflict detection over the `decisions__{org_id}` Qdrant collection); **workload** (sum of open `action_items` per assignee, flagged at 80%/95% of capacity); **missing-stakeholder** (org-graph traversal over departments/teams).
- **Output:** `notifications` via `agent-results` (conflict banner, workload warning, stakeholder/dependency card).
- **Confidence handling:** Opus-tier high-stakes reasoning; conflict false-positive rate is an eval gate.

---

## 3. `BaseAgent` lifecycle & `AgentState`

All six agents share one LangGraph contract: a `BaseAgent` lifecycle (`spawn` / `process` / `terminate`), the shared `AgentState` dataclass, and a versioned prompt-template loader.

Lifecycle: `spawn(meeting_id, user_id, org_id)` → `Processing` on each 30s window ready → `Terminated` on meeting end / leave.

`AgentState` carries the meeting/user identifiers, the 30-second `TranscriptWindow` (a window, **not** a single raw chunk), the `get_user_context()` result, and `retrieved_facts` (RAG / peer-agent fills), plus the tenant key and control fields:

- `org_id` - tenant key, injected by authz middleware; `spawn` asserts it is present (fail-closed).
- `agent_id` - stable per-agent identity for agent-to-agent authz + audit.
- `relevance_score` - PersonalListener pre-gate (0–1).
- `confidence` - surfacing gate (0–1), see §6.
- `prompt_version` - resolved template version (staged rollout).
- `token_budget_remaining` - mirror of `agent_budgets` for a fast quota check.
- `correlation_id` - set when this state is part of a collaboration.

`BaseAgent.process` denies-by-default when `token_budget_remaining <= 0` (hard quota). The `PromptLoader` resolves `{prompt_key}@{version}` per tenant so a staged rollout can pin a tenant to a known-good version (e.g. `decision_agent@v7`); tenant pins override the global default and both are reviewable in git.

---

## 4. The cost-controlled LLM pipeline

This is the architectural heart of the agent tier and the fix for **R4 (the LLM cost spiral)**. Five levers, in order:

1. **Relevance pre-gating (Haiku triage).** One cheap Haiku call per window per participant scores relevance; ~35% pass. The 65% that fail are logged and dropped - no extraction call is ever made for them.
2. **Windowed batching.** Deepgram utterances are merged by speaker turn into **30-second windows** (90 windows/meeting versus 360 raw `transcript-chunks`). Agents run **per window, not per raw chunk**.
3. **Tiered model routing (LiteLLM, Haiku → Sonnet → Opus).** Haiku triages/scores; Sonnet extracts (Listener/Decision/Action/Question); Opus is reserved for confirmed high-stakes reasoning (ProactiveAgent conflict, ContextGap fallback synthesis). The router picks the cheapest model that can do the job. Tier mix by call volume: Haiku ~45%, Sonnet ~55%, Opus <1%.
4. **Semantic + Anthropic prompt caching.** Stable prefixes (role taxonomy, system prompt, prior-window context) are Anthropic-prompt-cached; semantically near-duplicate windows hit a semantic cache. Caching is applied *after* tiering.
5. **Per-tenant token budgets + hard quotas.** Every call checks `agent_budgets`, mirrored hot in Redis at **`nexus:{org_id}:budget:{period}`**. At 80%/95% an alert fires; at the `hard_cap` the call is denied (state → `blocked`), deny-by-default. Token usage is emitted on `usage-events` to `billing-metering-service`.

Together these defeat the **R4 cost-spiral risk**: the controlled path runs ~1,206 LLM calls/meeting versus a naive per-chunk ~8,640 - an ~86% reduction in call volume *before* caching savings. The per-tenant Redis budget mirror is what makes the quota enforcement fast enough to gate every call inline.

```python
async def check_budget(org_id: UUID, est_tokens: int) -> Literal["ok", "throttle", "block"]:
    b = await redis.get(f"nexus:{org_id}:budget:{period()}")   # hot mirror of agent_budgets
    if b.tokens_consumed + est_tokens >= b.hard_cap:
        await emit("billing-events", quota_exceeded(org_id))   # → billing-metering-service
        return "block"                                          # deny-by-default
    ...
    return "ok"
```

---

## 5. RAG pipeline

`rag-service` owns the frozen five-stage `RAGService`: **`expand → retrieve → rerank → assemble → cite`**, behind `POST /api/rag/query`. Every retrieval flows through all five stages; there is no shortcut that skips citation tracking.

1. **Expand** - LiteLLM on the Haiku tier returns 2–3 query variants to widen recall (skippable when a tenant is throttled/blocked; RAG degrades to single-query recall rather than failing).
2. **Retrieve (hybrid)** - three arms run **in parallel** per variant and are fused with RRF:
   - **Vector** - `embedding-service` query-embed, then Qdrant search over `transcript_chunks__{org_id}` / `document_chunks__{org_id}`, with the mandatory `org_id` payload filter (defense in depth on top of the per-tenant collection).
   - **Graph** - `knowledge-graph-service` `RELATES_TO` traversal (+ `FOLLOWS_FROM` lineage) in `nexus_knowledge`, every Cypher template parameterized by `$org_id`.
   - **FTS** - `search-service` Postgres `tsvector` over `transcript_chunks` + `chat_messages`, under RLS by `org_id`.
3. **Rerank** - RRF fusion (`k = 60`) → top-50 candidates → `reranker-service` BGE-reranker-base cross-encoder → **top-10**.
4. **Assemble** - token-budget packing (default 3,000 tokens, configurable per surface), dedupe, source diversity. The cap is a cost-control surface (every assembled token is metered to `usage-events`) and tightens automatically as a tenant approaches its `agent_budgets` hard quota.
5. **Cite** - every passage binds its provenance: transcript → `meeting_id` / `chunk_id` / `speaker`; document → `document_id` / `element_id`. **A passage that cannot be bound to a citation is dropped before return** - uncited text never reaches the answer.

The whole result is cached in Redis at `nexus:{org_id}:rag:{user_id}:{query_hash}` (10-min TTL; per-user because expansion and graph traversal are role-tuned). The **warm p95 budget is < 500 ms** (cold < 1.5 s); the cross-encoder rerank is the long pole on a cold path, which is why `reranker-service` keeps a warm-replica floor. `rag-service` is the **sync orchestrator** of the hybrid-retrieval fan-out and reaches the model servers, graph, and search over the internal **gRPC** mesh; `org_id` rides in gRPC metadata on every hop.

---

## 6. Confidence, prompt versioning, and hallucination controls

**Confidence thresholds** (frozen canon, overridable per tenant via `tenant_settings.agent_confidence_overrides`):

| Confidence | Action |
|---|---|
| `> 0.7` | **Surface** to the user (sidebar card / toast) |
| `0.4 – 0.7` | **Silent log** to `worklogs` (no surfacing) |
| `< 0.4` | **Drop** |

**Prompt versioning + staged rollout.** A prompt change is reviewed in a PR, must beat the current default in the eval harness, then canaries on ~1% of tenants (via tenant prompt pins) for 48h before the global default advances. A regression rolls the pin back instantly without redeploying.

**Hallucination controls (layered):** confidence gating; the citation contract (every fill carries resolvable provenance; an unresolvable citation blocks surfacing); confirmation-before-commit for decisions/actions; PII redaction before LLM egress; and silent-on-empty for `ContextGapAgent`.

---

## 7. Inference serving

`embedding-service` and `reranker-service` are dedicated GPU model servers on **Ray Serve / vLLM**, **off the API tier**, with their own autoscalers and cost center. This decoupling is the explicit fix for the inference-on-API-tier coupling and the precondition for both R4 and R5.

- **`embedding-service`** - **BGE-small-en, 384-dim**. Two and only two call patterns: **embed-on-ingest** (consume `transcript-chunks` / `document-chunks`, split 600/60, batch 32, sub-100 ms/batch, upsert to Qdrant) and **query-embed only** (`POST /embed`, internal-only, 1–3 variants, inside the warm-RAG budget). Query embeddings are Redis-cached at `nexus:{org_id}:qembed:{query_hash}`. The bulk of embedding work happens once at ingest (~525 embeds/sec peak); retrieval only ever embeds the short query. **`embedding-service` is the only writer of Qdrant.**
- **`reranker-service`** - **BGE-reranker-base** cross-encoder. Sole job: top-50 RRF candidates → top-10 by joint `(query, passage)` relevance. SLOs split on warm vs cold: **warm p95 < 500 ms**, cold p95 < 1.5 s; a **minimum warm replica floor of ≥ 1 per active region** keeps p95 on the warm side. Owns no data.

Cross-encoder vs bi-encoder: the embedding model encodes query and document independently (precomputable at ingest, cheap query); the reranker concatenates `(query, passage)` and runs a full forward pass per pair (accurate, ~50× the work, only affordable over the 50 RRF survivors). Opposite cost profiles and scaling signals → separate deployments even though both are "BGE on GPU."

---

## 8. Agent-to-agent collaboration

`ContextGapAgent` can ask **other users' agents** for context over `agent-requests` / `agent-responses`. The `AgentRequest` / `AgentResponse` Avro envelopes carry (names verbatim): `requester_agent_id`, `responder_target`, `query`, `deadline`, `correlation_id` (and `org_id`).

**Fallback chain (verbatim):** peer-agents → related-meeting agents → RAG → silent. Each hop has a deadline; on miss the agent falls through. If everything misses, the agent stays silent (no fabricated fill). The **p95 round-trip budget is < 1.5 s** so a fill lands inside the live conversation. Every round-trip - success, fallthrough, or denial - is logged to `agent_collaborations` (`requester`, `responder`, `query`, `result`, `success`, `latency`).

**Authorization - deny-by-default.** The authz middleware enforces three fail-closed rules on every `agent-requests` message:
1. **Same-tenant only.** `request.org_id` must equal the requester agent's bound `org_id` (derived from `requester_agent_id`, not the payload). Agents can never cross org boundaries.
2. **Role scope.** The requester's role must be authorized to query the target's scope; grants are deny-by-default (no grant ⇒ deny).
3. **Audit.** Every grant and denial is appended to `audit_logs` (`actor_type = 'agent'`, append-only) and reflected in `agent_collaborations.success`.

Because all topics are tenant-partitioned, a request is physically constrained to its tenant's partitions; the authz check is defense-in-depth on top of that.

---

## 9. Knowledge graph

`knowledge-graph-service` is the **only writer** of the Apache AGE **`nexus_knowledge`** graph (co-located in Postgres). The graph is **org-partitioned: every node carries `org_id`, and every Cypher template is parameterized by `$org_id`** so a traversal cannot leave the org subgraph.

- **Node labels (verbatim):** `Person`, `Meeting`, `Decision`, `Action`, `Topic`.
- **Edge labels (verbatim):** `ATTENDED`, `PRODUCED`, `ASSIGNED_TO`, `RELATES_TO`, `FOLLOWS_FROM`.

**Writes only on confirmation.** Decisions and actions are *proposed* by agents (candidates on `agent-results`) and *committed* by users. On confirmation, `core-api` writes the row and publishes the confirmed `decisions` / `actions` event through its transactional outbox; `knowledge-graph-service` consumes those and creates the `PRODUCED` / `ASSIGNED_TO` edge. No silent graph writes. The graph is read at request time by `rag-service` (the `RELATES_TO` + `FOLLOWS_FROM` arm) and by `ProactiveAgent` (conflict / lineage queries).

---

## 10. Data ownership

| Store | Owning service (writer) | Tenancy mechanism |
|---|---|---|
| Qdrant (per-tenant collections) | **`embedding-service` only** (upserts) | Physical per-tenant collection `transcript_chunks__{org_id}` / `document_chunks__{org_id}` / `decisions__{org_id}` + mandatory `org_id` payload filter |
| Apache AGE `nexus_knowledge` | **`knowledge-graph-service` only** | Org-partitioned: `org_id` on every node; Cypher parameterized by `$org_id` |
| `document_chunks` / `document_elements` (Postgres) + original blobs (object storage) | **`ingestion-service`** | RLS by `org_id`; per-tenant object key prefix `{org_id}/…` |
| Redis (answer cache, query-embed cache, budget mirror, dedupe inbox) | `rag-service`, `agent-runtime-service` | Every key prefixed `nexus:{org_id}:…` |

The AI tier owns **no other Postgres tables** and **never writes Postgres directly** - see §13.

---

## 11. Events consumed & produced

**Consumed (in):**
- `transcript-chunks` - `agent-runtime-service`, `embedding-service`, `search-service`.
- `document-chunks` - `agent-runtime-service`, `embedding-service`.
- `agent-responses` - `agent-runtime-service` (the requester side of A2A collaboration).
- `decisions` / `actions` (confirmed, from `core-api`) - `knowledge-graph-service`.
- `integration-events` - `ingestion-service` (attachment-to-ingest).

**Produced (out):**
- `agent-results` - agent output (notes, scores, suggestions, candidates).
- `agent-stream` - streamed LLM token deltas to the UI (via `realtime-gateway`).
- `agent-requests` - A2A context-fill requests (`ContextGapAgent`).
- `usage-events` - metered tokens (`agent-runtime-service`, `rag-service`) → `billing-metering-service`.
- `document-chunks` - `ingestion-service` (typed, split document elements).

Every async edge is at-least-once with idempotency: producers write through a transactional outbox; consumers dedupe on `idempotency_key` via an inbox table + the `nexus:{org_id}:dedupe:{topic}:{idempotency_key}` Redis cache; poison messages land in a per-topic DLQ. Every topic is partitioned by `org_id`. (Topic and schema names are frozen in `07-event-architecture.md` / `_canon/`.)

---

## 12. Multi-tenancy

`org_id` is enforced at every layer of this tier:
- **gRPC** - `org_id` travels in gRPC metadata on every internal hop; the callee's authz middleware applies it identically to sync paths.
- **Qdrant** - per-tenant collection `{collection}__{org_id}` **and** mandatory `org_id` payload filter (belt-and-suspenders).
- **AGE** - every Cypher template parameterized by `$org_id`; traversal cannot leave the org subgraph.
- **Redis** - every key prefixed `nexus:{org_id}:…` (answer cache, query-embed cache, budget mirror, dedupe inbox), enabling per-tenant flush on RTBF or plan change.
- **Redpanda** - every topic partitioned by `org_id`; a request is physically constrained to its tenant's partitions.

Agent-to-agent calls are same-tenant-only and fail closed (§8).

---

## 13. Statelessness

The AI tier is **stateless against the relational heart**. `agent-runtime-service` reads context (via `get_user_context` over gRPC) and emits events; it **never writes Postgres directly**. Durable agent output reaches the relational core by emitting `agent-results` and relying on `core-api`'s transactional outbox - `core-api` (not the agent) is the producer of confirmed `decisions` / `actions` events. The agent tier proposes; `core-api` commits. Every service container is horizontally scalable and holds no durable state outside the stores in §10.

---

## 14. Scaling & SLOs

The tier autoscales (KEDA + Ray Serve) on **queue depth, GPU utilization, and token throughput** - its own cost center, distinct from CRUD req/s.

| Surface | Target |
|---|---|
| Agent end-to-end output latency | **p50 < 2 s**, p95 < 4 s |
| A2A collaboration round-trip | **p95 < 1.5 s** |
| `POST /api/rag/query` (warm / cold) | **p95 < 500 ms** / < 1.5 s |
| `reranker-service` rerank (warm) | **p95 < 500 ms** |
| `embedding-service` embed (batch of 32) | < 100 ms |
| Embedding consumer lag | < 5 s (alert at 30 s) |
| Per-tenant token budget | 100% hard-cap enforcement (deny-by-default) |

The per-tenant token budget ties the agent tier to the SaaS business platform: usage is metered on `usage-events`, aggregated, quota-enforced, and surfaced in the tenant admin console.

---

## 15. Dev practices

- **Typing** - full Python type hints; `mypy` clean.
- **Style** - `ruff` + `black`.
- **Tests** - `pytest` with **TDD**; mock GPU model servers and LLM calls in CI.
- **Eval / quality gates** - golden-set eval harness for the agents (extraction F1, confidence calibration via Brier, relevance precision/recall, question usefulness, conflict false-positive rate) and for RAG (citation validity, retrieval quality). Prompt changes must pass before rollout.
- **Commits** - conventional commits; **trunk-based** development with PRs. Commits in this project must **not** include a `Co-Authored-By` footer.

---

## 16. CI/CD

- **GitHub Actions** for build, test, lint, and the eval harness.
- **GPU-runners or mocked inference in CI** - the eval suite runs against mocked model servers / a small GPU runner so prompt-change gates are cheap and deterministic.
- **Model artifact handling** - BGE model weights are pulled from a managed artifact store / Hugging Face cache at build or container-init time, not committed to git (see `.gitignore`).

---

## 17. Observability

- **OpenTelemetry** traces span the gRPC fan-out (`rag-service` → `embedding-service` / `reranker-service` / `knowledge-graph-service` / `search-service`) and the agent pipeline, correlated by `correlation_id`.
- **Token / cost metrics** - per-tenant token volume on `usage-events`, tier mix (Haiku/Sonnet/Opus), cache hit rates (answer + query-embed), budget consumption.
- **Prometheus / Grafana** - per-stage RAG latencies, RRF candidate counts, reranker GPU utilization and warm-replica count, embedding consumer lag (all tenant-tagged, structured JSON logs).
- **Sentry** - error reporting.

---

## References

- `08-ai-agent-architecture.md` - the six agents, `BaseAgent`/`AgentState`, confidence thresholds, the cost-controlled LLM pipeline, A2A collaboration.
- `09-rag-and-inference-serving.md` - `RAGService` pipeline, hybrid retrieval + RRF, `embedding-service` / `reranker-service`, inference off the API tier.
- `03-container-architecture.md` - containers, the gRPC mesh, communication styles, data-store ownership, cross-cutting rules.
- `05-data-architecture.md` - Qdrant per-tenant collections, AGE `nexus_knowledge`, store selection and retention.
- `07-event-architecture.md` - the topic catalog, Avro envelope, outbox/inbox, DLQs, tenant partitioning.
