# PREVIOUS.md

Running log of what was built here and why, so any human or agent can pick up cleanly.

## Session 1 — Plan 0, Phase 0D: scaffold the AI/inference repo

Executed **Phase 0D** of `mytheus-docs/plan/plan-0-scaffold-and-contracts.md`: stood up the seven-service workspace so the AI tier builds and boots, with zero domain logic yet. This tier NEVER writes Postgres; it consumes/produces events and calls the gRPC mesh. The scaffold only proves buildability — agents, RAG, GPU serving, and the graph land in later bands.

### What exists now

- **`pyproject.toml`** — one workspace (hatchling, `packages = ["services", "libs"]`). Real forward deps declared per the plan: `fastapi`, `uvicorn`, `pydantic`, `langgraph`, `litellm`, `confluent-kafka[avro,schemaregistry]`, `qdrant-client`, `grpcio`, `httpx`, `redis[hiredis]`. GPU serving is an **optional `gpu` extra** (`ray[serve]`) so the default dev loop stays GPU-free and fast. Dev group: pytest, pytest-asyncio, ruff, black, mypy.
- **Seven services** under `services/<module>/`:
  - Five FastAPI services — `agent_runtime_service`, `rag_service`, `knowledge_graph_service`, `search_service`, `ingestion_service` — each a 3-line `main.py` that calls `create_app("<canonical-name>")` and answers `/healthz`.
  - Two GPU services — `embedding_service`, `reranker_service` — each a `serve.py` with a **Ray Serve placeholder** `@serve.deployment` returning a `{"status": "stub", ...}` dev response. The Ray import is **guarded** (`try/except ModuleNotFoundError`) so the module imports fine when Ray is absent; the deployment is simply not defined in dev.
- **`libs/shared/`** — `app.py` (`create_app(service_name)` → FastAPI + `/healthz`) and `clients.py` (`load_events_client()`, a **lazy** importer of the published `mytheus_events` Avro client; fails closed with an actionable message until that wheel is installed, so importing the module never hard-fails the scaffold — the AI tier consumes contracts, never hand-rolls serializers).
- **`proto/rag.proto`** — placeholder for the internal gRPC mesh (`rag-service` → embedding/reranker/knowledge-graph/search). `org_id` rides in gRPC metadata, never the message body.
- **`tests/test_healthz.py`** — parametrized smoke test hitting `/healthz` on all five FastAPI apps + a test asserting `load_events_client()` fails closed.
- **`.env.example`**, **`Makefile`** (uniform `install/lint/typecheck/test/ci` targets, matching the other repos).

### Deviations from the plan (and why)

- **Flat `services/<underscore_module>/main.py` layout instead of the plan's `services/<hyphen-name>/src/<module>/`.** Rationale: this matches the sibling `mytheus-backend` layout *exactly*, so both Python repos share one identical hatchling + mypy + import convention (`uvicorn services.<svc>.main:app`) with no src-layout `mypy_path` gymnastics. The canonical hyphenated service name is preserved verbatim as the `create_app("agent-runtime-service")` label (and will be the deploy-descriptor name), so nothing about canon is lost — only the on-disk folder uses the underscore form, same as `identity_tenant_service` in the backend.
- **GPU services are import-guarded rather than hard-requiring Ray.** The plan's acceptance explicitly allows "skipped behind an import guard when Ray is absent"; this keeps `uv sync` (no `gpu` extra) light and the gates GPU-free.

### Verification (all green)

- `uv run ruff check .` — clean.
- `uv run black --check .` — clean (21 files).
- `uv run mypy .` — clean, 19 source files, `strict = true`. Two localized `# type: ignore[attr-defined]` on the Ray `.bind()` calls (Ray's `@serve.deployment` rewrites the class at runtime; mypy can't see it) — the same class of documented escape hatch used for generated stubs elsewhere.
- `uv run pytest -q` — 6 passed (5 `/healthz` + 1 lazy-import fail-closed).
- Sanity: both GPU `serve.py` modules import with Ray absent; `rag-service` `/healthz` returns `200 {"status":"ok","service":"rag-service"}`.
