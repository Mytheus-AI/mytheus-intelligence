# knowledge-graph-service

The only service that writes the Apache AGE graph `nexus_knowledge`.

- **Where AGE runs:** a dedicated `apache/age:release_PG16_1.6.0` container (`age` in `../../docker-compose.yml`, port 5434). The shared `supabase/postgres:15.8.1.060` does not ship the `age` extension. This is risk R1 and its pre-authorized fallback: self-hosted Postgres with AGE.
- **Production:** a dedicated `apache/age` container beside the system-of-record Postgres (mytheus-docs ADR-0031 D1).
- **Writes** (`worker.py`, two consumers on two threads):
  - `tenant-events` (group `knowledge-graph-service.topic-seed`): for each `provisioned` event, one `Topic {name, org_id}` per distinct taxonomy topic (25 in v1).
  - `decisions`, `actions`, `meeting-room-events` (group `knowledge-graph-service.confirmed-facts`, `facts_consumer.py` -> `confirmed.py`): confirmed decisions write `Decision -PRODUCED-> Meeting` and, from `supersedes`, `-FOLLOWS_FROM->` the prior decision (marked `superseded`); confirmed actions write `Action -PRODUCED-> Meeting` and `-ASSIGNED_TO-> Person`; a user `joined` writes `Person -ATTENDED-> Meeting`. One `graph-events` record per edge, through `intelligence.outbox`. Nothing else writes edges: candidates never reach these topics.
- **Reads** (`templates.py`): `decision_lineage`, `decisions_by_person`, `actions_blocked_on_decision`, `related_decisions_for_topic`. `$org_id` on every node and edge pattern, depth <= 10, at most 100 rows, soft-deleted nodes filtered, READ ONLY transaction with a 2 s statement timeout. Served by `POST /graph/query/{template}` and `POST /graph/cypher` (`main.py`, read-only, `$org_id` required) and gRPC `KnowledgeGraphService.QueryTemplate` (`grpc_main.py`, `proto/knowledge_graph.proto`). Callers authenticate with `X-Mytheus-Service` / `X-Mytheus-Service-Token`; the tenant is `X-Mytheus-Org`. Allowed callers: `bff`, `rag-service`, `agent-runtime-service`.
- **Rebuild:** `rebuild.py` re-applies confirmed rows (JSON lines exported from `core-api`) through the same writers; `--purge` drops the tenant's subgraph first.
- **Safety:** all Cypher goes through `libs/shared/age.py`. Every statement must bind `$org_id`, and `org_id` is injected from a typed UUID. Values travel only as `cypher()` agtype parameters. Labels come from the frozen sets.
- **Taxonomy:** `taxonomy/role_topics.v1.json` is a copy of the backend's immutable `role_topics.v1.yaml`. `tests/test_knowledge_graph_topics.py` checks that the two still match.

| Env | Meaning |
| --- | --- |
| `AGE_DATABASE_URL` | Required, with no default. Example: `postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph` |
| `ROLE_TOPICS_TAXONOMY_VERSION` | Default `v1` |
| `MYTHEUS_KG_CALLER_TOKENS` | `service=secret,...` (32+ chars each) for the read surfaces. Unset refuses every caller |
| `KG_GRPC_BIND` | gRPC bind address, default `0.0.0.0:50104` |
| `KG_CONSUMERS` | Which consumers `worker.py` starts, default `topic-seed,confirmed-facts,erasure` (`erasure`: `admin-events` erasure requests, `erasure_consumer.py`) |
| `DATABASE_URL`, `REDIS_URL`, `REDPANDA_BROKERS`, `SCHEMA_REGISTRY_URL` | The same as the other consumers (used for the inbox, dedupe and DLQ) |

```bash
docker compose up -d age
python -m services.knowledge_graph_service.worker
python -m services.knowledge_graph_service.seed_topics --org-id <uuid> [--verify-only]
python -m services.knowledge_graph_service.rebuild --org-id <uuid> --rows rows.jsonl [--purge]
uvicorn services.knowledge_graph_service.main:app --port 8202
python -m services.knowledge_graph_service.grpc_main
MYTHEUS_REQUIRE_AGE=1 pytest tests/test_knowledge_graph_age_live.py tests/test_knowledge_graph_facts_live.py tests/test_knowledge_graph_api.py
```
