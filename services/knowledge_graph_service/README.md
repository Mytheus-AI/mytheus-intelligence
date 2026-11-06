# knowledge-graph-service

The only service that writes the Apache AGE graph `nexus_knowledge`.

- **Where AGE runs:** a dedicated `apache/age:release_PG16_1.6.0` container (`age` in `../../docker-compose.yml`, port 5434). The shared `supabase/postgres:15.8.1.060` does not ship the `age` extension. This is risk R1 and its pre-authorized fallback: self-hosted Postgres with AGE.
- **Writes:** `worker.py` consumes `tenant-events` (group `knowledge-graph-service.topic-seed`). For each `provisioned` event it MERGEs one `Topic {name, org_id}` per distinct taxonomy topic. v1 has 25 of them. The producer is `identity-tenant-service`'s `provision_organization`, which both org provisioning and `make seed` already call.
- **Safety:** all Cypher goes through `libs/shared/age.py`. Every statement must bind `$org_id`, and `org_id` is injected from a typed UUID. Values travel only as `cypher()` agtype parameters. Labels come from the frozen sets.
- **Taxonomy:** `taxonomy/role_topics.v1.json` is a copy of the backend's immutable `role_topics.v1.yaml`. `tests/test_knowledge_graph_topics.py` checks that the two still match.

| Env | Meaning |
| --- | --- |
| `AGE_DATABASE_URL` | Required, with no default. Example: `postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph` |
| `ROLE_TOPICS_TAXONOMY_VERSION` | Default `v1` |
| `DATABASE_URL`, `REDIS_URL`, `REDPANDA_BROKERS`, `SCHEMA_REGISTRY_URL` | The same as the other consumers (used for the inbox, dedupe and DLQ) |

```bash
docker compose up -d age
python -m services.knowledge_graph_service.worker
python -m services.knowledge_graph_service.seed_topics --org-id <uuid> [--verify-only]
MYTHEUS_REQUIRE_AGE=1 pytest tests/test_knowledge_graph_age_live.py
```
