"""0003 search-service: transcript_chunk_search (this repo's OWN FTS read model)

Plan 5-6 task 10 (`search-service`). Plan step 1's literal pseudocode says:

    ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS tsv tsvector
      GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
    CREATE INDEX ... ON transcript_chunks USING gin(tsv);
    (same for chat_messages)

This migration deliberately does NOT do that. `public.transcript_chunks` is owned by
`transcription-service` in `mytheus-realtime` (confirmed live: a partitioned table, RANGE
(created_at), granted only to `mytheus_app`/`mytheus_admin`/`postgres` — `mytheus_ai_app` and
`mytheus_ai_relay` hold ZERO grants on it today), and `public.chat_messages` is owned by
`core-api` in `mytheus-backend` (confirmed live: FKs to organizations/meetings/users, same
grant shape). Both tables live in the SAME physical `mytheus` Postgres this repo also uses, but
neither is this repo's schema to ALTER. The task brief is explicit: "If FTS requires a tsvector
column or index on someone else's table, that is a real cross-repo ownership question... Do NOT
silently ALTER another service's table." This band has already refused two equivalent
second-writer temptations (task 1: teams/departments; task 4: the team_members write) and both
refusals were correct.

Instead: `intelligence.transcript_chunk_search`, this repo's OWN read model, built ONLY from the
real `transcript-chunks` Kafka topic (the exact same topic `embedding-service` already consumes
for the vector arm; CLAUDE.md's service-map row for search-service literally reads "Events in:
transcript-chunks" / "Builds and owns its FTS indexes"). This satisfies the plan's actual intent
(a `GENERATED ALWAYS AS (to_tsvector(...)) STORED` column + a GIN index, auto-maintained on
every insert) without a schema write on a table this repo does not own. See
`services/search_service/consumer.py`'s module docstring for the full reasoning, and the task
report for the `chat_messages` arm, which is BLOCKED (no event topic exists for it at all — see
that docstring too) rather than built around by touching `core-api`'s table directly.

Same schema/RLS/grant conventions as migrations `9c2b7e4a1d03` (task 7) and `b3f7a1c9e402`
(task 9): `intelligence` schema, FORCE RLS + the identical fail-closed `tenant_isolation`
predicate, and no new GRANT statements for `mytheus_ai_app` (migration `9c2b7e4a1d03`'s
`ALTER DEFAULT PRIVILEGES IN SCHEMA intelligence` already covers any table created here from
that point forward — proven, not assumed, in `tests/test_search_migration.py`, the same
discipline task 9 used for its own tables).

Idempotent redelivery. `UNIQUE (org_id, chunk_id)` lets `consumer.py`'s writes use
`ON CONFLICT (org_id, chunk_id) DO NOTHING`, so redelivering the same `transcript-chunks` event
(BaseConsumer's own Redis+inbox dedupe notwithstanding) never inserts a duplicate row — the same
belt-and-suspenders idempotency shape `embedding_service/consumer.py`'s stable Qdrant point id
gives the vector arm, applied here to a Postgres upsert instead.

No partitioning. This task's data volume does not warrant it yet (unlike `transcript_chunks`'s
own documented range-partitioning at scale, which this repo does not own or replicate); nothing
here required revoking a direct grant from an application role for a partition, per the task
brief's conditional instruction.

Revision ID: d415f023220e
Revises: b3f7a1c9e402
Create Date: 2026-08-06

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d415f023220e"
down_revision: str | None = "b3f7a1c9e402"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "intelligence"


def _force_rls(table: str) -> None:
    """Identical predicate shape to migrations `9c2b7e4a1d03`/`b3f7a1c9e402` (duplicated, not
    imported — Alembic revision files are each self-contained by this repo's convention)."""
    op.execute(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY;")
    op.execute(f"ALTER TABLE {SCHEMA}.{table} FORCE ROW LEVEL SECURITY;")
    predicate = "org_id = NULLIF(current_setting('app.current_org', true), '')::uuid"
    op.execute(
        f"CREATE POLICY tenant_isolation ON {SCHEMA}.{table} "
        f"USING ({predicate}) WITH CHECK ({predicate});"
    )


def upgrade() -> None:
    op.execute(f"""
        CREATE TABLE {SCHEMA}.transcript_chunk_search (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id      UUID NOT NULL,
            meeting_id  UUID NOT NULL,
            chunk_id    UUID NOT NULL,
            speaker_id  TEXT NOT NULL,
            start_ms    INT NOT NULL,
            end_ms      INT NOT NULL,
            text        TEXT NOT NULL,
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT transcript_chunk_search_org_chunk_uq UNIQUE (org_id, chunk_id)
        );
        """)
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_transcript_chunk_search_org_id "
        f"ON {SCHEMA}.transcript_chunk_search (org_id);"
    )
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_transcript_chunk_search_org_meeting "
        f"ON {SCHEMA}.transcript_chunk_search (org_id, meeting_id);"
    )
    # The plan's own literal ask: a GIN index over the generated tsvector column, so `GET
    # /search`'s `tsv @@ websearch_to_tsquery(...)` is index-scanned, not sequential.
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_transcript_chunk_search_tsv "
        f"ON {SCHEMA}.transcript_chunk_search USING gin(tsv);"
    )

    _force_rls("transcript_chunk_search")


def downgrade() -> None:
    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.transcript_chunk_search;")
