"""0002 ingestion tables: document_chunks, document_elements, ingestion_jobs

Plan 5-6 task 9 (`ingestion-service`). Adds the three tables the task brief's plan text calls
for: the two content tables named verbatim in the plan (`document_chunks`, `document_elements`)
plus "an ingestion job state table" for `GET /ingest/jobs/:id`. All three land in the
`intelligence` schema established by migration `9c2b7e4a1d03` — never `public` (see that
migration's docstring and `libs/shared/db.py` for why: `mytheus-backend` owns unrelated
`public` tables in the same physical database, and a dedicated schema keeps every grant
explicit rather than inherited).

No new GRANT statements are needed for `mytheus_ai_app`: migration `9c2b7e4a1d03` already ran
`ALTER DEFAULT PRIVILEGES IN SCHEMA intelligence GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES
TO mytheus_ai_app`, scoped to this schema, which — by Postgres's own semantics — applies to
every table created in `intelligence` from that point forward, including these three. That
migration's docstring said as much explicitly ("Tables/sequences created by LATER migrations in
THIS schema (tasks 8-10) inherit these grants automatically"); this migration is the first to
rely on it, and `tests/test_migration_rls_grants.py` (existing) plus the new assertions in
`tests/test_ingestion_migration.py` prove the inheritance actually happened rather than assuming
it.

No new roles: `mytheus_ai_relay` never touches these tables (they are not drained by the
outbox relay; only `intelligence.outbox` is), so it gets no grant here, matching the least-
privilege posture migration `9c2b7e4a1d03` established.

RLS: FORCE + the identical fail-closed `tenant_isolation` predicate as migration `9c2b7e4a1d03`,
on all three tables, per the task brief ("RLS FORCE + tenant_isolation policy on all three").

Idempotent re-run. The plan's acceptance check ("a re-run is idempotent (stable element_id ->
dedupe)") is enforced here with a UNIQUE constraint on `(org_id, document_id, element_id)` for
both `document_chunks` and `document_elements`, so `services/ingestion_service/pipeline.py`'s
writes use `ON CONFLICT (...) DO NOTHING` and a re-parse of the same document that produces the
same element boundaries never inserts a duplicate row (the same guarantee the plan's own outbox
already provides on the `document-chunks` event side via `(topic, idempotency_key)`).

No partitioning. This task's data volume does not warrant it yet (unlike `transcript_chunks`'s
documented range-partitioning at scale); nothing here required revoking a direct grant from an
application role for a partition, per the task brief's conditional instruction.

Revision ID: b3f7a1c9e402
Revises: 9c2b7e4a1d03
Create Date: 2026-08-06

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b3f7a1c9e402"
down_revision: str | None = "9c2b7e4a1d03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "intelligence"


def _force_rls(table: str) -> None:
    """Identical predicate shape to migration `9c2b7e4a1d03`'s `_force_rls` (duplicated, not
    imported — Alembic revision files are each self-contained by convention in this repo, and a
    cross-migration-file import would couple this file's correctness to that file never
    changing)."""
    op.execute(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY;")
    op.execute(f"ALTER TABLE {SCHEMA}.{table} FORCE ROW LEVEL SECURITY;")
    predicate = "org_id = NULLIF(current_setting('app.current_org', true), '')::uuid"
    op.execute(
        f"CREATE POLICY tenant_isolation ON {SCHEMA}.{table} "
        f"USING ({predicate}) WITH CHECK ({predicate});"
    )


def upgrade() -> None:
    op.execute(f"""
        CREATE TABLE {SCHEMA}.document_chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id      UUID NOT NULL,
            document_id UUID NOT NULL,
            element_id  TEXT NOT NULL,
            element_type TEXT NOT NULL,
            text        TEXT NOT NULL,
            "position"  INT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT document_chunks_org_doc_element_uq UNIQUE (org_id, document_id, element_id)
        );
        """)
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_document_chunks_org_id ON {SCHEMA}.document_chunks (org_id);"
    )
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_document_chunks_org_document "
        f"ON {SCHEMA}.document_chunks (org_id, document_id);"
    )

    op.execute(f"""
        CREATE TABLE {SCHEMA}.document_elements (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id      UUID NOT NULL,
            document_id UUID NOT NULL,
            element_id  TEXT NOT NULL,
            element_type TEXT NOT NULL,
            raw         JSONB NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT document_elements_org_doc_element_uq UNIQUE (org_id, document_id, element_id)
        );
        """)
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_document_elements_org_id ON {SCHEMA}.document_elements (org_id);"
    )
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_document_elements_org_document "
        f"ON {SCHEMA}.document_elements (org_id, document_id);"
    )

    # `id` doubles as the `document_id` used by document_chunks/document_elements and the
    # object-storage key (libs/shared/object_storage.py) — one identifier, minted server-side
    # (uuid4()) the moment a job is created, never caller-supplied. GET /ingest/jobs/:id looks
    # up by this id.
    op.execute(f"""
        CREATE TABLE {SCHEMA}.ingestion_jobs (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id       UUID NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
            source       TEXT NOT NULL CHECK (source IN ('upload', 'calendar_attachment')),
            bucket       TEXT NOT NULL,
            object_key   TEXT NOT NULL,
            chunk_count  INT NOT NULL DEFAULT 0,
            error        TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """)
    op.execute(
        f"CREATE INDEX ix_{SCHEMA}_ingestion_jobs_org_id ON {SCHEMA}.ingestion_jobs (org_id);"
    )

    _force_rls("document_chunks")
    _force_rls("document_elements")
    _force_rls("ingestion_jobs")


def downgrade() -> None:
    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.ingestion_jobs;")
    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.document_elements;")
    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.document_chunks;")
