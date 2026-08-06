"""0001 intelligence schema: outbox, inbox, least-privilege roles, RLS

Plan 5-6 section 3.6 called for outbox/inbox migrations for `ingestion-service` and
`search-service`; `mytheus-intelligence` had NO Alembic history at all, so this is routed to
task 7 (`embedding-service`), the first task to touch this repo, to establish. This migration
is the ONE thing every later consumer/producer in this repo (tasks 8-10) will build on, so it
follows `mytheus-backend`'s reference implementation deliberately rather than inventing a
second dialect: same outbox/inbox column shapes, same `FOR UPDATE SKIP LOCKED` relay-drain
policy shape, same fail-closed RLS predicate.

Three decisions this migration makes, each because of a real, checked constraint:

1. Schema, not `public`. `mytheus-backend` already owns `public.outbox` / `public.inbox` in
   this same physical Postgres (`mytheus`, confirmed live before writing this migration). A
   table of ours in `public` would collide on name, AND would silently inherit
   `mytheus-backend` migration `a1c4e7f20b18`'s `ALTER DEFAULT PRIVILEGES IN SCHEMA public`
   grants to `mytheus_app`/`mytheus_admin` — roles belonging to a different repo's request
   path. A dedicated `intelligence` schema makes every grant explicit (nothing is inherited)
   and gives tasks 8-10 a clean place to add their own owned tables
   (`document_chunks`/`document_elements`/FTS indexes) without re-litigating this.

2. Two roles, not `mytheus-backend`'s three. `mytheus_ai_app` (the consumer/request path,
   subject to `tenant_isolation`) and `mytheus_ai_relay` (drains `intelligence.outbox` only,
   nothing else — same `outbox_relay_drain`-shaped permissive policy `mytheus-backend`
   migration `a1c4e7f20b18` uses to avoid a BYPASSRLS role for the identical reason: the relay
   must read every tenant to drain a shared table, and scoping that to one table and one role
   keeps the cross-tenant reach visible in `pg_policies` instead of hidden in a role
   attribute). No `mytheus_ai_admin` yet: this repo has no operator/ops-console seam analogous
   to `admin-ops-service` at this point in the plan; add one when a task needs it rather than
   pre-provisioning an unused role.

3. No foreign key from `intelligence.outbox`/`intelligence.inbox` to `public.organizations`.
   `org_id` here is an opaque tenant UUID carried by the event envelope header, not a row this
   repo is ever allowed to look up by joining into another repo's table (CLAUDE.md: "cross-tenant
   access... never a cross-service table write" applies equally to reads across a bounded
   context in the same physical database). A hard FK would also wrongly couple this repo's
   migration order to `mytheus-backend`'s tenant-provisioning lifecycle.

Passwords are deliberately absent, mirroring `mytheus-backend` migration `a1c4e7f20b18`: the
roles are created without one, so they cannot authenticate until an operator (or, in dev/test,
`tests/conftest.py`) sets one. A password in a migration is a password in git history.

Revision ID: 9c2b7e4a1d03
Revises:
Create Date: 2026-08-06

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c2b7e4a1d03"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "intelligence"
APP_ROLE = "mytheus_ai_app"
RELAY_ROLE = "mytheus_ai_relay"
ALL_ROLES = (APP_ROLE, RELAY_ROLE)


def _create_role(name: str) -> None:
    """Create a login role idempotently, with no password and no elevated attributes.
    NOSUPERUSER/NOBYPASSRLS stated explicitly (not relied on as defaults) so a later `ALTER
    ROLE` granting either shows up as a diff against a stated intent — same rationale as
    `mytheus-backend` migration `a1c4e7f20b18`."""
    op.execute(f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{name}') THEN
            CREATE ROLE {name} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
          ELSE
            ALTER ROLE {name} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
          END IF;
        END $$;
        """)


def _force_rls(table: str) -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY;")
    op.execute(f"ALTER TABLE {SCHEMA}.{table} FORCE ROW LEVEL SECURITY;")
    # NULLIF(..., '') so an unset OR reset-to-empty app.current_org yields NULL, and the
    # predicate is false (fail-closed: zero rows), never an error. Identical predicate shape
    # to mytheus-backend's `_force_rls`.
    predicate = "org_id = NULLIF(current_setting('app.current_org', true), '')::uuid"
    op.execute(
        f"CREATE POLICY tenant_isolation ON {SCHEMA}.{table} "
        f"USING ({predicate}) WITH CHECK ({predicate});"
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")

    op.execute(f"""
        CREATE TABLE {SCHEMA}.outbox (
            id              BIGSERIAL PRIMARY KEY,
            org_id          UUID NOT NULL,
            topic           TEXT NOT NULL,
            partition_key   TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            schema_subject  TEXT NOT NULL,
            payload         JSONB NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            published_at    TIMESTAMPTZ,
            attempts        INTEGER NOT NULL DEFAULT 0,
            CONSTRAINT outbox_idem_uq UNIQUE (topic, idempotency_key)
        );
        """)
    op.execute(
        f"CREATE INDEX outbox_unpublished_idx ON {SCHEMA}.outbox (created_at) "
        f"WHERE published_at IS NULL;"
    )
    op.execute(f"CREATE INDEX ix_{SCHEMA}_outbox_org_id ON {SCHEMA}.outbox (org_id);")

    op.execute(f"""
        CREATE TABLE {SCHEMA}.inbox (
            consumer_group   TEXT NOT NULL,
            org_id           UUID NOT NULL,
            idempotency_key  TEXT NOT NULL,
            processed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (consumer_group, org_id, idempotency_key)
        );
        """)
    op.execute(f"CREATE INDEX ix_{SCHEMA}_inbox_org_id ON {SCHEMA}.inbox (org_id);")

    # --- Row-Level Security: FORCE + tenant_isolation on both tables ----------------------
    _force_rls("outbox")
    _force_rls("inbox")

    # --- Roles ------------------------------------------------------------------------------
    for role in ALL_ROLES:
        _create_role(role)
    for role in ALL_ROLES:
        op.execute(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {role};")

    # --- Grants: mytheus_ai_app (the consumer/request path) --------------------------------
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {SCHEMA}.outbox TO {APP_ROLE};")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {SCHEMA}.inbox TO {APP_ROLE};")
    op.execute(f"GRANT USAGE, SELECT ON {SCHEMA}.outbox_id_seq TO {APP_ROLE};")

    # --- Grants + drain policy: mytheus_ai_relay --------------------------------------------
    #
    # Same D4-shaped fix as mytheus-backend: the relay must read every tenant to drain the
    # shared outbox, which the fail-closed tenant_isolation policy forbids by default. One
    # permissive policy scoped to this table and this role keeps the cross-tenant reach
    # inspectable in pg_policies rather than hidden in a BYPASSRLS role attribute.
    op.execute(f"GRANT SELECT, UPDATE ON {SCHEMA}.outbox TO {RELAY_ROLE};")
    op.execute(f"GRANT USAGE, SELECT ON {SCHEMA}.outbox_id_seq TO {RELAY_ROLE};")
    op.execute(
        f"CREATE POLICY outbox_relay_drain ON {SCHEMA}.outbox TO {RELAY_ROLE} "
        f"USING (true) WITH CHECK (true);"
    )
    op.execute(f"""
        COMMENT ON POLICY outbox_relay_drain ON {SCHEMA}.outbox IS
        'The relay must read every tenant to drain the shared outbox. Scoped to this one '
        'table and this one role so the cross-tenant reach is visible in pg_policies rather '
        'than hidden in a role attribute. mytheus_ai_relay has no grant on any other table.';
        """)

    # Tables/sequences created by LATER migrations in THIS schema (tasks 8-10) inherit these
    # grants automatically. Scoped to `intelligence` only — it has no effect on `public` or any
    # other schema, which is exactly the point (see migration docstring, decision 1).
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {SCHEMA} "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE};"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {SCHEMA} "
        f"GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE};"
    )


def downgrade() -> None:
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {SCHEMA} "
        f"REVOKE USAGE, SELECT ON SEQUENCES FROM {APP_ROLE};"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {SCHEMA} "
        f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {APP_ROLE};"
    )
    op.execute(f"DROP POLICY IF EXISTS outbox_relay_drain ON {SCHEMA}.outbox;")

    for role in ALL_ROLES:
        op.execute(f"""
            DO $$ BEGIN
              IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
                REVOKE ALL ON ALL TABLES IN SCHEMA {SCHEMA} FROM {role};
                REVOKE ALL ON ALL SEQUENCES IN SCHEMA {SCHEMA} FROM {role};
                REVOKE ALL ON SCHEMA {SCHEMA} FROM {role};
                DROP ROLE {role};
              END IF;
            END $$;
            """)

    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.inbox;")
    op.execute(f"DROP TABLE IF EXISTS {SCHEMA}.outbox;")
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;")
