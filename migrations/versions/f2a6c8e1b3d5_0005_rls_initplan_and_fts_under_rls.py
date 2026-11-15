"""0005 every tenant policy in InitPlan form, and transcript FTS that survives RLS

Journal plan Part 3 fix 1; the benchmark and plans are in mytheus-backend
`research/rls_fts_bench/` and the research metrics folder.

1. Every policy in the `intelligence` schema that reads `app.current_org` is rewritten from
   `org_id = NULLIF(current_setting('app.current_org', true), '')::uuid` to the InitPlan form
   `org_id = (SELECT NULLIF(current_setting('app.current_org', true), '')::uuid)`. Same NULLIF,
   same fail-closed meaning (unset or empty => NULL => no row), evaluated once per statement
   instead of per row. ALTER POLICY changes only the expressions: command, roles, kind,
   ENABLE/FORCE and the presence of WITH CHECK are untouched. `public` tables belong to
   mytheus-backend and mytheus-realtime, whose own migrations rewrite theirs.

2. `search-service`'s query (`services/search_service/fts.py`) tests `tsv @@ q` on
   `transcript_chunk_search`. Under RLS a user qual becomes an index condition only if it is
   LEAKPROOF, and core does not mark `ts_match_vq` leakproof, so the GIN index on `tsv` was
   never used for any tenant: every search read the tenant's whole slice (262 ms for a
   200k-row tenant against 14 ms without RLS, in the benchmark). The fix:
   - `ts_match_vq` / `ts_match_qv` are marked LEAKPROOF when this runs as a superuser (it does
     in production: this history migrates as `supabase_admin`). Their errors depend only on the
     query constant, never on a row's contents. Elsewhere mytheus-platform's
     `deploy/postgres-init/harden-roles.sql` applies the same idempotent statement.
   - The plain GIN index on `tsv` is replaced by a composite `(org_id, tsv)` GIN (btree_gin),
     so the index itself intersects "this tenant" with "matches", rather than matching every
     tenant's rows and filtering. Every caller is tenant-scoped, so nothing used the plain one
     in a way the composite does not serve.

Revision ID: f2a6c8e1b3d5
Revises: e7c1a9d2b4f5
Create Date: 2026-11-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f2a6c8e1b3d5"
down_revision: str | None = "e7c1a9d2b4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "intelligence"
_PER_ROW = "(NULLIF(current_setting('app.current_org'::text, true), ''::text))::uuid"
_INITPLAN = f"( SELECT {_PER_ROW})"
_INITPLAN_DEPARSED = f'( SELECT {_PER_ROW} AS "nullif")'


def _rewrite_policies(old: str, new: str, *, skip_if_contains: str) -> None:
    op.execute(f"""
        DO $$
        DECLARE
          p record;
          q text;
          c text;
          clause text;
        BEGIN
          FOR p IN
            SELECT pol.polname, n.nspname, cl.relname,
                   pg_get_expr(pol.polqual, pol.polrelid) AS qual,
                   pg_get_expr(pol.polwithcheck, pol.polrelid) AS chk
            FROM pg_policy pol
            JOIN pg_class cl ON cl.oid = pol.polrelid
            JOIN pg_namespace n ON n.oid = cl.relnamespace
            WHERE n.nspname = '{SCHEMA}'
              AND coalesce(pg_get_expr(pol.polqual, pol.polrelid), '')
                  || coalesce(pg_get_expr(pol.polwithcheck, pol.polrelid), '')
                  LIKE '%app.current_org%'
          LOOP
            q := p.qual;
            c := p.chk;
            IF q IS NOT NULL AND position($s${skip_if_contains}$s$ IN q) = 0 THEN
              q := replace(q, $o${old}$o$, $n${new}$n$);
            END IF;
            IF c IS NOT NULL AND position($s${skip_if_contains}$s$ IN c) = 0 THEN
              c := replace(c, $o${old}$o$, $n${new}$n$);
            END IF;
            clause := '';
            IF q IS DISTINCT FROM p.qual THEN
              clause := clause || ' USING (' || q || ')';
            END IF;
            IF c IS DISTINCT FROM p.chk THEN
              clause := clause || ' WITH CHECK (' || c || ')';
            END IF;
            IF clause <> '' THEN
              EXECUTE format('ALTER POLICY %I ON %I.%I', p.polname, p.nspname, p.relname)
                      || clause;
            END IF;
          END LOOP;
        END $$;
        """)


def upgrade() -> None:
    _rewrite_policies(_PER_ROW, _INITPLAN, skip_if_contains="SELECT")
    op.execute("""
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'extensions') THEN
            CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA extensions;
          ELSE
            CREATE EXTENSION IF NOT EXISTS btree_gin;
          END IF;
        END $$;
        """)
    op.execute(
        f"CREATE INDEX ix_intelligence_transcript_chunk_search_org_tsv "
        f"ON {SCHEMA}.transcript_chunk_search USING gin (org_id, tsv);"
    )
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_intelligence_transcript_chunk_search_tsv;")
    op.execute("""
        DO $$
        BEGIN
          IF (SELECT bool_and(proleakproof) FROM pg_proc
              WHERE oid IN ('ts_match_vq(tsvector,tsquery)'::regprocedure,
                            'ts_match_qv(tsquery,tsvector)'::regprocedure)) THEN
            RETURN;
          END IF;
          IF (SELECT rolsuper FROM pg_roles WHERE rolname = current_user) THEN
            ALTER FUNCTION ts_match_vq(tsvector, tsquery) LEAKPROOF;
            ALTER FUNCTION ts_match_qv(tsquery, tsvector) LEAKPROOF;
          ELSE
            RAISE WARNING 'ts_match_vq is not LEAKPROOF and % is not a superuser: FTS under '
              'RLS cannot use its GIN index until mytheus-platform '
              'deploy/postgres-init/harden-roles.sql runs as a superuser', current_user;
          END IF;
        END $$;
        """)


def downgrade() -> None:
    # LEAKPROOF and btree_gin are database-wide and shared with mytheus-backend; left in place.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_intelligence_transcript_chunk_search_tsv "
        f"ON {SCHEMA}.transcript_chunk_search USING gin (tsv);"
    )
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_intelligence_transcript_chunk_search_org_tsv;")
    _rewrite_policies(_INITPLAN_DEPARSED, _PER_ROW, skip_if_contains="#never-present#")
