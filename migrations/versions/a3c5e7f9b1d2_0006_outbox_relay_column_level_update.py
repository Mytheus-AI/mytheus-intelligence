"""0006 the outbox relay may UPDATE only the drain columns

Journal plan Part 3 fix 4 (outbox), the intelligence twin of mytheus-backend `c9f1d3e5a7b2`.
0001 granted `mytheus_ai_relay` table-level `SELECT, UPDATE ON intelligence.outbox` next to the
permissive `outbox_relay_drain` policy, so the relay could rewrite any pending event's
`org_id`, `topic` or `payload` (re-key it into another tenant, or forge it) before publishing.
`libs/shared/relay.py` writes only `published_at` and `attempts`; the grant becomes a column
grant on exactly those two. SELECT stays table-level; `FOR UPDATE SKIP LOCKED` needs UPDATE on
at least one column, which it still has.

Revision ID: a3c5e7f9b1d2
Revises: f2a6c8e1b3d5
Create Date: 2026-11-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a3c5e7f9b1d2"
down_revision: str | None = "f2a6c8e1b3d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "intelligence"
RELAY_ROLE = "mytheus_ai_relay"
DRAIN_COLUMNS = "published_at, attempts"


def upgrade() -> None:
    op.execute(f"REVOKE UPDATE ON {SCHEMA}.outbox FROM {RELAY_ROLE};")
    op.execute(f"GRANT UPDATE ({DRAIN_COLUMNS}) ON {SCHEMA}.outbox TO {RELAY_ROLE};")


def downgrade() -> None:
    op.execute(f"REVOKE UPDATE ({DRAIN_COLUMNS}) ON {SCHEMA}.outbox FROM {RELAY_ROLE};")
    op.execute(f"GRANT UPDATE ON {SCHEMA}.outbox TO {RELAY_ROLE};")
