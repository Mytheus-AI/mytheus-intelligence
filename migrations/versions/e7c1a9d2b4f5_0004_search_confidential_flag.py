"""0004 search: a confidential flag on the FTS read model

Polish plan, Phase 0 ("never record everywhere"): a chunk of a confidential meeting
(`meetings.confidential`, ADR-0039 B2) is still indexed, so its own participants can ask about it
through the participant-scoped RAG path (ADR-0033 D1, the `meeting_ids` allowlist), but every
org-wide query (`GET /search`, an unscoped FTS call) excludes it. The consumer sets the flag from
core-api's `GET /internal/meetings/{id}/index-policy` (`libs/shared/index_guard.py`).

Existing rows default to false: they were indexed before the flag existed, exactly as before.
No new grant: `mytheus_ai_app` already holds SELECT/INSERT/UPDATE/DELETE on the schema's tables
(migration `9c2b7e4a1d03`'s default privileges); DELETE is what the never-record purge uses.

Revision ID: e7c1a9d2b4f5
Revises: d415f023220e
Create Date: 2026-11-11
"""

from __future__ import annotations

from alembic import op

revision: str = "e7c1a9d2b4f5"
down_revision: str | None = "d415f023220e"
branch_labels = None
depends_on = None

SCHEMA = "intelligence"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE {SCHEMA}.transcript_chunk_search "
        "ADD COLUMN confidential boolean NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.transcript_chunk_search DROP COLUMN confidential")
