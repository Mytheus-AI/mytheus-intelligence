"""Operator CLI: rebuild one tenant's decision/action graph from confirmed rows.

    python -m services.knowledge_graph_service.rebuild --org-id <uuid> --rows rows.jsonl [--purge]

The graph is a projection of confirmed `decisions` and `action_items` rows (ADR-0005, ADR-0031
D1), so it must be rebuildable from them. Each input line is one confirmed row as JSON, in the
same shape the topics carry, plus `kind`:

    {"kind": "decision", "decision_id": ..., "meeting_id": ..., "content": ...,
     "confidence": 0.9, "confirmed_at": "2026-11-09T10:00:00+00:00", "supersedes": null}
    {"kind": "action", "action_id": ..., "meeting_id": ..., "content": ..., "confidence": 0.8,
     "confirmed_at": ..., "assignee_id": ..., "due_date": "2026-11-20"}

Only `status = confirmed` rows (and decisions later `superseded`) belong in the file; a
`proposed` or `dismissed` row is refused. The rows come from `core-api`'s tables, which this tier
does not read: export them there (`COPY (...) TO STDOUT`) and pass the file here. Rows apply
through the same functions the consumer uses, so order does not matter and a re-run is a no-op.
`--purge` first removes the tenant's whole subgraph (Topics included; re-seed with
`seed_topics.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from services.knowledge_graph_service.confirmed import (
    ConfirmedAction,
    ConfirmedDecision,
    apply_action,
    apply_decision,
    purge_org,
)
from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.graph import ensure_graph

_APPLIABLE = {"confirmed", "superseded", "done", None}


def _opt_uuid(value: Any) -> UUID | None:
    return UUID(str(value)) if value else None


def parse_row(row: dict[str, Any]) -> ConfirmedDecision | ConfirmedAction:
    if row.get("status") not in _APPLIABLE:
        raise ValueError(f"refusing a {row.get('status')!r} row: only confirmed facts build edges")
    if row.get("kind") == "decision":
        return ConfirmedDecision(
            decision_id=UUID(str(row["decision_id"])),
            meeting_id=UUID(str(row["meeting_id"])),
            content=str(row["content"]),
            confidence=float(row["confidence"]),
            confirmed_at=str(row["confirmed_at"]),
            supersedes=_opt_uuid(row.get("supersedes")),
        )
    if row.get("kind") == "action":
        return ConfirmedAction(
            action_id=UUID(str(row["action_id"])),
            meeting_id=UUID(str(row["meeting_id"])),
            content=str(row["content"]),
            confidence=float(row["confidence"]),
            confirmed_at=str(row["confirmed_at"]),
            assignee_id=_opt_uuid(row.get("assignee_id")),
            due_date=str(row["due_date"]) if row.get("due_date") else None,
        )
    raise ValueError(f"unknown kind {row.get('kind')!r}")


async def run(org_id: UUID, rows_path: Path, *, purge: bool) -> int:
    facts = [
        parse_row(json.loads(line))
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    async with age_connection() as conn:
        await ensure_graph(conn)
        if purge:
            await purge_org(conn, org_id)
        for fact in facts:
            if isinstance(fact, ConfirmedDecision):
                await apply_decision(conn, org_id, fact)
            else:
                await apply_action(conn, org_id, fact)
    print(f"OK org={org_id} applied={len(facts)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", type=UUID, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.org_id, args.rows, purge=args.purge)))


if __name__ == "__main__":
    main()
