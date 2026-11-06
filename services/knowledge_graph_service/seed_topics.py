"""Operator CLI: seed and/or verify an org's `Topic` nodes directly (backfill, smoke checks).

    python -m services.knowledge_graph_service.seed_topics --org-id <uuid> [--org-id ...]
    python -m services.knowledge_graph_service.seed_topics --org-id <uuid> --verify-only

Still `knowledge-graph-service` writing its own store, so the sole-writer rule holds. Exits
non-zero if, for any org, `MATCH (t:Topic {org_id:$org}) RETURN count(t)` differs from the
number of distinct taxonomy topics (Plan 5-6 3.2's third acceptance check).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from uuid import UUID

from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.graph import count_topics, ensure_graph, merge_topics
from services.knowledge_graph_service.taxonomy import distinct_topics


async def run(org_ids: list[UUID], *, verify_only: bool) -> int:
    expected = len(distinct_topics())
    failures = 0
    async with age_connection() as conn:
        if not verify_only:
            await ensure_graph(conn)
        for org_id in org_ids:
            if not verify_only:
                await merge_topics(conn, org_id, distinct_topics())
            got = await count_topics(conn, org_id)
            status = "OK" if got == expected else "MISMATCH"
            failures += got != expected
            print(f"{status} org={org_id} topics={got} expected={expected}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", action="append", type=UUID, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.org_id, verify_only=args.verify_only)))


if __name__ == "__main__":
    main()
