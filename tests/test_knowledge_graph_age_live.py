"""Live Apache AGE test for Topic seeding: Plan 5-6 3.2's third acceptance check, for real.

Needs the `age` service from this repo's docker-compose.yml (`docker compose up -d age`,
port 5434). Set `AGE_DATABASE_URL` to override the default DSN. Skips LOUDLY when unreachable;
`MYTHEUS_REQUIRE_AGE=1` turns the skip into a failure (same convention as conftest.py's
`MYTHEUS_REQUIRE_*` flags).
"""

from __future__ import annotations

import os
import socket
import uuid
from urllib.parse import urlparse

import pytest

from services.knowledge_graph_service.connection import age_connection
from services.knowledge_graph_service.graph import count_topics, ensure_graph, merge_topics
from services.knowledge_graph_service.taxonomy import distinct_topics

_DSN = os.environ.get(
    "AGE_DATABASE_URL", "postgresql://nexus_graph:nexus-graph-dev@localhost:5434/nexus_graph"
)


def _reachable() -> bool:
    u = urlparse(_DSN)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 5432), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_age() -> None:
    if _reachable():
        return
    msg = (
        f"LIVE AGE TEST NOT RUN: no Apache AGE Postgres at {_DSN}. The Topic-seeding acceptance "
        "check (Plan 5-6 3.2: MATCH (t:Topic {org_id:$org}) RETURN count(t) == distinct "
        "taxonomy topics) is UNVERIFIED. Start it with `docker compose up -d age`."
    )
    if os.environ.get("MYTHEUS_REQUIRE_AGE") == "1":
        raise RuntimeError(msg)
    pytest.skip(msg)


async def test_seed_is_idempotent_and_count_matches_taxonomy_and_orgs_are_isolated() -> None:
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    names = distinct_topics()
    async with age_connection(_DSN) as conn:
        await ensure_graph(conn)
        await ensure_graph(conn)  # bootstrap is idempotent
        await merge_topics(conn, org_a, names)
        await merge_topics(conn, org_a, names)  # redelivery creates nothing new
        assert await count_topics(conn, org_a) == len(names)
        assert await count_topics(conn, org_b) == 0
        await merge_topics(conn, org_b, names[:3])
        assert await count_topics(conn, org_b) == 3
        assert await count_topics(conn, org_a) == len(names)
