"""`nexus_knowledge` write operations. Everything that mutates the graph lives here, and only
`knowledge-graph-service` imports it.

Built on `libs/shared/age.py`, which enforces `$org_id` binding, parameter-only tenant data and
the frozen label sets. This module adds the domain rules on top:

- `ensure_graph`: idempotent bootstrap of the graph, its ten frozen labels, and a btree index on
  the `org_id` agtype property of every label table (`05-data-architecture.md` section 5.2).
  Run by the worker at startup and by the seed CLI, not by Alembic: the graph lives in the
  dedicated AGE Postgres (see `libs/shared/age.py`), not in the database this repo's Alembic
  history targets.
- `merge_topics`: Plan 5-6 3.2 step 3, `MERGE (t:Topic {name: $name, org_id: $org_id})` once
  per distinct taxonomy topic. AGE has no uniqueness constraints, so two concurrent MERGEs of
  the same node could both create it; a transaction-scoped advisory lock keyed on the org
  serialises writers per tenant, which makes MERGE genuinely idempotent under redelivery.
- `merge_edge`: the edge writer for later bands (confirmed `decisions`/`actions`). Frozen edge
  labels only, and only between the endpoint label pairs canon lists. Not wired to any consumer
  yet: Plan 5-6 says "edges still not written this band".
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from libs.shared.age import (
    EDGE_LABELS,
    GRAPH_NAME,
    NODE_LABELS,
    AgeConnection,
    AgeQueryRejected,
    agtype_to_int,
    require_edge_label,
    require_node_label,
    run_cypher,
)

# The natural key each node label is matched on (canon key properties, `_canon/02-data-model.md`
# section 4). Only these property names may be formatted into a MATCH pattern.
NODE_KEYS: dict[str, str] = {
    "Person": "user_id",
    "Meeting": "meeting_id",
    "Decision": "decision_id",
    "Action": "action_id",
    "Topic": "name",
}

# Allowed (from_label, to_label) pairs per edge label, verbatim from the canon edge table.
EDGE_ENDPOINTS: dict[str, frozenset[tuple[str, str]]] = {
    "ATTENDED": frozenset({("Person", "Meeting")}),
    "PRODUCED": frozenset({("Decision", "Meeting"), ("Action", "Meeting")}),
    "ASSIGNED_TO": frozenset({("Action", "Person")}),
    "RELATES_TO": frozenset({("Decision", "Topic"), ("Action", "Topic"), ("Meeting", "Topic")}),
    "FOLLOWS_FROM": frozenset({("Decision", "Decision")}),
}

_MERGE_TOPIC = "MERGE (t:Topic {name: $name, org_id: $org_id}) RETURN t"
_COUNT_TOPICS = "MATCH (t:Topic {org_id: $org_id}) RETURN count(t)"
_ORG_LOCK = "SELECT pg_advisory_xact_lock(hashtext('nexus_knowledge:org:' || $1::text))"


async def ensure_graph(conn: AgeConnection) -> None:
    """Create the graph, labels and org_id indexes if absent. Safe to run concurrently and
    repeatedly. Requires a role allowed to create objects (dev: the container superuser)."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS age")
    await conn.execute("LOAD 'age'")
    await conn.execute("BEGIN")
    try:
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext('nexus_knowledge:ensure'))")
        rows = await conn.fetch("SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1", GRAPH_NAME)
        if not rows:
            await conn.fetch("SELECT ag_catalog.create_graph($1)", GRAPH_NAME)
        for label in sorted(NODE_LABELS | EDGE_LABELS):
            exists = await conn.fetch(
                "SELECT 1 FROM ag_catalog.ag_label l JOIN ag_catalog.ag_graph g "
                "ON l.graph = g.graphid WHERE g.name = $1 AND l.name = $2",
                GRAPH_NAME,
                label,
            )
            if not exists:
                fn = "create_vlabel" if label in NODE_LABELS else "create_elabel"
                await conn.fetch(f"SELECT ag_catalog.{fn}($1, $2)", GRAPH_NAME, label)
            # DDL identifiers cannot be bound as parameters; `label` comes only from the frozen
            # constant sets above, never from input.
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{label.lower()}_org_idx" '
                f'ON {GRAPH_NAME}."{label}" USING btree '
                "(ag_catalog.agtype_access_operator(properties, '\"org_id\"'::ag_catalog.agtype))"
            )
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise


def _clean_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise AgeQueryRejected(f"invalid Topic name {name!r}")
        seen.setdefault(name.strip(), None)
    if not seen:
        raise AgeQueryRejected("refusing to seed an empty topic set")
    return list(seen)


async def merge_topics(conn: AgeConnection, org_id: UUID, names: Sequence[str]) -> int:
    """Upsert one `Topic` node per distinct name for `org_id`, in one transaction. Returns the
    number of distinct names merged. Idempotent: re-running creates nothing new."""
    clean = _clean_names(names)
    await conn.execute("BEGIN")
    try:
        await conn.execute(_ORG_LOCK, str(org_id))
        for name in clean:
            await run_cypher(conn, org_id, _MERGE_TOPIC, {"name": name}, columns=("t",))
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise
    return len(clean)


async def count_topics(conn: AgeConnection, org_id: UUID) -> int:
    """Plan 5-6 3.2's acceptance query, verbatim: `MATCH (t:Topic {org_id:$org}) RETURN
    count(t)`."""
    rows = await run_cypher(conn, org_id, _COUNT_TOPICS, columns=("n",))
    return agtype_to_int(rows[0][0]) if rows else 0


async def merge_edge(
    conn: AgeConnection,
    org_id: UUID,
    *,
    label: str,
    from_label: str,
    from_id: str,
    to_label: str,
    to_id: str,
) -> None:
    """Upsert one edge between two EXISTING same-org nodes. Both endpoints are matched with
    `org_id: $org_id`, so an edge can never join two tenants' subgraphs."""
    require_edge_label(label)
    require_node_label(from_label)
    require_node_label(to_label)
    if (from_label, to_label) not in EDGE_ENDPOINTS[label]:
        raise AgeQueryRejected(f"{label} is not defined from {from_label} to {to_label}")
    fk, tk = NODE_KEYS[from_label], NODE_KEYS[to_label]
    cypher = (
        f"MATCH (a:{from_label} {{org_id: $org_id, {fk}: $from_id}}), "
        f"(b:{to_label} {{org_id: $org_id, {tk}: $to_id}}) "
        f"MERGE (a)-[e:{label} {{org_id: $org_id}}]->(b) RETURN e"
    )
    await conn.execute("BEGIN")
    try:
        await conn.execute(_ORG_LOCK, str(org_id))
        rows = await run_cypher(
            conn, org_id, cypher, {"from_id": from_id, "to_id": to_id}, columns=("e",)
        )
        if not rows:
            raise AgeQueryRejected(f"{label}: one or both endpoints do not exist in this org")
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise
