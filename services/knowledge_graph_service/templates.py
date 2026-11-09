"""The four named, read-only `nexus_knowledge` templates (ADR-0005 D2/D3) and the guarded ad-hoc
Cypher path. Backs `POST /graph/query/{template}`, `POST /graph/cypher` and the gRPC
`KnowledgeGraphService`.

Every template:
- binds `$org_id` on every node AND every relationship pattern (edges carry `org_id` too), so a
  traversal cannot step outside one tenant's subgraph even through a mis-written edge;
- is bounded: variable-length paths stop at depth 10, result sets at `MAX_LIMIT`;
- filters soft-deleted nodes (`deleted_at IS NULL`);
- runs in a READ ONLY transaction with a statement timeout, so even a template bug cannot write
  or run away with the agent latency budget (p50 < 2 s).

Templates return `GraphNode`s, one uniform shape for all four, so the HTTP and gRPC surfaces and
their callers (`bff`, `rag-service`, `ProactiveAgent`) need no per-template decoding.

`actions_blocked_on_decision`: canon has no Action-to-Decision edge, so "blocked on" is read as
"still-open actions PRODUCED by the meeting that PRODUCED the decision". A true blocking
relation needs a source field (`action_items.blocked_on_decision_id`, open in the worksplit);
until then this is the closest answer the frozen labels can give, and it is recorded as a
deviation in the Plan 7-9 band file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from libs.shared.age import AgeConnection, AgeQueryRejected, build_cypher_sql, encode_params

MAX_LIMIT = 100
DEFAULT_LIMIT = 50
STATEMENT_TIMEOUT_MS = 2000
TEMPLATE_NAMES = (
    "decision_lineage",
    "decisions_by_person",
    "actions_blocked_on_decision",
    "related_decisions_for_topic",
)

_ID_PROPS = {
    "Person": "user_id",
    "Meeting": "meeting_id",
    "Decision": "decision_id",
    "Action": "action_id",
    "Topic": "name",
}


class TemplateNotFound(LookupError):
    pass


@dataclass(frozen=True)
class GraphNode:
    label: str
    id: str
    role: str  # "subject" | "ancestor" | "descendant" | "match"
    depth: int = 0
    properties: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "id": self.id,
            "role": self.role,
            "depth": self.depth,
            "properties": self.properties,
        }


def parse_agtype(value: Any) -> Any:
    """Decode an agtype text value (`{...}::vertex`, `"x"`, `3`) to Python."""
    if value is None:
        return None
    text = str(value).strip()
    for suffix in ("::vertex", "::edge", "::path"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return json.loads(text)


def _node(raw: Any, role: str, depth: int = 0) -> GraphNode:
    v = parse_agtype(raw)
    label = v["label"]
    props = dict(v["properties"])
    return GraphNode(
        label=label, id=str(props.get(_ID_PROPS[label], "")), role=role, depth=depth,
        properties=props,
    )  # fmt: skip


def _uuid_param(params: Mapping[str, Any], name: str) -> str:
    raw = params.get(name)
    try:
        return str(UUID(str(raw)))
    except (TypeError, ValueError):
        raise AgeQueryRejected(f"{name} must be a UUID") from None


def _limit(params: Mapping[str, Any]) -> int:
    raw = params.get("limit", DEFAULT_LIMIT)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise AgeQueryRejected("limit must be an integer") from None
    if not 1 <= n <= MAX_LIMIT:
        raise AgeQueryRejected(f"limit must be 1..{MAX_LIMIT}")
    return n


async def _read(
    conn: AgeConnection,
    org_id: UUID,
    cypher: str,
    params: Mapping[str, Any],
    columns: Sequence[str],
) -> list[Any]:
    return await conn.fetch(build_cypher_sql(cypher, columns), encode_params(org_id, params))


async def _read_only(conn: AgeConnection) -> None:
    await conn.execute("BEGIN READ ONLY")
    await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")


_LINEAGE_SUBJECT = (
    "MATCH (d:Decision {org_id: $org_id, decision_id: $decision_id}) "
    "WHERE d.deleted_at IS NULL RETURN d"
)
_LINEAGE_ANCESTORS = (
    "MATCH p = (d:Decision {org_id: $org_id, decision_id: $decision_id})"
    "-[:FOLLOWS_FROM*1..10 {org_id: $org_id}]->(x:Decision {org_id: $org_id}) "
    "WHERE x.deleted_at IS NULL RETURN x, length(p)"
)
_LINEAGE_DESCENDANTS = (
    "MATCH p = (x:Decision {org_id: $org_id})-[:FOLLOWS_FROM*1..10 {org_id: $org_id}]->"
    "(d:Decision {org_id: $org_id, decision_id: $decision_id}) "
    "WHERE x.deleted_at IS NULL RETURN x, length(p)"
)


async def decision_lineage(
    conn: AgeConnection, org_id: UUID, params: Mapping[str, Any]
) -> list[GraphNode]:
    """The FOLLOWS_FROM chain around one decision, oldest first: ancestors (what it superseded,
    deepest first), the decision itself, then descendants (what superseded it). Empty when the
    decision is not in this org's graph."""
    p = {"decision_id": _uuid_param(params, "decision_id")}
    subject = await _read(conn, org_id, _LINEAGE_SUBJECT, p, ("d",))
    if not subject:
        return []

    def nearest(rows: list[Any], role: str) -> list[GraphNode]:
        best: dict[str, GraphNode] = {}
        for row in rows:
            n = _node(row[0], role, int(parse_agtype(row[1])))
            if n.id not in best or n.depth < best[n.id].depth:
                best[n.id] = n
        return list(best.values())

    ancestors = nearest(await _read(conn, org_id, _LINEAGE_ANCESTORS, p, ("x", "n")), "ancestor")
    descendants = nearest(
        await _read(conn, org_id, _LINEAGE_DESCENDANTS, p, ("x", "n")), "descendant"
    )
    return (
        sorted(ancestors, key=lambda n: -n.depth)
        + [_node(subject[0][0], "subject")]
        + sorted(descendants, key=lambda n: n.depth)
    )


async def decisions_by_person(
    conn: AgeConnection, org_id: UUID, params: Mapping[str, Any]
) -> list[GraphNode]:
    """Decisions PRODUCED by meetings the person ATTENDED, newest confirmation first."""
    p = {"user_id": _uuid_param(params, "user_id")}
    limit = _limit(params)  # a validated int: the only value ever formatted into Cypher text
    cypher = (
        "MATCH (:Person {org_id: $org_id, user_id: $user_id})-[:ATTENDED {org_id: $org_id}]->"
        "(m:Meeting {org_id: $org_id})<-[:PRODUCED {org_id: $org_id}]-"
        "(d:Decision {org_id: $org_id}) WHERE d.deleted_at IS NULL "
        f"WITH DISTINCT d RETURN d ORDER BY d.confirmed_at DESC LIMIT {limit}"
    )
    return [_node(r[0], "match") for r in await _read(conn, org_id, cypher, p, ("d",))]


async def actions_blocked_on_decision(
    conn: AgeConnection, org_id: UUID, params: Mapping[str, Any]
) -> list[GraphNode]:
    """Still-open actions from the meeting that produced the decision (see module docstring)."""
    p = {"decision_id": _uuid_param(params, "decision_id")}
    limit = _limit(params)
    cypher = (
        "MATCH (:Decision {org_id: $org_id, decision_id: $decision_id})"
        "-[:PRODUCED {org_id: $org_id}]->(m:Meeting {org_id: $org_id})"
        "<-[:PRODUCED {org_id: $org_id}]-(a:Action {org_id: $org_id}) "
        "WHERE a.deleted_at IS NULL AND a.status = 'confirmed' "
        f"WITH DISTINCT a RETURN a ORDER BY a.due_date LIMIT {limit}"
    )
    return [_node(r[0], "match") for r in await _read(conn, org_id, cypher, p, ("a",))]


async def related_decisions_for_topic(
    conn: AgeConnection, org_id: UUID, params: Mapping[str, Any]
) -> list[GraphNode]:
    """Decisions RELATES_TO a taxonomy Topic. Empty until topic extraction writes RELATES_TO."""
    topic = params.get("topic")
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 200:
        raise AgeQueryRejected("topic must be a non-empty string of at most 200 characters")
    limit = _limit(params)
    cypher = (
        "MATCH (:Topic {org_id: $org_id, name: $topic})<-[:RELATES_TO {org_id: $org_id}]-"
        "(d:Decision {org_id: $org_id}) WHERE d.deleted_at IS NULL "
        f"WITH DISTINCT d RETURN d ORDER BY d.confirmed_at DESC LIMIT {limit}"
    )
    rows = await _read(conn, org_id, cypher, {"topic": topic.strip()}, ("d",))
    return [_node(r[0], "match") for r in rows]


_TEMPLATES = {
    "decision_lineage": decision_lineage,
    "decisions_by_person": decisions_by_person,
    "actions_blocked_on_decision": actions_blocked_on_decision,
    "related_decisions_for_topic": related_decisions_for_topic,
}


async def run_template(
    conn: AgeConnection, org_id: UUID, name: str, params: Mapping[str, Any]
) -> list[GraphNode]:
    fn = _TEMPLATES.get(name)
    if fn is None:
        raise TemplateNotFound(name)
    await _read_only(conn)
    try:
        return await fn(conn, org_id, params)
    finally:
        await conn.execute("ROLLBACK")  # read only: nothing to keep


_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|LOAD|CALL|DROP)\b", re.IGNORECASE
)


async def run_adhoc(
    conn: AgeConnection,
    org_id: UUID,
    cypher: str,
    params: Mapping[str, Any] | None,
    columns: Sequence[str],
) -> list[list[Any]]:
    """`POST /graph/cypher`: internal-only, read-only, fail-closed. Refused unless the text binds
    `$org_id` (checked by `build_cypher_sql`); refused on any write clause; and run inside a READ
    ONLY transaction anyway, so a clause the regex misses still cannot write."""
    if _WRITE_CLAUSE.search(cypher):
        raise AgeQueryRejected("ad-hoc Cypher is read-only")
    sql = build_cypher_sql(cypher, columns)
    await _read_only(conn)
    try:
        rows = await conn.fetch(sql, encode_params(org_id, params))
    finally:
        await conn.execute("ROLLBACK")
    return [[parse_agtype(v) for v in row] for row in rows][:MAX_LIMIT]
