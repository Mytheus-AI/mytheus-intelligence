"""Graph writes driven by confirmed facts: the Plan 8 edge-on-confirmation writer.

Every function here is called only from `facts_consumer.py` (confirmed `decisions` / `actions`
from `core-api`, and `meeting-room-events` joins) or from `rebuild.py` (the same facts read back
from confirmed rows). Nothing here can be reached from an agent candidate: the `Decision` and
`Action` Avro `status` enums are `{confirmed}`, so an unconfirmed claim cannot even be encoded
on those topics.

Shape of every write:
- One transaction, holding the per-org advisory lock `graph.py` already uses, so concurrent
  MERGEs of the same node cannot both create it (AGE has no uniqueness constraints).
- Every node and edge is MERGEd with `org_id: $org_id` in its pattern, and `$org_id` is injected
  by `libs/shared/age.py` from a typed UUID. An edge can only ever join two same-org nodes.
- MERGE, never CREATE: re-applying a fact (redelivery past the inbox, or a rebuild) is a no-op.
- Order-independent. A decision that supersedes one the graph has not seen yet MERGEs a stub for
  the prior node; the prior's own event fills it in later and keeps its `superseded` status.

Each function returns the `GraphFact`s it wrote, which the consumer turns into `graph-events`.

A personal task (created outside any meeting) arrives on `actions` with the nil uuid as its
`meeting_id`. It means "no meeting": the `Action` node and its `ASSIGNED_TO` edge are written,
no `Meeting` node and no `PRODUCED` edge (`ConfirmedAction.meeting_id` is None).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from libs.shared.age import AgeConnection, AgeQueryRejected, run_cypher
from services.knowledge_graph_service.graph import _ORG_LOCK, NODE_KEYS

GraphEventType = Literal["edge_created", "lineage_updated"]
NIL_MEETING = UUID(int=0)  # "no meeting": a personal task created outside any meeting


def meeting_or_none(meeting_id: UUID | None) -> UUID | None:
    return None if meeting_id is None or meeting_id == NIL_MEETING else meeting_id


@dataclass(frozen=True)
class GraphFact:
    """One edge write, as `graph-events` reports it (`GraphEvent`: event_type, edge_label,
    from_id, to_id)."""

    event_type: GraphEventType
    edge_label: str
    from_id: str
    to_id: str


@dataclass(frozen=True)
class ConfirmedDecision:
    decision_id: UUID
    meeting_id: UUID
    content: str
    confidence: float
    confirmed_at: str  # ISO-8601, from the event header's occurred_at
    supersedes: UUID | None = None


@dataclass(frozen=True)
class ConfirmedAction:
    action_id: UUID
    meeting_id: UUID | None  # None: a personal task, no meeting produced it
    content: str
    confidence: float
    confirmed_at: str
    assignee_id: UUID | None = None
    due_date: str | None = None  # ISO date


# The confirmed decision. A node already marked `superseded` (a later decision replaced it and
# was applied first) keeps that status: confirmation is not allowed to un-supersede it.
_DECISION = (
    "MERGE (m:Meeting {org_id: $org_id, meeting_id: $meeting_id}) "
    "MERGE (d:Decision {org_id: $org_id, decision_id: $decision_id}) "
    "SET d.status = CASE WHEN d.status = 'superseded' THEN 'superseded' ELSE 'confirmed' END, "
    "d.confidence = $confidence, d.content = $content, d.meeting_id = $meeting_id, "
    "d.confirmed_at = $confirmed_at "
    "MERGE (d)-[:PRODUCED {org_id: $org_id}]->(m) "
    "RETURN d"
)

# Lineage: the new decision FOLLOWS_FROM the one it replaces, which becomes `superseded`.
_SUPERSEDES = (
    "MATCH (d:Decision {org_id: $org_id, decision_id: $decision_id}) "
    "MERGE (p:Decision {org_id: $org_id, decision_id: $supersedes}) "
    "SET p.status = 'superseded' "
    "MERGE (d)-[:FOLLOWS_FROM {org_id: $org_id}]->(p) "
    "RETURN p"
)

_ACTION = (
    "MERGE (m:Meeting {org_id: $org_id, meeting_id: $meeting_id}) "
    "MERGE (a:Action {org_id: $org_id, action_id: $action_id}) "
    "SET a.status = 'confirmed', a.confidence = $confidence, a.content = $content, "
    "a.meeting_id = $meeting_id, a.due_date = $due_date, a.confirmed_at = $confirmed_at "
    "MERGE (a)-[:PRODUCED {org_id: $org_id}]->(m) "
    "RETURN a"
)

# A personal task: the same Action node, no Meeting and no PRODUCED edge.
_ACTION_NO_MEETING = (
    "MERGE (a:Action {org_id: $org_id, action_id: $action_id}) "
    "SET a.status = 'confirmed', a.confidence = $confidence, a.content = $content, "
    "a.due_date = $due_date, a.confirmed_at = $confirmed_at "
    "RETURN a"
)

_ASSIGNED_TO = (
    "MATCH (a:Action {org_id: $org_id, action_id: $action_id}) "
    "MERGE (p:Person {org_id: $org_id, user_id: $assignee_id}) "
    "MERGE (a)-[:ASSIGNED_TO {org_id: $org_id}]->(p) "
    "RETURN p"
)

_ATTENDED = (
    "MERGE (p:Person {org_id: $org_id, user_id: $user_id}) "
    "MERGE (m:Meeting {org_id: $org_id, meeting_id: $meeting_id}) "
    "MERGE (p)-[:ATTENDED {org_id: $org_id}]->(m) "
    "RETURN m"
)

_PURGE_ORG = "MATCH (n {org_id: $org_id}) DETACH DELETE n RETURN count(*)"


@asynccontextmanager
async def org_transaction(conn: AgeConnection, org_id: UUID) -> AsyncIterator[None]:
    """One graph transaction serialised per tenant."""
    await conn.execute("BEGIN")
    try:
        await conn.execute(_ORG_LOCK, str(org_id))
        yield
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise


async def _write(conn: AgeConnection, org_id: UUID, cypher: str, params: dict[str, Any]) -> None:
    rows = await run_cypher(conn, org_id, cypher, params, columns=("n",))
    if not rows:
        raise AgeQueryRejected("graph write matched nothing; refusing a silent no-op")


async def apply_decision(
    conn: AgeConnection, org_id: UUID, decision: ConfirmedDecision
) -> list[GraphFact]:
    did, mid = str(decision.decision_id), str(decision.meeting_id)
    facts = [GraphFact("edge_created", "PRODUCED", did, mid)]
    async with org_transaction(conn, org_id):
        await _write(
            conn,
            org_id,
            _DECISION,
            {
                "decision_id": did,
                "meeting_id": mid,
                "content": decision.content,
                "confidence": float(decision.confidence),
                "confirmed_at": decision.confirmed_at,
            },
        )
        if decision.supersedes is not None and decision.supersedes != decision.decision_id:
            prior = str(decision.supersedes)
            await _write(conn, org_id, _SUPERSEDES, {"decision_id": did, "supersedes": prior})
            facts.append(GraphFact("lineage_updated", "FOLLOWS_FROM", did, prior))
    return facts


async def apply_action(
    conn: AgeConnection, org_id: UUID, action: ConfirmedAction
) -> list[GraphFact]:
    aid = str(action.action_id)
    meeting = meeting_or_none(action.meeting_id)
    params: dict[str, Any] = {
        "action_id": aid,
        "content": action.content,
        "confidence": float(action.confidence),
        "due_date": action.due_date,
        "confirmed_at": action.confirmed_at,
    }
    facts: list[GraphFact] = []
    if meeting is not None:
        params["meeting_id"] = str(meeting)
        facts.append(GraphFact("edge_created", "PRODUCED", aid, str(meeting)))
    async with org_transaction(conn, org_id):
        await _write(conn, org_id, _ACTION if meeting is not None else _ACTION_NO_MEETING, params)
        if action.assignee_id is not None:
            uid = str(action.assignee_id)
            await _write(conn, org_id, _ASSIGNED_TO, {"action_id": aid, "assignee_id": uid})
            facts.append(GraphFact("edge_created", "ASSIGNED_TO", aid, uid))
    return facts


async def apply_attended(
    conn: AgeConnection, org_id: UUID, *, user_id: UUID, meeting_id: UUID
) -> list[GraphFact]:
    uid, mid = str(user_id), str(meeting_id)
    async with org_transaction(conn, org_id):
        await _write(conn, org_id, _ATTENDED, {"user_id": uid, "meeting_id": mid})
    return [GraphFact("edge_created", "ATTENDED", uid, mid)]


async def soft_delete(
    conn: AgeConnection, org_id: UUID, *, label: str, node_id: str, deleted_at: str
) -> bool:
    """Tag one node `deleted_at`; every read template filters it out. Returns whether the node
    existed in this org."""
    if label not in NODE_KEYS:
        raise AgeQueryRejected(f"{label!r} is not a frozen nexus_knowledge node label")
    key = NODE_KEYS[label]
    cypher = (
        f"MATCH (n:{label} {{org_id: $org_id, {key}: $node_id}}) "
        "SET n.deleted_at = $deleted_at RETURN n"
    )
    async with org_transaction(conn, org_id):
        rows = await run_cypher(
            conn, org_id, cypher, {"node_id": node_id, "deleted_at": deleted_at}, columns=("n",)
        )
    return bool(rows)


async def purge_org(conn: AgeConnection, org_id: UUID) -> None:
    """The RTBF step: remove every node (and with it every edge) of one tenant."""
    async with org_transaction(conn, org_id):
        await run_cypher(conn, org_id, _PURGE_ORG, columns=("n",))
