"""Finishing an admin data erasure in `nexus_knowledge` (store `knowledge_graph`):
`libs/shared/erasure.py` has the shared rules; this is knowledge-graph-service's share, as the
only writer of the graph.

`GraphErasureConsumer` (group `knowledge-graph-service.erasure`, `admin-events`):

- MEETING: the `Meeting` node is DETACH DELETEd, which removes every edge it has (`ATTENDED`
  from people, `PRODUCED` from decisions and actions, `RELATES_TO` to topics). The graph only
  ever holds CONFIRMED decisions and actions, which core-api keeps: those nodes stay with their
  content, status, lineage (`FOLLOWS_FROM`) and assignee, but lose their `meeting_id` and gain
  `meeting_erased: true`, so nothing in the graph points at the erased meeting any more.
- PERSON: the `Person` node is DETACH DELETEd, which removes their `ATTENDED` and
  `ASSIGNED_TO` edges. Actions they were assigned stay, unassigned, exactly as core-api sets
  `action_items.assignee_id` to NULL; decisions never referenced a person in the graph.

Every statement binds `$org_id` (`libs/shared/age.py` refuses one that does not), and all of one
erasure runs in one graph transaction under the per-org advisory lock the other writers hold,
so a concurrent MERGE cannot interleave. Atomicity across databases is as in
`facts_consumer.py`: the graph commits first, then the inbox row and the completion outbox row;
a crash in between redelivers the event and the erasure re-runs, matching nothing.

`rebuild.py` builds the graph from exported core-api rows; core-api keeps confirmed decisions
of an erased meeting with their `meeting_id`, so an operator rebuilding a tenant must re-run
the erasures afterwards (or export with the erased meetings' ids nulled), or the rebuilt graph
gets the `Meeting` node back.

`ErasureAwareFactsConsumer` is `ConfirmedFactsConsumer` with the erasure tombstone check after
each write (`libs/shared/erasure.py` "Late arrivals"): a late `decisions` / `actions` /
`meeting-room-events` fact about an erased meeting, or about an erased person from before their
erasure, is written and then erased again, and its `graph-events` are not emitted.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.age import AgeConnection, agtype_to_int, run_cypher
from libs.shared.consumer import ConsumedEvent
from libs.shared.erasure import (
    ErasureConsumer,
    ErasureRequest,
    meeting_erased,
    person_erased_before,
)
from services.knowledge_graph_service.confirmed import GraphFact, org_transaction
from services.knowledge_graph_service.facts_consumer import ConfirmedFactsConsumer

logger = logging.getLogger(__name__)

GraphConnect = Callable[[], AbstractAsyncContextManager[AgeConnection]]

_KEEP_DECISIONS = (
    "MATCH (d:Decision {org_id: $org_id, meeting_id: $meeting_id}) "
    "SET d.meeting_id = null, d.meeting_erased = true RETURN count(d)"
)
_KEEP_ACTIONS = (
    "MATCH (a:Action {org_id: $org_id, meeting_id: $meeting_id}) "
    "SET a.meeting_id = null, a.meeting_erased = true RETURN count(a)"
)
_MEETING_EDGES = (
    "MATCH (m:Meeting {org_id: $org_id, meeting_id: $meeting_id})-[e]-() RETURN count(e)"
)
_DELETE_MEETING = (
    "MATCH (m:Meeting {org_id: $org_id, meeting_id: $meeting_id}) DETACH DELETE m "
    "RETURN count(*)"
)
_PERSON_EDGES = "MATCH (p:Person {org_id: $org_id, user_id: $user_id})-[e]-() RETURN count(e)"
_DELETE_PERSON = (
    "MATCH (p:Person {org_id: $org_id, user_id: $user_id}) DETACH DELETE p RETURN count(*)"
)


async def _count(
    conn: AgeConnection, org_id: uuid.UUID, cypher: str, params: dict[str, Any]
) -> int:
    rows = await run_cypher(conn, org_id, cypher, params, columns=("n",))
    return agtype_to_int(rows[0][0]) if rows else 0


async def erase_meeting_graph(
    conn: AgeConnection, org_id: uuid.UUID, meeting_id: uuid.UUID
) -> dict[str, int]:
    params = {"meeting_id": str(meeting_id)}
    async with org_transaction(conn, org_id):
        decisions = await _count(conn, org_id, _KEEP_DECISIONS, params)
        actions = await _count(conn, org_id, _KEEP_ACTIONS, params)
        edges = await _count(conn, org_id, _MEETING_EDGES, params)
        nodes = await _count(conn, org_id, _DELETE_MEETING, params)
    return {
        "meeting_nodes_deleted": nodes,
        "edges_deleted": edges,
        "decisions_kept": decisions,
        "actions_kept": actions,
    }


async def erase_person_graph(
    conn: AgeConnection, org_id: uuid.UUID, user_id: uuid.UUID
) -> dict[str, int]:
    params = {"user_id": str(user_id)}
    async with org_transaction(conn, org_id):
        edges = await _count(conn, org_id, _PERSON_EDGES, params)
        nodes = await _count(conn, org_id, _DELETE_PERSON, params)
    return {"person_nodes_deleted": nodes, "edges_deleted": edges}


class GraphErasureConsumer(ErasureConsumer):
    group: ClassVar[str] = "knowledge-graph-service.erasure"
    store: ClassVar[str] = "knowledge_graph"
    producer_name: ClassVar[str] = "knowledge-graph-service"

    def __init__(self, *, graph_connect: GraphConnect, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph_connect = graph_connect

    async def erase(self, req: ErasureRequest, session: AsyncSession) -> dict[str, int]:
        del session  # the graph is not in the relational database
        async with self._graph_connect() as conn:
            if req.kind == "meeting":
                return await erase_meeting_graph(conn, req.org_id, req.target_id)
            return await erase_person_graph(conn, req.org_id, req.target_id)


def _as_uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value not in (None, "") else None
    except ValueError:
        return None


class ErasureAwareFactsConsumer(ConfirmedFactsConsumer):
    async def write_graph(self, evt: ConsumedEvent) -> list[GraphFact]:
        facts = await super().write_graph(evt)
        if not facts:
            return facts
        erased: set[str] = set()
        meeting = _as_uuid(evt.body.get("meeting_id"))
        if meeting is not None and await meeting_erased(self._redis, evt.org_id, meeting):
            async with self._graph_connect() as conn:
                await erase_meeting_graph(conn, evt.org_id, meeting)
            erased.add(str(meeting))
        person_field = "assignee_id" if evt.topic == "actions" else "participant_id"
        person = _as_uuid(evt.body.get(person_field))
        if person is not None and await person_erased_before(
            self._redis, evt.org_id, person, evt.header
        ):
            async with self._graph_connect() as conn:
                await erase_person_graph(conn, evt.org_id, person)
            erased.add(str(person))
        if erased:
            logger.info("kg: re-applied an erasure after a late %s (org %s)", evt.topic, evt.org_id)
        return [f for f in facts if f.from_id not in erased and f.to_id not in erased]
