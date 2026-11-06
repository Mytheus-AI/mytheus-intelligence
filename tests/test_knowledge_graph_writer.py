"""Unit tests for the `nexus_knowledge` writer, against a fake asyncpg cursor (no database).

Covers `libs/shared/age.py` (tenant-safety checks) and `services/knowledge_graph_service/
graph.py` + `consumer.py` (Topic seeding, edge rules). The live-AGE counterpart is
`tests/test_knowledge_graph_age_live.py`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from libs.shared.age import (
    AgeQueryRejected,
    agtype_to_int,
    build_cypher_sql,
    encode_params,
    prepare_session,
    run_cypher,
)
from libs.shared.consumer import ConsumedEvent
from services.knowledge_graph_service.consumer import TenantTopicSeedConsumer
from services.knowledge_graph_service.graph import (
    count_topics,
    ensure_graph,
    merge_edge,
    merge_topics,
)
from services.knowledge_graph_service.taxonomy import distinct_topics

ORG_A = uuid.UUID("00000000-0000-4000-8000-00000000000a")


class FakeCursor:
    def __init__(self, fetch_result: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.fetch_result = fetch_result if fetch_result is not None else [("{}",)]
        self.codecs: list[str] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append(("execute", query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.calls.append(("fetch", query, args))
        return self.fetch_result

    async def set_type_codec(self, typename: str, **_: Any) -> None:
        self.codecs.append(typename)

    def cypher_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (q, json.loads(a[0])) for kind, q, a in self.calls if kind == "fetch" and "cypher(" in q
        ]


# --- libs/shared/age.py -------------------------------------------------------------------------


def test_cypher_without_org_id_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        build_cypher_sql("MATCH (t:Topic) RETURN t", ("t",))


def test_dollar_quote_in_cypher_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        build_cypher_sql("MATCH (t:Topic {org_id: $org_id}) $$; DROP TABLE x; --", ("t",))


def test_bad_result_column_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        build_cypher_sql("MATCH (t:Topic {org_id: $org_id}) RETURN t", ("t agtype); --",))


def test_sql_wraps_graph_and_binds_single_placeholder() -> None:
    sql = build_cypher_sql("MATCH (t:Topic {org_id: $org_id}) RETURN t", ("t",))
    assert sql.startswith("SELECT * FROM ag_catalog.cypher('nexus_knowledge', $$")
    assert sql.endswith("$$, $1) AS (t ag_catalog.agtype)")


def test_org_id_must_be_a_uuid() -> None:
    with pytest.raises(AgeQueryRejected):
        encode_params(str(ORG_A), {})  # type: ignore[arg-type]
    with pytest.raises(AgeQueryRejected):
        encode_params(None, {})  # type: ignore[arg-type]


def test_org_id_cannot_be_smuggled_through_params() -> None:
    other = uuid.uuid4()
    with pytest.raises(AgeQueryRejected):
        encode_params(ORG_A, {"org_id": str(other)})


def test_non_scalar_param_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        encode_params(ORG_A, {"name": {"nested": "map"}})


async def test_tenant_value_travels_only_in_params_never_in_sql_text() -> None:
    hostile = "x'}) MATCH (n) DETACH DELETE n //"
    cur = FakeCursor()
    await run_cypher(
        cur, ORG_A, "MERGE (t:Topic {name: $name, org_id: $org_id})", {"name": hostile}
    )
    _, sql, args = cur.calls[0]
    assert hostile not in sql
    assert str(ORG_A) not in sql
    assert json.loads(args[0]) == {"name": hostile, "org_id": str(ORG_A)}


async def test_prepare_session_loads_age_and_registers_agtype() -> None:
    cur = FakeCursor()
    await prepare_session(cur)
    statements = [q for _, q, _ in cur.calls]
    assert statements[0] == "LOAD 'age'"
    assert "ag_catalog" in statements[1]
    assert cur.codecs == ["agtype"]


def test_agtype_to_int() -> None:
    assert agtype_to_int("25") == 25
    assert agtype_to_int("3::integer") == 3


# --- graph.py: Topic seeding --------------------------------------------------------------------


async def test_merge_topics_one_locked_transaction_one_merge_per_distinct_name() -> None:
    cur = FakeCursor()
    n = await merge_topics(cur, ORG_A, ["pricing", "roadmap", "pricing", " roadmap "])
    assert n == 2
    kinds = [(k, q) for k, q, _ in cur.calls]
    assert kinds[0] == ("execute", "BEGIN")
    assert "pg_advisory_xact_lock" in kinds[1][1]
    assert cur.calls[1][2] == (str(ORG_A),)
    assert kinds[-1] == ("execute", "COMMIT")
    merges = cur.cypher_calls()
    assert [p["name"] for _, p in merges] == ["pricing", "roadmap"]
    assert all(p["org_id"] == str(ORG_A) for _, p in merges)
    assert all("MERGE (t:Topic {name: $name, org_id: $org_id})" in q for q, _ in merges)


async def test_merge_topics_rolls_back_on_failure() -> None:
    class Boom(FakeCursor):
        async def fetch(self, query: str, *args: Any) -> list[Any]:
            raise RuntimeError("db down")

    cur = Boom()
    with pytest.raises(RuntimeError):
        await merge_topics(cur, ORG_A, ["ux"])
    assert cur.calls[-1][1] == "ROLLBACK"


async def test_merge_topics_refuses_empty_or_blank() -> None:
    with pytest.raises(AgeQueryRejected):
        await merge_topics(FakeCursor(), ORG_A, [])
    with pytest.raises(AgeQueryRejected):
        await merge_topics(FakeCursor(), ORG_A, ["  "])


async def test_count_topics_uses_the_acceptance_query() -> None:
    cur = FakeCursor(fetch_result=[("25",)])
    assert await count_topics(cur, ORG_A) == 25
    q, params = cur.cypher_calls()[0]
    assert "MATCH (t:Topic {org_id: $org_id}) RETURN count(t)" in q
    assert params == {"org_id": str(ORG_A)}


async def test_ensure_graph_creates_graph_and_all_ten_labels_when_absent() -> None:
    cur = FakeCursor(fetch_result=[])
    await ensure_graph(cur)
    sql = [q for _, q, _ in cur.calls]
    assert any("create_graph" in q for q in sql)
    assert sum("create_vlabel" in q for q in sql) == 5
    assert sum("create_elabel" in q for q in sql) == 5
    assert sum("CREATE INDEX IF NOT EXISTS" in q for q in sql) == 10
    assert sql[-1] == "COMMIT"


# --- graph.py: edges ----------------------------------------------------------------------------


async def test_edge_with_non_frozen_label_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        await merge_edge(
            FakeCursor(),
            ORG_A,
            label="KNOWS",
            from_label="Person",
            from_id="u",
            to_label="Person",
            to_id="v",
        )


async def test_edge_between_wrong_label_pair_is_refused() -> None:
    with pytest.raises(AgeQueryRejected):
        await merge_edge(
            FakeCursor(),
            ORG_A,
            label="ATTENDED",
            from_label="Topic",
            from_id="ux",
            to_label="Meeting",
            to_id="m",
        )


async def test_edge_matches_both_endpoints_inside_the_org() -> None:
    cur = FakeCursor()
    await merge_edge(
        cur,
        ORG_A,
        label="RELATES_TO",
        from_label="Decision",
        from_id="d1",
        to_label="Topic",
        to_id="pricing",
    )
    q, params = cur.cypher_calls()[0]
    assert "(a:Decision {org_id: $org_id, decision_id: $from_id})" in q
    assert "(b:Topic {org_id: $org_id, name: $to_id})" in q
    assert "MERGE (a)-[e:RELATES_TO {org_id: $org_id}]->(b)" in q
    assert params == {"from_id": "d1", "to_id": "pricing", "org_id": str(ORG_A)}


async def test_edge_with_missing_endpoint_fails_and_rolls_back() -> None:
    cur = FakeCursor(fetch_result=[])
    with pytest.raises(AgeQueryRejected):
        await merge_edge(
            cur,
            ORG_A,
            label="ATTENDED",
            from_label="Person",
            from_id="u",
            to_label="Meeting",
            to_id="m",
        )
    assert cur.calls[-1][1] == "ROLLBACK"


# --- consumer.py --------------------------------------------------------------------------------


def _consumer(cur: FakeCursor) -> TenantTopicSeedConsumer:
    @asynccontextmanager
    async def connect() -> AsyncIterator[FakeCursor]:
        yield cur

    return TenantTopicSeedConsumer(
        graph_connect=connect,  # type: ignore[arg-type]
        session_factory=None,  # type: ignore[arg-type]
        redis=None,  # type: ignore[arg-type]
        codec=None,  # type: ignore[arg-type]
        dlq_producer=None,  # type: ignore[arg-type]
    )


def _evt(event_type: str) -> ConsumedEvent:
    return ConsumedEvent(
        topic="tenant-events",
        org_id=ORG_A,
        idempotency_key=f"org:{ORG_A}:{event_type}",
        header={"org_id": str(ORG_A)},
        body={"event_type": event_type, "org_slug": "acme", "plan_code": "team"},
        partition=0,
        offset=0,
    )


async def test_provisioned_event_seeds_every_distinct_taxonomy_topic_for_header_org() -> None:
    cur = FakeCursor()
    await _consumer(cur).apply(_evt("provisioned"), None)  # type: ignore[arg-type]
    merges = cur.cypher_calls()
    assert [p["name"] for _, p in merges] == distinct_topics("v1")
    assert {p["org_id"] for _, p in merges} == {str(ORG_A)}


@pytest.mark.parametrize("event_type", ["suspended", "deleted"])
async def test_other_lifecycle_events_do_not_touch_the_graph(event_type: str) -> None:
    cur = FakeCursor()
    await _consumer(cur).apply(_evt(event_type), None)  # type: ignore[arg-type]
    assert cur.calls == []


def test_consumer_identity() -> None:
    assert TenantTopicSeedConsumer.topics == ["tenant-events"]
    assert TenantTopicSeedConsumer.group == "knowledge-graph-service.topic-seed"
    assert TenantTopicSeedConsumer.producer_name == "knowledge-graph-service"
