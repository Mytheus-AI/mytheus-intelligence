"""The AGE tenant guard against the four statements that passed the old substring check
(research cross-store study, chain G1), plus the shapes the shipped code relies on."""

import pytest

from libs.shared.age import (
    AgeQueryRejected,
    build_cypher_sql,
    cypher_code,
    require_fully_scoped,
)
from services.knowledge_graph_service import templates

BYPASSES = {
    "anchor_only_varlength": "MATCH (d:Decision {org_id: $org_id})-[*1..5]-(x) RETURN x",
    "org_in_where_only_on_anchor": "MATCH (d)-[*1..3]-(x) WHERE d.org_id = $org_id RETURN x",
    "org_in_comment": "MATCH (x:Decision) RETURN x // $org_id",
    "org_in_string": "MATCH (x:Decision) WHERE x.content <> '$org_id' RETURN x",
}

MORE_REFUSED = [
    "MATCH (d:Decision {org_id: $org_id})-->(x) RETURN x",  # bracketless edge
    "MATCH (d:Decision {org_id: $org_id})-[r:FOLLOWS_FROM]->(x {org_id: $org_id}) RETURN x",
    "MATCH (d:Decision {org_id: $org_id}), (x:Decision) RETURN x",  # second, unscoped node
    "MATCH (x:Decision) /* $org_id */ RETURN x",
    'MATCH (x:Decision) WHERE x.content = "$org_id" RETURN x',
    "MATCH (d:Decision {org_id: $org_idx}) RETURN d",  # not the parameter
]

ALLOWED = [
    "MATCH (d:Decision {org_id: $org_id}) RETURN count(d)",
    "MATCH (d:Decision {org_id: $org_id, decision_id: $id}) RETURN d",
    "MATCH (a:Decision {org_id: $org_id})-[r:FOLLOWS_FROM {org_id: $org_id}]->"
    "(b:Decision {org_id: $org_id}) WHERE a.deleted_at IS NULL RETURN b",
    "MATCH (d:Decision {org_id: $org_id}) WITH d MATCH (d)-[r:RELATES_TO {org_id: $org_id}]->"
    "(t:Topic {org_id: $org_id}) RETURN t",
]


@pytest.mark.parametrize("name", sorted(BYPASSES))
def test_each_known_bypass_is_refused(name: str) -> None:
    with pytest.raises(AgeQueryRejected):
        require_fully_scoped(BYPASSES[name])


@pytest.mark.parametrize("cypher", MORE_REFUSED)
def test_other_unscoped_shapes_are_refused(cypher: str) -> None:
    with pytest.raises(AgeQueryRejected):
        require_fully_scoped(cypher)


@pytest.mark.parametrize("cypher", ALLOWED)
def test_fully_scoped_statements_pass(cypher: str) -> None:
    require_fully_scoped(cypher)
    build_cypher_sql(cypher, ["x"])


@pytest.mark.parametrize("name", ["org_in_comment", "org_in_string"])
def test_the_base_check_ignores_comments_and_strings(name: str) -> None:
    """build_cypher_sql (used by every caller, templates included) no longer counts an
    `$org_id` that only appears inside a comment or a string literal."""
    with pytest.raises(AgeQueryRejected):
        build_cypher_sql(BYPASSES[name], ["x"])


def test_cypher_code_keeps_code_and_empties_literals() -> None:
    code = cypher_code("MATCH (d {org_id: $org_id}) WHERE d.x = '$org_id' // $org_id\nRETURN d")
    assert code.count("$org_id") == 1
    assert "RETURN d" in code


def test_every_shipped_template_still_builds() -> None:
    """The guard tightening must not reject the code-authored templates."""
    texts = [
        v
        for v in vars(templates).values()
        if isinstance(v, str) and "MATCH" in v and "$org_id" in v
    ]
    assert texts, "found no template text to check"
    for text in texts:
        build_cypher_sql(text, ["x"])
