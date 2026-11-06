"""The vendored role -> topics taxonomy must equal `identity-tenant-service`'s source of truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.knowledge_graph_service.taxonomy import (
    TaxonomyVersionNotFound,
    distinct_topics,
    load_taxonomy,
)

_BACKEND_YAML = (
    Path(__file__).resolve().parents[2]
    / "mytheus-backend/services/identity_tenant_service/taxonomy/role_topics.v1.yaml"
)


def test_v1_has_25_distinct_topics_with_pricing_once() -> None:
    topics = distinct_topics("v1")
    assert len(topics) == 25
    assert topics.count("pricing") == 1
    assert topics == distinct_topics("v1")  # deterministic


def test_unknown_version_fails_closed() -> None:
    with pytest.raises(TaxonomyVersionNotFound):
        load_taxonomy("v999")


def test_vendored_copy_matches_backend_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    if not _BACKEND_YAML.is_file():
        pytest.skip(
            f"DRIFT CHECK NOT RUN: {_BACKEND_YAML} not found (mytheus-backend not checked out "
            "alongside). The vendored taxonomy copy is unverified against its source of truth."
        )
    source = yaml.safe_load(_BACKEND_YAML.read_text())
    vendored = load_taxonomy("v1")
    assert source == {"version": vendored.version, "roles": vendored.roles}
