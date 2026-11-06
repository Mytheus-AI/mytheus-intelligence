"""The role -> topics taxonomy as `knowledge-graph-service` consumes it.

The source of truth is `mytheus-backend/services/identity_tenant_service/taxonomy/
role_topics.{version}.yaml` (Plan 5-6 section 3.2), owned by `identity-tenant-service`. This
package holds a byte-for-byte-equivalent JSON copy of each published version.

Why a copy and not a live fetch: `GET /taxonomy/role-topics` sits behind a Clerk member-tier
token, and this consumer has no user token. Why that is safe: a published version is immutable
by the owner's own policy (`loader.py` point 1: a change is a NEW file, never an edit), so a
copy of an immutable artifact cannot go stale; `tests/test_knowledge_graph_topics.py` asserts
the copy equals the backend YAML whenever both repos are checked out side by side, so a
violation of that policy fails a test rather than drifting silently.

`ROLE_TOPICS_TAXONOMY_DIR` may point at another directory of `role_topics.{version}.json`
files; `ROLE_TOPICS_TAXONOMY_VERSION` (default `v1`) pins the version, the same variable and
default the backend uses.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_DEFAULT_DIR = Path(__file__).parent


class RoleTopicsTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    roles: dict[str, list[str]]


class TaxonomyVersionNotFound(LookupError):
    """No artifact for that version, or its `version` field disagrees with its filename. No
    silent fallback to another version."""


def active_version() -> str:
    return os.environ.get("ROLE_TOPICS_TAXONOMY_VERSION", "v1")


def _taxonomy_dir() -> Path:
    override = os.environ.get("ROLE_TOPICS_TAXONOMY_DIR")
    return Path(override) if override else _DEFAULT_DIR


@cache
def _load(directory: Path, version: str) -> RoleTopicsTaxonomy:
    path = directory / f"role_topics.{version}.json"
    if not path.is_file():
        raise TaxonomyVersionNotFound(version)
    taxonomy = RoleTopicsTaxonomy.model_validate(json.loads(path.read_text()))
    if taxonomy.version != version:
        raise TaxonomyVersionNotFound(version)
    return taxonomy


def load_taxonomy(version: str | None = None) -> RoleTopicsTaxonomy:
    return _load(_taxonomy_dir(), version or active_version())


def distinct_topics(version: str | None = None) -> list[str]:
    """Every distinct topic across all roles, first-seen order (deterministic). `pricing`
    appears under both `product_manager` and `sales` in v1 and is ONE `Topic` node, so v1
    yields 25 names from 26 entries. This is the number Plan 5-6 3.2's acceptance check
    compares `MATCH (t:Topic {org_id:$org}) RETURN count(t)` against."""
    seen: dict[str, None] = {}
    for topics in load_taxonomy(version).roles.values():
        for name in topics:
            seen.setdefault(name, None)
    return list(seen)
