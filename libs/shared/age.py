"""Apache AGE access: the one sanctioned way code in this repo issues Cypher against the
`nexus_knowledge` graph.

Plan 5-6 gap 16. `knowledge-graph-service` is the ONLY writer of `nexus_knowledge` (ADR-0005,
`_canon/02-data-model.md` section 4); this module is the low-level client it writes through.
It enforces, in code and fail-closed, the three rules canon states for the graph:

1. Tenant scoping (ADR-0005 D2). Every Cypher statement MUST bind `$org_id`, and the value of
   `$org_id` is ALWAYS injected here from a typed `uuid.UUID` argument - a caller can neither
   omit it nor smuggle a different one in through `params` (a `params["org_id"]` key is
   rejected, not merged). A statement that does not reference `$org_id` is refused before it
   reaches the database.
2. No string-interpolated tenant data. AGE's `cypher(graph, query, params)` takes its
   parameters as ONE agtype map bound to a real SQL placeholder (`$1`); tenant values only ever
   travel through that map. The Cypher text itself is code-authored. The only thing that is
   ever formatted into it is a node/edge LABEL, and labels are checked against the frozen canon
   sets below (Cypher cannot parameterize a label) - never caller data.
3. Frozen vocabulary. Node labels `Person`, `Meeting`, `Decision`, `Action`, `Topic` and edge
   labels `ATTENDED`, `PRODUCED`, `ASSIGNED_TO`, `RELATES_TO`, `FOLLOWS_FROM` (verbatim from the
   PDF baseline, carried into canon). Anything else is rejected.

Where AGE runs. `supabase/postgres:15.8.1.060` (the image `mytheus-backend/docker-compose.yml`
uses for the shared dev Postgres) does NOT ship the `age` extension - verified 2026-11-06 by
listing every `*.control` file across all of that image's layers: there is no `age.control`.
This is exactly risk R1 ("Apache AGE on managed Supabase may be operationally awkward") and its
pre-authorized fallback "self-host Postgres+AGE". The graph therefore lives in a dedicated
`apache/age` Postgres (`mytheus-intelligence/docker-compose.yml` service `age`, port 5434),
addressed by `AGE_DATABASE_URL` - deliberately a different variable from `DATABASE_URL`, so a
misconfiguration cannot silently point graph writes at the relational core.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

GRAPH_NAME = "nexus_knowledge"

NODE_LABELS: frozenset[str] = frozenset({"Person", "Meeting", "Decision", "Action", "Topic"})
EDGE_LABELS: frozenset[str] = frozenset(
    {"ATTENDED", "PRODUCED", "ASSIGNED_TO", "RELATES_TO", "FOLLOWS_FROM"}
)

_COLUMN_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_SCALAR_TYPES = (str, int, float, bool, type(None))


class AgeQueryRejected(ValueError):
    """A Cypher statement or its parameters failed a tenant-safety check. Never retried: this
    is a programming error, not a transient fault."""


class AgeConnection(Protocol):
    """The subset of `asyncpg.Connection` this module uses. Tests substitute a fake cursor."""

    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def set_type_codec(
        self,
        typename: str,
        *,
        schema: str = ...,
        encoder: Any,
        decoder: Any,
        format: str = ...,
    ) -> None: ...


def age_database_url() -> str:
    """The AGE Postgres DSN (plain asyncpg form, `postgresql://...`). Fail closed: no default,
    because the obvious default (the shared `DATABASE_URL`) is a Postgres without AGE."""
    url = os.environ.get("AGE_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "AGE_DATABASE_URL not set. Expected postgresql://<user>:<pw>@localhost:5434/"
            "nexus_graph (the dedicated apache/age container in "
            "mytheus-intelligence/docker-compose.yml)."
        )
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def prepare_session(conn: AgeConnection) -> None:
    """Per-connection setup AGE requires: load the extension library, put `ag_catalog` on the
    search path, and teach the driver to pass `agtype` as text (asyncpg has no built-in codec
    for it). Use as an asyncpg pool `init=` hook so every pooled connection is prepared."""
    await conn.execute("LOAD 'age'")
    await conn.execute('SET search_path = ag_catalog, "$user", public')
    await conn.set_type_codec(
        "agtype", schema="ag_catalog", encoder=str, decoder=str, format="text"
    )


def _check_param_value(key: str, value: Any) -> None:
    if isinstance(value, _SCALAR_TYPES):
        return
    if isinstance(value, list | tuple) and all(isinstance(v, _SCALAR_TYPES) for v in value):
        return
    raise AgeQueryRejected(f"parameter {key!r} has unsupported type {type(value).__name__}")


def encode_params(org_id: UUID, params: Mapping[str, Any] | None = None) -> str:
    """The agtype parameter map, JSON-encoded, with `org_id` injected from the typed argument.

    Fails closed on a non-UUID `org_id` (a string that merely looks like one is refused: the
    caller must have parsed it already, which is where a malformed tenant id gets caught) and on
    any attempt to supply `org_id` through `params`."""
    if not isinstance(org_id, UUID):
        raise AgeQueryRejected("org_id must be a uuid.UUID, never a raw string or None")
    merged: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key == "org_id":
            raise AgeQueryRejected("org_id may not be passed in params; it is injected")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise AgeQueryRejected(f"invalid parameter name {key!r}")
        _check_param_value(key, value)
        merged[key] = list(value) if isinstance(value, tuple) else value
    merged["org_id"] = str(org_id)
    return json.dumps(merged, separators=(",", ":"), sort_keys=True)


def build_cypher_sql(cypher: str, columns: Sequence[str]) -> str:
    """Wrap a code-authored Cypher statement in AGE's `cypher()` SQL function.

    The statement goes inside a dollar-quoted literal, so it must not itself contain `$$`
    (which would terminate the literal early). It must reference `$org_id`. The single SQL
    placeholder `$1` carries the parameter map; AGE requires that third argument to be a bare
    parameter, so it is not cast."""
    if "$$" in cypher:
        raise AgeQueryRejected("Cypher text may not contain '$$'")
    if not re.search(r"\$org_id\b", cypher):
        raise AgeQueryRejected("every nexus_knowledge statement must bind $org_id")
    if not columns:
        raise AgeQueryRejected("at least one result column is required")
    for col in columns:
        if not _COLUMN_RE.fullmatch(col):
            raise AgeQueryRejected(f"invalid result column name {col!r}")
    cols = ", ".join(f"{c} ag_catalog.agtype" for c in columns)
    return f"SELECT * FROM ag_catalog.cypher('{GRAPH_NAME}', $$ {cypher} $$, $1) AS ({cols})"


async def run_cypher(
    conn: AgeConnection,
    org_id: UUID,
    cypher: str,
    params: Mapping[str, Any] | None = None,
    *,
    columns: Sequence[str] = ("result",),
) -> list[Any]:
    """Execute one tenant-scoped Cypher statement. Returns the driver's rows (agtype values
    decoded as text by `prepare_session`'s codec)."""
    sql = build_cypher_sql(cypher, columns)
    return await conn.fetch(sql, encode_params(org_id, params))


def agtype_to_int(value: Any) -> int:
    """Decode an agtype scalar such as `25` (text form) to an int."""
    text = str(value).strip()
    if "::" in text:
        text = text.split("::", 1)[0]
    return int(json.loads(text))


def require_node_label(label: str) -> str:
    if label not in NODE_LABELS:
        raise AgeQueryRejected(f"{label!r} is not a frozen nexus_knowledge node label")
    return label


def require_edge_label(label: str) -> str:
    if label not in EDGE_LABELS:
        raise AgeQueryRejected(f"{label!r} is not a frozen nexus_knowledge edge label")
    return label
