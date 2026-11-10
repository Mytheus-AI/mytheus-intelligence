"""The Qdrant tenancy chokepoint — ADR-0006, resolved and no longer DECISION-REQUIRED.

Per-tenant physical collections (`{logical}__{org_id_hex}`) PLUS a mandatory `org_id` payload
filter on every query. Both layers, always — the collection boundary is the primary
isolation, the payload filter is the compensating control if a collection name is ever
computed wrongly (ADR-0006 "Consequences: Positive"). `resolve_collection()` is the ONLY
function in this repo allowed to turn a logical name + org_id into a physical collection name;
nothing else may string-format `__{org_id}` inline, or the belt-and-suspenders design
degrades into "everyone remembers to do it right."

`org_id` is rendered as 32-char hex (`UUID.hex`), never the client-supplied string, and never
anything derived from request/event *body* fields — it comes from the verified event envelope
header (`libs/shared/tenancy.parse_org_id`) or the authenticated gRPC caller, per ADR-0006 §1.

`embedding-service` is the ONLY writer of Qdrant in this program (CLAUDE.md, `system.md` §10);
this module's `upsert()` is therefore the single write path other services must not duplicate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

EMBEDDING_DIM = 384  # BGE-small-en

# Logical collection names — frozen verbatim by ADR-0006 / _canon/02-data-model.md §3.
LOGICAL_TRANSCRIPT_CHUNKS = "transcript_chunks"
LOGICAL_DOCUMENT_CHUNKS = "document_chunks"
LOGICAL_DECISIONS = "decisions"


class QdrantAuthNotConfigured(RuntimeError):
    """Raised at construction when the credential this client needs is absent or misconfigured.

    Deliberately not a warning-and-continue. A client built with no key keeps working against a
    Qdrant that does not enforce one and fails the first time it meets a Qdrant that does, which
    moves the failure from startup in dev to a request in production.
    """


def _url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def _distinct_or_raise(write: str | None, read: str | None) -> None:
    """Both keys set to the same value is not a control, and it is invisible without this check.

    Every isolation test still passes in that configuration, because the "read-only" client is
    the write client wearing another name. This is the one coupling between the two factories
    and it exists precisely because the failure is silent.
    """
    if write and read and write == read:
        raise QdrantAuthNotConfigured(
            "QDRANT_API_KEY and QDRANT_READ_ONLY_API_KEY must not be the same value: "
            "a shared key gives every reader write access while looking correctly configured"
        )


def write_client() -> AsyncQdrantClient:
    """The privileged client. `embedding-service` is the ONLY service that may call this.

    Holding the write key is what makes the sole-writer property real rather than a convention
    the next contributor has to know about (status-overview gap 19).
    """
    key = os.environ.get("QDRANT_API_KEY")
    if not key:
        raise QdrantAuthNotConfigured(
            "QDRANT_API_KEY is not set. embedding-service cannot write to Qdrant without the "
            "write credential, and there is deliberately no unauthenticated fallback."
        )
    _distinct_or_raise(key, os.environ.get("QDRANT_READ_ONLY_API_KEY"))
    return AsyncQdrantClient(url=_url(), api_key=key)


def read_client() -> AsyncQdrantClient:
    """The client every non-writing consumer uses: `rag-service`, `search-service`, tooling.

    Qdrant refuses collection creation and upserts for this credential with `403 Forbidden:
    Global manage access is required`, verified against `qdrant/qdrant:v1.12.4`. Searching and
    listing are unaffected, which is all a reader needs.
    """
    key = os.environ.get("QDRANT_READ_ONLY_API_KEY")
    if not key:
        raise QdrantAuthNotConfigured(
            "QDRANT_READ_ONLY_API_KEY is not set. A reader must not fall back to the write "
            "credential, which is exactly the collapse this split exists to prevent."
        )
    _distinct_or_raise(os.environ.get("QDRANT_API_KEY"), key)
    return AsyncQdrantClient(url=_url(), api_key=key)


def physical_collection(logical: str, org_id: UUID) -> str:
    """The ADR-0006 §1 resolution: `{logical}__{org_id.hex}`. `org_id` is NEVER client-supplied
    — it must already have been verified (event envelope header / authz middleware) by the
    time it reaches this function."""
    return f"{logical}__{org_id.hex}"


async def ensure_collection(client: AsyncQdrantClient, name: str) -> None:
    """Create the collection with ADR-0006 §4's locked, version-pinned physical config if it
    does not already exist. Idempotent: safe to call on every upsert (the embed-on-ingest
    consumer does not know in advance whether a tenant's collection has been provisioned by
    the `identity-tenant-service` activation workflow yet)."""
    if await client.collection_exists(name):
        return
    await client.create_collection(
        name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
        ),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=50_000),
        on_disk_payload=True,
    )


def _org_filter(org_id: UUID) -> Filter:
    """The mandatory payload backstop (ADR-0006 §1: "every query is org-scoped TWICE"). An
    org-wide search never returns a chunk of a confidential meeting (`confidential: true`,
    `libs/shared/index_guard.py`): those are reachable only through a participant-scoped query
    (rag-service's `meeting_ids` allowlist)."""
    return Filter(
        must=[FieldCondition(key="org_id", match=MatchValue(value=str(org_id)))],
        must_not=[FieldCondition(key="confidential", match=MatchValue(value=True))],
    )


async def delete_window(
    client: AsyncQdrantClient,
    logical: str,
    org_id: UUID,
    meeting_id: UUID,
    start_ms: int,
    end_ms: int,
) -> None:
    """Remove a meeting's points whose `start_ms` is in [start_ms, end_ms): the never-record
    purge of a window indexed before the chunk that named the topic arrived. Org-scoped twice,
    like every other operation; a collection that does not exist yet has nothing to remove."""
    name = physical_collection(logical, org_id)
    if not await client.collection_exists(name):
        return
    await client.delete(
        name,
        points_selector=Filter(
            must=[
                FieldCondition(key="org_id", match=MatchValue(value=str(org_id))),
                FieldCondition(key="meeting_id", match=MatchValue(value=str(meeting_id))),
                FieldCondition(key="start_ms", range=Range(gte=start_ms, lt=end_ms)),
            ]
        ),
    )


async def upsert(
    client: AsyncQdrantClient,
    logical: str,
    org_id: UUID,
    points: list[PointStruct],
) -> str:
    """Resolve the physical collection, provision it if needed, and upsert. Returns the
    physical collection name (used for the `embedding-events` ack payload's `collection`
    field). Every point's payload MUST already carry `org_id` (checked by the caller building
    the points, not re-derived here, so the write and the read use the same backstop
    contract)."""
    name = physical_collection(logical, org_id)
    await ensure_collection(client, name)
    await client.upsert(name, points)
    return name


async def search_with_tenant_filter(
    client: AsyncQdrantClient,
    logical: str,
    org_id: UUID,
    query_vector: list[float],
    *,
    limit: int = 50,
) -> list[Any]:
    """The only sanctioned search path: physical collection boundary AND payload filter,
    together. Used directly by the isolation-proof tests, and is the shape `rag-service`'s
    vector retrieval arm must call through (ADR-0006 §1's `resolve_collection` + search
    example)."""
    name = physical_collection(logical, org_id)
    result = await client.query_points(
        collection_name=name,
        query=query_vector,
        query_filter=_org_filter(org_id),
        limit=limit,
    )
    return list(result.points)


@dataclass(frozen=True)
class CollectionRef:
    """Placeholder for the future pooled-vs-cell cluster resolution (ADR-0006 §2). This repo
    runs a single pooled Qdrant cluster today (`QDRANT_URL`); `data_residency`-driven cluster
    routing is out of this task's scope and is left for the cell-promotion work."""

    name: str
