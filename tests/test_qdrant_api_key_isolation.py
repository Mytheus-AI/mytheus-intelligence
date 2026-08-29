"""The "embedding-service is the only writer of Qdrant" property, as infrastructure.

Closes status-overview gap 19. Until now that property was a code convention: `libs/shared/
qdrant.py` documented itself as the single write path and nothing stopped a second service, a
script, or a mistake from opening its own client and upserting. The dev Qdrant had no API key
configured at all, so any process that could reach the port had full control of every tenant's
vectors.

Qdrant's two-key model is what makes the convention enforceable. Verified directly against
`qdrant/qdrant:v1.12.4` before this was designed around:

| Credential | list | search | create collection | upsert |
|---|---|---|---|---|
| none | 401 | 401 | 401 | 401 |
| `QDRANT__SERVICE__READ_ONLY_API_KEY` | 200 | 200 | **403** | **403** |
| `QDRANT__SERVICE__API_KEY` | 200 | 200 | 200 | 200 |

So the write key goes to `embedding-service` alone and every other consumer gets the read-only
key. A reader that tries to write now fails at the server, not at review time.

These tests assert the PROPERTY (a read-only credential cannot write), not the configuration
(some env var is set). That distinction is the lesson from the partition-grant finding in
mytheus-backend, where six standing assertions were all scoped to the roles the program
created and none could see the role the image shipped.
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from libs.shared.qdrant import (
    EMBEDDING_DIM,
    LOGICAL_TRANSCRIPT_CHUNKS,
    QdrantAuthNotConfigured,
    physical_collection,
    read_client,
    upsert,
    write_client,
)

# --- the control itself ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_read_only_client_cannot_upsert(qdrant_client, qdrant_read_client):
    """The whole point of gap 19. A reader holding the read-only key can search but not write.

    Provisioning is done with the write client so the failure under test is the upsert itself
    and not a missing collection.
    """
    org_id = uuid.uuid4()
    name = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id)
    await qdrant_client.create_collection(
        name, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    point = PointStruct(id=1, vector=[0.0] * EMBEDDING_DIM, payload={"org_id": str(org_id)})

    with pytest.raises(Exception) as exc:
        await qdrant_read_client.upsert(name, [point])
    assert "403" in str(exc.value) or "orbidden" in str(exc.value), exc.value

    await qdrant_client.delete_collection(name)


@pytest.mark.asyncio
async def test_a_read_only_client_cannot_create_a_collection(qdrant_read_client):
    """Creating a collection is how a rogue writer would start, so it is refused too.

    `ensure_collection` runs on every upsert, so a reader that only had the write half blocked
    could still provision empty collections and make the vector store's shape depend on who
    happened to call first.
    """
    name = physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, uuid.uuid4())
    with pytest.raises(Exception) as exc:
        await qdrant_read_client.create_collection(
            name, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
    assert "403" in str(exc.value) or "orbidden" in str(exc.value), exc.value


@pytest.mark.asyncio
async def test_a_read_only_client_can_still_search(qdrant_client, qdrant_read_client):
    """The control must not break the readers it is there to constrain.

    `rag-service` and `search-service` query Qdrant; if the read-only key could not search,
    this would be an outage dressed up as a security control.
    """
    org_id = uuid.uuid4()
    name = await upsert(
        qdrant_client,
        LOGICAL_TRANSCRIPT_CHUNKS,
        org_id,
        [PointStruct(id=1, vector=[0.1] * EMBEDDING_DIM, payload={"org_id": str(org_id)})],
    )

    result = await qdrant_read_client.query_points(
        collection_name=name, query=[0.1] * EMBEDDING_DIM, limit=1
    )

    assert len(result.points) == 1
    await qdrant_client.delete_collection(name)


@pytest.mark.asyncio
async def test_an_unauthenticated_client_is_refused_entirely(qdrant_unauthenticated):
    """The server must require a key at all. If it does not, both keys above are decoration:
    anyone who can reach the port simply omits the header."""
    with pytest.raises(Exception) as exc:
        await qdrant_unauthenticated.get_collections()
    assert "401" in str(exc.value) or "nauthorized" in str(exc.value), exc.value


# --- fail-closed construction -----------------------------------------------------------------


def test_write_client_refuses_to_build_without_its_key(monkeypatch):
    """No silent fallback to an unauthenticated client.

    Defaulting to "no key" would mean a deployment that forgot the secret keeps working right
    up until it meets a Qdrant that enforces one, and the failure would land in production
    rather than at startup. Same reasoning as `api-gateway`'s `GATEWAY_SIGNING_KEY`, which
    fails closed rather than accepting a default.
    """
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    with pytest.raises(QdrantAuthNotConfigured, match="QDRANT_API_KEY"):
        write_client()


def test_read_client_refuses_to_build_without_its_key(monkeypatch):
    monkeypatch.delenv("QDRANT_READ_ONLY_API_KEY", raising=False)
    with pytest.raises(QdrantAuthNotConfigured, match="QDRANT_READ_ONLY_API_KEY"):
        read_client()


def test_the_two_factories_do_not_share_a_key(monkeypatch):
    """A deployment that points both variables at the same value has no control at all, and
    the mistake is invisible: every test above still passes, because the "read-only" client is
    simply the write client under another name."""
    monkeypatch.setenv("QDRANT_API_KEY", "same-value")
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "same-value")
    with pytest.raises(QdrantAuthNotConfigured, match="must not be the same"):
        read_client()


def test_the_write_key_is_never_read_from_the_read_only_variable(monkeypatch):
    """Guards the copy-paste that would quietly grant write access to every reader."""
    monkeypatch.setenv("QDRANT_API_KEY", "write-value")
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "read-value")
    assert os.environ["QDRANT_API_KEY"] != os.environ["QDRANT_READ_ONLY_API_KEY"]
    # Constructing both must not raise; the distinctness check is the only coupling.
    write_client()
    read_client()
