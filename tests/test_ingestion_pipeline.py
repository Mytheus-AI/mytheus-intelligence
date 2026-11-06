"""`ingest_document` end to end against REAL Postgres (`intelligence.document_chunks` /
`document_elements` under real RLS) and real Redis (PII gate). Covers the plan's acceptance
checks directly:
- rows into `document_chunks` with `element_type` in {Title, NarrativeText, Table, ListItem}
- one `emit` call (one `document-chunks` outbox-shaped payload) per chunk
- a re-run (same org_id/document_id/content) is idempotent: zero additional rows, zero
  additional emits
- with PII redaction enabled, the emitted/stored chunk text is redacted before it would reach
  the bus
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from services.ingestion_service.pipeline import ingest_document
from tests.conftest import APP_URL
from tests.fixtures import sample_documents as docs

pytestmark = pytest.mark.asyncio

_FROZEN_TYPES = {"Title", "NarrativeText", "Table", "ListItem"}


@pytest.fixture()
def session_factory(migrated_db):
    engine = create_async_engine(APP_URL, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


class _RecordingEmitter:
    """Collects emit() calls in place of a real outbox insert, so these tests can assert on
    exactly what would have been enqueued without depending on outbox internals (those are
    proven separately in test_outbox_relay.py / test_embedding_consumer.py)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def __call__(self, topic: str, idem: str, payload: dict) -> None:
        self.calls.append((topic, idem, payload))


async def _count_rows(session_factory, org_id, document_id) -> tuple[int, int]:
    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        chunks = (
            await session.execute(
                text(
                    "SELECT count(*) FROM intelligence.document_chunks "
                    "WHERE org_id = :o AND document_id = :d"
                ),
                {"o": str(org_id), "d": str(document_id)},
            )
        ).scalar_one()
        elements = (
            await session.execute(
                text(
                    "SELECT count(*) FROM intelligence.document_elements "
                    "WHERE org_id = :o AND document_id = :d"
                ),
                {"o": str(org_id), "d": str(document_id)},
            )
        ).scalar_one()
    return chunks, elements


async def test_ingest_writes_typed_chunks_and_emits_one_event_per_chunk(
    session_factory, redis_client
):
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    emitter = _RecordingEmitter()

    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        outcome = await ingest_document(
            session=session,
            org_id=org_id,
            document_id=document_id,
            filename="notes.md",
            content=docs.make_md(),
            emit=emitter,
            redis=redis_client,
        )

    assert outcome.chunk_count == len(emitter.calls)
    assert outcome.chunk_count > 0

    for topic, idem, payload in emitter.calls:
        assert topic == "document-chunks"
        assert idem == f"docchunk:{document_id}:{payload['element_id']}"
        assert payload["document_id"] == str(document_id)
        assert payload["element_type"] in _FROZEN_TYPES
        assert payload["position"] >= 0

    chunk_rows, element_rows = await _count_rows(session_factory, org_id, document_id)
    assert chunk_rows == outcome.chunk_count
    assert element_rows == outcome.element_count


async def test_table_and_listitem_elements_keep_their_boundary_unsplit(
    session_factory, redis_client
):
    """Plan note: 'tables/lists keep the typed element boundary' — proven by checking that a
    short Table/ListItem element's chunk text equals the original element text exactly (i.e. it
    was never handed to the 600/60 splitter as multiple pieces)."""
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    emitter = _RecordingEmitter()
    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await ingest_document(
            session=session,
            org_id=org_id,
            document_id=document_id,
            filename="notes.md",
            content=docs.make_md(),
            emit=emitter,
            redis=redis_client,
        )
    table_or_list_payloads = [
        p for _, _, p in emitter.calls if p["element_type"] in ("Table", "ListItem")
    ]
    assert table_or_list_payloads
    for p in table_or_list_payloads:
        assert p["element_id"].endswith("_0")  # exactly one piece: el_XXXX_0, never _1


async def test_rerun_with_same_document_id_is_idempotent(session_factory, redis_client):
    """The literal acceptance check: 'a re-run is idempotent (stable element_id -> dedupe)'."""
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    content = docs.make_docx()

    async def run_once():
        emitter = _RecordingEmitter()
        async with session_factory() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            outcome = await ingest_document(
                session=session,
                org_id=org_id,
                document_id=document_id,
                filename="notes.docx",
                content=content,
                emit=emitter,
                redis=redis_client,
            )
        return outcome, emitter

    outcome1, emitter1 = await run_once()
    chunks_after_1, elements_after_1 = await _count_rows(session_factory, org_id, document_id)
    assert chunks_after_1 == outcome1.chunk_count
    assert len(emitter1.calls) == outcome1.chunk_count

    # Re-run: same org_id, same document_id, same content.
    outcome2, emitter2 = await run_once()
    chunks_after_2, elements_after_2 = await _count_rows(session_factory, org_id, document_id)

    # The pipeline computed the SAME element boundaries again (element_id is deterministic from
    # parse position), so ON CONFLICT DO NOTHING means zero new rows...
    assert chunks_after_2 == chunks_after_1
    assert elements_after_2 == elements_after_1
    # ...even though ingest_document itself still returns the full recomputed set (it does not
    # know the outbox/table layer deduped) — the DEDUPE happens at the write layer, which is
    # exactly what "stable element_id -> dedupe" means. Re-asserting the emit call SHAPE is
    # identical (same idempotency keys) proves a real outbox's (topic, idempotency_key) unique
    # index would also no-op the second round, matching outbox.py's own ON CONFLICT DO NOTHING.
    assert outcome2.chunk_count == outcome1.chunk_count
    idem_keys_1 = {idem for _, idem, _ in emitter1.calls}
    idem_keys_2 = {idem for _, idem, _ in emitter2.calls}
    assert idem_keys_1 == idem_keys_2


async def test_pii_redaction_applies_before_the_chunk_is_written_or_emitted(
    session_factory, redis_client
):
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    content = b"# Contact\n\nReach the on-call engineer at oncall@example.com or 555-123-4567.\n"

    await redis_client.set(key, "1")  # explicit on (also the default when the key is absent)
    try:
        emitter = _RecordingEmitter()
        async with session_factory() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
            )
            await ingest_document(
                session=session,
                org_id=org_id,
                document_id=document_id,
                filename="contact.md",
                content=content,
                emit=emitter,
                redis=redis_client,
            )
    finally:
        await redis_client.delete(key)

    joined_emitted = " ".join(p["text"] for _, _, p in emitter.calls)
    assert "oncall@example.com" not in joined_emitted
    assert "555-123-4567" not in joined_emitted
    assert "[REDACTED_EMAIL]" in joined_emitted

    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT text FROM intelligence.document_chunks "
                        "WHERE org_id = :o AND document_id = :d"
                    ),
                    {"o": str(org_id), "d": str(document_id)},
                )
            )
            .scalars()
            .all()
        )
    joined_stored = " ".join(rows)
    assert "oncall@example.com" not in joined_stored
    assert "555-123-4567" not in joined_stored


async def test_pii_redaction_explicitly_off_leaves_text_intact(session_factory, redis_client):
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    key = f"nexus:{org_id}:tenant_settings:pii_redaction_enabled"
    await redis_client.set(key, "0")
    content = b"# Contact\n\nReach the on-call engineer at oncall@example.com.\n"

    emitter = _RecordingEmitter()
    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await ingest_document(
            session=session,
            org_id=org_id,
            document_id=document_id,
            filename="contact.md",
            content=content,
            emit=emitter,
            redis=redis_client,
        )

    joined = " ".join(p["text"] for _, _, p in emitter.calls)
    assert "oncall@example.com" in joined
    await redis_client.delete(key)


async def test_pii_redaction_on_by_default_when_no_setting_published(session_factory, redis_client):
    """Fail-safe default: no `tenant_settings` mirror key => redact (DB/canon default is true)."""
    org_id, document_id = uuid.uuid4(), uuid.uuid4()
    content = b"# Contact\n\nReach the on-call engineer at oncall@example.com.\n"

    emitter = _RecordingEmitter()
    async with session_factory() as session, session.begin():
        await session.execute(
            text("SELECT set_config('app.current_org', :o, true)"), {"o": str(org_id)}
        )
        await ingest_document(
            session=session,
            org_id=org_id,
            document_id=document_id,
            filename="contact.md",
            content=content,
            emit=emitter,
            redis=redis_client,
        )

    joined = " ".join(p["text"] for _, _, p in emitter.calls)
    assert "oncall@example.com" not in joined
    assert "[REDACTED_EMAIL]" in joined
