"""`TranscriptChunkSearchConsumer` — the Postgres FTS arm's own read model over
`transcript-chunks`, subclassing `libs.shared.consumer.BaseConsumer` per this repo's established
convention (tasks 7-9: subclass, never re-implement dedupe/DLQ/offset handling).

--- Why this consumer exists instead of the plan's literal step 1 ---

Plan 3.10 step 1 shows an `ALTER TABLE transcript_chunks ADD COLUMN ... tsv tsvector GENERATED
ALWAYS AS (to_tsvector('english', text)) STORED` (and the same for `chat_messages`). This module
deliberately does NOT do that. `public.transcript_chunks` is owned by `transcription-service` in
`mytheus-realtime` — confirmed live before writing any code: a partitioned table (RANGE on
`created_at`), granted only to `mytheus_app` / `mytheus_admin` / `postgres`; `mytheus_ai_app` and
`mytheus_ai_relay` (this repo's roles) hold ZERO grants on it. The task brief is explicit: "If
FTS requires a tsvector column or index on someone else's table, that is a real cross-repo
ownership question, not a detail... Do NOT silently ALTER another service's table." This band has
already refused two equivalent second-writer temptations (task 1: teams/departments; task 4: the
team_members write) and both refusals were correct — this is the third.

Instead, this consumer builds `intelligence.transcript_chunk_search` (migration `d415f023220e`):
this repo's OWN FTS index, materialized ONLY from the real `transcript-chunks` Kafka topic — the
SAME topic `embedding-service` already consumes for the vector arm. This is not a workaround
bolted on to dodge a permission error; it is this program's own stated rule applied literally
("One writer per store. Cross-context reads via API or events, never a cross-service table
write" — the band's global constraints), and it is exactly what `mytheus-intelligence/CLAUDE.md`
already says search-service does: "Events in: transcript-chunks" / "Builds and owns its FTS
indexes" (not "alters another service's indexes"). The `GENERATED ALWAYS AS (to_tsvector(...))
STORED` column and the GIN index the plan calls for both exist here, just on this repo's own copy
of the searchable text, auto-maintained on every consumed insert exactly the way the plan's
generated column would auto-maintain on the source table.

--- `chat_messages` — BLOCKED, reported, not built around ---

The plan also asks for FTS over `chat_messages`, owned by `core-api` in `mytheus-backend`'s
`public` schema (confirmed live: FKs to `organizations`/`meetings`/`users`, same grant shape as
`transcript_chunks`). Unlike transcript_chunks, there is no way to build an equivalent read model
here: the full, live, registered topic catalog (21 subjects at
`http://localhost:8081/subjects`, cross-checked against `mytheus-events/schemas/`) has NO
`chat-messages` topic at all, and `chat_messages` rows are written directly by
`core_api/main.py` with no outbox/event production path whatsoever. Closing this gap needs
either (a) a new event topic — a canon change owned by `mytheus-events`/`mytheus-docs`, out of
this worktree's scope to invent unilaterally, or (b) a direct cross-repo SELECT grant on
`core-api`'s table — a real coordination ask between two different repos' owners, not a decision
one task in one worktree gets to make on its own. Reported as a routed gap in the task report,
matching this band's established discipline (task 2's Topic-node blocker, task 4's
transcript_chunks-has-no-write-path finding) rather than reached past silently.

--- `redacted` and `is_final` ---

The REGISTERED `transcript-chunks-value` schema (checked live, version 2, not just the local
`.avsc` file) carries a `redacted` field (default `false`) alongside the fields the local v1 file
shows. A chunk arriving with `redacted=true` is never indexed here: this mirrors
`ingestion-service`'s PII-redaction discipline (`pii.py`: redact before writing OR emitting)
applied to a field this topic already carries, rather than inventing a second redaction
mechanism for transcripts. `is_final=false` is also skipped, defensively — the schema's own doc
string says this topic carries "Finalized diarized utterances (finals only)", so this branch is
not expected to trigger against a conforming producer, but trusting an upstream doc string over a
field the message actually carries is exactly the kind of assumption this repo has been burned by
before (task 9's `payload_json` contract, stated plainly as an assumption rather than silently
trusted).
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent

_INSERT = text("""
    INSERT INTO intelligence.transcript_chunk_search
        (org_id, meeting_id, chunk_id, speaker_id, start_ms, end_ms, text)
    VALUES (:org_id, :meeting_id, :chunk_id, :speaker_id, :start_ms, :end_ms, :text)
    ON CONFLICT (org_id, chunk_id) DO NOTHING
    """)


class TranscriptChunkSearchConsumer(BaseConsumer):
    group = "search-service.transcript-index"
    topics = ["transcript-chunks"]
    # search-service produces no events (CLAUDE.md's service-map row: "Events out: (sync gRPC
    # from rag-service)" only), so `producer_name` is never used to stamp an emitted
    # EventEnvelope by this consumer — kept for BaseConsumer's contract and in case a future
    # task ever adds one (e.g. a search-index-refreshed observability event).
    producer_name = "search-service"

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        body = evt.body

        if not body.get("is_final", True):
            # Defensive only — see module docstring. This topic is documented finals-only.
            return

        if body.get("redacted", False):
            # Do not index text flagged as redacted. See module docstring.
            return

        await session.execute(
            _INSERT,
            {
                "org_id": str(evt.org_id),
                "meeting_id": str(body["meeting_id"]),
                "chunk_id": str(body["chunk_id"]),
                "speaker_id": body["speaker_id"],
                "start_ms": body["start_ms"],
                "end_ms": body["end_ms"],
                "text": body["text"],
            },
        )
        # No `emit_outbox` call: nothing downstream consumes a "search index updated" fact
        # today. `BaseConsumer.process_raw`'s own inbox insert (run right after `apply`
        # returns) is still the at-least-once dedupe record of THIS consumer group's progress;
        # the `ON CONFLICT DO NOTHING` above is a second, independent guard against a rare
        # replay that bypasses both the Redis cache and the inbox row — the same
        # belt-and-suspenders idempotency shape `embedding_service/consumer.py`'s stable Qdrant
        # point id gives the vector arm, applied here to a Postgres upsert instead of a vector
        # point.
