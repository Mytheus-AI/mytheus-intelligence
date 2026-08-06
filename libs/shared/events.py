"""Canonical Redpanda event envelope.

Mirrors `mytheus-backend`'s `libs/shared/events.py` `EventEnvelope` verbatim (same field set,
same `to_value()` composition). Reused here rather than reinvented because the outbox/inbox/
relay pattern only holds together if every producer in the program builds the same envelope
shape; see `mytheus-events` `CLAUDE.md` "The envelope" for the frozen field list this
implements (`EventEnvelopeHeader`, namespace `io.nexus.events`).

`payload` holds ONLY the topic-specific fields; `to_value()` composes the full Avro value
``{header, **payload}`` that the outbox stores and the relay serializes against
``{topic}-value``. Every topic in this program is partitioned by `org_id` (ADR-0017 is the
sole, narrow exception, for `transcript-chunks`/`agent-stream`, and only on the PRODUCER side
of those two topics — `embedding-service` is a consumer of `transcript-chunks`, never its
producer, and `embedding-events` (what this repo produces) keeps the plain `org_id` key).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_micros() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000)


@dataclass
class EventEnvelope:
    topic: str
    payload: dict[str, Any]  # topic-specific fields ONLY (no header)
    org_id: str
    producer: str  # owning service name (kebab-case)
    idempotency_key: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    occurred_at: int = field(default_factory=_now_micros)  # timestamp-micros
    correlation_id: str | None = None
    causation_id: str | None = None

    @property
    def schema_subject(self) -> str:
        return f"{self.topic}-value"

    @property
    def partition_key(self) -> str:
        return self.org_id

    def header(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "idempotency_key": self.idempotency_key,
            "org_id": self.org_id,
            "schema_version": self.schema_version,
            "producer": self.producer,
            "occurred_at": self.occurred_at,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }

    def to_value(self) -> dict[str, Any]:
        """The full Avro-serialisable event value: header + topic payload."""
        return {"header": self.header(), **self.payload}
