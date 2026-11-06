"""Real Schema-Registry-backed Avro ser/deser, shared by the relay (producer) and every
consumer in this repo.

Why this exists instead of importing the generated `mytheus_events` client package: that
package (`mytheus-events/clients/python/mytheus_events`) is pydantic dataclasses only — it has
no Avro serialize/deserialize methods, and the root `mytheus-events/pyproject.toml` has no
`[build-system]` table, so `pip install -e` against it does not currently work (a pre-existing
packaging gap in that repo, out of scope for this worktree to fix unilaterally). Rather than
block on that or hand-roll a second Avro encoder, this module talks directly to the REAL,
already-registered Schema Registry (`http://localhost:8085`, all 21 `{topic}-value` subjects
registered under BACKWARD per this task's scope note) via `confluent_kafka.schema_registry`,
which is the same underlying mechanism the generated client would use if it were installable.
Every schema this module resolves is the actual registered one, not a local copy, so a
producer/consumer built on it is provably compatible with the real contract.

`AvroCodec.serialize_for(topic, value)` and `.deserialize_for(topic, raw)` cache one
(De)serializer per subject (schema id resolution happens once per subject after warm-up, per
`mytheus-events` CLAUDE.md).
"""

from __future__ import annotations

from typing import Any, cast

from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext


class AvroCodec:
    def __init__(self, schema_registry_url: str) -> None:
        self._client = SchemaRegistryClient({"url": schema_registry_url})
        self._serializers: dict[str, AvroSerializer] = {}
        self._deserializers: dict[str, AvroDeserializer] = {}

    def _serializer(self, subject: str) -> AvroSerializer:
        if subject not in self._serializers:
            # AvroSerializer needs the writer schema up front for encoding; it does not fetch
            # "latest" lazily. We fetch the ALREADY-REGISTERED latest version for the subject
            # (never register a new one from here — schema evolution is `mytheus-events`'s
            # job) and fail loudly if the subject the plan requires is not registered.
            latest = self._client.get_latest_version(subject)
            self._serializers[subject] = AvroSerializer(
                self._client,
                schema_str=latest.schema.schema_str,
                conf={"auto.register.schemas": False},
            )
        return self._serializers[subject]

    def _deserializer(self, subject: str) -> AvroDeserializer:
        if subject not in self._deserializers:
            self._deserializers[subject] = AvroDeserializer(self._client)
        return self._deserializers[subject]

    def serialize_for(self, topic: str, value: dict[str, Any]) -> bytes:
        """`topic` is the bare topic name; the subject is `{topic}-value` (frozen naming)."""
        subject = f"{topic}-value"
        ctx = SerializationContext(topic, MessageField.VALUE)
        result = self._serializer(subject)(value, ctx)
        assert result is not None
        return result

    def deserialize_for(self, topic: str, raw: bytes) -> dict[str, Any]:
        subject = f"{topic}-value"
        ctx = SerializationContext(topic, MessageField.VALUE)
        result = self._deserializer(subject)(raw, ctx)
        assert result is not None
        return cast(dict[str, Any], result)
