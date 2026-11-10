"""`usage-events` `agent_token` records for a service whose model calls are not agent windows
(the indexing consumers' never-record check, `libs/shared/index_guard.py`).

The same record and idempotency key `agent-runtime-service` and `rag-service` produce
(`usage:{org_id}:agent_token:{occurred_at}:{call_id}`), straight to Redpanda with an idempotent
producer, so the tenant's budget sees every model call wherever it was made. `tokens` carries
budget units (`libs/shared/cost.py`), as the router meters them.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from libs.shared.avro_codec import AvroCodec
from libs.shared.events import EventEnvelope

logger = logging.getLogger(__name__)


class UsageEmitter:
    def __init__(self, producer: Any, codec: AvroCodec, *, producer_name: str) -> None:
        self._producer = producer
        self._codec = codec
        self._name = producer_name

    async def usage(
        self, *, org_id: UUID, user_id: UUID | None, tokens: int, model_tier: str, call_id: str
    ) -> None:
        env = EventEnvelope(
            topic="usage-events",
            org_id=str(org_id),
            producer=self._name,
            idempotency_key="",
            payload={
                "user_id": str(user_id) if user_id else None,
                "event_type": "agent_token",
                "quantity": float(tokens),
                "source_id": call_id,
                "model_tier": model_tier,
            },
        )
        env.idempotency_key = f"usage:{org_id}:agent_token:{env.occurred_at}:{call_id}"
        value = self._codec.serialize_for(env.topic, env.to_value())

        def on_delivery(err: Any, _msg: Any) -> None:
            if err is not None:
                logger.error("%s: usage-events delivery failed org=%s: %s", self._name, org_id, err)

        self._producer.produce(env.topic, key=str(org_id), value=value, on_delivery=on_delivery)
        self._producer.poll(0)
