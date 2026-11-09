"""Budget awareness and `usage-events` metering for rag-service.

rag-service is the second sanctioned producer of `usage-events` (`agent_token` only, mytheus-docs
ADR-0011 section 3.1). Every LLM call goes through `LiteLLMRouter`, which meters it exactly
once through `UsageEmitter` with idempotency key
`usage:{org_id}:agent_token:{occurred_at}:{call_id}`, the same shape `agent-runtime-service`
uses. Produced straight to Redpanda with an idempotent producer: rag-service writes no
database, so there is nothing for an outbox to be atomic with.

`GateProbe` reads the same Redis budget mirror the router gates on, so the pipeline can decide
up front whether to run the LLM stages at all (degrade, never fail).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from libs.shared.avro_codec import AvroCodec
from libs.shared.budget import BudgetGate
from libs.shared.events import EventEnvelope
from services.rag_service.pipeline import BudgetState

logger = logging.getLogger(__name__)

PRODUCER = "rag-service"
EST_QUERY_TOKENS = 1500  # one expansion plus one answer, roughly


class GateProbe:
    def __init__(self, gate: BudgetGate) -> None:
        self._gate = gate

    async def state(self, org_id: UUID) -> tuple[BudgetState, float]:
        view = await self._gate.view(org_id)
        if view is None:
            return "block", 1.0  # unknown budget: no LLM work, retrieval still answers
        used = view.tokens_consumed / view.token_quota if view.token_quota > 0 else 1.0
        decision = view.decide(EST_QUERY_TOKENS)
        if view.enforcement != "enforce" and decision != "ok":
            logger.info("rag: budget would %s org=%s (alert_only)", decision, org_id)
            return "ok", used
        return decision, used


class UsageEmitter:
    def __init__(self, producer: Any, codec: AvroCodec) -> None:
        self._producer = producer
        self._codec = codec

    async def usage(
        self, *, org_id: UUID, user_id: UUID | None, tokens: int, model_tier: str, call_id: str
    ) -> None:
        env = EventEnvelope(
            topic="usage-events",
            org_id=str(org_id),
            producer=PRODUCER,
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
                logger.error("rag: usage-events delivery failed org=%s: %s", org_id, err)

        self._producer.produce(env.topic, key=str(org_id), value=value, on_delivery=on_delivery)
        self._producer.poll(0)

    def flush(self, timeout_s: float = 10.0) -> None:
        self._producer.flush(timeout_s)
