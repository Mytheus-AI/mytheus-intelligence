"""BaseConsumer: a failed apply releases the Redis dedupe mark, so the redelivery is applied.

The mark is set before `apply` (cheap duplicate short-circuit). Before this fix a failure left it
set for 24 hours, so after a restart the redelivered event was dropped as a "duplicate" without
ever having been applied, by every consumer in this repo.
"""

from __future__ import annotations

from typing import Any, ClassVar

import fakeredis.aioredis
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.consumer import BaseConsumer, ConsumedEvent
from libs.shared.inbox import dedupe_key
from tests.test_erasure_consumers import ORG, Db, Dlq, JsonCodec, admin_value, feed


class Flaky(BaseConsumer):
    group: ClassVar[str] = "test.flaky"
    topics: ClassVar[list[str]] = ["admin-events"]

    def __init__(self, *, failures: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.failures = failures
        self.applied = 0

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        if self.failures:
            self.failures -= 1
            raise ConnectionError("store down")
        self.applied += 1


@pytest.mark.asyncio
async def test_a_failed_apply_releases_the_mark_and_the_redelivery_applies() -> None:
    redis, db = fakeredis.aioredis.FakeRedis(), Db()
    consumer = Flaky(
        failures=1,
        session_factory=db,  # type: ignore[arg-type]
        redis=redis,
        codec=JsonCodec(),  # type: ignore[arg-type]
        dlq_producer=Dlq(),  # type: ignore[arg-type]
    )
    value = admin_value(ORG, kind="meeting", target=str(ORG))
    with pytest.raises(ConnectionError):
        await feed(consumer, value)
    idem = value["header"]["idempotency_key"]
    assert not await redis.exists(dedupe_key(str(ORG), "admin-events", Flaky.group, idem))
    assert await feed(consumer, value) == "applied"
    assert consumer.applied == 1
    assert await feed(consumer, value) == "duplicate"
