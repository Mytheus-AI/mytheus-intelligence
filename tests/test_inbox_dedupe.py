"""The Redis dedupe cache is scoped per consumer group, like the durable inbox (ADR-0031 D6).

Two groups consume the same topic (`decisions`: knowledge-graph-service and embedding-service;
`transcript-chunks`: embedding-service and search-service). With a group-less key, whichever
group handled a message second saw the first group's key and skipped its own effect.
"""

from __future__ import annotations

import uuid

import fakeredis.aioredis

from libs.shared.inbox import dedupe_key, is_duplicate


async def test_each_group_applies_a_fact_once_independently() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    org = str(uuid.uuid4())
    key = f"decision:{uuid.uuid4()}:confirmed"

    assert not await is_duplicate(redis, org, "decisions", key, consumer_group="kg")
    assert not await is_duplicate(redis, org, "decisions", key, consumer_group="embed")
    assert await is_duplicate(redis, org, "decisions", key, consumer_group="kg")
    assert await is_duplicate(redis, org, "decisions", key, consumer_group="embed")


def test_key_keeps_the_tenant_prefix() -> None:
    org = str(uuid.uuid4())
    assert dedupe_key(org, "decisions", "kg", "k").startswith(f"nexus:{org}:dedupe:decisions:")
