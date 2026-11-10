"""A2A round-trip latency harness (plan 10-13 3.3 acceptance: p95 < 1.5 s under E7 = 12
round-trips per meeting). Offline: the real requester (`A2AClient`), PEP (Redis-cached PDP),
responder and `ResponseRouter` over the in-memory bus, with realistic delays faked:

- each broker hop (produce -> consume, both directions): lognormal, median 8 ms, capped 60 ms;
- the responder's participant-scoped retrieval: lognormal, median 450 ms, sigma 0.4 (p95 about
  0.87 s, matching rag-service's measured CPU passage path, work-done 07 section 11.5 item 8),
  capped at 1.3 s;
- the PDP: 15 ms, behind the 60 s decision cache.

20 meetings x 12 concurrent round-trips (4 meetings at a time), seeded. The deadline is set
well past the budget so the measured value is the real round trip, not one cut off by the
deadline. Asserts p95 < 1500 ms and writes the numbers to stdout.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
import uuid
from typing import Any

from services.agent_runtime_service.a2a.pep import Pep
from tests.a2a_helpers import (
    FakePdp,
    Loopback,
    ScriptedRag,
    allow_all,
    lognormal_delay,
    rag_passage,
)

MEETINGS = 20
ROUND_TRIPS_PER_MEETING = 12  # E7
CONCURRENT_MEETINGS = 4
BUDGET_MS = 1500


class SlowPdp(FakePdp):
    async def authorize(self, org_id: uuid.UUID, role: str, qc: str, scope: str) -> str | None:
        await asyncio.sleep(0.015)
        return await super().authorize(org_id, role, qc, scope)


async def _meeting(rng: random.Random, samples: list[float]) -> None:
    org = uuid.uuid4()
    users = [uuid.uuid4() for _ in range(4)]
    team = str(uuid.uuid4())
    contexts: dict[uuid.UUID, dict[str, Any]] = {
        u: {"rbac_tier": "member", "team_ids": [team], "department_id": ""} for u in users
    }
    rag = ScriptedRag(
        {
            u: [rag_passage("The team chose the enterprise tier in July.", 0.9, kind="decision")]
            for u in users
        },
        delay=lambda: lognormal_delay(rng, 0.45, 0.4, 1.3),
    )
    pdp = SlowPdp(allow_all())
    bus = Loopback(
        org_contexts={org: contexts},
        rag=rag,
        pdp=pdp,
        hop_delay=lambda: lognormal_delay(rng, 0.008, 0.5, 0.06),
    )
    pep: Pep = bus.pep

    async def one(i: int) -> None:
        requester, responder = users[i % 4], users[(i + 1) % 4]
        start = time.monotonic()
        # The requester's PEP pre-publish check is part of the round trip it pays for.
        assert await pep.authorize(org, "member", "decision_lineage", "same_team")
        outcome = await bus.client.ask(
            org_id=org,
            requester=requester,
            responder=responder,
            query="Q3 pricing tier decision",
            query_class="decision_lineage",
            deadline_s=time.time() + 4.5,  # inside the responder's 5 s horizon
        )
        assert outcome.result == "answered", outcome
        samples.append((time.monotonic() - start) * 1000)

    await asyncio.gather(*(one(i) for i in range(ROUND_TRIPS_PER_MEETING)))
    await bus.drain()
    assert len(pdp.calls) <= 2  # the decision cache absorbed the rest (the responder also asks)


async def test_a2a_round_trip_p95_is_under_1_5_s():
    rng = random.Random(1013)
    samples: list[float] = []
    for batch in range(0, MEETINGS, CONCURRENT_MEETINGS):
        await asyncio.gather(
            *(_meeting(rng, samples) for _ in range(min(CONCURRENT_MEETINGS, MEETINGS - batch)))
        )
    assert len(samples) == MEETINGS * ROUND_TRIPS_PER_MEETING
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[int(0.95 * len(ordered)) - 1]
    p99 = ordered[int(0.99 * len(ordered)) - 1]
    print(
        f"\nA2A round trip over {len(samples)} runs: p50 {p50:.0f} ms, p95 {p95:.0f} ms, "
        f"p99 {p99:.0f} ms, max {ordered[-1]:.0f} ms"
    )
    assert p95 < BUDGET_MS
