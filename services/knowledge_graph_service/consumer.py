"""`TenantTopicSeedConsumer`: seeds each new org's `Topic` nodes from `tenant-events`.

Plan 5-6 gap 16. Plan 3.2 step 3 says the per-org `Topic` seed runs "in the seeder (3.4) and at
org provisioning". Both of those already go through ONE producer path:
`mytheus-backend/services/identity_tenant_service/provisioning.provision_organization`, which
writes the org row and a `tenant-events` `provisioned` outbox row in the same transaction
(`seed.seed_organization` calls it). So the producer side needs no new code and no new topic:
`identity-tenant-service` produces the fact, this consumer is the sole writer that turns it into
graph state. That is the canon sole-writer shape (other services produce events, the owner
writes its store) without inventing a schema `mytheus-events` would have to freeze.

Behaviour:
- `event_type == "provisioned"`: MERGE one `Topic` node per distinct taxonomy topic for the
  envelope header's `org_id` (never the payload's; `BaseConsumer` has already DLQ'd any event
  whose header `org_id` is missing or malformed).
- `suspended` / `deleted`: no graph effect here. Suspension does not change topics; deletion is
  the RTBF purge workflow's job (`12-compliance-and-privacy.md`), which will call into this
  service rather than being inferred from a lifecycle event.

Atomicity note. `BaseConsumer.process_raw` records the inbox row in the relational Postgres;
the graph lives in a different database (the AGE container), so the two are NOT one
transaction. Order is: graph MERGE commits inside `apply`, then the inbox row commits. A crash
between them redelivers the event and re-runs a MERGE that creates nothing - harmless because
`merge_topics` is idempotent. The reverse order could lose a seed, so it is not used.

`tenant-events` is log-compacted and this group starts at `earliest`, so orgs provisioned
before this consumer first ran (including `make seed`'s acme/globex) are backfilled on first
start, provided the backend relay has published their outbox rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession

from libs.shared.age import AgeConnection
from libs.shared.consumer import BaseConsumer, ConsumedEvent
from services.knowledge_graph_service.graph import merge_topics
from services.knowledge_graph_service.taxonomy import distinct_topics

logger = logging.getLogger(__name__)

GraphConnect = Callable[[], AbstractAsyncContextManager[AgeConnection]]


class TenantTopicSeedConsumer(BaseConsumer):
    group: ClassVar[str] = "knowledge-graph-service.topic-seed"
    topics: ClassVar[list[str]] = ["tenant-events"]
    producer_name: ClassVar[str] = "knowledge-graph-service"

    def __init__(
        self,
        *,
        graph_connect: GraphConnect,
        taxonomy_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._graph_connect = graph_connect
        self._taxonomy_version = taxonomy_version

    async def apply(self, evt: ConsumedEvent, session: AsyncSession) -> None:
        del session  # the graph is not in the relational database; see module docstring
        event_type = evt.body.get("event_type")
        if event_type != "provisioned":
            logger.info("topic-seed: ignoring tenant-events %s for org %s", event_type, evt.org_id)
            return
        names = distinct_topics(self._taxonomy_version)
        async with self._graph_connect() as conn:
            merged = await merge_topics(conn, evt.org_id, names)
        logger.info("topic-seed: merged %d Topic nodes for org %s", merged, evt.org_id)
