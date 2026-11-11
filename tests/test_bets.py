"""Decisions as bets: the after-meeting proposer (`bets.py`), deterministic.

- grounding: a proposal survives only when its quote is in a transcript line; that line's start
  becomes `quote_start_ms`;
- filters: unknown refs, low confidence, a bad direction for the kind, "reach" without a level,
  dates out of range (missing -> 90 days; more than 3 years -> dropped), one per decision,
  never-record text;
- the proposer calls nothing without a recorded decision or a transcript, and emits nothing
  when nothing is kept;
- the loop: one record (DecisionAgent / decision_candidate / bets:{generation}, user = host),
  claimed once, retried on a retryable failure, flushed.
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from typing import Any

import fakeredis
import httpx

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from services.agent_runtime_service import bets
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.minutes import Resolver, SourceUnavailable, assign_labels

HOST = uuid.uuid4()
OTHER = uuid.uuid4()
D1, D2 = uuid.uuid4(), uuid.uuid4()
DAY = date(2026, 11, 4)


def body(**over: Any) -> dict[str, Any]:
    b: dict[str, Any] = {
        "meeting_id": str(uuid.uuid4()),
        "title": "Billing",
        "scheduled_start": "2026-11-04T15:00:00Z",
        "host_user_id": str(HOST),
        "participants": [
            {"user_id": str(HOST), "name": "Maya Patel", "attendance": "live"},
            {"user_id": str(OTHER), "name": "Raj Mehta", "attendance": "live"},
        ],
        "turns": [
            {
                "speaker_id": str(HOST),
                "start_ms": 0,
                "end_ms": 4000,
                "text": "Annual billing from January.",
            },
            {
                "speaker_id": str(OTHER),
                "start_ms": 65000,
                "end_ms": 70000,
                "text": "This should cut churn by 3 points by March, I am sure.",
            },
        ],
        "decisions": [
            {"id": str(D1), "content": "Annual billing from January", "status": "confirmed"},
            {"id": str(D2), "content": "Hire a support lead", "status": "proposed"},
        ],
    }
    b.update(over)
    return b


def source(**over: Any) -> bets.BetSource:
    s = bets.parse_bet_source(body(**over), date(2026, 1, 1))
    assert s is not None
    return s


def item(**over: Any) -> dict[str, Any]:
    e = {
        "ref": "D1",
        "statement": "Annual billing cuts churn by 3 points by March",
        "kind": "metric",
        "metric": "Monthly churn",
        "direction": "decrease",
        "magnitude": 3,
        "unit": "points",
        "check_by": "2027-03-31",
        "date_said": True,
        "source": "metrics",
        "category": "pricing",
        "assumption": "Participant 2 says monthly customers take the discount",
        "quote": "this should cut churn by 3 points by March",
        "confidence": 0.85,
    }
    e.update(over)
    return e


def parse(
    items: list[dict[str, Any]], policy: NeverRecordPolicy | None = None
) -> list[dict[str, Any]]:
    s = source()
    labels = assign_labels(s.source)
    resolver = Resolver.build(s.source, labels, policy or NeverRecordPolicy.from_org(None))
    out = bets.parse_expectations(
        json.dumps({"expectations": items}),
        source=s.source,
        resolver=resolver,
        meeting_date=s.meeting_date,
    )
    assert out is not None
    return out


def test_source_carries_the_meeting_date():
    assert source().meeting_date == DAY
    assert bets.parse_bet_source(body(scheduled_start=None), DAY) is not None
    assert bets.parse_bet_source({"nope": 1}, DAY) is None


def test_a_grounded_proposal_is_kept_with_its_moment_and_names():
    [e] = parse([item()])
    assert e["decision_id"] == str(D1) and e["direction"] == "decrease"
    assert e["quote_start_ms"] == 65000 and e["check_date"] == "2027-03-31"
    assert e["assumption"] == "Raj Mehta says monthly customers take the discount"


def test_ungrounded_low_confidence_and_malformed_proposals_are_dropped():
    assert parse([item(quote="churn will fall a lot")]) == []  # never said
    assert parse([item(confidence=0.4)]) == []
    assert parse([item(ref="D9")]) == []
    assert parse([item(kind="event")]) == []  # an event cannot "decrease"
    assert parse([item(direction="reach", magnitude=None)]) == []
    assert parse([item(check_by="2031-01-01")]) == []
    assert parse([item(), item(statement="Again")]) and len(parse([item(), item()])) == 1


def test_a_missing_date_defaults_to_ninety_days():
    [e] = parse([item(check_by=None, date_said=False)])
    assert e["check_date"] == "2027-02-02" and e["date_said"] is False


def test_never_record_text_is_dropped():
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("churn",)))
    assert parse([item()], policy) == []


def test_unparseable_answer_is_none():
    s = source()
    resolver = Resolver.build(s.source, assign_labels(s.source), NeverRecordPolicy.from_org(None))
    assert (
        bets.parse_expectations("not json", source=s.source, resolver=resolver, meeting_date=DAY)
        is None
    )


class FakeRouter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> Any:
        self.calls.append(kw)
        text = self.text

        class Out:
            pass

        o = Out()
        o.text = text  # type: ignore[attr-defined]
        return o


class Src:
    def __init__(self, value: bets.BetSource | None, fail: int = 0) -> None:
        self.value = value
        self.fail = fail

    async def source(self, _org: uuid.UUID, _m: uuid.UUID) -> bets.BetSource | None:
        if self.fail:
            self.fail -= 1
            raise SourceUnavailable("down")
        return self.value


def proposer(router: FakeRouter, src: Src) -> bets.BetProposer:
    return bets.BetProposer(PromptLoader(PROMPTS_DIR), router, source=src)  # type: ignore[arg-type]


async def test_proposer_skips_the_call_without_decisions_or_transcript():
    router = FakeRouter(json.dumps({"expectations": [item()]}))
    assert (
        await proposer(router, Src(source(decisions=[]))).propose(uuid.uuid4(), uuid.uuid4(), "g")
        is None
    )
    assert (
        await proposer(router, Src(source(turns=[]))).propose(uuid.uuid4(), uuid.uuid4(), "g")
        is None
    )
    assert router.calls == []


async def test_proposer_builds_the_record_for_the_host():
    router = FakeRouter(json.dumps({"expectations": [item()]}))
    p = await proposer(router, Src(source())).propose(uuid.uuid4(), uuid.uuid4(), "g1")
    assert p is not None and p.user_id == HOST
    assert p.payload["prompt_version"] == "decision_bet@v1"
    [call] = router.calls
    assert call["tier"] == "extract" and "Meeting date: 2026-11-04" in call["user"]
    assert "Maya" not in call["user"] + call["cached_prefix"]  # labels only
    r = p.result("g1")
    assert r.kind == "decision_candidate" and r.window_id == "bets:g1"
    empty = FakeRouter(json.dumps({"expectations": []}))
    assert await proposer(empty, Src(source())).propose(uuid.uuid4(), uuid.uuid4(), "g") is None


class Sink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.flushed = 0

    async def agent_result(self, **kw: Any) -> None:
        self.records.append(kw)

    def flush(self, timeout_s: float = 10.0) -> None:
        self.flushed += 1


def ended(org: uuid.UUID, meeting: uuid.UUID) -> dict[str, Any]:
    return {
        "event_type": "ended",
        "meeting_id": str(meeting),
        "header": {"org_id": str(org), "idempotency_key": f"meeting:{meeting}:ended"},
    }


async def _no_sleep(_s: float) -> None:
    return None


async def test_loop_emits_once_retries_and_claims():
    org, meeting = uuid.uuid4(), uuid.uuid4()
    router = FakeRouter(json.dumps({"expectations": [item()]}))
    sink = Sink()
    claims = fakeredis.aioredis.FakeRedis()
    loop = bets.BetsLoop(
        proposer(router, Src(source(), fail=1)), sink, claims=claims, sleep=_no_sleep
    )
    assert await loop.handle(ended(org, meeting)) == 1
    [rec] = sink.records
    assert rec["agent"] == "DecisionAgent" and rec["user_id"] == HOST
    assert rec["window_id"].startswith("bets:") and sink.flushed == 1
    assert await loop.handle(ended(org, meeting)) == 0  # claimed: no second bill
    assert len(router.calls) == 1
    assert await loop.handle({"event_type": "started"}) == 0


async def test_http_source_maps_status_codes():
    def handler(request: httpx.Request) -> httpx.Response:
        if "missing" in str(request.url):
            return httpx.Response(404)
        if "busy" in str(request.url):
            return httpx.Response(409)
        return httpx.Response(200, json=body())

    t = httpx.MockTransport(handler)
    src = bets.HttpBetSource("http://core", "tok", transport=t)
    got = await src.source(uuid.uuid4(), uuid.uuid4())
    assert got is not None and got.meeting_date == DAY
    src_missing = bets.HttpBetSource("http://core/missing", "tok", transport=t)
    assert await src_missing.source(uuid.uuid4(), uuid.uuid4()) is None
    busy = bets.HttpBetSource("http://core/busy", "tok", transport=t)
    try:
        await busy.source(uuid.uuid4(), uuid.uuid4())
    except SourceUnavailable:
        pass
    else:  # pragma: no cover
        raise AssertionError("409 must be retryable")
