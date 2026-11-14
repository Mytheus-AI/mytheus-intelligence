"""Ask Myte v2 (mytheus-docs ADR-0049), intelligence half: per-person metering and the daily
allowance (R3), latency telemetry (R4), the new proposal kinds (F1/F3/F4), memories (F4),
attachments (F7) and peer-answer mode (F1). Real route, loop and router; a fake bff, a fake
identity-tenant-service, fakeredis and a scripted model. No network, no model calls."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import fakeredis
import httpx
import pytest
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from services.rag_service import main as rag_main
from services.rag_service.myte import proposals
from services.rag_service.myte.allowance import (
    OVER_ALLOWANCE,
    UID_TTL_S,
    UNITS_TTL_S,
    Allowance,
    daily_units,
    identity_url,
    uid_key,
    units_key,
)
from services.rag_service.myte.contracts import MyteRequest
from services.rag_service.myte.sources import attachment_drafts, passages
from services.rag_service.myte.tools import PROPOSAL_KINDS, model_tools
from tests.conftest import sign_test_token
from tests.myte_kit import (
    FIXED_NOW,
    FakeBff,
    ScriptedModel,
    Sink,
    check_stream,
    make_loop,
    parse_sse,
)

ORG = uuid.uuid4()
USER = uuid.UUID("99999999-aaaa-4bbb-8ccc-999999999999")
MAYA = "d0000000-0000-4000-8000-000000000001"
ARJUN = "d0000000-0000-4000-8000-000000000002"
M1 = "a0000000-0000-4000-8000-000000000001"
D1 = "b0000000-0000-4000-8000-000000000001"
PEOPLE = {
    "items": [
        {"id": MAYA, "name": "Maya Chen", "role": "Product manager"},
        {"id": ARJUN, "name": "Arjun Rao", "role": "Engineer"},
    ]
}
SEARCH = {
    "items": [
        {"kind": "meeting", "id": M1, "text": "Pricing review"},
        {
            "kind": "decision",
            "id": D1,
            "text": "No discounts above 15%",
            "meeting_id": M1,
            "meeting_title": "Pricing review",
        },
    ]
}


class FakeIdentity:
    def __init__(self, status: int = 200, body: Any = None, down: bool = False) -> None:
        self.status = status
        self.body = body if body is not None else {"id": str(USER), "email": "me@example.com"}
        self.down = down
        self.calls: list[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.down:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(self.status, json=self.body)


def allowance(
    identity: FakeIdentity | None = None,
    redis: Any = None,
    cap: int = 300_000,
) -> tuple[Allowance, Any, FakeIdentity]:
    identity = identity or FakeIdentity()
    redis = redis if redis is not None else fakeredis.aioredis.FakeRedis()
    http = httpx.AsyncClient(transport=httpx.MockTransport(identity.handler))
    return (
        Allowance(redis, http, base_url="http://identity:8000", cap=cap, clock=lambda: FIXED_NOW),
        redis,
        identity,
    )


def run(
    turns: list[dict[str, Any]],
    body: dict[str, Any],
    fixtures: dict[str, Any] | None = None,
    token: str | None = None,
    **kw: Any,
) -> tuple[list[tuple[str, dict[str, Any]]], FakeBff, ScriptedModel]:
    bff = FakeBff(fixtures=fixtures or {})
    model = ScriptedModel(list(turns))
    client = TestClient(rag_main.build_app(myte=make_loop(model, bff, **kw)))
    res = client.post(
        "/api/rag/myte",
        content=json.dumps(body),
        headers={"Authorization": f"Bearer {token or sign_test_token(ORG)}"},
    )
    assert res.status_code == 200
    return parse_sse(res.text), bff, model


def of(events: list[tuple[str, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [d for n, d in events if n == name]


def first_round(model: ScriptedModel) -> dict[str, Any]:
    return next(c for c in model.calls if c.get("stream"))


def context_of(model: ScriptedModel) -> str:
    return str(first_round(model)["messages"][0]["content"][1]["text"])


def system_of(model: ScriptedModel) -> str:
    return str(first_round(model)["messages"][0]["content"][0]["text"])


def offered(model: ScriptedModel) -> set[str]:
    return {t["function"]["name"] for t in first_round(model).get("tools") or []}


# --- R3: who is asking -----------------------------------------------------------------------


async def test_the_asker_is_resolved_with_their_own_bearer_and_cached_ten_minutes():
    allow, redis, identity = allowance()
    assert await allow.user_id(ORG, "user_abc", "Bearer tok") == USER
    assert await allow.user_id(ORG, "user_abc", "Bearer tok") == USER
    assert len(identity.calls) == 1
    call = identity.calls[0]
    assert str(call.url) == "http://identity:8000/users/me"
    assert call.headers["authorization"] == "Bearer tok"
    key = uid_key(ORG, "user_abc")
    assert key == f"nexus:{ORG}:myte:uid:user_abc"
    assert (await redis.get(key)).decode() == str(USER)
    assert 0 < await redis.ttl(key) <= UID_TTL_S


@pytest.mark.parametrize(
    "identity",
    [
        FakeIdentity(status=401),
        FakeIdentity(status=404),
        FakeIdentity(status=500),
        FakeIdentity(body={"email": "no id"}),
        FakeIdentity(down=True),
    ],
)
async def test_an_identity_failure_is_unknown_not_a_refusal(identity, caplog):
    allow, redis, _ = allowance(identity)
    with caplog.at_level(logging.WARNING):
        assert await allow.user_id(ORG, "user_abc", "Bearer tok") is None
    assert "identity lookup" in caplog.text
    assert await redis.get(uid_key(ORG, "user_abc")) is None
    assert await allow.over(ORG, None) is False


async def test_no_clerk_subject_means_no_lookup():
    allow, _, identity = allowance()
    assert await allow.user_id(ORG, None, "Bearer tok") is None
    assert identity.calls == []


def test_env_defaults(monkeypatch):
    monkeypatch.delenv("IDENTITY_URL", raising=False)
    monkeypatch.delenv("MYTE_USER_DAILY_UNITS", raising=False)
    assert identity_url() == "http://identity-tenant-service:8000"
    assert daily_units() == 300_000
    monkeypatch.setenv("IDENTITY_URL", "http://id:9000/")
    monkeypatch.setenv("MYTE_USER_DAILY_UNITS", "1234")
    assert identity_url() == "http://id:9000"
    assert daily_units() == 1234
    monkeypatch.setenv("MYTE_USER_DAILY_UNITS", "lots")
    assert daily_units() == 300_000


def test_every_model_call_is_metered_to_the_person():
    allow, _, _ = allowance()
    sink = Sink()
    events, _, _ = run(
        [{"tools": [{"tool": "list_my_open_actions", "args": {}}]}, {"text": "Nothing due."}],
        {"question": "What's on me?"},
        sink=sink,
        allowance=allow,
    )
    assert check_stream(events) == []
    assert len(sink.calls) == 3  # two rounds + follow-ups
    assert {c["user_id"] for c in sink.calls} == {USER}


def test_an_unresolved_asker_still_gets_an_answer_on_the_workspace_budget():
    allow, _, _ = allowance(FakeIdentity(down=True))
    sink = Sink()
    events, _, _ = run([{"text": "Hello."}], {"question": "Hi"}, sink=sink, allowance=allow)
    assert of(events, "done") and not of(events, "error")
    assert {c["user_id"] for c in sink.calls} == {None}


# --- R3: the daily allowance -----------------------------------------------------------------


def test_each_rounds_units_are_added_to_todays_counter():
    allow, redis, _ = allowance()
    sink = Sink()
    run(
        [{"tools": [{"tool": "list_my_open_actions", "args": {}}]}, {"text": "Done."}],
        {"question": "What's on me?"},
        sink=sink,
        allowance=allow,
    )
    import asyncio

    key = units_key(ORG, USER, "2026-11-14")
    assert key == f"nexus:{ORG}:myte:units:{USER}:2026-11-14"
    used = asyncio.run(redis.get(key))
    assert int(used) == sum(c["tokens"] for c in sink.calls) > 0
    ttl = asyncio.run(redis.ttl(key))
    assert UNITS_TTL_S - 60 < ttl <= UNITS_TTL_S


def test_over_the_allowance_is_a_budget_error_before_any_model_call():
    import asyncio

    allow, redis, _ = allowance(cap=1000)
    asyncio.run(redis.set(units_key(ORG, USER, "2026-11-14"), 1000))
    events, bff, model = run([{"text": "never"}], {"question": "Hi"}, allowance=allow)
    assert events == [("error", {"reason": "budget", "message": OVER_ALLOWANCE})]
    assert "tomorrow" in OVER_ALLOWANCE
    assert model.calls == [] and bff.calls == []


def test_the_allowance_is_checked_before_each_round():
    # One round costs more than the cap: the second round is refused.
    allow, _, _ = allowance(cap=1)
    events, _, model = run(
        [{"tools": [{"tool": "list_my_open_actions", "args": {}}]}, {"text": "never"}],
        {"question": "What's on me?"},
        allowance=allow,
    )
    assert of(events, "error") == [{"reason": "budget", "message": OVER_ALLOWANCE}]
    assert len([c for c in model.calls if c.get("stream")]) == 1
    assert check_stream(events) == []


async def test_yesterdays_units_do_not_count_today():
    redis = fakeredis.aioredis.FakeRedis()
    allow, _, _ = allowance(redis=redis, cap=10)
    await redis.set(units_key(ORG, USER, "2026-11-13"), 999)
    assert await allow.over(ORG, USER) is False
    await allow.add(ORG, USER, 10)
    assert await allow.over(ORG, USER) is True


async def test_a_cap_of_zero_turns_the_per_person_cap_off():
    allow, redis, _ = allowance(cap=0)
    await redis.set(units_key(ORG, USER, "2026-11-14"), 10**9)
    assert await allow.over(ORG, USER) is False


async def test_redis_down_fails_open():
    class Broken:
        async def get(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError("down")

        set = incrby = expire = get

    allow, _, identity = allowance(redis=Broken())
    assert await allow.user_id(ORG, "user_abc", "Bearer t") == USER  # looked up, not cached
    assert await allow.over(ORG, USER) is False
    await allow.add(ORG, USER, 5)  # no raise
    assert len(identity.calls) == 1


# --- R4: latency ------------------------------------------------------------------------------


def _count(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(f"{name}_count", labels) or 0.0


def test_latency_is_observed_and_logged_once_per_answer(caplog):
    before = (
        _count("myte_first_step_seconds", {"mode": "ask"}),
        _count("myte_first_text_seconds", {"mode": "ask"}),
        _count("myte_answer_seconds", {"mode": "ask", "outcome": "done"}),
    )
    with caplog.at_level(logging.INFO, logger="services.rag_service.myte.latency"):
        run(
            [{"tools": [{"tool": "list_my_open_actions", "args": {}}]}, {"text": "Done."}],
            {"question": "What's on me?"},
        )
    after = (
        _count("myte_first_step_seconds", {"mode": "ask"}),
        _count("myte_first_text_seconds", {"mode": "ask"}),
        _count("myte_answer_seconds", {"mode": "ask", "outcome": "done"}),
    )
    assert [a - b for a, b in zip(after, before, strict=True)] == [1, 1, 1]
    lines = [r for r in caplog.records if r.getMessage().startswith("myte answer")]
    assert len(lines) == 1
    msg = lines[0].getMessage()
    assert "outcome=done" in msg and "rounds=2" in msg and "tools=1" in msg
    assert "first_step_s=" in msg and "first_text_s=" in msg and "answer_s=" in msg


def test_an_error_answer_is_timed_with_its_reason():
    before = _count("myte_answer_seconds", {"mode": "ask", "outcome": "budget"})
    run([{"text": "x"}], {"question": "Hi"}, budget=_blocked_gate())
    assert _count("myte_answer_seconds", {"mode": "ask", "outcome": "budget"}) == before + 1


def _blocked_gate() -> Any:
    from tests.myte_kit import gate

    return gate(consumed=10**9, quota=1, cap=1)


# --- F3/F1/F4: proposal kinds ------------------------------------------------------------------


NOW = datetime(2026, 11, 14, 15, 30, tzinfo=UTC)


def known() -> proposals.Known:
    return proposals.Known(
        meetings={M1: "Pricing review"},
        decisions={D1: ("No discounts above 15%", M1)},
        people={MAYA: "Maya Chen", ARJUN: "Arjun Rao"},
        asker_id=str(USER),
    )


def test_the_tool_offers_every_adr_kind():
    assert set(PROPOSAL_KINDS) == {
        "task",
        "promise",
        "attendance",
        "agenda_item",
        "schedule_meeting",
        "invite_people",
        "slack_post",
        "export_decisions",
        "share_decision",
        "ask_myte",
        "memory",
    }
    tool = next(t for t in model_tools([]) if t["function"]["name"] == "propose_action")
    assert tool["function"]["parameters"]["properties"]["kind"]["enum"] == list(PROPOSAL_KINDS)


def test_schedule_meeting_is_normalised_with_names():
    p = proposals.build(
        {
            "kind": "schedule_meeting",
            "title": "Pricing sync",
            "start": "2026-11-16T10:00:00+05:30",
            "end": "2026-11-16T10:30:00+05:30",
            "attendee_ids": [MAYA, MAYA, ARJUN],
        },
        known(),
        NOW,
    )
    assert p["start"] == "2026-11-16T10:00:00+05:30" and p["end"] == "2026-11-16T10:30:00+05:30"
    assert p["attendee_ids"] == [MAYA, ARJUN]
    assert p["attendee_names"] == ["Maya Chen", "Arjun Rao"]
    assert p["title"] == "Pricing sync"


@pytest.mark.parametrize(
    "args,error",
    [
        ({"start": "2026-11-16T10:00:00", "end": "2026-11-16T10:30:00+05:30"}, "offset"),
        ({"start": "tomorrow", "end": "2026-11-16T10:30:00+05:30"}, "ISO 8601"),
        ({"start": "2026-11-16T10:30:00+00:00", "end": "2026-11-16T10:00:00+00:00"}, "after"),
        ({"start": "2026-11-16T10:00:00+00:00", "end": "2026-11-18T10:00:00+00:00"}, "24 hours"),
        ({"start": "2026-11-01T10:00:00+00:00", "end": "2026-11-01T11:00:00+00:00"}, "past"),
        (
            {
                "start": "2026-11-16T10:00:00+00:00",
                "end": "2026-11-16T11:00:00+00:00",
                "attendee_ids": [str(uuid.uuid4())],
            },
            "find_people",
        ),
        ({"start": "2026-11-16T10:00:00Z", "end": "2026-11-16T11:00:00Z", "title": ""}, "title"),
    ],
)
def test_schedule_meeting_refusals(args, error):
    with pytest.raises(proposals.Invalid, match=error):
        proposals.build({"kind": "schedule_meeting", "title": "Sync", **args}, known(), NOW)


def test_invite_people_needs_a_known_meeting_and_known_people():
    p = proposals.build(
        {"kind": "invite_people", "meeting_id": M1, "user_ids": [ARJUN]}, known(), NOW
    )
    assert p["user_ids"] == [ARJUN] and p["user_names"] == ["Arjun Rao"]
    assert p["title"] == "Invite Arjun Rao to Pricing review"
    for bad in (
        {"meeting_id": str(uuid.uuid4()), "user_ids": [ARJUN]},
        {"meeting_id": M1, "user_ids": []},
        {"meeting_id": M1, "user_ids": [str(uuid.uuid4())]},
        {"user_ids": [ARJUN]},
    ):
        with pytest.raises(proposals.Invalid):
            proposals.build({"kind": "invite_people", **bad}, known(), NOW)


def test_slack_post_text_and_channel():
    p = proposals.build(
        {"kind": "slack_post", "text": "  Launch is Friday.  ", "channel": "Product"}, known(), NOW
    )
    assert p["text"] == "Launch is Friday." and p["channel"] == "#product"
    assert p["title"] == "Post to Slack in #product"
    p = proposals.build({"kind": "slack_post", "text": "x" * 5000}, known(), NOW)
    assert len(p["text"]) == 4000 and "channel" not in p
    with pytest.raises(proposals.Invalid, match="text"):
        proposals.build({"kind": "slack_post", "text": " "}, known(), NOW)
    with pytest.raises(proposals.Invalid, match="channel"):
        proposals.build({"kind": "slack_post", "text": "hi", "channel": "my team"}, known(), NOW)


def test_export_decisions():
    p = proposals.build(
        {"kind": "export_decisions", "destination": "notion", "meeting_id": M1}, known(), NOW
    )
    assert p["title"] == "Export decisions from Pricing review to Notion"
    p = proposals.build(
        {"kind": "export_decisions", "destination": "slack", "decision_ids": [D1]}, known(), NOW
    )
    assert p["decision_ids"] == [D1] and p["title"] == "Export 1 decision to Slack"
    for bad in (
        {"destination": "drive", "meeting_id": M1},
        {"destination": "notion"},
        {"destination": "notion", "decision_ids": [str(uuid.uuid4())]},
        {"destination": "notion", "decision_ids": "x"},
    ):
        with pytest.raises(proposals.Invalid):
            proposals.build({"kind": "export_decisions", **bad}, known(), NOW)


def test_share_decision_carries_its_meeting():
    p = proposals.build({"kind": "share_decision", "decision_id": D1}, known(), NOW)
    assert p == {
        "id": p["id"],
        "kind": "share_decision",
        "decision_id": D1,
        "meeting_id": M1,
        "title": "Share: No discounts above 15%",
    }
    with pytest.raises(proposals.Invalid):
        proposals.build({"kind": "share_decision", "decision_id": str(uuid.uuid4())}, known(), NOW)


def test_ask_myte():
    p = proposals.build(
        {
            "kind": "ask_myte",
            "person_id": MAYA,
            "person_name": "Maya Chen",
            "question": "Is the pricing copy final?",
        },
        known(),
        NOW,
    )
    assert p["person_id"] == MAYA and p["person_name"] == "Maya Chen"
    assert p["question"] == "Is the pricing copy final?" and p["title"] == "Ask Maya Chen's Myte"
    # the name falls back to the source's name
    p = proposals.build({"kind": "ask_myte", "person_id": ARJUN, "question": "Q?"}, known(), NOW)
    assert p["person_name"] == "Arjun Rao"
    for bad in (
        {"person_id": str(uuid.uuid4()), "question": "Q?"},
        {"person_id": MAYA, "question": " "},
        {"person_id": str(USER), "question": "Q?"},
    ):
        k = known()
        k.people[str(USER)] = "Me"
        with pytest.raises(proposals.Invalid):
            proposals.build({"kind": "ask_myte", **bad}, k, NOW)


def test_memory():
    p = proposals.build({"kind": "memory", "text": " Prefers   short answers "}, known(), NOW)
    assert p["text"] == "Prefers short answers"
    assert p["title"] == "Remember: Prefers short answers"
    with pytest.raises(proposals.Invalid, match="300"):
        proposals.build({"kind": "memory", "text": "x" * 301}, known(), NOW)
    with pytest.raises(proposals.Invalid):
        proposals.build({"kind": "memory", "text": ""}, known(), NOW)


def test_unknown_kind_and_v1_kinds_still_work():
    with pytest.raises(proposals.Invalid, match="kind must be one of"):
        proposals.build({"kind": "delete_everything"}, known(), NOW)
    p = proposals.build({"kind": "task", "title": "Upload", "due_date": "2026-11-16"}, known(), NOW)
    assert p["title"] == "Upload" and p["due_date"] == "2026-11-16"
    p = proposals.build(
        {"kind": "attendance", "title": "Pricing", "meeting_id": M1, "choice": "agent"},
        known(),
        NOW,
    )
    assert p["choice"] == "agent"


def test_a_schedule_proposal_end_to_end_uses_ids_from_find_people():
    events, _, model = run(
        [
            {"tools": [{"tool": "find_people", "args": {"query": "Maya"}}]},
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {
                            "kind": "schedule_meeting",
                            "title": "Pricing",
                            "start": "2026-11-16T10:00:00+00:00",
                            "end": "2026-11-16T10:30:00+00:00",
                            "attendee_ids": [MAYA],
                        },
                    }
                ]
            },
            {"text": "I suggested a 30-minute meeting with Maya on Monday."},
        ],
        {"question": "Set up 30 minutes with Maya Monday at 10 UTC"},
        fixtures={"find_people": PEOPLE},
    )
    assert check_stream(events) == []
    (p,) = of(events, "proposal")
    assert p["kind"] == "schedule_meeting" and p["attendee_names"] == ["Maya Chen"]
    assert of(events, "done")[0]["abstained"] is False


def test_an_invented_id_goes_back_to_the_model_and_no_card_is_shown():
    events, _, model = run(
        [
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {"kind": "ask_myte", "person_id": MAYA, "question": "Q?"},
                    }
                ]
            },
            {"text": "I need to find Maya first."},
        ],
        {"question": "Ask Maya's Myte about pricing"},
    )
    assert of(events, "proposal") == []
    tool_msgs = [m for m in model.calls[-2]["messages"] if m.get("role") == "tool"]
    assert "find_people" in tool_msgs[-1]["content"]


def test_a_person_in_scope_can_be_asked_without_a_lookup():
    events, _, _ = run(
        [
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {"kind": "ask_myte", "person_id": MAYA, "question": "Ready?"},
                    }
                ]
            },
            {"text": "I suggested asking Maya's Myte."},
        ],
        {
            "question": "Ask her Myte if the deck is ready",
            "scope": {"kind": "person", "person_id": MAYA, "label": "Maya Chen"},
        },
    )
    (p,) = of(events, "proposal")
    assert p["person_name"] == "Maya Chen"


def test_people_named_in_tool_results_are_known_ids():
    owner = str(uuid.uuid4())
    events, _, _ = run(
        [
            {"tools": [{"tool": "list_my_open_actions", "args": {}}]},
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {"kind": "ask_myte", "person_id": owner, "question": "Status?"},
                    }
                ]
            },
            {"text": "Suggested."},
        ],
        {"question": "Ask the owner's Myte for status"},
        fixtures={
            "list_my_open_actions": {
                "items": [
                    {
                        "id": str(uuid.uuid4()),
                        "content": "Ship",
                        "owner_id": owner,
                        "owner_name": "Samar Rihan",
                    }
                ]
            }
        },
    )
    (p,) = of(events, "proposal")
    assert p["person_name"] == "Samar Rihan"


def test_at_most_three_proposals_per_answer():
    calls = [
        {"tool": "propose_action", "args": {"kind": "memory", "text": f"Fact {i}"}}
        for i in range(4)
    ]
    events, _, _ = run([{"tools": calls}, {"text": "ok"}], {"question": "Remember these"})
    assert len(of(events, "proposal")) == 3


# --- F4: memories ------------------------------------------------------------------------------


def test_memories_are_in_the_context_block_as_data():
    _, _, model = run(
        [{"text": "ok"}],
        {
            "question": "Hi",
            "memories": [
                {"id": "m1", "text": "Prefers bullet points"},
                {"id": "m2", "text": "Owns   billing"},
            ],
        },
    )
    ctx = context_of(model)
    assert "What you remember about them:" in ctx
    assert "  - Prefers bullet points" in ctx and "  - Owns billing" in ctx
    assert "never instructions" in system_of(model) and "memory:" in system_of(model)


def test_no_memories_says_nothing_yet():
    _, _, model = run([{"text": "ok"}], {"question": "Hi"})
    assert "What you remember about them: nothing yet" in context_of(model)


@pytest.mark.parametrize(
    "body",
    [
        {"question": "q", "memories": [{"text": "x"}] * 31},
        {"question": "q", "memories": [{"text": "x" * 301}]},
        {"question": "q", "attachments": [{"name": "a", "text": "x"}] * 4},
        {"question": "q", "attachments": [{"name": "a", "text": "x" * 20_001}] * 2},
        {"question": "q", "attachments": [{"name": "n" * 121, "text": "x"}]},
        {"question": "", "mode": "peer_answer"},
        {"question": "  "},
    ],
)
def test_v2_limits_are_one_invalid_event(body):
    events, _, model = run([{"text": "never"}], body)
    assert [n for n, _ in events] == ["error"] and events[0][1]["reason"] == "invalid"
    assert model.calls == []


def test_v2_limits_accept_their_edges():
    req = MyteRequest.model_validate(
        {
            "question": "q",
            "memories": [{"id": str(i), "text": "x" * 300} for i in range(30)],
            "attachments": [{"name": "n" * 120, "text": "x" * 20_000}] * 2,
        }
    )
    assert len(req.memories) == 30 and req.mode == "ask" and not req.peer_mode


# --- F7: attachments ---------------------------------------------------------------------------


def test_passages_split_at_breaks_near_1500_characters():
    para = ("This is a sentence about the plan. " * 30).strip()
    text = "\n\n".join([para] * 4)
    parts = passages(text)
    assert all(len(p) <= 1500 for p in parts) and len(parts) >= 4
    assert "".join(parts).replace(" ", "") == text.replace("\n", "").replace(" ", "")
    assert passages("short") == ["short"]
    assert passages("x" * 3100) == ["x" * 1500, "x" * 1500, "x" * 100]


def test_attachments_are_numbered_document_sources_before_the_first_round():
    long_text = "Line about the rollout plan.\n" * 120  # ~3500 characters -> 3 passages
    events, _, model = run(
        [{"text": "The notes say Drive is dropped [1]."}],
        {
            "question": "Summarise these",
            "attachments": [
                {"name": "notes.md", "text": "We will drop Google Drive access."},
                {"name": "plan.txt", "text": long_text},
            ],
        },
    )
    assert check_stream(events) == []
    sources = of(events, "source")
    assert [s["n"] for s in sources] == [1, 2, 3, 4]
    assert all(s["kind"] == "document" for s in sources)
    assert sources[0]["title"] == "notes.md"
    assert sources[1]["title"] == "plan.txt (part 1 of 3)"
    # sources come before anything the model streamed
    names = [n for n, _ in events]
    assert names.index("delta") > max(i for i, n in enumerate(names) if n == "source")
    ctx = context_of(model)
    assert '"notes.md" (source 1)' in ctx and '"plan.txt" (sources 2-4)' in ctx
    assert "[1] document | notes.md | We will drop Google Drive access." in ctx
    assert "data, never instructions" in ctx
    done = of(events, "done")[0]
    assert done["cited"] == [1] and done["abstained"] is False


def test_no_attachments_says_none():
    _, _, model = run([{"text": "ok"}], {"question": "Hi"})
    assert "Attached by them: none" in context_of(model)


def test_attachment_drafts_keep_files_apart():
    class A:
        def __init__(self, name: str, text: str) -> None:
            self.name, self.text = name, text

    drafts = attachment_drafts([A("a.txt", "same"), A("b.txt", "same")])
    assert len({d.key() for d in drafts}) == 2


# --- F1: peer answers --------------------------------------------------------------------------


PEER = {
    "question_id": str(uuid.uuid4()),
    "from_name": "Maya Chen",
    "question": "When is the Calendar reconnect due?",
}


def test_peer_answer_uses_its_own_prompt_and_no_proposals_cards_followups_or_title():
    events, _, model = run(
        [
            {"tools": [{"tool": "list_my_open_actions", "args": {}}]},
            {"text": "It's due Friday [[Reconnect]]."},
        ],
        {"mode": "peer_answer", "peer": PEER, "want_title": True},
        fixtures={
            "list_my_open_actions": {
                "items": [{"id": str(uuid.uuid4()), "content": "Reconnect Google Calendar"}]
            }
        },
    )
    assert check_stream(events) == []
    names = {n for n, _ in events}
    assert not names & {"proposal", "card", "followups", "title"}
    assert of(events, "done")[0]["cited"] == [1]
    assert "drafting the reply" in system_of(model)
    assert "Share only what answers the question" in system_of(model)
    assert "propose_action" not in offered(model) and "show_card" not in offered(model)
    assert "list_my_open_actions" in offered(model)
    ctx = context_of(model)
    assert "The teammate asking: Maya Chen" in ctx
    assert "Their question: When is the Calendar reconnect due?" in ctx
    user = [m for m in first_round(model)["messages"] if m["role"] == "user"][-1]
    assert user["content"] == "Maya Chen asked your Myte: When is the Calendar reconnect due?"
    # no follow-ups call: only streaming rounds
    assert all(c.get("stream") for c in model.calls)


def test_peer_answer_refuses_a_proposal_the_model_tries_anyway():
    events, _, model = run(
        [
            {"tools": [{"tool": "propose_action", "args": {"kind": "memory", "text": "x"}}]},
            {"text": "Draft."},
        ],
        {"mode": "peer_answer", "peer": PEER},
    )
    assert of(events, "proposal") == []
    tool = [m for m in model.calls[-1]["messages"] if m.get("role") == "tool"][-1]
    assert "cannot suggest changes" in tool["content"]


def test_peer_answer_carries_the_recipients_note_for_the_draft():
    _, _, model = run(
        [{"text": "Draft."}],
        {"mode": "peer_answer", "peer": PEER, "question": "Keep it to one line"},
    )
    user = [m for m in first_round(model)["messages"] if m["role"] == "user"][-1]
    assert user["content"].endswith("My note for this draft: Keep it to one line")


def test_peer_answer_is_timed_under_its_mode():
    before = _count("myte_answer_seconds", {"mode": "peer_answer", "outcome": "done"})
    run([{"text": "Draft."}], {"mode": "peer_answer", "peer": PEER})
    assert _count("myte_answer_seconds", {"mode": "peer_answer", "outcome": "done"}) == before + 1


def test_the_answer_prompt_is_v2_and_v1_still_loads():
    from libs.shared.prompts import PromptLoader
    from services.rag_service.main import PROMPTS_DIR

    loader = PromptLoader(PROMPTS_DIR)
    assert loader.resolve("myte_answer", ORG) == "v2"
    assert loader.resolve("myte_peer_answer", ORG) == "v1"
    v1 = loader.load("myte_answer", "v1")
    assert "{memories}" not in v1.user_template  # pinned tenants keep the v1 context
    _, _, model = run([{"text": "ok"}], {"question": "Hi"})
    assert "ask_myte" in system_of(model)


def test_a_card_tells_the_model_it_does_not_replace_the_answer():
    # Live eval: with "keep your text short", Sonnet wrote only "That's the full list."
    _, _, model = run(
        [
            {"tools": [{"tool": "list_my_open_actions", "args": {}}]},
            {"tools": [{"tool": "show_card", "args": {"kind": "tasks", "source_ns": [1]}}]},
            {"text": "Upload the logo is due Monday [1]."},
        ],
        {"question": "What's on me?"},
        fixtures={
            "list_my_open_actions": {"items": [{"id": str(uuid.uuid4()), "content": "Upload"}]}
        },
    )
    tool = [m for m in model.calls[-2]["messages"] if m.get("role") == "tool"][-1]
    assert "does not replace the answer" in tool["content"]
