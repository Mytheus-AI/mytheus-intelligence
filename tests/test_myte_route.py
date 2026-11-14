"""`POST /api/rag/myte` (ADR-0047 D2-D5, D8, D10): the Server-Sent Events contract end to end,
with the real loop and router, a fake bff MCP server and a scripted model. No network, no
model calls."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest
from starlette.testclient import TestClient

from services.rag_service import main as rag_main
from services.rag_service.myte.loop import LoopConfig
from services.rag_service.myte.route import sse
from tests.conftest import auth_headers, sign_test_token
from tests.myte_kit import FakeBff, ScriptedModel, Sink, check_stream, gate, make_loop, parse_sse

ORG = uuid.uuid4()
M1 = "11111111-aaaa-4bbb-8ccc-111111111111"
D1 = "22222222-aaaa-4bbb-8ccc-222222222222"
A1 = "33333333-aaaa-4bbb-8ccc-333333333333"

OUTCOME = {
    "title": "Feature Update - Backend",
    "status": "ended",
    "scheduled_start": "2026-11-12T22:00:00Z",
    "decisions": [{"id": D1, "content": "Drop Google Drive access; Calendar read only"}],
    "actions": [{"id": A1, "content": "Reconnect Google Calendar", "due_date": "2026-11-13"}],
}
ACTIONS = {"items": [{"id": A1, "content": "Reconnect Google Calendar", "due_date": "2026-11-13"}]}


def post(client: TestClient, body: Any, headers: dict[str, str] | None = None) -> Any:
    return client.post(
        "/api/rag/myte",
        content=json.dumps(body) if not isinstance(body, str) else body,
        headers=headers if headers is not None else auth_headers(ORG),
    )


def run(
    turns: list[dict[str, Any]],
    body: dict[str, Any] | None = None,
    fixtures: dict[str, Any] | None = None,
    **kw: Any,
) -> tuple[list[tuple[str, dict[str, Any]]], FakeBff, ScriptedModel, Any]:
    bff = FakeBff(fixtures=fixtures or {})
    model = ScriptedModel(list(turns))
    loop = make_loop(model, bff, **kw)
    client = TestClient(rag_main.build_app(myte=loop))
    res = post(client, body or {"question": "What did we decide in the backend update?"})
    return parse_sse(res.text), bff, model, res


def names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [n for n, _ in events]


def of(events: list[tuple[str, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [d for n, d in events if n == name]


# --- the happy path ------------------------------------------------------------------------


def test_a_full_answer_streams_steps_sources_deltas_followups_title_done_in_order():
    events, bff, model, res = run(
        [
            {
                "tools": [
                    {"tool": "get_meeting_outcome", "args": {"meeting_id": M1}},
                    {"tool": "list_my_open_actions", "args": {}},
                ]
            },
            {
                "text": "Two decisions were confirmed: Drive is dropped [[Drop Google Drive]]. "
                "One thing is on you: reconnect Calendar [[Reconnect Google Calendar]]."
            },
        ],
        body={"question": "What did we decide, and is anything on me?", "want_title": True},
        fixtures={"get_meeting_outcome": OUTCOME, "list_my_open_actions": ACTIONS},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["cache-control"] == "no-cache"
    assert check_stream(events) == [] and model.problems == []
    seq = names(events)
    assert seq[:2] == ["step", "step"]  # both lookups start at once (one round, concurrent)
    assert seq[-3:] == ["followups", "title", "done"]
    assert seq.index("delta") > max(i for i, n in enumerate(seq) if n == "source")
    labels = {d["label"] for d in of(events, "step")}
    assert labels == {"Opening the meeting", "Checking your tasks"}
    sources = of(events, "source")
    # meeting, decision, action from the outcome; the same action from the task list is reused
    assert [s["kind"] for s in sources] == ["meeting", "decision", "action"]
    assert sources[1]["decision_id"] == D1 and sources[2]["action_id"] == A1
    text = "".join(d["text"] for d in of(events, "delta"))
    assert "[2]" in text and "[3]" in text
    assert of(events, "done") == [{"abstained": False, "degraded": False, "cited": [2, 3]}]
    assert of(events, "followups")[0]["items"][0] == "Who was there?"
    assert of(events, "title") == [{"text": "Test"}]
    # every bff call went out as the person
    assert {c.authorization.split()[0] for c in bff.calls} == {"Bearer"}
    assert all(c.authorization == bff.calls[0].authorization for c in bff.calls)


def test_the_persons_token_is_forwarded_verbatim_to_bff():
    bff = FakeBff(fixtures={"list_my_open_actions": ACTIONS})
    model = ScriptedModel(
        [{"tools": [{"tool": "list_my_open_actions"}]}, {"text": "One [[Reconnect]]."}]
    )
    client = TestClient(rag_main.build_app(myte=make_loop(model, bff)))
    token = sign_test_token(ORG)
    post(client, {"question": "what do I owe"}, {"Authorization": f"Bearer {token}"})
    assert bff.calls[0].authorization == f"Bearer {token}"


def test_the_model_sees_compact_numbered_lines_and_the_tools_minus_writes():
    _events, _bff, model, _ = run(
        [{"tools": [{"tool": "list_my_open_actions"}]}, {"text": "ok [[Reconnect]]"}],
        fixtures={"list_my_open_actions": ACTIONS},
    )
    stream_calls = [c for c in model.calls if c.get("stream")]
    offered = {t["function"]["name"] for t in stream_calls[0]["tools"]}
    assert "create_action" not in offered and "add_promise" not in offered
    assert {"propose_action", "show_card", "search_help", "list_my_schedule"} <= offered
    tool_msg = next(m for m in stream_calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("[1] action | Reconnect Google Calendar | ")
    assert f"action_id={A1}" in tool_msg["content"]


def test_the_system_prompt_is_in_character_and_versioned():
    _events, _bff, model, _ = run(
        [{"text": "Hi, I'm Pip."}],
        body={
            "question": "hi",
            "persona": {
                "name": "Pip",
                "personality": "friendly",
                "instructions": "Mention deadlines",
            },
            "timezone": "Asia/Kolkata",
            "page": {"kind": "meeting", "id": M1, "title": "Feature Update - Backend"},
        },
    )
    call = next(c for c in model.calls if c.get("stream"))
    system = call["messages"][0]["content"]
    assert "You are a Myte" in system[0]["text"] and "cache_control" in system[0]
    context = system[1]["text"]
    assert "Your name: Pip" in context and "friendly" in context and "Mention deadlines" in context
    assert "Saturday, November 14, 2026, 21:00" in context and "Asia/Kolkata" in context
    assert 'meeting page for "Feature Update - Backend"' in context
    # No thinking (owner's call: token cost); Sonnet 5.5's "off" with tools is between_tools.
    assert call["thinking"] == {"type": "between_tools"} and "output_config" not in call
    assert call["model"].endswith("sonnet-5-5")


def test_history_becomes_alternating_turns_before_the_question():
    _events, _bff, model, _ = run(
        [{"text": "ok"}],
        body={
            "question": "and who owns it?",
            "history": [
                {"role": "myte", "content": "(a leading Myte message is dropped)"},
                {"role": "user", "content": "what did we decide about pricing?"},
                {"role": "myte", "content": "No discounts [1]."},
            ],
        },
    )
    msgs = next(c for c in model.calls if c.get("stream"))["messages"][1:]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "what did we decide about pricing?"),
        ("assistant", "No discounts [1]."),
        ("user", "and who owns it?"),
    ]


# --- scope ---------------------------------------------------------------------------------


def test_meeting_scope_passes_meeting_id_to_tools_that_take_it():
    _events, bff, _model, _ = run(
        [
            {"tools": [{"tool": "ask_meetings", "args": {"question": "open questions?"}}]},
            {"text": "x"},
        ],
        body={"question": "open questions?", "scope": {"kind": "meeting", "meeting_id": M1}},
        fixtures={"ask_meetings": {"answer": None, "citations": []}},
    )
    assert bff.calls[0].arguments == {"question": "open questions?", "meeting_id": M1}


def test_as_of_scope_fills_the_date_and_timezone_reaches_capacity():
    _events, bff, _model, _ = run(
        [
            {
                "tools": [
                    {"tool": "company_as_of", "args": {}},
                    {"tool": "check_my_capacity", "args": {}},
                ]
            },
            {"text": "x"},
        ],
        body={
            "question": "what was true?",
            "scope": {"kind": "as_of", "as_of": "2026-10-01"},
            "timezone": "Europe/London",
        },
        fixtures={"company_as_of": {"decisions": []}, "check_my_capacity": {"days": []}},
    )
    args = {c.name: c.arguments for c in bff.calls}
    assert args["company_as_of"] == {"date": "2026-10-01"}
    assert args["check_my_capacity"] == {"timezone": "Europe/London"}


# --- proposals and cards -------------------------------------------------------------------


def test_a_change_is_a_proposal_event_and_nothing_is_written():
    events, bff, model, _ = run(
        [
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {
                            "kind": "task",
                            "title": "Upload the logo",
                            "due_date": "2026-11-16",
                        },
                    }
                ]
            },
            {"text": "I suggested a task for the logo upload."},
        ],
        body={"question": "Add a task to upload the logo by Monday"},
    )
    [proposal] = of(events, "proposal")
    assert proposal["kind"] == "task" and proposal["title"] == "Upload the logo"
    assert proposal["due_date"] == "2026-11-16" and uuid.UUID(proposal["id"])
    assert bff.calls == []  # a proposal calls nothing
    tool_result = next(
        m for m in [c for c in model.calls if c.get("stream")][1]["messages"] if m["role"] == "tool"
    )
    assert "Nothing is saved until they confirm" in tool_result["content"]
    assert of(events, "done")[0]["abstained"] is False
    assert "step" not in names(events)


def test_a_model_that_calls_a_write_tool_directly_is_refused_before_bff():
    events, bff, _model, _ = run(
        [
            {"tools": [{"tool": "create_action", "args": {"content": "x"}}]},
            {"text": "I can't do that directly."},
        ],
    )
    assert bff.calls == []
    assert of(events, "step")[-1]["state"] == "failed"


def test_attendance_needs_a_known_meeting_and_a_choice():
    events, _bff, model, _ = run(
        [
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {"kind": "attendance", "title": "Test", "meeting_id": M1},
                    }
                ]
            },
            {"tools": [{"tool": "list_my_schedule"}]},
            {
                "tools": [
                    {
                        "tool": "propose_action",
                        "args": {
                            "kind": "attendance",
                            "title": "Test",
                            "meeting_id": M1,
                            "choice": "agent",
                        },
                    }
                ]
            },
            {"text": "I suggested sending your Myte to Test."},
        ],
        body={"question": "Send my Myte to Test"},
        fixtures={
            "list_my_schedule": {
                "items": [{"id": M1, "title": "Test", "start": "2026-11-15T23:00:00Z"}]
            }
        },
    )
    results = [
        m["content"]
        for m in [c for c in model.calls if c.get("stream")][-1]["messages"]
        if m["role"] == "tool"
    ]
    assert results[0].startswith("Error: use a meeting_id from your sources")
    [proposal] = of(events, "proposal")
    assert proposal["choice"] == "agent" and proposal["meeting_id"] == M1


def test_show_card_builds_a_card_from_registered_sources_only():
    events, _bff, _model, _ = run(
        [
            {"tools": [{"tool": "list_my_open_actions"}]},
            {"tools": [{"tool": "show_card", "args": {"kind": "tasks", "source_ns": [1, 99]}}]},
            {"text": "Here they are [[Reconnect]]."},
        ],
        fixtures={"list_my_open_actions": ACTIONS},
    )
    [card] = of(events, "card")
    assert card["kind"] == "tasks" and card["source_ns"] == [1]
    assert card["data"]["items"][0]["action_id"] == A1 and card["data"]["detail"] is None


def test_help_questions_are_answered_from_the_help_corpus():
    events, bff, _model, _ = run(
        [
            {"tools": [{"tool": "search_help", "args": {"query": "connect slack"}}]},
            {"text": "From Settings, Integrations [[Connect Slack]]."},
        ],
        body={"question": "How do I connect Slack?"},
    )
    assert bff.calls == []
    assert of(events, "source")[0]["title"] == "Help: Connect Slack"
    assert of(events, "step")[0]["label"] == "Looking in the Mytheus help"
    assert of(events, "done")[0]["cited"] == [1]


# --- abstention, degradation, limits -------------------------------------------------------


def test_nothing_found_abstains():
    events, _bff, _model, _ = run(
        [
            {"tools": [{"tool": "search_meetings", "args": {"query": "refund policy"}}]},
            {"text": "I couldn't find that in your meetings or the company brain."},
        ],
        fixtures={"search_meetings": {"items": []}},
    )
    assert of(events, "done") == [{"abstained": True, "degraded": False, "cited": []}]
    assert of(events, "step")[-1]["state"] == "done"


def test_a_tool_error_is_shown_to_the_model_and_the_step_fails_without_degrading():
    events, _bff, model, _ = run(
        [
            {"tools": [{"tool": "get_meeting_outcome", "args": {"meeting_id": M1}}]},
            {"text": "I can only see meetings you took part in."},
        ],
        fixtures={
            "get_meeting_outcome": ["error", "Not found, or not from a meeting you took part in."]
        },
    )
    assert of(events, "step")[-1]["state"] == "failed"
    msg = next(
        m for m in [c for c in model.calls if c.get("stream")][1]["messages"] if m["role"] == "tool"
    )
    assert msg["content"] == "Error: Not found, or not from a meeting you took part in."
    assert of(events, "done") == [{"abstained": True, "degraded": False, "cited": []}]


def test_a_lookup_that_cannot_reach_bff_degrades_the_answer():
    events, _bff, _model, _ = run(
        [{"tools": [{"tool": "list_my_inbox"}]}, {"text": "I couldn't check your inbox."}],
        fixtures={"list_my_inbox": ["http", 502]},
    )
    assert of(events, "step")[-1]["state"] == "failed"
    assert of(events, "done")[0]["degraded"] is True


def test_the_tool_time_budget_fails_slow_lookups_and_forces_an_answer():
    events, _bff, model, _ = run(
        [
            {"tools": [{"tool": "list_my_inbox"}]},
            {"text": "I ran out of time checking your inbox."},
        ],
        fixtures={"list_my_inbox": ["slow", 2.0]},
        config=LoopConfig(tool_budget_s=0.2),
    )
    assert of(events, "step")[-1]["state"] == "failed"
    assert of(events, "done")[0]["degraded"] is True
    assert [c.get("tool_choice") for c in model.calls if c.get("stream")] == [None, "none"]


def test_the_last_round_is_asked_to_answer_without_tools():
    turns = [{"tools": [{"tool": "list_my_inbox"}]}] * 3 + [{"text": "Done."}]
    events, bff, model, _ = run(
        turns, config=LoopConfig(max_rounds=3), fixtures={"list_my_inbox": {"items": []}}
    )
    rounds = [c for c in model.calls if c.get("stream")]
    assert len(rounds) == 3 and rounds[-1]["tool_choice"] == "none"
    assert len(bff.calls) == 2  # the third round's tool call is not run
    assert of(events, "done")[0]["degraded"] is True


def test_an_empty_answer_says_so_in_character_and_is_degraded():
    events, _bff, _model, _ = run([{"text": ""}])
    text = "".join(d["text"] for d in of(events, "delta"))
    assert text.startswith("I couldn't put an answer together")
    assert of(events, "done")[0]["degraded"] is True


def test_preamble_text_and_the_answer_are_separated():
    events, _bff, _model, _ = run(
        [
            {"tools": [{"tool": "list_my_inbox"}], "text": "Let me look."},
            {"text": "Nothing waiting."},
        ],
        fixtures={"list_my_inbox": {"items": []}},
    )
    assert "".join(d["text"] for d in of(events, "delta")) == "Let me look.\n\nNothing waiting."


def test_followups_failing_does_not_fail_the_answer_and_the_title_falls_back():
    bff = FakeBff()
    model = ScriptedModel([{"text": "ok"}], followups="not json")
    client = TestClient(rag_main.build_app(myte=make_loop(model, bff)))
    events = parse_sse(
        post(client, {"question": "What changed since Monday?", "want_title": True}).text
    )
    assert "followups" not in names(events)
    assert of(events, "title") == [{"text": "What changed since Monday"}]
    assert names(events)[-1] == "done"


def test_no_title_unless_asked():
    events, *_ = run([{"text": "ok"}], body={"question": "q", "want_title": False})
    assert "title" not in names(events)


def test_one_metering_record_per_round_plus_the_followups_call():
    sink = Sink()
    run(
        [{"tools": [{"tool": "list_my_inbox"}]}, {"text": "Nothing."}],
        fixtures={"list_my_inbox": {"items": []}},
        sink=sink,
    )
    assert [c["model_tier"] for c in sink.calls] == ["sonnet", "sonnet", "haiku"]


# --- errors --------------------------------------------------------------------------------


def test_auth_is_required_before_any_stream():
    loop = make_loop(ScriptedModel([]), FakeBff())
    client = TestClient(rag_main.build_app(myte=loop))
    assert post(client, {"question": "q"}, headers={}).status_code == 401
    assert post(client, {"question": "q"}, {"Authorization": "Bearer junk"}).status_code == 401


def test_service_credentials_are_refused():
    loop = make_loop(ScriptedModel([]), FakeBff())
    client = TestClient(rag_main.build_app(myte=loop))
    headers = {
        **auth_headers(ORG),
        "X-Mytheus-Service": "bff",
        "X-Mytheus-Service-Token": "a" * 40,
        "X-Mytheus-Org": str(ORG),
    }
    assert post(client, {"question": "q"}, headers).status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "history": [{"role": "user", "content": "h"}] * 13},
        {"question": "q", "history": [{"role": "user", "content": "x" * 4001}]},
        {"question": "q", "history": [{"role": "system", "content": "x"}]},
        {"question": "q", "scope": {"kind": "meeting", "meeting_id": "not-a-uuid"}},
        "not json",
        ["a", "list"],
    ],
)
def test_invalid_bodies_are_one_invalid_error_event(body: Any):
    model = ScriptedModel([])
    client = TestClient(rag_main.build_app(myte=make_loop(model, FakeBff())))
    res = post(client, body)
    events = parse_sse(res.text)
    assert res.status_code == 200 and names(events) == ["error"]
    assert events[0][1]["reason"] == "invalid" and events[0][1]["message"]
    assert model.calls == []


def test_the_limits_accept_their_edges():
    events, *_ = run(
        [{"text": "ok"}],
        body={"question": "x" * 2000, "history": [{"role": "user", "content": "y" * 4000}] * 12},
    )
    assert names(events)[-1] == "done"


def test_a_blocked_budget_is_a_budget_error():
    events, bff, model, _ = run([{"text": "never"}], budget=gate(consumed=200, quota=100, cap=100))
    assert names(events) == ["error"] and events[0][1]["reason"] == "budget"
    assert "limit" in events[0][1]["message"]
    assert [c for c in model.calls if c.get("stream")] == []


def test_a_bff_401_is_a_session_error():
    bff = FakeBff(list_status=401)
    client = TestClient(rag_main.build_app(myte=make_loop(ScriptedModel([]), bff)))
    events = parse_sse(post(client, {"question": "q"}).text)
    assert names(events) == ["error"] and events[0][1]["reason"] == "session"


def test_a_401_mid_answer_is_a_session_error():
    events, *_ = run(
        [{"tools": [{"tool": "list_my_inbox"}]}, {"text": "never"}],
        fixtures={"list_my_inbox": ["http", 401]},
    )
    assert names(events)[-1] == "error" and events[-1][1]["reason"] == "session"
    assert "done" not in names(events)


def test_a_bff_403_is_a_session_error_that_names_the_role():
    bff = FakeBff(list_status=403)
    client = TestClient(rag_main.build_app(myte=make_loop(ScriptedModel([]), bff)))
    events = parse_sse(post(client, {"question": "q"}).text)
    assert events[0][1]["reason"] == "session" and "role" in events[0][1]["message"]


def test_bff_down_is_unavailable():
    bff = FakeBff(list_status=503)
    client = TestClient(rag_main.build_app(myte=make_loop(ScriptedModel([]), bff)))
    events = parse_sse(post(client, {"question": "q"}).text)
    assert events == [("error", {"reason": "unavailable", "message": events[0][1]["message"]})]


def test_a_provider_auth_failure_is_not_connected_and_others_are_unavailable():
    class AuthenticationError(Exception):
        pass

    async def no_key(**_kw: Any) -> Any:
        raise AuthenticationError("no api key")

    async def flaky(**_kw: Any) -> Any:
        raise ConnectionError("reset")

    for fn, reason in ((no_key, "not_connected"), (flaky, "unavailable")):
        client = TestClient(rag_main.build_app(myte=make_loop(fn, FakeBff())))
        events = parse_sse(post(client, {"question": "q"}).text)
        assert names(events) == ["error"] and events[0][1]["reason"] == reason


def test_an_unwired_service_is_not_connected():
    client = TestClient(rag_main.build_app())
    events = parse_sse(post(client, {"question": "q"}).text)
    assert names(events) == ["error"] and events[0][1]["reason"] == "not_connected"


async def test_keepalive_comments_go_out_while_waiting():
    async def slow() -> Any:
        await asyncio.sleep(0.25)
        from services.rag_service.myte.loop import Event

        yield Event("done", {"abstained": True, "degraded": False, "cited": []})

    out = [chunk async for chunk in sse(slow(), keepalive_s=0.1)]
    assert out[0] == ": keepalive\n\n" and out[-1].startswith("event: done\n")


async def test_closing_the_stream_cancels_the_answer():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang(**_kw: Any) -> Any:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    from services.rag_service.domain import Caller
    from services.rag_service.myte.contracts import MyteRequest

    loop = make_loop(hang, FakeBff())
    stream = loop.stream(
        MyteRequest(question="q"), Caller(org_id=ORG, clerk_user_id="u"), "Bearer t"
    )
    task = asyncio.ensure_future(stream.__anext__())
    await started.wait()
    task.cancel()  # what Starlette does when the client goes away
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)
