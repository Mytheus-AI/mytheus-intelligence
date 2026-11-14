"""Ask Myte's numbered sources and citations (ADR-0047 D5), and tool-output mapping to the D8
source kinds."""

from __future__ import annotations

from services.rag_service.myte.sources import (
    MAX_SOURCES,
    Draft,
    SourceRegistry,
    map_output,
    remap_citations,
)

M1 = "11111111-1111-1111-1111-111111111111"
D1 = "22222222-2222-2222-2222-222222222222"
A1 = "33333333-3333-3333-3333-333333333333"


def test_numbers_are_one_based_stable_and_deduplicated():
    reg = SourceRegistry()
    a, new_a = reg.add(
        Draft(kind="decision", title="Drop Drive", text="Drop Drive", decision_id=D1)
    )
    b, new_b = reg.add(Draft(kind="action", title="Reconnect", text="Reconnect", action_id=A1))
    again, new_again = reg.add(
        Draft(kind="decision", title="Drop Drive (record)", text="longer", decision_id=D1)
    )
    assert (a.n, b.n, again.n) == (1, 2, 1)  # type: ignore[union-attr]
    assert (new_a, new_b, new_again) == (True, True, False)


def test_the_registry_stops_at_its_cap():
    reg = SourceRegistry()
    for i in range(MAX_SOURCES):
        reg.add(Draft(kind="document", title=f"t{i}", text=f"x{i}"))
    source, new = reg.add(Draft(kind="document", title="one more", text="y"))
    assert source is None and not new


def test_cited_reads_only_registered_numbers_in_every_bracket_shape():
    reg = SourceRegistry()
    for i in range(3):
        reg.add(Draft(kind="document", title=f"t{i}", text=f"x{i}"))
    text = "A [1]. B [2][3]. C [1, 3]. Not a source [9]. Not a cite [x]."
    assert reg.cited(text) == [1, 2, 3]
    assert reg.cited("Nothing found.") == []


def test_the_event_matches_d8_and_omits_empty_fields():
    reg = SourceRegistry()
    s, _ = reg.add(
        Draft(
            kind="transcript",
            title="Backend update",
            text="Let's drop Drive entirely.",
            meeting_id=M1,
            meeting_title="Backend update",
            speaker="Sushan",
            start_ms=840000,
            occurred_at="2026-11-12T23:14:00Z",
        )
    )
    assert s is not None
    assert s.event() == {
        "n": 1,
        "kind": "transcript",
        "title": "Backend update",
        "text": "Let's drop Drive entirely.",
        "meeting_id": M1,
        "meeting_title": "Backend update",
        "occurred_at": "2026-11-12T23:14:00Z",
        "speaker": "Sushan",
        "start_ms": 840000,
    }
    line = s.line()
    assert line.startswith("[1] transcript | Backend update | Let's drop Drive entirely.")
    assert f"meeting_id={M1}" in line


def test_the_model_line_is_one_line_and_pipe_safe():
    reg = SourceRegistry()
    s, _ = reg.add(Draft(kind="document", title="a|b", text="line one\nline | two"))
    assert s is not None and "\n" not in s.line() and s.line().count("|") == 2


def test_search_results_map_to_their_kinds():
    body = {
        "items": [
            {"kind": "meeting", "id": M1, "text": "Backend update", "occurred_at": "2026-11-12"},
            {
                "kind": "decision",
                "id": D1,
                "text": "Drop Drive",
                "meeting_title": "Backend update",
                "occurred_at": "2026-11-12",
            },
            {"kind": "action", "id": A1, "text": "Reconnect Calendar", "occurred_at": "2026-11-13"},
        ]
    }
    drafts = map_output("search_meetings", {}, body, "")
    assert [d.kind for d in drafts] == ["meeting", "decision", "action"]
    assert drafts[0].meeting_id == M1 and drafts[1].decision_id == D1 and drafts[2].action_id == A1


def test_ask_meetings_citations_keep_their_numbers_as_aliases():
    body = {
        "answer": "We drop Drive [1]. Calendar only [2].",
        "citations": [
            {"n": 1, "kind": "transcript", "text": "drop Drive", "meeting_id": M1, "start_ms": 5},
            {"n": 2, "kind": "decision", "text": "Calendar only", "decision_id": D1},
        ],
    }
    drafts = map_output("ask_meetings", {"question": "q"}, body, "")
    assert [(d.kind, d.alias) for d in drafts] == [("transcript", 1), ("decision", 2)]
    assert remap_citations(body["answer"], {1: 4, 2: 7}) == "We drop Drive [4]. Calendar only [7]."
    assert remap_citations("x [3]", {1: 4}) == "x "


def test_a_meeting_outcome_is_the_meeting_its_decisions_and_actions():
    body = {
        "title": "Backend update",
        "status": "ended",
        "scheduled_start": "2026-11-12T22:00:00Z",
        "agenda": [{"title": "Google verification", "status": "covered"}],
        "decisions": [{"id": D1, "content": "Drop Drive"}],
        "actions": [
            {"id": A1, "content": "Reconnect", "owner_name": "Sushan", "due_date": "2026-11-13"}
        ],
    }
    drafts = map_output("get_meeting_outcome", {"meeting_id": M1}, body, "")
    assert [d.kind for d in drafts] == ["meeting", "decision", "action"]
    assert all(d.meeting_id == M1 for d in drafts)
    assert "Google verification (covered)" in drafts[0].text
    assert "owner Sushan" in drafts[2].text and "due 2026-11-13" in drafts[2].text


def test_a_decision_record_hides_lineage_from_meetings_the_person_was_not_in():
    body = {
        "id": D1,
        "content": "Calendar read only",
        "status": "confirmed",
        "meeting_title": "Backend update",
        "rationale": "keeps us out of the security assessment",
        "lineage": [
            {"relation": "replaced", "visible": True, "content": "Read Drive attachments"},
            {"relation": "replaced", "visible": False, "content": "SECRET"},
            {"relation": "subject", "visible": True, "content": "Calendar read only"},
        ],
    }
    [d] = map_output("get_decision_record", {"decision_id": D1}, body, "")
    assert d.kind == "decision" and d.decision_id == D1
    assert "SECRET" not in d.text and "a decision from a meeting you were not in" in d.text
    assert "Why: keeps us out" in d.text


def test_actions_promises_people_inbox_and_schedule_map_to_d8_kinds():
    assert (
        map_output("list_my_open_actions", {}, {"items": [{"id": A1, "content": "Ship"}]}, "")[
            0
        ].kind
        == "action"
    )
    promise = map_output(
        "list_my_promises",
        {},
        {
            "items": [
                {
                    "id": "p1",
                    "content": "Send deck",
                    "due_at": "2026-11-20",
                    "source": {"label": "Slack"},
                }
            ]
        },
        "",
    )[0]
    assert promise.kind == "promise" and "from Slack" in promise.text
    person = map_output(
        "find_people",
        {"query": "maya"},
        {"items": [{"id": "u1", "name": "Maya", "role": "PM"}]},
        "",
    )[0]
    assert person.kind == "person" and person.title == "Maya" and "PM" in person.text
    note = map_output(
        "list_my_inbox", {}, {"items": [{"id": "n1", "title": "Review a decision"}]}, ""
    )[0]
    assert note.kind == "notification"
    meeting = map_output(
        "list_my_schedule",
        {},
        {
            "items": [
                {"id": M1, "title": "Test", "start": "2026-11-15T23:00:00Z", "attendance": "agent"}
            ]
        },
        "",
    )[0]
    assert (
        meeting.kind == "meeting"
        and meeting.meeting_id == M1
        and "your choice: agent" in meeting.text
    )


def test_capacity_and_my_myte_become_one_source_of_their_text():
    [cap] = map_output(
        "check_my_capacity", {"due_date": "2026-11-20"}, {"level": "full"}, "Friday is full."
    )
    assert cap.kind == "promise" and cap.text == "Friday is full." and "2026-11-20" in cap.title
    [mine] = map_output("get_my_myte", {}, {"name": "Pip"}, "Pip, balanced, listening on")
    assert mine.kind == "document"


def test_conflicts_map_to_decisions_and_none_is_no_source():
    body = {
        "conflicts": [{"decision_id": D1, "decision_content": "No discounts", "quote": "20% off"}]
    }
    [d] = map_output("check_against_decisions", {"text": "20% off"}, body, "")
    assert d.kind == "decision" and '"20% off"' in d.text
    assert map_output("check_against_decisions", {}, {"conflicts": []}, "No conflict.") == []


def test_an_unknown_tool_with_text_is_one_document_and_an_empty_list_is_nothing():
    [d] = map_output("brand_new_tool", {}, None, "Some answer")
    assert d.kind == "document" and d.text == "Some answer"
    assert map_output("list_my_open_actions", {}, {"items": []}, "You have no open actions.") == []


def test_a_fuller_find_of_the_same_item_upgrades_it_and_is_re_sent_with_its_number():
    reg = SourceRegistry()
    hit, _ = reg.add(Draft(kind="decision", title="Drop Drive", text="Drop Drive", decision_id=D1))
    fuller, send = reg.add(
        Draft(kind="decision", title="Drop Drive", text="Drop Drive. Why: security", decision_id=D1)
    )
    assert send and fuller is hit and fuller.n == 1 and "Why: security" in fuller.line()
    same, send_again = reg.add(Draft(kind="decision", title="x", text="short", decision_id=D1))
    assert same is hit and not send_again and "Why: security" in same.draft.text
