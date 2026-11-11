"""Fixes from the first live eval run (2026-11-11): tolerant parsing of valid-but-variant model
answers, and the prompt versions that replaced the failing defaults.

- `proactive_conflict@v2` answered one object, then "Correction: ... here it is:" and a second
  object; the first-"{"-to-last-"}" slice made that invalid JSON and lost a real contradiction.
- `meeting_agent@v2` answered `"raised": ["C2"]` (the label the prompt shows), dropped as not an
  integer.
- The new prompt versions change only the system prompt: a tenant pinned to an older version, and
  every user message, are unchanged.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from libs.shared.agent_metrics import output_ok
from libs.shared.json_answer import json_object, json_objects
from libs.shared.prompts import PromptLoader
from services.agent_runtime_service.agents.candidates import parse_candidates
from services.agent_runtime_service.agents.meeting_agent import parse_assessment
from services.agent_runtime_service.agents.proactive_agent import parse_conflicts
from services.agent_runtime_service.consumer import PROMPTS_DIR

CORRECTED = (
    '{"conflicts": [{"candidate": 1, "confidence": 0.93, "rationale": "a", "rationale_note": '
    "null}]}\n\nCorrection: the answer must follow the exact shape, so here it is:\n\n"
    '{"conflicts": [{"candidate": 1, "confidence": 0.91, "rationale": "b"}]}'
)


def test_a_self_corrected_answer_is_read_as_its_last_object():
    assert len(json_objects(CORRECTED)) == 2
    assert json_object(CORRECTED, "conflicts") == {
        "conflicts": [{"candidate": 1, "confidence": 0.91, "rationale": "b"}]
    }
    assert parse_conflicts(CORRECTED, 1) == [(0, 0.91, "b")]
    assert output_ok("proactive_conflict@v2", CORRECTED) is True


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"candidates": [1]}', [1]),
        ('```json\n{"candidates": [1]}\n```', [1]),
        ('Here you go:\n{"candidates": [1]}\nHope that helps.', [1]),
        ('{"note": "{not json}"} then {"candidates": [2]}', [2]),
        ('{"candidates": [1]} {"other": true}', [1]),  # the last object holding the key
        ('{"candidates": [1', []),  # truncated: nothing, never a guess
        ("no json at all", []),
        ('{"candidates": "x"}', []),
    ],
)
def test_the_candidate_parser_tolerates_variants_and_nothing_else(text: str, expected: list):
    assert parse_candidates(text) == expected


def test_the_parse_check_agrees_with_the_parsers():
    assert output_ok("decision_agent@v5", '```json\n{"candidates": []}\n```') is True
    assert output_ok("decision_agent@v5", '{"candidates": [') is False
    assert output_ok("decision_gate@v2", 'Sure: {"score": 0.8}') is True
    assert output_ok("decision_gate@v2", '{"score": "high"}') is False


def test_raised_accepts_the_label_the_prompt_shows():
    a = parse_assessment(
        '{"updates": [], "off_agenda": null, "raised": ["C2", 1, "c1", "x", true]}'
    )
    assert a.raised == [2, 1]


LOADER = PromptLoader(PROMPTS_DIR)
VALUES = {
    "company": "",
    "ground_rules": "",
    "profile": "Role: engineer",
    "focus": "",
    "voice": "",
    "attendance": "",
    "person_memory": "",
    "guidance": "",
    "memory_focus": "",
    "window": "Participant 1: hi",
    "agenda": "1. Roadmap (pending, 20 min planned)",
    "carried": "(none)",
    "topics": "pricing, roadmap",
}


@pytest.mark.parametrize(
    "key,new,old",
    [
        ("personal_listener_gate", "v6", "v5"),
        ("personal_listener", "v5", "v4"),
        ("meeting_agent", "v3", "v2"),
        ("proactive_stakeholder", "v3", "v2"),
    ],
)
def test_the_new_defaults_change_only_the_system_prompt(key: str, new: str, old: str):
    org = uuid4()
    assert LOADER.resolve(key, org) == new
    a, b = LOADER.load(key, new), LOADER.load(key, old)  # the old one stays for pins
    assert a.render_user(**VALUES) == b.render_user(**VALUES)
    assert a.system != b.system


def test_the_v6_gate_answer_with_its_outcome_field_parses():
    from services.agent_runtime_service.agents.personal_listener_agent import parse_relevance

    answer = '{"outcome": true, "relevance": 0.6}'
    assert parse_relevance(answer) == 0.6
    assert output_ok("personal_listener_gate@v6", answer) is True
    assert parse_relevance('{"outcome": false, "relevance": 0.0}') == 0.0
