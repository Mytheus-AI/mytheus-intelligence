"""Decisions that know why and the assumption watcher (`decision_why.py`, a4 contract 1 and 2).

Every model call, core-api, billing and the knowledge graph are faked; the router, the budget
gate, the prompt loader, the parsers and the real `Emitter` run as in production.

1. Parsing: the why-source and open-assumption bodies; assumptions (labels back to user ids, a
   quote that is not in the sources is never shown as one, watch fields, caps, never record,
   URLs); the ripple (labels to ids, effects, confidence floor, caps); breaks (verbatim evidence
   only, the 0.6 floor, "someone" in rationales).
2. The decision hook: the exact contract 1a payload, graph actions merged and deduped, gating
   (flags, plan, budget, no owner, already read, never record, KG down), no user id, name or
   email in any model call, never-record windows removed before the model sees the transcript.
3. Trigger A: the exact contract 1b payload, no model call without open assumptions, never from a
   confidential decision.
4. The proactive loop: one flush for conflict alerts and the why record; a failing why never
   stops the conflict alerts; a failing conflict check never stops the why.
5. Trigger B (`AssumptionWatchLoop`): the record, the Redis claim and its replay, confidential and
   never-record meetings, retries, the recipient.
6. The HTTP client: headers, query, 404 / 409 / 5xx.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date
from typing import Any
from uuid import UUID, uuid5

import fakeredis
import httpx
import pytest

from libs.shared.kg_client import KgNode, KgUnavailable
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, Goal
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service import decision_why as dw
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS
from services.agent_runtime_service.agents.proactive_agent import ConfirmedDecision
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.minutes import SourceUnavailable, parse_source
from services.agent_runtime_service.proactive.loop import ProactiveLoop
from tests.a2a_helpers import JsonCodec, LoopbackProducer, budget
from tests.test_proactive_agent import FakePlans
from tests.test_proactive_agent import make as make_proactive

ORG = uuid.uuid4()
DECISION, OLD_DECISION = uuid.uuid4(), uuid.uuid4()
MEETING, OLD_MEETING, OTHER_MEETING = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
PRIYA, SAM, ALEX = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
A1, A2, GRAPH_ACTION = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
GOAL1, GOAL2 = str(uuid.uuid4()), str(uuid.uuid4())
OPEN1, OPEN2 = uuid.uuid4(), uuid.uuid4()
EMAILS = {PRIYA: "priya@acme.example", SAM: "sam@acme.example", ALEX: "alex@acme.example"}
DAY = date(2026, 11, 11)

ASSUME = "write down the reasons it rests on"
RIPPLE = "You map the ripple"
CHECK = "You watch the reasons"
NEVER_CHECK = "never-record"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# the exact contract key sets
TOP_1A = {"alert", "why", "prompt", "model_tier"}
TOP_1B = {"alert", "breaks", "prompt", "model_tier"}
ALERT_KEYS = {
    "id",
    "alert_type",
    "priority",
    "surfaced",
    "title",
    "body",
    "confidence",
    "rationale",
    "subject",
    "related",
    "link",
}
WHY_KEYS = {"decision_id", "assumptions", "ripple"}
ASSUMPTION_KEYS = {"kind", "statement", "quote", "speaker_user_id", "confidence", "watch"}
WATCH_KEYS = {"date", "value", "comparator", "unit", "issue_key"}
GOAL_KEYS = {"goal_id", "label", "effect", "why", "confidence"}
ACTION_KEYS = {"action_id", "label", "effect", "why", "confidence", "source"}
BREAK_KEYS = {
    "assumption_id",
    "confidence",
    "evidence_quote",
    "rationale",
    "source",
    "source_meeting_id",
    "source_decision_id",
    "speaker_user_id",
}

ASSUMPTIONS_ANSWER = {
    "assumptions": [
        {
            "kind": "price",
            "statement": "Vendor B costs less than 40,000 USD a year.",
            "quote": "we go with Vendor B because it's under 40k a year",
            "speaker": "Participant 2",
            "confidence": 0.9,
            "watch": {
                "date": None,
                "value": 40000,
                "comparator": "lt",
                "unit": "USD",
                "issue_key": None,
            },
        },
        {
            "kind": "person",
            "statement": "Participant 2 owns the migration (see https://wiki.acme.example/x).",
            "quote": "Priya owns the migration, tracked in OPS-42",
            "speaker": "Participant 1",
            "confidence": 0.8,
            "watch": {"issue_key": "ops-42"},
        },
        {
            "kind": "date",
            "statement": "Vendor B ships SSO by March.",
            "quote": "they promised SSO by March for sure",  # never said: not a quote
            "speaker": "Participant 2",
            "confidence": 0.7,
            "watch": {"date": "2027-03-31", "issue_key": "SSO-999", "comparator": "lt"},
        },
        {"kind": "metric", "statement": "Too weak to keep.", "confidence": 0.3},
        {"kind": "weird", "statement": "Vendor B keeps its EU region.", "confidence": 0.6},
    ]
}
RIPPLE_ANSWER = {
    "goals": [
        {
            "goal": "G1",
            "effect": "advances",
            "why": "Cheaper tooling cuts spend.",
            "confidence": 0.8,
        },
        {"goal": "G9", "effect": "advances", "why": "No such goal.", "confidence": 0.9},
        {"goal": "G2", "effect": "risks", "why": "Weak link.", "confidence": 0.4},
    ],
    "actions": [
        {
            "action": "A1",
            "effect": "changes",
            "why": "The integration targets Vendor A.",
            "confidence": 0.85,
        },
        {"action": "A2", "effect": "explodes", "why": "Not an effect.", "confidence": 0.9},
    ],
}
CHECK_ANSWER = {
    "breaks": [
        {
            "assumption": "H1",
            "evidence_quote": "Go with Vendor B for observability",
            "speaker": None,
            "rationale": "The new decision moves observability off Vendor A.",
            "confidence": 0.85,
        },
        {
            "assumption": "H2",
            "evidence_quote": "words nobody wrote",
            "speaker": None,
            "rationale": "Made up.",
            "confidence": 0.95,
        },
    ]
}


class LLM:
    """Answers by prompt (system text marker). An answer may be a dict (JSON), a string, an
    exception (raised) or a callable of the call's kwargs."""

    def __init__(
        self,
        assumptions: Any = None,
        ripple: Any = None,
        check: Any = None,
        never: Any = '{"about": false, "score": 0.0}',
    ) -> None:
        self.answers = {
            ASSUME: ASSUMPTIONS_ANSWER if assumptions is None else assumptions,
            RIPPLE: RIPPLE_ANSWER if ripple is None else ripple,
            CHECK: CHECK_ANSWER if check is None else check,
        }
        self.never = never
        self.calls: list[dict[str, Any]] = []

    def of(self, marker: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if marker in c["messages"][0]["content"][0]["text"]]

    @staticmethod
    def user(call: dict[str, Any]) -> str:
        content = call["messages"][1]["content"]
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content)

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system = kw["messages"][0]["content"][0]["text"]
        answer: Any = self.never
        for marker, value in self.answers.items():
            if marker in system:
                answer = value
        if callable(answer):
            answer = answer(kw)
        if isinstance(answer, Exception):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 80},
        }


def source_body(**decision: Any) -> dict[str, Any]:
    base = {
        "id": str(DECISION),
        "meeting_id": str(MEETING),
        "content": "Go with Vendor B for observability.",
        "rationale": "Vendor B is under 40k a year and they ship SSO by March.",
        "quote": "we go with Vendor B because it's under 40k a year",
        "quote_speaker_id": "spk-1",
        "quote_start_ms": 60_000,
        "confirmed_by": str(PRIYA),
        "confidential": False,
        "status": "confirmed",
    }
    return {
        "decision": {**base, **decision},
        "meeting_title": "Tooling review",
        "participants": [
            {"user_id": str(PRIYA), "name": EMAILS[PRIYA]},
            {"user_id": str(SAM), "name": EMAILS[SAM]},
            {"user_id": str(ALEX), "name": EMAILS[ALEX]},
        ],
        "speakers": {"spk-1": str(PRIYA), "spk-2": str(SAM)},
        "turns": [
            {"speaker_id": "spk-2", "start_ms": 0, "end_ms": 4_000, "text": "What about cost?"},
            {
                "speaker_id": "spk-1",
                "start_ms": 60_000,
                "end_ms": 64_000,
                "text": "Honestly we go with Vendor B because it's under 40k a year.",
            },
            {
                "speaker_id": "spk-2",
                "start_ms": 65_000,
                "end_ms": 69_000,
                "text": "And Priya owns the migration, tracked in OPS-42.",
            },
        ],
        "candidate_actions": [
            {
                "id": str(A1),
                "content": "Integrate Vendor A tracing into checkout",
                "due_date": "2026-11-30",
                "meeting_id": str(OTHER_MEETING),
                "meeting_title": "Checkout sync",
                "customer": False,
            },
            {
                "id": str(A2),
                "content": "Send Acme the observability rollout plan",
                "due_date": None,
                "meeting_id": str(OTHER_MEETING),
                "meeting_title": "Acme QBR",
                "customer": True,
            },
        ],
        "already_read": False,
    }


def open_items() -> list[dict[str, Any]]:
    return [
        {
            "id": str(OPEN1),
            "decision_id": str(OLD_DECISION),
            "decision_content": "Standardise observability on Vendor A.",
            "kind": "dependency",
            "statement": "Vendor A stays our observability vendor through 2027.",
            "quote": "Vendor A is our tool for the next two years",
            "meeting_id": str(OLD_MEETING),
        },
        {
            "id": str(OPEN2),
            "decision_id": str(OLD_DECISION),
            "decision_content": "Standardise observability on Vendor A.",
            "kind": "price",
            "statement": "Vendor A costs under 50k.",
            "quote": None,
            "meeting_id": str(OLD_MEETING),
        },
    ]


class FakeCore:
    def __init__(
        self,
        body: dict[str, Any] | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        fail_open: int = 0,
    ) -> None:
        self.body = source_body() if body is None else body
        self.items = open_items() if items is None else items
        self.fail_open = fail_open
        self.calls: list[tuple[str, Any]] = []

    async def why_source(self, org_id: UUID, decision_id: UUID) -> dw.WhySource | None:
        self.calls.append(("why", decision_id))
        return dw.parse_why_source(self.body) if self.body else None

    async def open_assumptions(
        self,
        org_id: UUID,
        *,
        exclude_meeting_id: UUID | None = None,
        exclude_decision_id: UUID | None = None,
    ) -> list[dw.OpenAssumption]:
        self.calls.append(("open", (exclude_meeting_id, exclude_decision_id)))
        if self.fail_open > 0:
            self.fail_open -= 1
            raise SourceUnavailable("core-api down")
        return dw.parse_open({"items": self.items})


class FakeKg:
    def __init__(self, nodes: list[KgNode] | None = None, *, down: bool = False) -> None:
        self.nodes = nodes if nodes is not None else graph_nodes()
        self.down = down
        self.calls: list[tuple[UUID, str, dict[str, str]]] = []

    async def query(self, org_id: UUID, template: str, params: Any) -> list[KgNode]:
        self.calls.append((org_id, template, dict(params)))
        if self.down:
            raise KgUnavailable("down")
        return list(self.nodes)


def graph_nodes() -> list[KgNode]:
    return [
        KgNode("Action", str(GRAPH_ACTION), "match", 0, {"content": "Migrate the dashboards"}),
        KgNode("Action", str(A1), "match", 0, {"content": "Integrate Vendor A tracing"}),
        KgNode("Action", str(uuid.uuid4()), "match", 0, {"content": "Plan the layoffs"}),
        KgNode("Decision", str(uuid.uuid4()), "match", 0, {"content": "not an action"}),
    ]


ORG_CTX = CompanyContext(
    goals=(Goal(id=GOAL1, title="Cut infra spend 20%"), Goal(id=GOAL2, title="Hire 5 engineers")),
    never_record_topics=("layoffs",),
)


class FakeMinutes:
    def __init__(self, body: dict[str, Any] | None = None, *, fail: int = 0) -> None:
        self.body = body if body is not None else minutes_body()
        self.fail = fail
        self.calls = 0

    async def source(self, org_id: UUID, meeting_id: UUID) -> Any:
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise SourceUnavailable("not yet")
        return parse_source(self.body) if self.body else None


def minutes_body(**over: Any) -> dict[str, Any]:
    body = {
        "meeting_id": str(OTHER_MEETING),
        "title": "Platform sync",
        "confidential": False,
        "host_user_id": str(ALEX),
        "participants": [
            {"user_id": str(SAM), "name": EMAILS[SAM]},
            {"user_id": str(ALEX), "name": EMAILS[ALEX]},
        ],
        "speakers": {"s1": str(SAM), "s2": str(ALEX)},
        "turns": [
            {"speaker_id": "s1", "start_ms": 0, "end_ms": 5_000, "text": "Quick update on tools."},
            {
                "speaker_id": "s2",
                "start_ms": 6_000,
                "end_ms": 9_000,
                "text": "We are dropping Vendor A at the end of this quarter, it is signed.",
            },
            {
                "speaker_id": "s1",
                "start_ms": 70_000,
                "end_ms": 74_000,
                "text": "Also the layoffs list is final.",
            },
            {
                "speaker_id": "s2",
                "start_ms": 95_000,
                "end_ms": 99_000,
                "text": "Secret detail after that remark.",
            },
            {"speaker_id": "s1", "start_ms": 130_000, "end_ms": 134_000, "text": "Thanks all."},
        ],
    }
    body.update(over)
    return body


def meeting_check_answer() -> dict[str, Any]:
    return {
        "breaks": [
            {
                "assumption": "H1",
                "evidence_quote": "We are dropping Vendor A at the end of this quarter",
                "speaker": "Participant 2",
                "rationale": "Participant 2 says Vendor A is dropped this quarter.",
                "confidence": 0.9,
            },
            {
                "assumption": "H2",
                "evidence_quote": "Quick update on tools",
                "speaker": "Participant 1",
                "rationale": "Too weak.",
                "confidence": 0.5,
            },
        ]
    }


def make(
    llm: LLM | None = None,
    *,
    core: Any = "default",
    kg: Any = "default",
    minutes: Any = "default",
    plans: Any = "default",
    org: CompanyContext | None = ORG_CTX,
    settings: dw.WhySettings | None = None,
    consumed: int = 0,
    quota: int = 10**6,
    cap: int = 10**7,
) -> tuple[dw.DecisionWhy, LLM, dict[str, Any]]:
    llm = llm or LLM()

    async def org_context(_org: UUID) -> CompanyContext:
        if org is None:
            raise RuntimeError("brain down")
        return org

    parts: dict[str, Any] = {
        "core": FakeCore() if core == "default" else core,
        "kg": FakeKg() if kg == "default" else kg,
        "minutes": FakeMinutes() if minutes == "default" else minutes,
        "plans": FakePlans(True) if plans == "default" else plans,
        "usage": [],
    }

    class Usage:
        async def usage(self, **kw: Any) -> None:
            parts["usage"].append(kw)

    router = LiteLLMRouter(budget(consumed, quota, cap), Usage(), llm)
    why = dw.DecisionWhy(
        PromptLoader(PROMPTS_DIR),
        router,
        deps=dw.WhyDeps(
            core=parts["core"],
            minutes=parts["minutes"],
            kg=parts["kg"],
            plans=parts["plans"],
            org_context=org_context,
        ),
        settings=settings or dw.WhySettings(settle_s=0),
    )
    return why, llm, parts


def decision(**kw: Any) -> ConfirmedDecision:
    base: dict[str, Any] = {
        "org_id": ORG,
        "decision_id": DECISION,
        "meeting_id": MEETING,
        "content": "Go with Vendor B for observability.",
        "rationale": "Vendor B is under 40k a year and they ship SSO by March.",
        "confirmed_by": PRIYA,
        "supersedes": None,
    }
    return ConfirmedDecision(**{**base, **kw})


def by_type(records: list[dw.Record], alert_type: str) -> list[dw.Record]:
    return [r for r in records if r.result.payload["alert"]["alert_type"] == alert_type]


def assert_no_identity(llm: LLM) -> None:
    for call in llm.calls:
        text = json.dumps(call["messages"])
        assert not _UUID.search(text), "a user / object id reached the model"
        assert "@acme.example" not in text, "an email reached the model"


# --- 1. parsing ---------------------------------------------------------------------------------


def test_the_why_source_and_open_bodies_parse_tolerantly():
    s = dw.parse_why_source(source_body())
    assert s is not None and s.decision_id == DECISION and s.confirmed_by == PRIYA
    assert [a.id for a in s.candidate_actions] == [A1, A2] and s.candidate_actions[1].customer
    assert s.speakers == {"spk-1": PRIYA, "spk-2": SAM} and len(s.turns) == 3
    assert not s.confidential and not s.already_read
    body = source_body()
    body["decision"].pop("confidential")
    body["turns"].append({"speaker_id": "x", "text": "no start"})
    body["candidate_actions"].append({"id": "nope", "content": "bad id"})
    s2 = dw.parse_why_source(body)
    assert s2 is not None and s2.confidential  # unknown counts as confidential
    assert len(s2.turns) == 3 and len(s2.candidate_actions) == 2
    assert dw.parse_why_source({"decision": {"id": "x"}}) is None
    assert dw.parse_why_source([]) is None
    items = dw.parse_open({"items": [*open_items(), {"id": "bad"}, open_items()[0]]})
    assert [a.id for a in items] == [OPEN1, OPEN2]
    assert dw.parse_open({"items": [{**open_items()[0], "kind": "vibes"}]})[0].kind == "other"
    assert dw.parse_open(None) == []


def _labels() -> dw.Labels:
    s = dw.parse_why_source(source_body())
    assert s is not None
    return dw.Labels.build(s.turns, s.speakers, s.participants)


def test_labels_number_speakers_first_then_silent_participants():
    labels = _labels()
    # SAM speaks first, PRIYA second, ALEX never speaks
    assert labels.of[SAM] == "Participant 1" and labels.of[PRIYA] == "Participant 2"
    assert labels.of[ALEX] == "Participant 3"
    assert labels.person("participant #2") == PRIYA and labels.person("Participant 9") is None
    assert labels.of_speaker("spk-1") == "Participant 2"
    assert labels.swap("Participant 2 owns it", named=True) == "priya owns it"
    assert labels.swap("Participant 2 owns it", named=False) == "someone owns it"


def test_assumptions_map_labels_verify_quotes_and_validate_watch_fields():
    s = dw.parse_why_source(source_body())
    assert s is not None
    haystack = dw._norm(" ".join([s.content, s.rationale, s.quote, *(t.text for t in s.turns)]))
    out = dw.parse_assumptions(
        json.dumps(ASSUMPTIONS_ANSWER),
        labels=_labels(),
        sources=haystack,
        policy=NeverRecordPolicy.from_org(None),
    )
    assert out is not None
    assert [a["kind"] for a in out] == ["price", "person", "date", "other"]
    price, person, when, other = out
    assert price["speaker_user_id"] == str(PRIYA) and price["confidence"] == 0.9
    assert price["watch"] == {
        "date": None,
        "value": 40000.0,
        "comparator": "lt",
        "unit": "USD",
        "issue_key": None,
    }
    # label -> display name, URL removed, issue key said literally (upper-cased)
    assert person["statement"].startswith("priya owns the migration")
    assert "https" not in person["statement"] and "[link removed]" in person["statement"]
    assert person["speaker_user_id"] == str(SAM) and person["watch"]["issue_key"] == "OPS-42"
    # a quote nobody said is not a quote (and so has no speaker); a key nobody said is dropped;
    # a comparator without a value is dropped
    assert when["quote"] is None and when["speaker_user_id"] is None
    assert when["watch"]["date"] == "2027-03-31" and when["watch"]["issue_key"] is None
    assert when["watch"]["comparator"] is None
    assert other["statement"] == "Vendor B keeps its EU region."
    assert all(set(a) == ASSUMPTION_KEYS and set(a["watch"]) == WATCH_KEYS for a in out)


def test_assumptions_cap_dedupe_never_record_and_unparseable():
    many = {
        "assumptions": [
            {"kind": "other", "statement": f"Reason number {i}.", "confidence": 0.9}
            for i in range(8)
        ]
        + [{"kind": "other", "statement": "Reason number 0.", "confidence": 0.9}]
    }
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("layoffs",)))
    labels = _labels()
    out = dw.parse_assumptions(json.dumps(many), labels=labels, sources="", policy=policy)
    assert out is not None and len(out) == 5
    leak = {"assumptions": [{"kind": "other", "statement": "The layoffs stay", "confidence": 1}]}
    assert dw.parse_assumptions(json.dumps(leak), labels=labels, sources="", policy=policy) == []
    assert dw.parse_assumptions("no json", labels=labels, sources="", policy=policy) is None
    fenced = "```json\n" + json.dumps({"assumptions": []}) + "\n```"
    assert dw.parse_assumptions(fenced, labels=labels, sources="", policy=policy) == []


def test_verbatim_ignores_case_punctuation_and_allows_an_ellipsis():
    hay = dw._norm("Honestly, we go with Vendor B -- because it's under 40k a year.")
    assert dw.verbatim("we go with vendor B because it's under 40k", hay)
    assert dw.verbatim("we go with Vendor B ... under 40k a year", hay)
    assert not dw.verbatim("we go with Vendor C", hay)
    assert not dw.verbatim("...", hay)


def test_ripple_maps_labels_to_ids_and_drops_bad_entries():
    s = dw.parse_why_source(source_body())
    assert s is not None
    parsed = dw.parse_ripple(
        json.dumps(RIPPLE_ANSWER),
        goals=list(ORG_CTX.goals),
        actions=s.candidate_actions,
        policy=NeverRecordPolicy.from_org(None),
    )
    assert parsed is not None
    goals, actions = parsed
    assert goals == [
        {
            "goal_id": GOAL1,
            "label": "Cut infra spend 20%",
            "effect": "advances",
            "why": "Cheaper tooling cuts spend.",
            "confidence": 0.8,
        }
    ]
    assert [a["action_id"] for a in actions] == [str(A1)]
    assert actions[0]["source"] == "agent" and set(actions[0]) == ACTION_KEYS
    assert dw.parse_ripple("nope", goals=[], actions=[], policy=NeverRecordPolicy()) is None
    assert dw.resolve_ref("g2", "G", 3) == 1 and dw.resolve_ref(2, "G", 3) == 1
    assert dw.resolve_ref("A2", "G", 3) is None and dw.resolve_ref(True, "G", 3) is None


def test_breaks_need_a_verbatim_quote_and_the_floor():
    items = dw.parse_open({"items": open_items()})
    hay = dw._norm("Decision: Go with Vendor B for observability.")
    out = dw.parse_breaks(
        json.dumps(CHECK_ANSWER),
        open_items=items,
        labels=_labels(),
        evidence=hay,
        policy=NeverRecordPolicy(),
    )
    assert out is not None and [a.id for a, _b in out] == [OPEN1]
    low = {"breaks": [{**CHECK_ANSWER["breaks"][0], "confidence": 0.59}]}
    found = dw.parse_breaks(
        json.dumps(low),
        open_items=items,
        labels=_labels(),
        evidence=hay,
        policy=NeverRecordPolicy(),
    )
    assert found == []
    labelled = {"breaks": [{**CHECK_ANSWER["breaks"][0], "rationale": "Participant 2 moved it."}]}
    (_a, b), *_ = dw.parse_breaks(
        json.dumps(labelled),
        open_items=items,
        labels=_labels(),
        evidence=hay,
        policy=NeverRecordPolicy(),
    ) or [(None, {})]
    assert b["rationale"] == "someone moved it."


# --- 2. the decision hook ----------------------------------------------------------------------


async def test_a_decision_gets_one_why_record_with_the_exact_contract_payload():
    why, llm, parts = make()
    records = await why.on_decision(decision(), DAY)
    [rec] = by_type(records, "decision_why")
    assert rec.org_id == ORG and rec.meeting_id == MEETING and rec.user_id == PRIYA
    assert rec.window_id == f"why:{DECISION}" and rec.result.window_id == rec.window_id
    assert rec.result.kind == "alert" and rec.result.content == (
        "Why: Go with Vendor B for observability."
    )
    p = rec.result.payload
    assert set(p) == TOP_1A and set(p["alert"]) == ALERT_KEYS and set(p["why"]) == WHY_KEYS
    assert p["prompt"] == "decision_assumptions@v1" and p["model_tier"] == "extract"
    alert = p["alert"]
    assert alert["id"] == str(uuid5(CANDIDATE_NS, f"ProactiveAgent:decision_why:{DECISION}"))
    assert alert["alert_type"] == "decision_why" and alert["priority"] == 3
    assert alert["surfaced"] is False and alert["title"] == "Why this decision was made"
    assert alert["subject"] == {"kind": "decision", "id": str(DECISION)}
    assert alert["related"] == [] and alert["link"] == f"/decisions/{DECISION}"
    assert alert["body"] == "" and alert["rationale"] == ""
    assert rec.result.confidence == alert["confidence"] == 0.9  # the best assumption
    w = p["why"]
    assert w["decision_id"] == str(DECISION) and len(w["assumptions"]) == 4
    assert all(set(a) == ASSUMPTION_KEYS for a in w["assumptions"])
    assert set(w["ripple"]) == {"goals", "actions"}
    assert all(set(g) == GOAL_KEYS for g in w["ripple"]["goals"])
    assert all(set(a) == ACTION_KEYS for a in w["ripple"]["actions"])
    # the agent's A1 wins over the graph's copy; the never-record and non-action nodes are out
    actions = {a["action_id"]: a for a in w["ripple"]["actions"]}
    assert list(actions) == [str(A1), str(GRAPH_ACTION)]
    assert actions[str(A1)]["source"] == "agent"
    assert actions[str(GRAPH_ACTION)] == {
        "action_id": str(GRAPH_ACTION),
        "label": "Migrate the dashboards",
        "effect": "changes",
        "why": "An open task from the meeting that made this decision.",
        "confidence": 0.7,
        "source": "graph",
    }
    assert parts["kg"].calls == [
        (ORG, "actions_blocked_on_decision", {"decision_id": str(DECISION)})
    ]
    assert json.loads(json.dumps(p)) == p  # serializable as payload_json


async def test_every_call_is_metered_to_the_tenant_and_never_names_anyone():
    why, llm, parts = make()
    await why.on_decision(decision(), DAY)
    assert len(llm.of(ASSUME)) == 1 and len(llm.of(RIPPLE)) == 1 and len(llm.of(CHECK)) == 1
    assert all(u["user_id"] is None and u["org_id"] == ORG for u in parts["usage"])
    assert {c["model"] for c in llm.calls} == {"anthropic/claude-sonnet-5-5"}
    assert_no_identity(llm)
    assume = LLM.user(llm.of(ASSUME)[0])
    assert "Meeting date: 2026-11-11" in assume
    assert "Participant 2: Honestly we go with Vendor B" in assume
    assert "(said by Participant 2)" in assume
    ripple = LLM.user(llm.of(RIPPLE)[0])
    assert "G1. Cut infra spend 20%" in ripple and "A2. [customer] Send Acme" in ripple
    assert ripple.count("Cut infra spend") == 1  # the goals are not listed twice


async def test_empty_answers_still_send_the_record_so_core_api_marks_it_read():
    why, llm, _ = make(
        LLM(assumptions={"assumptions": []}, ripple={"goals": [], "actions": []}),
        kg=FakeKg([]),
    )
    [rec] = by_type(await why.on_decision(decision(), DAY), "decision_why")
    w = rec.result.payload["why"]
    assert w["assumptions"] == [] and w["ripple"] == {"goals": [], "actions": []}
    assert rec.result.confidence == 0.0


async def test_no_ripple_call_without_goals_or_actions_and_kg_down_degrades():
    body = source_body()
    body["candidate_actions"] = []
    why, llm, _ = make(core=FakeCore(body), kg=FakeKg(down=True), org=CompanyContext())
    [rec] = by_type(await why.on_decision(decision(), DAY), "decision_why")
    assert llm.of(RIPPLE) == [] and rec.result.payload["why"]["ripple"]["actions"] == []


async def test_a_failing_ripple_keeps_the_assumptions_and_an_unreadable_answer_sends_nothing():
    why, _, _ = make(LLM(ripple=RuntimeError("provider down")), kg=FakeKg([]))
    [rec] = by_type(await why.on_decision(decision(), DAY), "decision_why")
    assert rec.result.payload["why"]["ripple"]["goals"] == []
    assert len(rec.result.payload["why"]["assumptions"]) == 4
    why2, _, _ = make(LLM(assumptions="I cannot help with that"))
    assert by_type(await why2.on_decision(decision(), DAY), "decision_why") == []


@pytest.mark.parametrize(
    "case",
    ["why_off", "plan_off", "blocked", "throttled", "no_owner", "already_read", "brain_down"],
)
async def test_gating_sends_nothing_and_calls_no_model(case: str):
    kw: dict[str, Any] = {}
    d = decision()
    if case == "why_off":
        kw["settings"] = dw.WhySettings(why_enabled=False, watch_enabled=False)
    elif case == "plan_off":
        kw["plans"] = FakePlans(False)
    elif case == "blocked":
        kw.update(consumed=10**7 + 1, quota=10**6, cap=10**7)
    elif case == "throttled":
        kw.update(consumed=10**6 + 1, quota=10**6, cap=10**7)
    elif case == "no_owner":
        d = decision(confirmed_by=None)
    elif case == "already_read":
        kw["core"] = FakeCore({**source_body(), "already_read": True})
    elif case == "brain_down":
        kw["org"] = None
    why, llm, _ = make(**kw)
    assert await why.on_decision(d, DAY) == []
    assert llm.calls == []


async def test_all_plans_overrides_the_plan_flag():
    why, _, _ = make(plans=None, settings=dw.WhySettings(all_plans=True))
    assert by_type(await why.on_decision(decision(), DAY), "decision_why")
    why2, llm2, _ = make(plans=None)
    assert await why2.on_decision(decision(), DAY) == [] and llm2.calls == []


async def test_why_only_and_watch_only_flags():
    why, llm, _ = make(settings=dw.WhySettings(watch_enabled=False))
    records = await why.on_decision(decision(), DAY)
    assert by_type(records, "decision_why") and llm.of(CHECK) == []
    why2, llm2, _ = make(settings=dw.WhySettings(why_enabled=False))
    records2 = await why2.on_decision(decision(), DAY)
    assert by_type(records2, "assumption_broken") and not by_type(records2, "decision_why")
    assert llm2.of(ASSUME) == [] and llm2.of(RIPPLE) == []


async def test_a_never_record_decision_gets_nothing():
    why, llm, _ = make()
    assert await why.on_decision(decision(content="Plan the layoffs for Q1."), DAY) == []
    body = source_body(quote="the layoffs make this cheaper")
    why2, llm2, _ = make(core=FakeCore(body))
    assert await why2.on_decision(decision(), DAY) == []
    assert llm.calls == [] and llm2.calls == []


async def test_never_record_windows_are_removed_before_the_model_sees_the_transcript():
    body = source_body()
    body["turns"] += [
        {"speaker_id": "spk-2", "start_ms": 90_000, "end_ms": 92_000, "text": "About layoffs."},
        {"speaker_id": "spk-1", "start_ms": 125_000, "end_ms": 127_000, "text": "Canary-77 x."},
        {"speaker_id": "spk-1", "start_ms": 160_000, "end_ms": 162_000, "text": "Back to tools."},
    ]
    why, llm, _ = make(core=FakeCore(body))
    await why.on_decision(decision(), DAY)
    sent = LLM.user(llm.of(ASSUME)[0]).split("Transcript around the decision:")[1]
    assert "layoffs" not in sent.lower() and "Canary-77" not in sent
    assert "Back to tools." in sent and "Vendor B because" in sent


# --- 3. trigger A ------------------------------------------------------------------------------


async def test_trigger_a_sends_the_exact_contract_1b_payload():
    why, llm, parts = make()
    [rec] = by_type(await why.on_decision(decision(), DAY), "assumption_broken")
    assert parts["core"].calls[-1] == ("open", (MEETING, DECISION))
    window = f"assumption-check:decision:{DECISION}"
    assert rec.window_id == window and rec.user_id == PRIYA and rec.meeting_id == MEETING
    p = rec.result.payload
    assert set(p) == TOP_1B and set(p["alert"]) == ALERT_KEYS
    assert p["prompt"] == "assumption_check@v1" and p["model_tier"] == "extract"
    alert = p["alert"]
    assert alert["id"] == str(uuid5(CANDIDATE_NS, window))
    assert alert["alert_type"] == "assumption_broken" and alert["priority"] == 2
    assert alert["surfaced"] is False and alert["link"] == "" and alert["related"] == []
    assert alert["title"] == "An assumption may no longer hold"
    assert alert["subject"] == {"kind": "decision", "id": str(DECISION)}
    [brk] = p["breaks"]  # H2's quote was never written: dropped
    assert set(brk) == BREAK_KEYS
    assert brk == {
        "assumption_id": str(OPEN1),
        "confidence": 0.85,
        "evidence_quote": "Go with Vendor B for observability",
        "rationale": "The new decision moves observability off Vendor A.",
        "source": "decision",
        "source_meeting_id": str(MEETING),
        "source_decision_id": str(DECISION),
        "speaker_user_id": None,
    }
    sent = LLM.user(llm.of(CHECK)[0])
    assert "H1. [dependency] Vendor A stays" in sent and "a newly confirmed decision" in sent
    assert_no_identity(llm)


async def test_trigger_a_makes_no_call_without_open_assumptions_or_from_a_confidential_decision():
    why, llm, _ = make(core=FakeCore(items=[]))
    assert not by_type(await why.on_decision(decision(), DAY), "assumption_broken")
    assert llm.of(CHECK) == []
    why2, llm2, parts = make(core=FakeCore(source_body(confidential=True)))
    records = await why2.on_decision(decision(), DAY)
    assert by_type(records, "decision_why") and not by_type(records, "assumption_broken")
    assert llm2.of(CHECK) == [] and all(c[0] != "open" for c in parts["core"].calls)


async def test_trigger_a_core_down_still_sends_the_why():
    why, llm, _ = make(core=FakeCore(fail_open=1))
    records = await why.on_decision(decision(), DAY)
    assert by_type(records, "decision_why") and not by_type(records, "assumption_broken")


# --- 4. the proactive loop ---------------------------------------------------------------------


def _decision_value() -> dict[str, Any]:
    from tests.test_proactive_consumer import decision_value

    value = decision_value()
    value["decision_id"], value["meeting_id"] = str(DECISION), str(MEETING)
    value["confirmed_by"] = str(PRIYA)
    value["header"]["org_id"] = str(ORG)
    return value


def _loop(why: dw.DecisionWhy | None) -> tuple[ProactiveLoop, LoopbackProducer, Any]:
    from tests.test_proactive_agent import FakeCore as ProactiveCore
    from tests.test_proactive_agent import FakeKg as ProactiveKg
    from tests.test_proactive_agent import lineage_node, person

    prior = uuid.uuid4()
    agent, _llm, _ = make_proactive(
        kg=ProactiveKg(
            {("decision_lineage", str(DECISION)): [lineage_node(prior, "Use Vendor A.", 1)]}
        ),
        core=ProactiveCore(participants={MEETING: [person(PRIYA), person(SAM)]}),
    )
    producer = LoopbackProducer()
    return ProactiveLoop(agent, Emitter(producer, JsonCodec()), why=why), producer, agent


async def test_the_loop_sends_conflict_alerts_and_the_why_records_in_one_flush():
    why, _, _ = make()
    loop, producer, _ = _loop(why)
    assert await loop.handle("decisions", _decision_value()) == 4  # 2 conflict + why + broken
    assert producer.flushes == 1
    records = producer.on("agent-results")
    windows = sorted(r["window_id"] for r in records)
    assert windows == sorted(
        [
            f"proactive:conflict:{DECISION}",
            f"proactive:conflict:{DECISION}",
            f"why:{DECISION}",
            f"assumption-check:decision:{DECISION}",
        ]
    )
    why_rec = next(r for r in records if r["window_id"] == f"why:{DECISION}")
    assert why_rec["agent"] == "ProactiveAgent" and why_rec["kind"] == "alert"
    assert why_rec["user_id"] == str(PRIYA) and why_rec["header"]["org_id"] == str(ORG)
    assert set(json.loads(why_rec["payload_json"])) == TOP_1A


async def test_a_failing_why_never_stops_the_conflict_alerts(monkeypatch):
    why, _, _ = make()

    async def boom(*_a: Any, **_k: Any) -> list[dw.Record]:
        raise RuntimeError("bug in the why")

    monkeypatch.setattr(why, "on_decision", boom)
    loop, producer, _ = _loop(why)
    assert await loop.handle("decisions", _decision_value()) == 2
    assert {r["window_id"] for r in producer.on("agent-results")} == {
        f"proactive:conflict:{DECISION}"
    }


async def test_a_failing_conflict_check_never_stops_the_why(monkeypatch):
    why, _, _ = make()
    loop, producer, agent = _loop(why)

    async def boom(*_a: Any, **_k: Any) -> list[Any]:
        raise RuntimeError("bug in the conflict check")

    monkeypatch.setattr(agent, "check_decision", boom)
    assert await loop.handle("decisions", _decision_value()) == 2
    assert {r["window_id"] for r in producer.on("agent-results")} == {
        f"why:{DECISION}",
        f"assumption-check:decision:{DECISION}",
    }


async def test_actions_and_a_loop_without_why_are_unchanged():
    loop, producer, _ = _loop(None)
    assert await loop.handle("decisions", _decision_value()) == 2
    why, llm, _ = make()
    loop2, _, _ = _loop(why)
    from tests.test_proactive_consumer import action_value

    await loop2.handle("actions", action_value())
    assert llm.calls == []


def test_event_date_reads_micros_datetimes_and_falls_back_to_today():
    from datetime import UTC, datetime

    micros = int(datetime(2026, 3, 4, 12, tzinfo=UTC).timestamp() * 1_000_000)
    assert dw.event_date({"header": {"occurred_at": micros}}) == date(2026, 3, 4)
    assert dw.event_date({"header": {"occurred_at": datetime(2026, 3, 5, 1)}}) == date(2026, 3, 5)
    assert isinstance(dw.event_date({}), date)


# --- 5. trigger B ------------------------------------------------------------------------------


def ended_value(org: UUID = ORG, meeting: UUID = OTHER_MEETING, key: str = "k1") -> dict[str, Any]:
    from datetime import UTC, datetime

    return {
        "header": {
            "org_id": str(org),
            "idempotency_key": key,
            "occurred_at": int(datetime(2026, 11, 11, 15, tzinfo=UTC).timestamp() * 1_000_000),
        },
        "meeting_id": str(meeting),
        "event_type": "ended",
    }


def watch_loop(
    why: dw.DecisionWhy, **kw: Any
) -> tuple[dw.AssumptionWatchLoop, LoopbackProducer, Any, list[float]]:
    producer = LoopbackProducer()
    redis = fakeredis.aioredis.FakeRedis()
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    loop = dw.AssumptionWatchLoop(
        why, Emitter(producer, JsonCodec()), claims=redis, sleep=sleep, **kw
    )
    return loop, producer, redis, slept


async def test_an_ended_meeting_breaks_an_assumption_with_the_exact_payload_and_claims_it():
    why, llm, parts = make(LLM(check=meeting_check_answer()))
    loop, producer, redis, slept = watch_loop(why, settle_s=15)
    assert await loop.handle(ended_value()) == 1
    assert producer.flushes == 1 and slept == [15]
    [rec] = producer.on("agent-results")
    generation = dw.generation_of("k1", OTHER_MEETING)
    window = f"assumption-check:meeting:{OTHER_MEETING}:{generation}"
    assert rec["window_id"] == window and rec["meeting_id"] == str(OTHER_MEETING)
    assert rec["user_id"] == str(ALEX)  # the host
    p = json.loads(rec["payload_json"])
    assert set(p) == TOP_1B and set(p["alert"]) == ALERT_KEYS
    assert p["alert"]["subject"] == {"kind": "meeting", "id": str(OTHER_MEETING)}
    assert p["alert"]["id"] == str(uuid5(CANDIDATE_NS, window))
    [brk] = p["breaks"]  # H2 is under 0.6
    assert brk == {
        "assumption_id": str(OPEN1),
        "confidence": 0.9,
        "evidence_quote": "We are dropping Vendor A at the end of this quarter",
        "rationale": "someone says Vendor A is dropped this quarter.",
        "source": "meeting",
        "source_meeting_id": str(OTHER_MEETING),
        "source_decision_id": None,
        "speaker_user_id": str(ALEX),
    }
    assert parts["core"].calls == [("open", (OTHER_MEETING, None))]
    sent = LLM.user(llm.of(CHECK)[0])
    assert "the transcript of a meeting that just ended, 2026-11-11" in sent
    evidence = sent.split("New evidence")[1]
    assert "layoffs" not in evidence.lower() and "Secret detail" not in evidence
    assert "Thanks all." in evidence
    assert_no_identity(llm)
    assert await redis.exists(dw.claim_key(ORG, OTHER_MEETING, generation))
    # a replay of the same `ended` is claimed: no second call, nothing sent
    assert await loop.handle(ended_value()) == 0 and len(llm.of(CHECK)) == 1


async def test_ended_meeting_skips_confidential_unknown_quiet_and_unrelated_records():
    for minutes in (
        FakeMinutes(minutes_body(confidential=True)),
        FakeMinutes({}),
        FakeMinutes(minutes_body(turns=[])),
    ):
        why, llm, _ = make(minutes=minutes)
        loop, producer, _, _ = watch_loop(why, settle_s=0)
        assert await loop.handle(ended_value()) == 0
        assert llm.calls == [] and producer.flushes == 0
    why, llm, _ = make()
    loop, _, _, _ = watch_loop(why, settle_s=0)
    assert await loop.handle({**ended_value(), "event_type": "started"}) == 0
    assert await loop.handle({"event_type": "ended", "header": {}}) == 0
    why2, llm2, _ = make(core=FakeCore(items=[]))
    loop2, _, _, _ = watch_loop(why2, settle_s=0)
    assert await loop2.handle(ended_value()) == 0 and llm2.calls == []
    why3, llm3, _ = make(settings=dw.WhySettings(watch_enabled=False))
    loop3, _, _, _ = watch_loop(why3, settle_s=0)
    assert await loop3.handle(ended_value()) == 0 and llm3.calls == []


async def test_ended_meeting_retries_core_and_the_brain_then_gives_up_unclaimed():
    why, llm, _ = make(
        LLM(check=meeting_check_answer()),
        minutes=FakeMinutes(fail=1),
        core=FakeCore(fail_open=1),
    )
    loop, producer, redis, slept = watch_loop(why, settle_s=0, attempts=3, backoff_s=10)
    assert await loop.handle(ended_value()) == 1 and slept == [10, 20]
    why2, llm2, _ = make(org=None)
    loop2, producer2, redis2, _ = watch_loop(why2, settle_s=0, attempts=2, backoff_s=1)
    assert await loop2.handle(ended_value()) == 0 and producer2.flushes == 0
    generation = dw.generation_of("k1", OTHER_MEETING)
    assert not await redis2.exists(dw.claim_key(ORG, OTHER_MEETING, generation))


async def test_the_recipient_is_the_host_else_the_first_participant_else_nobody():
    s = parse_source(minutes_body(host_user_id=None))
    assert s is not None and dw.recipient(s) == SAM
    s2 = parse_source(minutes_body(host_user_id=None, participants=[]))
    assert s2 is not None and dw.recipient(s2) is None


def test_the_watch_consumer_is_its_own_group_from_the_end():
    cfg = dw.consumer_config("b:9092")
    assert cfg["group.id"] == "agent-runtime-assumptions" == dw.GROUP
    assert cfg["auto.offset.reset"] == "latest" and cfg["enable.auto.commit"] is False
    assert dw.TOPIC == "meeting-events"


def test_settings_read_the_env(monkeypatch):
    monkeypatch.setenv("DECISION_WHY_AGENT", "0")
    monkeypatch.setenv("ASSUMPTION_WATCH_AGENT", "off")
    monkeypatch.setenv("PROACTIVE_AGENT_ALL_PLANS", "1")
    monkeypatch.setenv("ASSUMPTION_WATCH_SETTLE_S", "3")
    s = dw.WhySettings.from_env()
    assert not s.why_enabled and not s.watch_enabled and s.all_plans and s.settle_s == 3.0
    monkeypatch.delenv("DECISION_WHY_AGENT")
    monkeypatch.delenv("ASSUMPTION_WATCH_AGENT")
    s2 = dw.WhySettings.from_env()
    assert s2.why_enabled and s2.watch_enabled


# --- 6. the HTTP client ------------------------------------------------------------------------


async def test_the_http_client_sends_the_caller_headers_and_reads_both_routes():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/why-source"):
            return httpx.Response(200, json=source_body())
        return httpx.Response(200, json={"items": open_items()})

    client = dw.HttpWhyCore("http://core/", "tok", transport=httpx.MockTransport(handler))
    s = await client.why_source(ORG, DECISION)
    assert s is not None and s.decision_id == DECISION
    items = await client.open_assumptions(
        ORG, exclude_meeting_id=MEETING, exclude_decision_id=DECISION
    )
    assert [a.id for a in items] == [OPEN1, OPEN2]
    first, second = seen
    assert first.url.path == f"/internal/decisions/{DECISION}/why-source"
    assert first.headers["X-Mytheus-Service"] == "agent-runtime-service"
    assert first.headers["X-Mytheus-Service-Token"] == "tok"
    assert first.headers["X-Mytheus-Org"] == str(ORG)
    assert second.url.path == "/internal/assumptions/open"
    assert dict(second.url.params) == {
        "exclude_meeting_id": str(MEETING),
        "exclude_decision_id": str(DECISION),
    }
    assert str(ORG) not in str(second.url)


@pytest.mark.parametrize("status", [404, 409, 500])
async def test_the_http_client_skips_on_an_error_and_open_retries_on_5xx(status: int):
    client = dw.HttpWhyCore(
        "http://core", "tok", transport=httpx.MockTransport(lambda r: httpx.Response(status))
    )
    assert await client.why_source(ORG, DECISION) is None
    if status >= 500:
        with pytest.raises(SourceUnavailable):
            await client.open_assumptions(ORG)
    else:
        assert await client.open_assumptions(ORG) == []

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    dead = dw.HttpWhyCore("http://core", "tok", transport=httpx.MockTransport(down))
    assert await dead.why_source(ORG, DECISION) is None
    with pytest.raises(SourceUnavailable):
        await dead.open_assumptions(ORG)


def test_the_prompts_are_registered_with_their_parse_formats():
    from libs.shared.agent_metrics import agent_for_prompt, output_ok

    loader = PromptLoader(PROMPTS_DIR)
    for key in (dw.ASSUMPTIONS_KEY, dw.RIPPLE_KEY, dw.CHECK_KEY):
        prompt = loader.load(key, loader.resolve(key, ORG))
        assert prompt.version == "v1" and "Never record" in prompt.system
        assert agent_for_prompt(prompt.ref) == "ProactiveAgent"
        assert "{company}{ground_rules}" in prompt.user_template.replace("\n", "")
    assert output_ok("decision_assumptions@v1", '{"assumptions": []}') is True
    assert output_ok("decision_ripple@v1", '{"goals": [], "actions": []}') is True
    assert output_ok("assumption_check@v1", "nope") is False
