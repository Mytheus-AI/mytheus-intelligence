"""`ProactiveAgent` (P11): plan 10-13 section 3.6, the P11 contract sections 1 and 2.

Every model call, rag-service, core-api, billing and the knowledge graph are faked here (the KG
gRPC client is exercised against the real servicer in `test_proactive_clients.py`; the Avro and
broker paths in `test_proactive_consumer.py`). The router, the budget gate, the prompt loader,
the runtime and the real `Emitter` run as in production.

1. conflict: fires on a contradicting candidate; a merely related decision (scripted "no
   conflict") does not; the superseded decision is never a candidate; KG down or slow -> the rag
   path still finds it (R1); the KG race (new decision not in the graph yet) walks from
   `supersedes`; lineage depth <= 3.
2. workload: at / below the threshold; the env threshold; recipients.
3. dependency: an unfinished prerequisite fires, all done does not.
4. missing_stakeholder: fires once per (meeting, person) through the runtime; KG down -> skipped
   before the gate is paid for; a topic outside the taxonomy is nothing.
5. Gating: budget throttled / blocked -> nothing and no model call; plan flag false -> nothing,
   true or `PROACTIVE_AGENT_ALL_PLANS=1` -> runs; `PROACTIVE_AGENT=0` -> nothing.
6. Output: the exact contract payload, deterministic ids, one result per recipient with the
   right priority, bands, recipients capped at 10, no user id or email in any model call.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any
from uuid import UUID, uuid5

import pytest

from libs.shared.kg_client import KgNode, KgUnavailable
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS
from services.agent_runtime_service.agents.proactive_agent import (
    ConfirmedAction,
    ConfirmedDecision,
    ProactiveAgent,
    ProactiveDeps,
    ProactiveSettings,
    RagDecisionSearch,
    alert_id,
    display_name,
    parse_action,
    parse_decision,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.proactive.core import ActionContext, Dependency, Participant
from services.agent_runtime_service.runtime import AgentRuntime
from tests.a2a_helpers import JsonCodec, LoopbackProducer, ScriptedRag, budget, rag_passage

ORG = uuid.uuid4()
MEETING, OLD_MEETING, OLDER_MEETING = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
PRIYA, SAM, ALEX, ADMIN, ORGANIZER, OUTSIDER = (uuid.uuid4() for _ in range(6))
DECISION, PRIOR, SUPERSEDED, GRANDPARENT = (uuid.uuid4() for _ in range(4))
ACTION, DEP_OPEN, DEP_DONE = (uuid.uuid4() for _ in range(3))
EMAILS = {u: f"user{u.hex[:6]}@acme.example" for u in (PRIYA, SAM, ALEX, ADMIN, ORGANIZER)}
EMAILS[PRIYA] = "priya@acme.example"

NEW_TEXT = "Launch the analytics beta on 2 December."
PRIOR_TEXT = "The analytics beta launches on 15 January, after the security review."

CONFLICT_MARKER = "newly confirmed team decision contradicts"
STAKEHOLDER_MARKER = "names the main topic"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class ScriptedLLM:
    """Sonnet answers the conflict judge with `conflict`; Haiku answers the stakeholder gate
    with `topic`; every other prompt scores 0."""

    def __init__(self, conflict: Any = None, topic: Any = None) -> None:
        self.conflict = conflict if conflict is not None else conflicts((1, 0.9))
        self.topic = topic if topic is not None else {"topic": "pricing", "score": 0.9}
        self.calls: list[dict[str, Any]] = []

    def of(self, marker: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if marker in c["messages"][0]["content"][0]["text"]]

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system = kw["messages"][0]["content"][0]["text"]
        if CONFLICT_MARKER in system:
            text = json.dumps(self.conflict)
        elif STAKEHOLDER_MARKER in system:
            text = json.dumps(self.topic)
        else:
            text = '{"score": 0}'
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40},
        }


def conflicts(*entries: tuple[int, float]) -> dict[str, Any]:
    return {
        "conflicts": [
            {"candidate": i, "confidence": c, "rationale": "The launch dates are incompatible."}
            for i, c in entries
        ]
    }


class UsageSink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)


class FakeKg:
    def __init__(
        self,
        answers: dict[tuple[str, str], list[KgNode]] | None = None,
        *,
        down: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.answers = answers or {}
        self.down = down
        self.delay = delay
        self.calls: list[tuple[UUID, str, dict[str, str]]] = []

    async def query(self, org_id: UUID, template: str, params: Any) -> list[KgNode]:
        self.calls.append((org_id, template, dict(params)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.down:
            raise KgUnavailable("down")
        key = params.get("decision_id") or params.get("topic") or ""
        return list(self.answers.get((template, key), []))


class FakeCore:
    def __init__(
        self,
        contexts: dict[UUID, ActionContext] | None = None,
        participants: dict[UUID, list[Participant]] | None = None,
    ) -> None:
        self.contexts = contexts or {}
        self.people = participants or {}
        self.calls: list[tuple[str, UUID, UUID]] = []

    async def action_context(self, org_id: UUID, action_id: UUID) -> ActionContext | None:
        self.calls.append(("context", org_id, action_id))
        return self.contexts.get(action_id)

    async def participants(self, org_id: UUID, meeting_id: UUID) -> list[Participant] | None:
        self.calls.append(("participants", org_id, meeting_id))
        return self.people.get(meeting_id)


class FakePlans:
    def __init__(self, on: bool = True) -> None:
        self.on = on
        self.calls: list[tuple[UUID, str]] = []

    async def enabled(self, org_id: UUID, feature: str) -> bool:
        self.calls.append((org_id, feature))
        return self.on


def person(uid: UUID, tier: str | None = "member") -> Participant:
    return Participant(uid, EMAILS.get(uid, f"{uid.hex[:6]}@acme.example"), tier)


def lineage_node(did: UUID, text: str, depth: int, role: str = "ancestor") -> KgNode:
    return KgNode(
        "Decision", str(did), role, depth, {"content": text, "meeting_id": str(OLD_MEETING)}
    )


def topic_node(did: UUID, text: str, meeting: UUID) -> KgNode:
    return KgNode("Decision", str(did), "match", 0, {"content": text, "meeting_id": str(meeting)})


def decision(**kw: Any) -> ConfirmedDecision:
    base: dict[str, Any] = {
        "org_id": ORG,
        "decision_id": DECISION,
        "meeting_id": MEETING,
        "content": NEW_TEXT,
        "rationale": "Marketing wants it before the holidays.",
        "confirmed_by": PRIYA,
        "supersedes": None,
    }
    return ConfirmedDecision(**{**base, **kw})


def action(**kw: Any) -> ConfirmedAction:
    base: dict[str, Any] = {
        "org_id": ORG,
        "action_id": ACTION,
        "meeting_id": MEETING,
        "content": "Draft the beta launch checklist",
        "assignee_id": PRIYA,
    }
    return ConfirmedAction(**{**base, **kw})


def ctx(**kw: Any) -> ActionContext:
    base: dict[str, Any] = {
        "assignee_id": PRIYA,
        "assignee_name": EMAILS[PRIYA],
        "meeting_id": MEETING,
        "open_action_count": 6,
        "organizer_id": ORGANIZER,
        "depends_on": [],
        "participants": [PRIYA, SAM, ALEX, ADMIN, ORGANIZER],
        "admins": [ADMIN, OUTSIDER],
    }
    return ActionContext(**{**base, **kw})


def make(
    llm: ScriptedLLM | None = None,
    *,
    kg: Any = "default",
    rag: Any = "default",
    core: Any = "default",
    plans: Any = "default",
    settings: ProactiveSettings | None = None,
    consumed: int = 0,
    quota: int = 10**6,
    cap: int = 10**7,
) -> tuple[ProactiveAgent, ScriptedLLM, dict[str, Any]]:
    llm = llm or ScriptedLLM()
    parts: dict[str, Any] = {
        "kg": (
            FakeKg({("decision_lineage", str(DECISION)): [lineage_node(PRIOR, PRIOR_TEXT, 1)]})
            if kg == "default"
            else kg
        ),
        "rag": ScriptedRag({}) if rag == "default" else rag,
        "core": (
            FakeCore(participants={MEETING: [person(PRIYA), person(SAM), person(ALEX)]})
            if core == "default"
            else core
        ),
        "plans": FakePlans(True) if plans == "default" else plans,
        "sink": UsageSink(),
    }
    router = LiteLLMRouter(budget(consumed, quota, cap), parts["sink"], llm)
    deps = ProactiveDeps(
        kg=parts["kg"],
        decisions=RagDecisionSearch(parts["rag"]) if parts["rag"] is not None else None,
        core=parts["core"],
        plans=parts["plans"],
        topics=["pricing", "roadmap", "infra"],
    )
    agent = ProactiveAgent(
        PromptLoader(PROMPTS_DIR), router, deps=deps, settings=settings or ProactiveSettings()
    )
    return agent, llm, parts


def payloads(alerts: list[Any]) -> list[tuple[UUID, dict[str, Any]]]:
    return [(u, r.payload) for a in alerts for u, r in a.results()]


# --- conflict ------------------------------------------------------------------------------


async def test_a_contradicting_lineage_ancestor_raises_one_red_alert_per_recipient():
    agent, llm, parts = make()
    alerts = await agent.check_decision(decision())
    assert len(alerts) == 1 and len(llm.of(CONFLICT_MARKER)) == 1
    results = alerts[0].results()
    assert [u for u, _ in results] == [PRIYA, SAM, ALEX]  # confirmed_by first, then participants
    user, result = results[0]
    assert result.kind == "alert" and result.window_id == f"proactive:conflict:{DECISION}"
    assert result.confidence == result.relevance_score == 0.9
    assert result.content == result.payload["alert"]["title"]
    alert = result.payload["alert"]
    assert set(result.payload) == {"alert", "prompt", "model_tier"}
    assert result.payload["prompt"] == "proactive_conflict@v1"
    assert result.payload["model_tier"] == "sonnet"
    assert set(alert) == {
        "id",
        "alert_type",
        "priority",
        "title",
        "body",
        "confidence",
        "surfaced",
        "rationale",
        "subject",
        "related",
        "link",
    }
    assert alert["alert_type"] == "conflict" and alert["priority"] == 1
    assert alert["surfaced"] is True
    assert alert["subject"] == {"kind": "decision", "id": str(DECISION)}
    assert alert["related"] == [{"kind": "decision", "id": str(PRIOR), "label": PRIOR_TEXT}]
    assert alert["link"] == f"/decisions/{DECISION}"
    assert len(alert["title"]) <= 120 and len(alert["body"]) <= 400
    assert alert["id"] == str(uuid5(CANDIDATE_NS, f"ProactiveAgent:conflict:{DECISION}:{PRIYA}"))
    assert alerts[0].meeting_id == MEETING
    # the KG lineage was asked for the new decision, scoped by the event's org
    assert parts["kg"].calls[0] == (ORG, "decision_lineage", {"decision_id": str(DECISION)})


async def test_a_merely_related_decision_is_not_a_conflict():
    agent, llm, _ = make(ScriptedLLM(conflict={"conflicts": []}))
    assert await agent.check_decision(decision()) == []
    assert len(llm.of(CONFLICT_MARKER)) == 1  # judged, and judged not a conflict


async def test_the_superseded_decision_is_never_a_candidate():
    kg = FakeKg(
        {("decision_lineage", str(DECISION)): [lineage_node(SUPERSEDED, "Launch on 1 Nov.", 1)]}
    )
    rag = ScriptedRag({PRIYA: [rag_passage("Launch on 1 Nov.", 0.9, kind="decision")]})
    rag.by_user[PRIYA] = [
        rag_passage("Launch on 1 Nov.", 0.9, kind="decision", decision_id=SUPERSEDED),
        rag_passage(NEW_TEXT, 0.99, kind="decision", decision_id=DECISION),  # itself, indexed
    ]
    agent, llm, _ = make(kg=kg, rag=rag)
    # the model would call it a conflict, but it is never asked: no candidate is left
    assert await agent.check_decision(decision(supersedes=SUPERSEDED)) == []
    assert llm.of(CONFLICT_MARKER) == []

    # with another candidate present, the superseded one is still not shown to the model
    rag.by_user[PRIYA].append(
        rag_passage(PRIOR_TEXT, 0.7, kind="decision", decision_id=PRIOR, meeting_id=OLD_MEETING)
    )
    agent, llm, _ = make(kg=kg, rag=rag)
    alerts = await agent.check_decision(decision(supersedes=SUPERSEDED))
    [call] = llm.of(CONFLICT_MARKER)
    prompt = call["messages"][-1]["content"]
    assert "1 Nov" not in prompt and PRIOR_TEXT in prompt
    assert alerts[0].related[0]["id"] == str(PRIOR)


@pytest.mark.parametrize("failure", ["down", "slow"])
async def test_kg_down_or_slow_still_finds_the_conflict_through_rag(failure: str):
    kg = FakeKg(down=failure == "down", delay=5.0 if failure == "slow" else 0.0)
    rag = ScriptedRag(
        {
            PRIYA: [
                rag_passage(PRIOR_TEXT, 0.8, kind="decision", decision_id=PRIOR),
                rag_passage("Standup notes about the beta.", 0.9),  # transcript: filtered out
            ]
        }
    )
    agent, llm, _ = make(kg=kg, rag=rag)
    start = time.monotonic()
    alerts = await agent.check_decision(decision())
    assert time.monotonic() - start < 1.5  # the 800 ms R1 deadline, not the 5 s hang
    assert len(alerts) == 1 and alerts[0].related[0]["id"] == str(PRIOR)
    prompt = llm.of(CONFLICT_MARKER)[0]["messages"][-1]["content"]
    assert "Standup notes" not in prompt
    # rag was asked AS the person who confirmed the decision, with the decision text
    assert rag.calls == [{"org_id": ORG, "user_id": PRIYA, "query": NEW_TEXT}]


async def test_without_kg_or_rag_candidates_there_is_no_model_call():
    agent, llm, _ = make(kg=FakeKg(down=True), rag=ScriptedRag({}))
    assert await agent.check_decision(decision()) == []
    assert llm.calls == []


async def test_lineage_walks_from_supersedes_when_the_graph_lags_and_stops_at_depth_3():
    kg = FakeKg(
        {
            ("decision_lineage", str(DECISION)): [],  # the KG worker has not applied it yet
            ("decision_lineage", str(SUPERSEDED)): [
                lineage_node(GRANDPARENT, PRIOR_TEXT, 1),  # depth 2 from the new decision
                lineage_node(PRIOR, "A depth-4 ancestor.", 3),  # depth 4: too far
                lineage_node(SUPERSEDED, "Launch on 1 Nov.", 0, role="subject"),
            ],
        }
    )
    agent, llm, _ = make(kg=kg)
    alerts = await agent.check_decision(decision(supersedes=SUPERSEDED))
    prompt = llm.of(CONFLICT_MARKER)[0]["messages"][-1]["content"]
    assert PRIOR_TEXT in prompt and "depth-4" not in prompt and "1 Nov" not in prompt
    assert alerts[0].related[0]["id"] == str(GRANDPARENT)


async def test_conflict_bands_and_the_best_candidate_wins():
    rag = ScriptedRag({PRIYA: [rag_passage("Budget is 10k.", 0.6, kind="decision")]})
    for confidence, surfaced in ((0.9, True), (0.55, False)):
        agent, _, _ = make(ScriptedLLM(conflict=conflicts((2, 0.3), (1, confidence))), rag=rag)
        [alert] = await agent.check_decision(decision())
        assert alert.related[0]["id"] == str(PRIOR)  # candidate 1 (lineage) is the best
        for _u, p in payloads([alert]):
            assert p["alert"]["surfaced"] is surfaced
            assert p["alert"]["confidence"] == confidence
    agent, _, _ = make(ScriptedLLM(conflict=conflicts((1, 0.35))))
    assert await agent.check_decision(decision()) == []
    # numbers outside the list are ignored (the model cannot point at something unseen)
    agent, _, _ = make(ScriptedLLM(conflict=conflicts((7, 0.95))))
    assert await agent.check_decision(decision()) == []


async def test_conflict_recipients_are_capped_at_10_and_core_down_means_confirmer_only():
    many = [person(uuid.uuid4()) for _ in range(15)]
    agent, _, _ = make(core=FakeCore(participants={MEETING: many}))
    [alert] = await agent.check_decision(decision())
    assert len(alert.to) == 10 and alert.to[0] == PRIYA
    agent, _, _ = make(core=None)
    [alert] = await agent.check_decision(decision())
    assert alert.to == [PRIYA]


# --- workload and dependency -----------------------------------------------------------------


async def test_workload_fires_at_the_threshold_without_a_model_call():
    core = FakeCore(contexts={ACTION: ctx(open_action_count=6)})
    agent, llm, _ = make(core=core)
    [alert] = await agent.check_action(action())
    assert llm.calls == []
    assert alert.alert_type == "workload" and alert.confidence == 0.9
    assert alert.title == "priya already owns 6 open actions"
    assert "@" not in alert.title + alert.body  # the local part, never the address
    # the meeting's participants who are org admins or the organizer, never a non-participant
    assert alert.to == [ADMIN, ORGANIZER]
    [(_, p), *_] = payloads([alert])
    assert p["alert"]["priority"] == 2 and p["alert"]["surfaced"] is True
    assert p["alert"]["subject"] == {"kind": "action", "id": str(ACTION)}
    assert p["alert"]["link"] == f"/meetings/{MEETING}"
    assert {"kind": "user", "id": str(PRIYA), "label": "priya"} in p["alert"]["related"]
    assert p["model_tier"] == "none" and p["prompt"] == ""


async def test_workload_below_the_threshold_and_the_env_threshold(monkeypatch):
    agent, _, _ = make(core=FakeCore(contexts={ACTION: ctx(open_action_count=5)}))
    assert await agent.check_action(action()) == []
    monkeypatch.setenv("PROACTIVE_WORKLOAD_THRESHOLD", "4")
    agent, _, _ = make(
        core=FakeCore(contexts={ACTION: ctx(open_action_count=5)}),
        settings=ProactiveSettings.from_env(),
    )
    [alert] = await agent.check_action(action())
    assert alert.alert_type == "workload"
    # no admin or organizer among the participants: the participants themselves
    agent, _, _ = make(
        core=FakeCore(contexts={ACTION: ctx(admins=[OUTSIDER], organizer_id=None)}),
    )
    [alert] = await agent.check_action(action())
    assert alert.to == [PRIYA, SAM, ALEX, ADMIN, ORGANIZER]
    # no assignee: nobody is overloaded
    agent, _, _ = make(core=FakeCore(contexts={ACTION: ctx(assignee_id=None)}))
    assert await agent.check_action(action()) == []


async def test_dependency_on_unfinished_work_fires_and_done_work_does_not():
    deps = [
        Dependency(DEP_DONE, "Security review", "done"),
        Dependency(DEP_OPEN, "Pick the analytics vendor", "confirmed"),
    ]
    core = FakeCore(contexts={ACTION: ctx(open_action_count=1, depends_on=deps)})
    agent, llm, _ = make(core=core)
    [alert] = await agent.check_action(action())
    assert llm.calls == []
    assert alert.alert_type == "dependency" and alert.confidence == 0.85
    assert alert.to[0] == PRIYA  # the assignee first
    assert alert.related == [
        {"kind": "action", "id": str(DEP_OPEN), "label": "Pick the analytics vendor"}
    ]
    [(_, p), *_] = payloads([alert])
    assert p["alert"]["priority"] == 2

    done = FakeCore(contexts={ACTION: ctx(open_action_count=1, depends_on=deps[:1])})
    agent, _, _ = make(core=done)
    assert await agent.check_action(action()) == []


async def test_workload_and_dependency_have_distinct_ids_and_unknown_actions_are_skipped():
    deps = [Dependency(DEP_OPEN, "Pick the analytics vendor", "proposed")]
    agent, _, _ = make(core=FakeCore(contexts={ACTION: ctx(depends_on=deps)}))
    alerts = await agent.check_action(action())
    assert {a.alert_type for a in alerts} == {"workload", "dependency"}
    ids = [p["alert"]["id"] for _u, p in payloads(alerts)]
    assert len(ids) == len(set(ids))
    agent, _, _ = make(core=FakeCore())  # 404 from core-api: None
    assert await agent.check_action(action()) == []


# --- missing stakeholder -------------------------------------------------------------------


def stakeholder_window(index: int = 0) -> TranscriptWindow:
    turns = [
        Turn(str(PRIYA), "Let's settle the Pro seat price for next quarter.", 0, 4000, ["c1"]),
        Turn(str(SAM), "I'd say 25 dollars, mail me at sam@acme.example.", 4000, 9000, ["c2"]),
    ]
    start = index * 30_000
    return TranscriptWindow(ORG, MEETING, f"{MEETING}:{start}", start, start + 30_000, turns)


def stakeholder_parts() -> tuple[FakeKg, FakeCore]:
    kg = FakeKg(
        {
            ("related_decisions_for_topic", "pricing"): [
                topic_node(PRIOR, "Pro is 25 dollars per seat.", OLD_MEETING),
                topic_node(GRANDPARENT, "Annual plans get two months free.", OLDER_MEETING),
                topic_node(uuid.uuid4(), "Pricing page goes live.", MEETING),  # this meeting
            ]
        }
    )
    core = FakeCore(
        participants={
            MEETING: [person(PRIYA), person(SAM)],
            OLD_MEETING: [person(PRIYA), person(ALEX)],
            OLDER_MEETING: [person(ALEX), person(ADMIN)],
        }
    )
    return kg, core


async def test_missing_stakeholder_fires_once_per_meeting_and_person_through_the_runtime():
    kg, core = stakeholder_parts()
    agent, llm, _ = make(kg=kg, core=core)
    sink = Emitter(LoopbackProducer(), JsonCodec())
    runtime = AgentRuntime(
        sink=sink,
        user_context=lambda org, user: _ctx(user),
        agents=[lambda: agent],
    )
    for u in (PRIYA, SAM):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    emitted = await runtime.process_window(stakeholder_window(0))
    records = sink._producer.on("agent-results")
    assert emitted == len(records) == 4  # ALEX and ADMIN, each to PRIYA and SAM
    by_person: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        payload = json.loads(rec["payload_json"])
        alert = payload["alert"]
        assert rec["agent"] == "ProactiveAgent" and rec["kind"] == "alert"
        assert alert["alert_type"] == "missing_stakeholder" and alert["priority"] == 3
        assert alert["subject"] == {"kind": "meeting", "id": str(MEETING)}
        assert payload["prompt"] == "proactive_stakeholder@v1" and payload["model_tier"] == "haiku"
        missing = alert["related"][0]["id"]
        assert rec["window_id"] == f"proactive:missing_stakeholder:{MEETING}:{missing}"
        assert rec["header"]["idempotency_key"].startswith(
            f"agentresult:{MEETING}:proactive:missing_stakeholder:"
        )
        by_person.setdefault(missing, []).append(rec)
    assert set(by_person) == {str(ALEX), str(ADMIN)}
    # ALEX attended both topic meetings (2 decisions) and ranks first
    alex = json.loads(by_person[str(ALEX)][0]["payload_json"])["alert"]
    assert "2 decision(s)" in alex["body"] and "@" not in alex["title"] + alex["body"]
    assert {r["user_id"] for r in by_person[str(ALEX)]} == {str(PRIYA), str(SAM)}
    assert len(llm.of(STAKEHOLDER_MARKER)) == 1

    # the next window on the same topic: the gate runs, but nobody is alerted twice
    assert await runtime.process_window(stakeholder_window(1)) == 0
    assert len(llm.of(STAKEHOLDER_MARKER)) == 2


async def _ctx(user: UUID) -> dict[str, Any]:
    return {
        "user_id": str(user),
        "email": EMAILS.get(user, ""),
        "role": "PM",
        "rbac_tier": "member",
    }


async def test_missing_stakeholder_is_skipped_when_kg_is_down_or_the_topic_is_unknown():
    _, core = stakeholder_parts()
    agent, llm, _ = make(kg=None, core=core)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    assert await agent.check_window(stakeholder_window(), {PRIYA, SAM}) == []
    assert llm.calls == []  # skipped before paying for the gate

    agent, llm, _ = make(kg=FakeKg(down=True), core=core)
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    assert await agent.check_window(stakeholder_window(), {PRIYA, SAM}) == []

    kg, core = stakeholder_parts()
    for topic in ({"topic": "weather", "score": 0.95}, {"topic": "pricing", "score": 0.3}):
        agent, llm, _ = make(ScriptedLLM(topic=topic), kg=kg, core=core)
        await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
        assert await agent.check_window(stakeholder_window(), {PRIYA, SAM}) == []
    assert all(c[1] != "related_decisions_for_topic" for c in kg.calls)


# --- gating --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "limits", [{"consumed": 2_000, "quota": 1_000}, {"consumed": 5_000, "cap": 5_000}]
)
async def test_throttled_or_blocked_budget_suspends_everything(limits: dict[str, int]):
    kg, core = stakeholder_parts()
    core.contexts[ACTION] = ctx(depends_on=[Dependency(DEP_OPEN, "x", "confirmed")])
    agent, llm, parts = make(kg=kg, core=core, **limits)
    assert await agent.check_decision(decision()) == []
    assert await agent.check_action(action()) == []
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    assert await agent.check_window(stakeholder_window(), {PRIYA, SAM}) == []
    assert llm.calls == [] and parts["sink"].usage_calls == []
    assert kg.calls == [] and core.calls == []


async def test_the_plan_flag_and_its_override(monkeypatch):
    agent, llm, parts = make(plans=FakePlans(False))
    assert await agent.check_decision(decision()) == []
    assert llm.calls == [] and parts["plans"].calls == [(ORG, "proactive_agent")]

    agent, llm, _ = make(plans=FakePlans(True))
    assert len(await agent.check_decision(decision())) == 1

    monkeypatch.setenv("PROACTIVE_AGENT_ALL_PLANS", "1")
    agent, llm, parts = make(plans=FakePlans(False), settings=ProactiveSettings.from_env())
    assert len(await agent.check_decision(decision())) == 1
    assert parts["plans"].calls == []  # not even asked

    monkeypatch.setenv("PROACTIVE_AGENT_ALL_PLANS", "0")
    agent, llm, _ = make(plans=None, settings=ProactiveSettings.from_env())
    assert await agent.check_decision(decision()) == []  # no billing client: suspended

    # the override does not bypass the budget
    monkeypatch.setenv("PROACTIVE_AGENT_ALL_PLANS", "1")
    agent, llm, _ = make(settings=ProactiveSettings.from_env(), consumed=2_000, quota=1_000)
    assert await agent.check_decision(decision()) == [] and llm.calls == []


async def test_proactive_agent_0_disables_it(monkeypatch):
    monkeypatch.setenv("PROACTIVE_AGENT", "0")
    agent, llm, parts = make(settings=ProactiveSettings.from_env())
    assert await agent.check_decision(decision()) == []
    assert llm.calls == [] and parts["plans"].calls == []
    monkeypatch.delenv("PROACTIVE_AGENT")
    s = ProactiveSettings.from_env()
    assert s.enabled and s.workload_threshold == 6 and not s.all_plans


# --- output invariants ---------------------------------------------------------------------


async def test_ids_are_deterministic_across_replays():
    first = payloads(await make()[0].check_decision(decision()))
    second = payloads(await make()[0].check_decision(decision()))
    assert [p["alert"]["id"] for _u, p in first] == [p["alert"]["id"] for _u, p in second]
    assert alert_id("conflict", str(DECISION), PRIYA) == first[0][1]["alert"]["id"]
    assert len({p["alert"]["id"] for _u, p in first}) == len(first)  # one per recipient


async def test_no_user_id_or_email_reaches_a_model():
    kg, core = stakeholder_parts()
    kg.answers[("decision_lineage", str(DECISION))] = [lineage_node(PRIOR, PRIOR_TEXT, 1)]
    rag = ScriptedRag(
        {PRIYA: [rag_passage("Contact sam@acme.example on pricing.", 0.8, kind="decision")]}
    )
    agent, llm, _ = make(kg=kg, core=core, rag=rag)
    await agent.check_decision(decision(content=f"{NEW_TEXT} Owner: priya@acme.example."))
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    await agent.check_window(stakeholder_window(), {PRIYA, SAM})
    assert llm.of(CONFLICT_MARKER) and llm.of(STAKEHOLDER_MARKER)
    for call in llm.calls:
        text = json.dumps(call["messages"])
        assert not _UUID.search(text), text
        assert "@acme.example" not in text, text


def test_display_names_and_record_parsing():
    assert display_name("priya@acme.example") == "priya"
    assert display_name(None) == display_name("  ") == "Someone"
    value = {
        "header": {"org_id": str(ORG)},
        "decision_id": DECISION,  # fastavro hands logical uuids back as UUID objects
        "meeting_id": str(MEETING),
        "content": NEW_TEXT,
        "rationale": "",
        "confirmed_by": str(PRIYA),
        "supersedes": None,
    }
    d = parse_decision(value)
    assert d is not None and d.decision_id == DECISION and d.supersedes is None
    assert parse_decision({**value, "header": {}}) is None
    assert parse_decision({**value, "content": "  "}) is None
    a = parse_action(
        {
            "header": {"org_id": str(ORG)},
            "action_id": str(ACTION),
            "meeting_id": str(MEETING),
            "content": "x",
            "assignee_id": None,
            "due_date": None,
        }
    )
    assert a is not None and a.assignee_id is None


def test_the_agent_is_registered_and_its_prompts_load():
    from libs.shared.budget import BudgetGate  # noqa: F401 - imported for the router type

    router = LiteLLMRouter(budget(), UsageSink(), ScriptedLLM())
    loader = PromptLoader(PROMPTS_DIR)
    names = [f().name for f in registered_agents(loader, router, proactive=ProactiveDeps())]
    assert names[-1] == "ProactiveAgent"
    for key in ("proactive_conflict", "proactive_stakeholder"):
        prompt = loader.load(key, loader.resolve(key, ORG))
        assert prompt.version == "v1" and prompt.user_template
