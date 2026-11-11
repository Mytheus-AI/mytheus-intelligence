"""Run the real agents over the eval corpus and golden sets; return one report per suite.

Suites (each through the agent's real code; only the model answer and the stores are fakes):

    decision        DecisionAgent.process over every corpus window
    action          ActionAgent.process over every corpus window
    persona         PersonalListenerAgent.process, two personas per window (persona_pairs.json)
    meeting         MeetingAgent.process_meeting over meeting_golden.json
    proactive_*     ProactiveAgent.check_decision / check_action / check_window
    never_record    AgentRuntime.process_window (PersonalListener + Decision + Action) over the
                    never-record windows and 10 controls: counts canaries and never-record terms
                    in everything that leaves the tier (results and streamed deltas)

Variants: `current` (the prompts registry.json resolves) and `worse` (the deliberately worse
fixtures in tests/eval/prompts, `WORSE`). Thresholds: tests/eval/thresholds.json.

Modes:

    # offline (default, CI): replays tests/eval/recordings; no network, no spend
    uv run pytest -m eval tests/test_eval_harness.py

    # live: real Haiku/Sonnet calls for both variants plus the note-quality judge; writes
    # tests/eval/reports/live-<variant>.json and fails on the thresholds. About 800 model calls,
    # roughly 3 to 5 USD at list price. Needs ANTHROPIC_API_KEY (env or the repo's .env).
    EVAL_LIVE=1 uv run pytest -m eval tests/test_eval_harness.py -k live -s

    # live + record: the same, and the answers replace the recordings (`source: recorded`), so
    # the offline gate replays real model output from then on. Run after any prompt change.
    EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_eval_harness.py -k live -s

    # judge tier: EVAL_JUDGE_TIER=extract (Sonnet, default) or triage (Haiku)

    # some suites only (cheaper iterations on one agent): EVAL_SUITES=persona,meeting. The
    # reports are then live-partial-<variant>.json; a prompt change still ends with a full run.
    EVAL_LIVE=1 EVAL_RECORD=1 EVAL_SUITES=persona \\
        uv run pytest -m eval tests/test_eval_harness.py -k live -s
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

import fakeredis

from libs.shared.budget import BudgetGate
from libs.shared.kg_client import KgNode
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.prompts import Prompt, PromptLoader, _read
from libs.shared.router import LiteLLMRouter
from libs.shared.state import AgentResult, AgentState, TranscriptWindow, Turn
from services.agent_runtime_service.agenda import Agenda, AgendaItem
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.meeting_agent import (
    MeetingAgent,
    MeetingDeps,
    MeetingSettings,
    carried_key,
)
from services.agent_runtime_service.agents.personal_listener_agent import PersonalListenerAgent
from services.agent_runtime_service.agents.proactive_agent import (
    ConfirmedAction,
    ConfirmedDecision,
    ProactiveAgent,
    ProactiveDeps,
    ProactiveSettings,
    RagDecisionSearch,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.proactive.core import ActionContext, Participant
from services.agent_runtime_service.runtime import AgentRuntime
from tests.a2a_helpers import ScriptedRag, rag_passage
from tests.evalkit import corpus, scoring
from tests.evalkit.models import Model

VARIANTS = ("current", "worse")
WORSE: dict[str, str] = {
    "decision_gate": "v0-worse",
    "decision_agent": "v0-worse",
    "action_gate": "v0-worse",
    "action_agent": "v0-worse",
    "personal_listener": "v0-worse",
    "meeting_agent": "v0-worse",
    "proactive_conflict": "v0-worse",
}
TOPICS = ["pricing", "roadmap", "infra", "hiring", "security"]
HOST = uuid5(corpus.NS, "host")
CONFIRMER = uuid5(corpus.NS, "confirmer")


class EvalLoader(PromptLoader):
    """The repo's prompts, with `overrides` (key -> version) read from the eval fixtures."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        super().__init__(PROMPTS_DIR)
        self.overrides = dict(overrides or {})

    def resolve(self, prompt_key: str, org_id: UUID) -> str:
        return self.overrides.get(prompt_key) or super().resolve(prompt_key, org_id)

    def load(self, prompt_key: str, version: str) -> Prompt:
        if (corpus.FIXTURE_PROMPTS_DIR / f"{prompt_key}@{version}.md").exists():
            return _read(corpus.FIXTURE_PROMPTS_DIR, prompt_key, version)
        return super().load(prompt_key, version)


def loader_for(variant: str) -> EvalLoader:
    return EvalLoader(WORSE if variant == "worse" else None)


@dataclass
class TokenSink:
    units: int = 0
    calls: int = 0

    async def usage(self, **kw: Any) -> None:
        self.units += int(kw["tokens"])
        self.calls += 1


async def _unlimited(_org: UUID) -> dict[str, Any]:
    return {
        "tokens_consumed": 0,
        "token_quota": 10**12,
        "hard_cap": 10**12,
        "enforcement": "enforce",
    }


def router_for(model: Model, sink: TokenSink | None = None) -> LiteLLMRouter:
    return LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), _unlimited), sink or TokenSink(), model
    )


def _org_context() -> dict[str, Any]:
    return {"org": corpus.NEVER_RECORD_ORG}


# --- decision / action ---------------------------------------------------------------------------


async def run_candidates(kind: str, model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    agent: DecisionAgent | ActionAgent
    if kind == "action":
        agent = ActionAgent(loader, router, today=lambda: corpus.REFERENCE_DATE)
    else:
        agent = DecisionAgent(loader, router)
    meeting = uuid5(corpus.NS, f"meeting:{kind}")
    await agent.spawn(org_id=corpus.ORG, meeting_id=meeting, user_id=meeting)
    rows = []
    for w in corpus.windows():
        built = corpus.build_window(w)
        model.case, model.persona = w["id"], ""
        state = AgentState(
            meeting_id=built.window.meeting_id,
            user_id=meeting,
            chunk=built.window,
            context=_org_context(),
            org_id=corpus.ORG,
        )
        result = await agent.process(state)
        emitted = []
        for c in result.payload["candidates"] if result is not None else []:
            cand = dict(c)
            if "assignee_id" in c:
                cand["assignee"] = built.label_of.get(c.get("assignee_id") or "")
            emitted.append(cand)
        expected = w["decisions"] if kind == "decision" else w["actions"]

        def ok(exp: dict[str, Any], label: str | None, w: dict[str, Any] = w) -> bool:
            return label in corpus.accepted_labels(w, exp["assignee"])

        rows.append(
            {
                "emitted": emitted,
                "expected": expected,
                "lookalikes": [lk for lk in w["lookalikes"] if lk["kind"] == kind],
                "assignee_ok": ok,
            }
        )
    report = scoring.candidates(rows)
    report["parse_failure_rate"] = round(model.parse_failure_rate(f"{kind}_"), 4)
    report["windows"] = len(rows)
    return report


# --- persona pairs -------------------------------------------------------------------------------


async def run_personas(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    agents: dict[str, PersonalListenerAgent] = {}
    notes: dict[tuple[str, str], str] = {}
    rows = []
    for pair in corpus.persona_pairs():
        w = corpus.window_by_id()[pair["window"]]
        built = corpus.build_window(w)
        for persona in (pair["a"], pair["b"]):
            if (w["id"], persona) in notes:
                continue
            if persona not in agents:
                user = uuid5(corpus.NS, f"persona:{persona}")
                agents[persona] = PersonalListenerAgent(loader, router)
                await agents[persona].spawn(org_id=corpus.ORG, meeting_id=user, user_id=user)
            agent = agents[persona]
            assert agent.user_id is not None
            model.case, model.persona = w["id"], persona
            state = AgentState(
                meeting_id=built.window.meeting_id,
                user_id=agent.user_id,
                chunk=built.window,
                context={**corpus.persona_context(persona), **_org_context()},
                org_id=corpus.ORG,
            )
            result = await agent.process(state)
            notes[(w["id"], persona)] = result.content if result is not None else ""
        rows.append((notes[(w["id"], pair["a"])], notes[(w["id"], pair["b"])], corpus.facts(w)))
    report = scoring.fact_overlap(rows)
    report["parse_failure_rate"] = round(model.parse_failure_rate("personal_listener"), 4)
    report["notes"] = {f"{k[0]}:{k[1]}": v for k, v in sorted(notes.items())}
    return report


# --- meeting agent -------------------------------------------------------------------------------


@dataclass
class StaticAgenda:
    value: Agenda

    async def agenda(self, org_id: UUID, meeting_id: UUID) -> Agenda | None:
        return self.value


def _turns_window(turns: list[dict[str, Any]], meeting: UUID, index: int) -> TranscriptWindow:
    built = [
        Turn(
            speaker_id=f"spk-{t['speaker']}",
            text=t["text"],
            start_ms=index * 30_000 + n * 4_000,
            end_ms=index * 30_000 + n * 4_000 + 3_500,
            chunk_ids=[f"{meeting}-{n}"],
        )
        for n, t in enumerate(turns)
    ]
    start = index * 30_000
    return TranscriptWindow(corpus.ORG, meeting, f"{meeting}:{start}", start, start + 30_000, built)


async def run_meeting(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    data = corpus.load("meeting_golden.json")
    got_updates: list[set[tuple[int, str]]] = []
    want_updates: list[set[tuple[int, str]]] = []
    got_off: list[bool] = []
    want_off: list[bool] = []
    raised_ok = 0
    for case in data["cases"]:
        meeting = uuid5(corpus.NS, f"meeting:{case['id']}")
        raw_items = case.get("agenda") or data["default_agenda"]
        items = [
            AgendaItem(
                id=uuid5(meeting, f"item:{n}"),
                position=n,
                title=i["title"],
                duration_minutes=i["duration_minutes"],
                status=i["status"],
                status_source=i["status_source"],
            )
            for n, i in enumerate(raw_items, start=1)
        ]
        carried = list(case.get("carried") or data["default_carried"])
        redis = fakeredis.aioredis.FakeRedis()
        agent = MeetingAgent(
            loader,
            router,
            deps=MeetingDeps(
                agenda=StaticAgenda(Agenda(HOST, items, open_actions=carried)), redis=redis
            ),
            settings=MeetingSettings(enabled=True, agenda_ttl_s=0),
        )
        await agent.spawn(org_id=corpus.ORG, meeting_id=meeting, user_id=meeting)
        model.case, model.persona = case["id"], ""
        window = _turns_window(case["turns"], meeting, index=10)
        results = await agent.process_meeting(window, {HOST: _org_context()})
        number = {str(i.id): n for n, i in enumerate(items, start=1)}
        updates: set[tuple[int, str]] = set()
        nudge = None
        for _, r in results:
            for u in r.payload.get("updates") or []:
                updates.add((number[u["item_id"]], u["status"]))
            nudge = r.payload.get("nudge") or nudge
        got_updates.append(updates)
        want_updates.append({(u["item"], u["status"]) for u in case["expected"]["updates"]})
        got_off.append(bool(nudge and nudge.get("type") == "off_agenda"))
        want_off.append(bool(case["expected"]["off_agenda"]))
        state = await redis.hgetall(f"nexus:{corpus.ORG}:meeting_agent:{meeting}")
        keys = {k.decode() for k in state}
        raised = {n for n, t in enumerate(carried, start=1) if f"raised:{carried_key(t)}" in keys}
        raised_ok += raised == set(case["expected"]["raised"])
    p, r, f1, tp, fp, fn = scoring.sets(got_updates, want_updates)
    off = scoring.binary(got_off, want_off)
    return {
        "cases": len(want_off),
        "update_precision": round(p, 4),
        "update_recall": round(r, 4),
        "update_f1": round(f1, 4),
        "off_agenda_precision": off["precision"],
        "off_agenda_recall": off["recall"],
        "false_nudge_rate": off["false_positive_rate"],
        "raised_accuracy": round(raised_ok / len(want_off), 4),
        "parse_failure_rate": round(model.parse_failure_rate("meeting_agent"), 4),
    }


# --- proactive -----------------------------------------------------------------------------------


@dataclass
class FakeKg:
    answers: dict[tuple[str, str], list[KgNode]] = field(default_factory=dict)

    async def query(self, org_id: UUID, template: str, params: Any) -> list[KgNode]:
        return list(self.answers.get((template, params.get("topic") or ""), []))


@dataclass
class FakeCore:
    contexts: dict[UUID, ActionContext] = field(default_factory=dict)
    people: dict[UUID, list[Participant]] = field(default_factory=dict)

    async def action_context(self, org_id: UUID, action_id: UUID) -> ActionContext | None:
        return self.contexts.get(action_id)

    async def participants(self, org_id: UUID, meeting_id: UUID) -> list[Participant] | None:
        return self.people.get(meeting_id)


def _person(key: str) -> UUID:
    return uuid5(corpus.NS, f"person:{key}")


async def run_conflict(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    cases = corpus.proactive_golden()["conflict"]
    confidences: list[float | None] = []
    for case in cases:
        rag = ScriptedRag({CONFIRMER: [rag_passage(case["candidate"], 0.8, kind="decision")]})
        agent = ProactiveAgent(
            loader,
            router,
            deps=ProactiveDeps(decisions=RagDecisionSearch(rag)),
            settings=ProactiveSettings(all_plans=True),
        )
        model.case, model.persona = case["id"], ""
        alerts = await agent.check_decision(
            ConfirmedDecision(
                org_id=corpus.ORG,
                decision_id=uuid5(corpus.NS, f"decision:{case['id']}"),
                meeting_id=uuid5(corpus.NS, f"meeting:{case['id']}"),
                content=case["decision"],
                rationale=case["rationale"],
                confirmed_by=CONFIRMER,
            )
        )
        confidences.append(alerts[0].confidence if alerts else None)
    want = [bool(c["contradicts"]) for c in cases]
    alert = scoring.binary([c is not None and c >= 0.4 for c in confidences], want)
    shown = scoring.binary([c is not None and c > 0.7 for c in confidences], want)
    return {
        "cases": len(cases),
        "precision": alert["precision"],
        "recall": alert["recall"],
        "false_positive_rate": alert["false_positive_rate"],
        "surfaced_precision": shown["precision"],
        "surfaced_false_positive_rate": shown["false_positive_rate"],
        "parse_failure_rate": round(model.parse_failure_rate("proactive_conflict"), 4),
    }


async def run_workload(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    cases = corpus.proactive_golden()["workload"]
    got: list[bool] = []
    for case in cases:
        action = uuid5(corpus.NS, f"action:{case['id']}")
        meeting = uuid5(corpus.NS, f"meeting:{case['id']}")
        people = [_person(k) for k in ("priya", "sam", "admin")]
        ctx = ActionContext(
            assignee_id=people[0] if case["assignee"] else None,
            assignee_name="priya@acme.example" if case["assignee"] else None,
            meeting_id=meeting,
            open_action_count=case["open_action_count"],
            organizer_id=people[2],
            participants=people,
            admins=[people[2]],
        )
        agent = ProactiveAgent(
            loader,
            router,
            deps=ProactiveDeps(core=FakeCore(contexts={action: ctx})),
            settings=ProactiveSettings(all_plans=True),
        )
        model.case, model.persona = case["id"], ""
        alerts = await agent.check_action(
            ConfirmedAction(
                org_id=corpus.ORG,
                action_id=action,
                meeting_id=meeting,
                content="Draft the beta launch checklist",
                assignee_id=ctx.assignee_id,
            )
        )
        got.append(any(a.alert_type == "workload" for a in alerts))
    m = scoring.binary(got, [bool(c["expected_alert"]) for c in cases])
    return {"cases": len(cases), **m}


async def run_stakeholder(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    cases = corpus.proactive_golden()["stakeholder"]
    got: list[set[str]] = []
    want: list[set[str]] = []
    surfaced_ok = 0
    for case in cases:
        meeting = uuid5(corpus.NS, f"meeting:{case['id']}")
        answers: dict[tuple[str, str], list[KgNode]] = {}
        people: dict[UUID, list[Participant]] = {
            meeting: [Participant(_person(k), f"{k}@acme.example") for k in case["present"]]
        }
        nodes = []
        for n, earlier in enumerate(case["earlier"]):
            mid = meeting if earlier.get("this_meeting") else uuid5(meeting, f"earlier:{n}")
            if mid != meeting:
                people[mid] = [
                    Participant(_person(k), f"{k}@acme.example") for k in earlier["attendees"]
                ]
            for i, text in enumerate(earlier["decisions"]):
                did = uuid5(mid, f"decision:{i}")
                nodes.append(
                    KgNode(
                        "Decision", str(did), "match", 0, {"content": text, "meeting_id": str(mid)}
                    )
                )
        answers[("related_decisions_for_topic", case["gate"]["topic"])] = nodes
        agent = ProactiveAgent(
            loader,
            router,
            deps=ProactiveDeps(kg=FakeKg(answers), core=FakeCore(people=people), topics=TOPICS),
            settings=ProactiveSettings(all_plans=True),
        )
        await agent.spawn(org_id=corpus.ORG, meeting_id=meeting, user_id=meeting)
        window = TranscriptWindow(
            corpus.ORG,
            meeting,
            f"{meeting}:0",
            0,
            30_000,
            [
                Turn(str(_person(t["speaker"])), t["text"], n * 4_000, n * 4_000 + 3_500, [f"c{n}"])
                for n, t in enumerate(case["turns"])
            ],
        )
        roster = {_person(k) for k in case["present"]}
        model.case, model.persona = case["id"], ""
        alerts = await agent.check_window(window, roster, {u: _org_context() for u in roster})
        keys = {k: _person(k) for k in ("alex", "dana", "evan", "morgan", "sam", "priya")}
        named = {k for k, uid in keys.items() for a in alerts if a.subject_key.endswith(str(uid))}
        got.append(named)
        want.append(set(case["expected"]))
        if case["expected"]:
            surfaced_ok += all(a.surfaced == case["surfaced"] for a in alerts) and bool(alerts)
    p, r, f1, *_ = scoring.sets(got, want)
    negatives = [i for i, w in enumerate(want) if not w]
    fpr = sum(1 for i in negatives if got[i]) / len(negatives) if negatives else 0.0
    positives = sum(1 for w in want if w)
    return {
        "cases": len(cases),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "band_accuracy": round(surfaced_ok / positives, 4) if positives else 1.0,
    }


# --- never record (through the runtime) ----------------------------------------------------------


@dataclass
class CaptureSink:
    texts: list[str] = field(default_factory=list)
    results: int = 0

    async def result(
        self, *, window: TranscriptWindow, user_id: UUID, agent: str, result: AgentResult
    ) -> None:
        import json

        self.results += 1
        self.texts.append(result.content)
        self.texts.append(json.dumps(result.payload))

    async def notification(self, **kw: Any) -> None:
        self.texts.append(f"{kw.get('title', '')} {kw.get('body', '')}")

    async def delta(
        self, *, window: TranscriptWindow, user_id: UUID, seq: int, text: str, done: bool = False
    ) -> None:
        self.texts.append(text)


def _leaks(texts: list[str], canary: str | None) -> int:
    """Texts that left the tier naming a never-record topic (the runtime's own matcher: word
    boundaries, aliases, plurals) or carrying the window's canary phrase."""
    policy = NeverRecordPolicy.from_org(corpus.NEVER_RECORD_ORG)
    marker = (canary or "").lower()
    return sum(1 for t in texts if policy.hits(t) or (marker and marker in t.lower()))


async def run_never_record(model: Model, variant: str) -> dict[str, Any]:
    loader, router = loader_for(variant), router_for(model)
    listener = uuid5(corpus.NS, f"persona:{corpus.LISTENER}")

    async def user_context(_org: UUID, user: UUID) -> dict[str, Any] | None:
        return corpus.persona_context(corpus.LISTENER) if user == listener else None

    async def org_context(_org: UUID) -> Any:
        return corpus.NEVER_RECORD_ORG

    runtime = AgentRuntime(
        sink=(sink := CaptureSink()),
        user_context=user_context,
        agents=[
            lambda: PersonalListenerAgent(loader, router),
            lambda: DecisionAgent(loader, router),
            lambda: ActionAgent(loader, router, today=lambda: corpus.REFERENCE_DATE),
        ],
        org_context=org_context,
    )
    never = [w for w in corpus.windows() if w["never_record"]]
    controls = [w for w in corpus.windows() if w["category"] == "decision_action"][:10]
    leaks = leaking_windows = 0
    blocked = outright = 0
    control_results = 0
    for w in [*never, *controls]:
        meeting = uuid5(corpus.NS, f"nr-meeting:{w['id']}")
        built = corpus.build_window(w, users=False, meeting=meeting)
        await runtime.on_room_event(corpus.ORG, meeting, "joined", str(listener))
        model.case, model.persona = w["id"], corpus.LISTENER
        before_texts, before_calls, before_results = len(sink.texts), len(model.calls), sink.results
        await runtime.process_window(built.window)
        await runtime.end_meeting(corpus.ORG, meeting)
        if w["never_record"]:
            n = _leaks(sink.texts[before_texts:], w["never_record"]["canary"])
            leaks += n
            leaking_windows += n > 0
            if w["never_record"]["mode"] in ("outright", "alias"):
                outright += 1
                blocked += len(model.calls) == before_calls
        else:
            control_results += sink.results - before_results
    return {
        "windows": len(never),
        "controls": len(controls),
        "leaks": leaks,
        "leaking_windows": leaking_windows,
        "outright_skip_rate": round(blocked / outright, 4) if outright else 1.0,
        "control_results": control_results,
    }


# --- all -----------------------------------------------------------------------------------------


SUITES: dict[str, Callable[[Model, str], Awaitable[dict[str, Any]]]] = {
    "decision": lambda m, v: run_candidates("decision", m, v),
    "action": lambda m, v: run_candidates("action", m, v),
    "persona": run_personas,
    "meeting": run_meeting,
    "proactive_conflict": run_conflict,
    "proactive_workload": run_workload,
    "proactive_stakeholder": run_stakeholder,
    "never_record": run_never_record,
}


def suites_from_env(raw: str | None) -> tuple[str, ...]:
    """`EVAL_SUITES=persona,meeting` -> those suites (in the canonical order); empty -> all.
    `persona` brings `never_record` along: both ask the PersonalListener the same cases, so a
    recording of one without the other cannot be replayed."""
    names = {n.strip() for n in (raw or "").split(",") if n.strip()}
    if not names:
        return tuple(SUITES)
    unknown = names - set(SUITES)
    if unknown:
        raise ValueError(f"unknown eval suites: {sorted(unknown)}; known: {list(SUITES)}")
    if "persona" in names:
        names.add("never_record")
    return tuple(n for n in SUITES if n in names)


async def run_all(
    model: Model, variant: str, suites: tuple[str, ...] | None = None
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in suites or tuple(SUITES):
        report[name] = await SUITES[name](model, variant)
    return report
