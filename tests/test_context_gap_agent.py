"""`ContextGapAgent` and the A2A PEP: plan 10-13 section 3.3, ADR-0013, the P10 contract
sections 2, 4 and 5. Every model call, rag-service, the PDP and the broker are faked
(`tests/a2a_helpers.py`); the requester, the PEP, the responder and the real `Emitter` run as in
production over an in-memory bus.

1. Detector hits / misses; one shared gate call per window; the query_class classifier.
2. Requester selection: participants who have the context get nothing; at most 2; weakest first.
3. The chain: peer answer before the deadline wins; a deadline miss falls through; an authz deny
   is an `authz_denied` entry and no publish; the related-meeting hop; the RAG hop; all miss is
   silent with the collaborations still emitted.
4. Throttled budget and `CONTEXT_GAP_A2A=0` go RAG-only (no agent-requests at all).
5. Bands < 0.4 / 0.4-0.7 / > 0.7; an uncited fill is dropped; deterministic ids.
6. No user id or email reaches the model.
7. The responder's PEP pre-process: bad signature DLQ; org spoof DLQ + entry under the derived
   org; foreign responder DLQ + entry; re-authorize deny; expired dropped; retrieval AS the
   responder.
8. Decision cache: hit, miss, 60 s TTL, PDP down = deny (not cached), unknown values.
9. Response routing by correlation_id; unknown ids and unasked responders ignored.
10. Identity: mint / verify; runtime registration and the result shape on agent-results.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import fakeredis
import httpx
import pytest

from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from libs.shared.state import TranscriptWindow, Turn
from services.agent_runtime_service.a2a import related as related_mod
from services.agent_runtime_service.a2a.bus import (
    DLQ_TOPIC,
    ResponseRouter,
    answer_json,
    parse_answer,
    response_group,
)
from services.agent_runtime_service.a2a.collab import NIL_MEETING, collaboration_id
from services.agent_runtime_service.a2a.identity import (
    IdentitySigner,
    parse_target,
    signer_from_env,
)
from services.agent_runtime_service.a2a.pep import (
    Pep,
    classify_query,
    grant_key,
    resolve_scope,
)
from services.agent_runtime_service.a2a.related import RelatedAttendeesClient, related_from_env
from services.agent_runtime_service.agents import registered_agents
from services.agent_runtime_service.agents.context_gap_agent import (
    ContextGapAgent,
    ContextGapDeps,
    ContextGapSettings,
    OwnContext,
    choose_requesters,
    detect,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.runtime import AgentRuntime
from tests.a2a_helpers import (
    SECRET,
    FakePdp,
    FakeRelated,
    Loopback,
    ScriptedRag,
    allow_all,
    budget,
    rag_passage,
)

ORG, OTHER_ORG = uuid.uuid4(), uuid.uuid4()
MEETING = uuid.uuid4()
ALICE, BOB, CARA, DAN = (uuid.uuid4() for _ in range(4))
TEAM, DEPT = str(uuid.uuid4()), str(uuid.uuid4())
QUERY = "Q3 pricing tier decision for the analytics vendor"


def ctx(user: uuid.UUID, tier: str = "member", **kw: Any) -> dict[str, Any]:
    return {
        "user_id": str(user),
        "email": f"{user.hex[:8]}@example.com",
        "role": "Engineer",
        "skills": [],
        "bio": "",
        "team_ids": kw.get("teams", [TEAM]),
        "department_id": kw.get("dept", DEPT),
        "rbac_tier": tier,
    }


CONTEXTS = {ALICE: ctx(ALICE), BOB: ctx(BOB), CARA: ctx(CARA)}
DAN_CTX = ctx(DAN, teams=[], dept="")


class Sink:
    def __init__(self) -> None:
        self.usage_calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.usage_calls.append(kw)

    async def result(self, **kw: Any) -> None:
        self.results.append(kw)

    async def notification(self, **kw: Any) -> None:
        self.notifications.append(kw)

    async def delta(self, **kw: Any) -> None:
        pass


def gate_json(score: float = 0.9, is_gap: bool = True, query: str = QUERY) -> str:
    return json.dumps({"is_gap": is_gap, "query": query, "score": score})


def fill_json(confidence: float = 0.86, sources: list[int] | None = None) -> str:
    return json.dumps(
        {
            "content": "The team chose the enterprise tier in July. Finance signed off on it.",
            "confidence": confidence,
            "rationale": "Explains the decision being referenced.",
            "sources": [1] if sources is None else sources,
        }
    )


class ScriptedLLM:
    """Haiku answers the context-gap gate with `gate` (every other gate scores 0); Sonnet
    answers the synthesis with `fill`."""

    def __init__(self, gate: str | None = None, fill: str | None = None) -> None:
        self.gate = gate if gate is not None else gate_json()
        self.fill = fill if fill is not None else fill_json()
        self.calls: list[dict[str, Any]] = []

    def of(self, marker: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if marker in c["messages"][0]["content"][0]["text"]]

    @property
    def gates(self) -> list[dict[str, Any]]:
        return self.of("in front of a meeting context assistant")

    @property
    def fills(self) -> list[dict[str, Any]]:
        return self.of("private context assistant")

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        system = kw["messages"][0]["content"][0]["text"]
        if "in front of a meeting context assistant" in system:
            text = self.gate
        elif "private context assistant" in system:
            text = self.fill
        elif "personal meeting note-taker" in system:
            text = '{"relevance": 0}'
        else:
            text = '{"score": 0}'
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }


def window(
    window_id: str = f"{MEETING}:0",
    text: str = "As we discussed last time, the pricing decision stands.",
    speaker: uuid.UUID = ALICE,
) -> TranscriptWindow:
    turns = [
        Turn(
            str(BOB),
            "Okay, so where are we on the analytics rollout?",
            0,
            4_000,
            [str(uuid.uuid4())],
        ),
        Turn(str(speaker), text, 4_000, 9_000, [str(uuid.uuid4())]),
    ]
    return TranscriptWindow(ORG, MEETING, window_id, 0, 30_000, turns)


# QUERY is a decision_lineage query, so only `decision` passages may answer it.
STRONG = [
    rag_passage(
        "In July the team chose the enterprise tier; finance signed off.", 0.92, kind="decision"
    )
]
WEAK = [rag_passage("Vendor call notes, pricing mentioned briefly.", 0.2, kind="decision")]


async def make(
    llm: ScriptedLLM,
    *,
    by_user: dict[uuid.UUID, list[Any]] | None = None,
    grants: set[tuple[str, str, str]] | None = None,
    contexts: dict[uuid.UUID, dict[str, Any]] | None = None,
    related: Any = None,
    settings: ContextGapSettings | None = None,
    gate: Any = None,
    hop_delay: float = 0.0,
    responder_rag: Any = None,
    a2a: bool = True,
    pdp: FakePdp | None = None,
    rag: Any = "default",
) -> tuple[ContextGapAgent, Loopback, Sink, ScriptedRag]:
    rag_client = ScriptedRag(by_user if by_user is not None else {ALICE: STRONG})
    people = {**(contexts or CONTEXTS), DAN: DAN_CTX}
    bus = Loopback(
        org_contexts={ORG: people},
        rag=rag_client,
        pdp=pdp or FakePdp(allow_all() if grants is None else grants),
        hop_delay=hop_delay,
        responder_rag=responder_rag,
    )
    sink = Sink()
    router = LiteLLMRouter(gate or budget(), sink, llm)
    deps = ContextGapDeps(
        rag=rag_client if rag == "default" else rag,
        a2a=bus.client if a2a else None,
        pep=bus.pep,
        related=related,
        user_context=bus.user_context,
    )
    agent = ContextGapAgent(
        PromptLoader(PROMPTS_DIR), router, deps=deps, settings=settings or ContextGapSettings()
    )
    await agent.spawn(org_id=ORG, meeting_id=MEETING, user_id=MEETING)
    return agent, bus, sink, rag_client


async def run(
    agent: ContextGapAgent, w: TranscriptWindow | None = None, contexts: Any = None
) -> dict[uuid.UUID, Any]:
    out = await agent.process_meeting(w or window(), contexts or CONTEXTS)
    return {user: r for user, r in out}


def hops(result: Any) -> list[tuple[str, str, str]]:
    return [(c["hop"], c["responder"], c["result"]) for c in result.payload["collaborations"]]


# --- 1. detector, gate, classifier -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Like we decided last time, we keep Postgres.",
        "As discussed, the rollout waits for QA.",
        "Is the pricing decision still on?",
        "What about the Q3 roadmap decision?",
        "Remember when we moved the launch?",
        "That came up in the previous meeting.",
        "Who owns the migration now?",
        "What happened with the vendor contract?",
        "Per the security review, we need SSO.",
        "Can you follow up on the hiring plan?",
        "We agreed to cut scope.",
    ],
)
def test_the_detector_hits(text: str):
    assert detect(window(text=text)) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Let's decide the launch date now.",
        "I think we should make a decision on this today.",
        "Sounds good, see you all tomorrow.",
        "The dashboard loads in two seconds.",
    ],
)
def test_the_detector_misses(text: str):
    w = window(text=text)
    w.turns[0].text = "Okay."
    assert detect(w) is None


async def test_no_reference_means_no_model_call_and_no_retrieval():
    llm = ScriptedLLM()
    agent, _bus, sink, rag = await make(llm)
    assert await run(agent, window(text="The dashboard loads fast now.")) == {}
    assert llm.calls == [] and rag.calls == [] and sink.usage_calls == []


async def test_one_shared_gate_call_per_window_for_every_participant():
    llm = ScriptedLLM()
    agent, _bus, sink, _rag = await make(llm, by_user={ALICE: STRONG})
    out = await run(agent)
    assert set(out) == {BOB, CARA}  # two requesters
    assert len(llm.gates) == 1  # ...and one gate
    assert "haiku" in llm.gates[0]["model"]
    assert all("sonnet" in c["model"] for c in llm.fills) and len(llm.fills) == 2
    assert sorted(u["model_tier"] for u in sink.usage_calls) == ["haiku", "sonnet", "sonnet"]


@pytest.mark.parametrize(
    "gate", [gate_json(0.3), gate_json(0.9, is_gap=False), gate_json(0.9, query=""), "nonsense"]
)
async def test_below_the_gate_or_no_query_nothing_else_runs(gate: str):
    llm = ScriptedLLM(gate=gate)
    agent, bus, _sink, rag = await make(llm)
    assert await run(agent) == {}
    assert len(llm.calls) == 1 and rag.calls == [] and bus.producer.produced == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Q3 pricing tier decision for the analytics vendor", "decision_lineage"),
        ("why we chose Postgres over Mongo", "decision_lineage"),
        ("what the team agreed on the launch scope", "decision_lineage"),
        ("who owns the auth migration task", "action_status"),
        ("status of the JIRA ticket for SSO", "action_status"),
        ("deadline for the security review follow-up", "action_status"),
        ("analytics vendor onboarding discussion", "meeting_context"),
        ("previous meeting about the mobile redesign", "meeting_context"),
    ],
)
def test_the_query_class_classifier(query: str, expected: str):
    assert classify_query(query) == expected


# --- 2. requesters ---------------------------------------------------------------------------


def test_requesters_skip_who_has_it_cap_at_two_weakest_first_and_skip_the_speaker():
    own = {
        ALICE: OwnContext(best=0.9),
        BOB: OwnContext(best=0.3),
        CARA: OwnContext(best=0.1),
        DAN: OwnContext(best=0.0),
    }
    assert choose_requesters(own, None, 0.5, 2) == [DAN, CARA]
    assert choose_requesters(own, DAN, 0.5, 2) == [CARA, BOB]  # the speaker presumably knows
    assert choose_requesters({DAN: OwnContext()}, DAN, 0.5, 2) == [DAN]  # unless alone
    assert choose_requesters({ALICE: OwnContext(best=0.9)}, None, 0.5, 2) == []


async def test_participants_who_have_the_context_get_nothing():
    llm = ScriptedLLM()
    agent, _bus, _sink, rag = await make(llm, by_user={ALICE: STRONG, BOB: STRONG})
    out = await run(agent)
    assert set(out) == {CARA}
    own_calls = [c for c in rag.calls if c["query"] == QUERY]
    assert {c["user_id"] for c in own_calls} >= {ALICE, BOB, CARA}  # each as themselves


async def test_nobody_lacks_it_nothing_is_emitted():
    llm = ScriptedLLM()
    agent, bus, _sink, _rag = await make(llm, by_user={u: STRONG for u in CONTEXTS})
    assert await run(agent) == {}
    assert bus.producer.on("agent-requests") == [] and llm.fills == []


# --- 3. the chain ----------------------------------------------------------------------------


async def test_a_peer_answer_before_the_deadline_wins():
    llm = ScriptedLLM()
    agent, bus, _sink, rag = await make(llm, by_user={ALICE: STRONG})
    out = await run(agent)
    bob = out[BOB]
    assert ("peer", f"user:{ALICE}", "answered") in hops(bob)
    fill = bob.payload["fill"]
    assert fill["hop"] == "peer" and fill["responder"] == f"user:{ALICE}"
    assert fill["surfaced"] is True and bob.confidence == pytest.approx(0.86)
    assert bob.content == fill["content"]
    winner = next(c for c in bob.payload["collaborations"] if c["id"] == fill["id"])
    assert winner["success"] is True and winner["correlation_id"]
    assert winner["target_scope"] == "same_team" and winner["query_class"] == "decision_lineage"
    assert fill["citations"][0]["kind"] == "decision" and "decision_id" in fill["citations"][0]
    # the responder retrieved AS Alice, never as Bob
    responder_calls = [c for c in rag.calls if c["user_id"] == ALICE]
    assert responder_calls and all(c["org_id"] == ORG for c in rag.calls)
    req = bus.producer.on("agent-requests")[0]
    assert req["header"]["org_id"] == str(ORG)
    assert req["header"]["idempotency_key"] == f"areq:{req['correlation_id']}"
    assert req["responder_target"].startswith("user:")
    ident = bus.signer.verify_request(
        req["requester_agent_id"],
        correlation_id=req["correlation_id"],
        responder_target=req["responder_target"],
        deadline_us=req["deadline"],
        query=req["query"],
    )
    assert ident is not None and ident.org_id == ORG
    keys = {p.key for p in bus.producer.produced if p.topic == "agent-requests"}
    assert keys == {str(ORG)}  # partitioned by org


async def test_a_deadline_miss_falls_through_to_rag():
    llm = ScriptedLLM()
    slow = ScriptedRag({ALICE: STRONG}, delay=1.0)  # the responders take longer than the hop
    agent, bus, _sink, _rag = await make(
        llm,
        by_user={ALICE: STRONG, BOB: WEAK},
        responder_rag=slow,
        settings=ContextGapSettings(deadline_ms=300),
    )
    t0 = time.monotonic()
    out = await run(agent)
    assert time.monotonic() - t0 < 1.0  # bounded by the deadline, not by the slow peer
    bob = out[BOB]
    results = hops(bob)
    assert ("peer", f"user:{ALICE}", "deadline_miss") in results
    assert results[-1] == ("rag", "rag", "rag_hit")
    assert bob.payload["fill"]["hop"] == "rag" and bob.payload["fill"]["responder"] == "rag"
    miss = next(c for c in bob.payload["collaborations"] if c["result"] == "deadline_miss")
    assert miss["success"] is False and miss["correlation_id"]
    await bus.drain()


async def test_an_authz_deny_is_logged_and_nothing_is_published():
    llm = ScriptedLLM()
    agent, bus, _sink, _rag = await make(llm, by_user={ALICE: STRONG, BOB: WEAK}, grants=set())
    out = await run(agent)
    bob = out[BOB]
    denied = [c for c in bob.payload["collaborations"] if c["result"] == "authz_denied"]
    assert {c["responder"] for c in denied} == {f"user:{ALICE}", f"user:{CARA}"}
    assert all(c["success"] is False and c["correlation_id"] is None for c in denied)
    assert bus.producer.on("agent-requests") == []  # no publish at all
    assert hops(bob)[-1] == ("rag", "rag", "rag_hit")


async def test_the_related_meeting_hop_asks_dan_when_peers_have_nothing():
    llm = ScriptedLLM()
    related = FakeRelated([DAN, BOB])  # Bob is in this meeting: never asked as "related"
    agent, bus, _sink, _rag = await make(
        llm,
        by_user={DAN: STRONG},
        related=related,
        settings=ContextGapSettings(max_requesters=3),
    )
    out = await run(agent, window(speaker=BOB))  # Bob referenced it: Alice and Cara ask
    alice = out[ALICE]
    results = hops(alice)
    assert ("peer", f"user:{BOB}", "no_answer") in results
    assert ("related_meeting", f"user:{DAN}", "answered") in results
    assert not any(r[0] == "related_meeting" and r[1] == f"user:{BOB}" for r in results)
    assert alice.payload["fill"]["hop"] == "related_meeting"
    dan = next(c for c in alice.payload["collaborations"] if c["hop"] == "related_meeting")
    assert dan["target_scope"] == "org"  # no shared team or department
    assert related.calls[0]["meeting_id"] == MEETING and related.calls[0]["org_id"] == ORG
    assert not any(c["hop"] == "rag" for c in alice.payload["collaborations"])
    await bus.drain()


async def test_no_related_route_means_the_hop_is_skipped_without_an_entry():
    for related in (None, FakeRelated(None)):
        agent, _bus, _sink, _rag = await make(ScriptedLLM(), by_user={}, related=related)
        out = await run(agent)
        assert out and all(
            c["hop"] != "related_meeting" for r in out.values() for c in r.payload["collaborations"]
        )


async def test_all_miss_is_silent_but_the_collaborations_are_emitted():
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(llm, by_user={}, related=FakeRelated([]))
    out = await run(agent)
    assert set(out) == {BOB, CARA}  # everyone lacks it; Alice spoke, so she is skipped
    for r in out.values():
        assert r.payload["fill"] is None and r.content == "" and r.confidence == 0.0
        assert hops(r)[-1] == ("rag", "rag", "rag_miss")
        assert any(c["result"] == "no_answer" for c in r.payload["collaborations"])
    assert llm.fills == []  # never synthesize from nothing


# --- 4. descope paths ------------------------------------------------------------------------


async def test_a_throttled_tenant_goes_rag_only():
    llm = ScriptedLLM()
    throttled = budget(consumed=2_000, quota=1_000, cap=10**7)
    agent, bus, sink, _rag = await make(llm, by_user={ALICE: STRONG, BOB: WEAK}, gate=throttled)
    out = await run(agent)
    assert bus.producer.on("agent-requests") == []
    for r in out.values():
        assert {c["hop"] for c in r.payload["collaborations"]} == {"rag"}
    assert out[BOB].payload["fill"]["hop"] == "rag"
    assert {u["model_tier"] for u in sink.usage_calls} == {"haiku"}  # the router downgrades


async def test_the_a2a_flag_off_goes_rag_only(monkeypatch):
    monkeypatch.setenv("CONTEXT_GAP_A2A", "0")
    agent, bus, _sink, _rag = await make(
        ScriptedLLM(), by_user={ALICE: STRONG, BOB: WEAK}, settings=ContextGapSettings.from_env()
    )
    out = await run(agent)
    assert bus.producer.on("agent-requests") == []
    assert all({c["hop"] for c in r.payload["collaborations"]} == {"rag"} for r in out.values())


async def test_no_rag_configured_means_a_silent_agent():
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(llm, rag=None)
    assert await run(agent) == {} and llm.calls == []


async def test_a_blocked_tenant_makes_no_call():
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(llm, gate=budget(consumed=10**7, cap=10**7))
    assert await run(agent) == {} and llm.calls == [] and agent.blocked


# --- 5. bands and ids ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "surfaced"),
    [(0.3, None), (0.4, False), (0.55, False), (0.7, False), (0.71, True)],
)
async def test_bands(confidence: float, surfaced: bool | None):
    agent, _bus, _sink, _rag = await make(ScriptedLLM(fill=fill_json(confidence)))
    bob = (await run(agent))[BOB]
    if surfaced is None:
        assert bob.payload["fill"] is None and bob.confidence == 0.0 and bob.content == ""
        assert bob.payload["collaborations"]  # still logged
    else:
        assert bob.payload["fill"]["surfaced"] is surfaced
        assert bob.confidence == pytest.approx(confidence)


async def test_an_uncited_fill_is_dropped():
    for sources in ([], [7]):  # nothing cited, or a passage it was never shown
        agent, _bus, _sink, _rag = await make(ScriptedLLM(fill=fill_json(0.9, sources)))
        assert (await run(agent))[BOB].payload["fill"] is None


async def test_the_cooldown_after_a_surfaced_fill():
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(llm)
    assert await run(agent, window(f"{MEETING}:0"))
    assert await run(agent, window(f"{MEETING}:30000")) == {}  # quiet for one window
    assert await run(agent, window(f"{MEETING}:60000"))


async def test_ids_are_deterministic():
    w = window()
    runs = []
    for _ in range(2):
        agent, _bus, _sink, _rag = await make(
            ScriptedLLM(), by_user={ALICE: STRONG, BOB: WEAK}, grants=set()
        )
        runs.append(await run(agent, w))
    a, b = runs
    for user in a:
        ids_a = [c["id"] for c in a[user].payload["collaborations"]]
        ids_b = [c["id"] for c in b[user].payload["collaborations"]]
        assert ids_a == ids_b
        assert ids_a == [
            str(
                uuid.uuid5(
                    uuid.UUID("6f1d3c2e-8a4b-5e7f-9c1d-2b3a4c5d6e7f"),
                    f"ContextGapAgent:{w.window_id}:{user}:{n}",
                )
            )
            for n in range(len(ids_a))
        ]
        assert ids_a == [collaboration_id(w.window_id, user, n) for n in range(len(ids_a))]
    assert a[BOB].payload["fill"]["id"] == a[BOB].payload["collaborations"][-1]["id"]


# --- 6. privacy ------------------------------------------------------------------------------


async def test_no_user_id_or_email_reaches_the_model():
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(llm, related=FakeRelated([DAN]), by_user={DAN: STRONG})
    await run(agent)
    assert llm.gates and llm.fills
    blob = json.dumps([c["messages"] for c in llm.calls])
    for user in (ALICE, BOB, CARA, DAN):
        assert str(user) not in blob and user.hex not in blob
        assert CONTEXTS.get(user, DAN_CTX)["email"] not in blob
    assert any("You:" in f["messages"][-1]["content"] for f in llm.fills)
    assert "Participant 1:" in llm.gates[0]["messages"][-1]["content"]


# --- 7. the responder's PEP pre-process ------------------------------------------------------


def request_value(
    bus: Loopback,
    *,
    requester: uuid.UUID = BOB,
    org: uuid.UUID = ORG,
    header_org: uuid.UUID | None = None,
    target: str | None = None,
    deadline_s: float | None = None,
    agent_id: str | None = None,
    query: str = QUERY,
) -> dict[str, Any]:
    cid = str(uuid.uuid4())
    target = target or f"user:{ALICE}"
    deadline = int((deadline_s or time.time() + 2) * 1_000_000)
    signed = bus.signer.mint_request(
        org,
        requester,
        correlation_id=cid,
        responder_target=target,
        deadline_us=deadline,
        query=query,
    )
    return {
        "header": {"org_id": str(header_org or org), "idempotency_key": f"areq:{cid}"},
        "requester_agent_id": agent_id or signed,
        "responder_target": target,
        "query": query,
        "deadline": deadline,
        "correlation_id": cid,
    }


def responder_bus(**kw: Any) -> Loopback:
    return Loopback(
        org_contexts={ORG: {**CONTEXTS}, OTHER_ORG: {DAN: DAN_CTX}},
        rag=ScriptedRag({ALICE: STRONG + WEAK + STRONG + STRONG}),
        pdp=FakePdp(kw.pop("grants", allow_all())),
        **kw,
    )


async def test_a_bad_signature_is_dead_lettered_and_only_logged():
    bus = responder_bus()
    good = request_value(bus)["requester_agent_id"]
    for forged in (good[:-1] + ("0" if good[-1] != "0" else "1"), "cga:x:y:z", "", None):
        value = request_value(bus, agent_id=forged or "nope")
        assert await bus.responder.handle(value, raw=b"raw") == "dlq_bad_signature"
    other = IdentitySigner("another-secret-0123456789abcdef-xyz")
    value = request_value(bus)
    value["requester_agent_id"] = other.mint_request(
        ORG,
        BOB,
        correlation_id=value["correlation_id"],
        responder_target=value["responder_target"],
        deadline_us=value["deadline"],
        query=value["query"],
    )
    assert await bus.responder.handle(value) == "dlq_bad_signature"
    dlq = bus.producer.on(DLQ_TOPIC)
    assert len(dlq) == 5 and all(d["failure_class"] == "BAD_SIGNATURE" for d in dlq)
    assert bus.producer.on("agent-results") == [] and bus.producer.on("agent-responses") == []
    assert bus.pdp.calls == []


async def test_an_org_spoof_is_dead_lettered_with_an_entry_under_the_derived_org():
    bus = responder_bus()
    value = request_value(bus, org=ORG, header_org=OTHER_ORG)
    assert await bus.responder.handle(value) == "dlq_org_spoof"
    [dlq] = bus.producer.on(DLQ_TOPIC)
    assert dlq["failure_class"] == "ORG_SPOOF" and dlq["org_id"] == str(ORG)
    dlq_msg = next(p for p in bus.producer.produced if p.topic == DLQ_TOPIC)
    assert dlq_msg.key == str(ORG).encode()  # keyed by the DERIVED org
    [res] = bus.producer.on("agent-results")
    assert res["header"]["org_id"] == str(ORG) and res["user_id"] == str(BOB)
    assert res["kind"] == "context_fill" and res["agent"] == "ContextGapAgent"
    assert res["meeting_id"] == str(NIL_MEETING)
    body = json.loads(res["payload_json"])
    assert body["fill"] is None
    [entry] = body["collaborations"]
    assert entry["result"] == "org_spoof" and entry["success"] is False
    assert entry["correlation_id"] == value["correlation_id"]
    assert bus.producer.on("agent-responses") == [] and bus.pdp.calls == []


async def test_a_foreign_responder_is_dead_lettered_with_an_entry():
    bus = responder_bus()
    for target in (f"user:{DAN}", f"user:{uuid.uuid4()}", "team:everyone"):  # Dan is OTHER_ORG
        assert await bus.responder.handle(request_value(bus, target=target)) == "dlq_foreign_org"
    dlq = bus.producer.on(DLQ_TOPIC)
    assert [d["failure_class"] for d in dlq] == ["FOREIGN_ORG"] * 3
    results = bus.producer.on("agent-results")
    assert len(results) == 3 and all(r["header"]["org_id"] == str(ORG) for r in results)
    entries = [json.loads(r["payload_json"])["collaborations"][0] for r in results]
    assert [e["result"] for e in entries] == ["foreign_org"] * 3
    assert bus.producer.on("agent-responses") == [] and bus.pdp.calls == []
    assert not any(c["user_id"] == DAN for c in bus.responder._rag.calls)  # type: ignore[union-attr]


async def test_the_responder_re_authorizes_and_a_deny_is_answered_and_logged():
    bus = responder_bus(grants=set())
    value = request_value(bus)
    assert await bus.responder.handle(value) == "denied"
    [resp] = bus.producer.on("agent-responses")
    assert resp["success"] is False and json.loads(resp["result"]) == {"error": "authz_denied"}
    assert resp["correlation_id"] == value["correlation_id"]
    ident = bus.signer.verify_response(resp["responder_agent_id"], value["correlation_id"])
    assert ident is not None and ident.user_id == ALICE
    [res] = bus.producer.on("agent-results")
    entry = json.loads(res["payload_json"])["collaborations"][0]
    assert entry["result"] == "authz_denied" and entry["target_scope"] == "same_team"
    assert bus.responder._rag.calls == []  # type: ignore[union-attr]


async def test_an_expired_request_is_dropped():
    bus = responder_bus()
    assert await bus.responder.handle(request_value(bus, deadline_s=time.time() - 1)) == "expired"
    assert bus.producer.produced == []


async def test_the_responder_retrieves_as_the_responder_and_returns_the_top_three():
    bus = responder_bus()
    value = request_value(bus)
    assert await bus.responder.handle(value) == "answered"
    calls = bus.responder._rag.calls  # type: ignore[union-attr]
    assert [(c["org_id"], c["user_id"]) for c in calls] == [(ORG, ALICE)]
    [resp] = bus.producer.on("agent-responses")
    assert resp["success"] is True and resp["header"]["org_id"] == str(ORG)
    assert resp["header"]["idempotency_key"] == f"aresp:{value['correlation_id']}"
    body = json.loads(resp["result"])
    assert len(body["passages"]) == 3
    assert [p["score"] for p in body["passages"]] == sorted(
        (p["score"] for p in body["passages"]), reverse=True
    )
    citation = body["passages"][0]["citation"]
    assert set(citation) == {"kind", "id", "meeting_id"} and citation["kind"] == "decision"
    assert parse_answer(resp["result"])[0].citation["decision_id"] == citation["id"]


async def test_nothing_found_is_no_answer():
    bus = Loopback(org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({}), pdp=FakePdp(allow_all()))
    assert await bus.responder.handle(request_value(bus)) == "no_answer"
    [resp] = bus.producer.on("agent-responses")
    assert resp["success"] is False and json.loads(resp["result"]) == {"error": "no_answer"}


async def test_without_a_secret_every_request_is_dead_lettered(monkeypatch):
    monkeypatch.delenv("AGENT_IDENTITY_SECRET", raising=False)
    assert signer_from_env() is None
    monkeypatch.setenv("AGENT_IDENTITY_SECRET", "short")
    assert signer_from_env() is None
    monkeypatch.setenv("AGENT_IDENTITY_SECRET", SECRET)
    assert signer_from_env() is not None


def test_answer_parsing_is_defensive():
    assert parse_answer("not json") == [] and parse_answer('{"passages": 3}') == []
    bad = json.dumps(
        {
            "passages": [
                {
                    "text": "x",
                    "score": 0.9,
                    "citation": {"kind": "secret", "id": str(uuid.uuid4())},
                },
                {"text": "x", "score": 0.9, "citation": {"kind": "transcript", "id": "nope"}},
                {"text": "", "score": 0.9, "citation": {"kind": "action", "id": str(uuid.uuid4())}},
                {
                    "text": "ok",
                    "score": True,
                    "citation": {"kind": "action", "id": str(uuid.uuid4())},
                },
            ]
        }
    )
    assert parse_answer(bad) == []
    assert answer_json([]) is None


# --- 8. decision cache -----------------------------------------------------------------------


async def test_decision_cache_miss_then_hit_with_a_60s_ttl():
    redis, pdp = fakeredis.aioredis.FakeRedis(), FakePdp({("member", "action_status", "org")})
    pep = Pep(redis, pdp)
    assert await pep.authorize(ORG, "member", "action_status", "org") is True
    key = grant_key(ORG, "member", "action_status", "org")
    assert key == f"nexus:{ORG}:a2a:grant:member:action_status:org"
    assert await redis.get(key) == b"allow" and 55 <= await redis.ttl(key) <= 60
    assert await pep.authorize(ORG, "member", "action_status", "org") is True
    assert len(pdp.calls) == 1  # the second check was a cache hit
    assert await pep.authorize(ORG, "member", "decision_lineage", "org") is False
    assert await redis.get(grant_key(ORG, "member", "decision_lineage", "org")) == b"deny"
    await redis.delete(key)  # what the TTL (or identity-tenant-service's invalidation) does
    pdp.grants = set()  # the grant was revoked meanwhile
    assert await pep.authorize(ORG, "member", "action_status", "org") is False
    assert len(pdp.calls) == 3


async def test_the_cache_is_namespaced_by_org():
    redis, pdp = fakeredis.aioredis.FakeRedis(), FakePdp(allow_all())
    pep = Pep(redis, pdp)
    await pep.authorize(ORG, "member", "meeting_context", "org")
    pdp.grants = set()
    assert await pep.authorize(OTHER_ORG, "member", "meeting_context", "org") is False


async def test_pdp_down_is_deny_and_not_cached():
    redis = fakeredis.aioredis.FakeRedis()
    pep = Pep(redis, FakePdp(allow_all(), down=True))
    assert await pep.authorize(ORG, "admin", "meeting_context", "same_team") is False
    assert await redis.keys("*") == []

    class Raising:
        async def authorize(self, *_a: Any) -> str:
            raise RuntimeError("UNAVAILABLE")

    assert await Pep(redis, Raising()).authorize(ORG, "admin", "meeting_context", "org") is False
    assert await redis.keys("*") == []


async def test_redis_down_still_asks_the_pdp():
    class Down:
        async def get(self, _k: str) -> Any:
            raise ConnectionError

        async def set(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError

    pdp = FakePdp(allow_all())
    assert await Pep(Down(), pdp).authorize(ORG, "owner", "meeting_context", "self") is True
    assert len(pdp.calls) == 1


@pytest.mark.parametrize(
    ("role", "qc", "scope"),
    [
        ("", "meeting_context", "org"),
        ("superuser", "meeting_context", "org"),
        ("member", "salaries", "org"),
        ("member", "meeting_context", "world"),
    ],
)
async def test_unknown_values_are_denied_without_asking(role: str, qc: str, scope: str):
    pdp = FakePdp(allow_all())
    assert await Pep(fakeredis.aioredis.FakeRedis(), pdp).authorize(ORG, role, qc, scope) is False
    assert pdp.calls == []


def test_target_scope_resolution():
    a = ctx(ALICE, teams=["t1", "t2"], dept="d1")
    assert resolve_scope(ALICE, a, ALICE, a) == "self"
    assert resolve_scope(ALICE, a, BOB, ctx(BOB, teams=["t2"], dept="d9")) == "same_team"
    assert resolve_scope(ALICE, a, BOB, ctx(BOB, teams=["t3"], dept="d1")) == "department"
    assert resolve_scope(ALICE, a, BOB, ctx(BOB, teams=[], dept="")) == "org"
    empty = ctx(ALICE, teams=[], dept="")
    assert resolve_scope(ALICE, empty, BOB, ctx(BOB, teams=[], dept="")) == "org"


# --- 9. response routing ---------------------------------------------------------------------


async def test_responses_route_by_correlation_id_and_unknown_ids_are_ignored():
    router = ResponseRouter()
    fut = router.expect("c1", "agent-a")
    assert router.resolve({"correlation_id": "nope", "responder_agent_id": "agent-a"}) is False
    assert router.resolve({"correlation_id": None}) is False
    assert router.resolve({"correlation_id": "c1", "responder_agent_id": "agent-b"}) is False
    assert not fut.done()  # an agent that was not asked cannot answer
    assert router.resolve({"correlation_id": "c1", "responder_agent_id": "agent-a"}) is True
    assert (await fut)["correlation_id"] == "c1"
    assert router.resolve({"correlation_id": "c1", "responder_agent_id": "agent-a"}) is False
    assert router.pending == 0
    assert response_group("pod-7") == "agent-runtime-resp-pod-7"


async def test_a_response_after_the_deadline_is_ignored():
    bus = Loopback(
        org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({ALICE: STRONG}), pdp=FakePdp(allow_all())
    )
    bus.hop_delay = 0.2
    outcome = await bus.client.ask(
        org_id=ORG,
        requester=BOB,
        responder=ALICE,
        query=QUERY,
        query_class="decision_lineage",
        deadline_s=time.time() + 0.1,
    )
    assert outcome.result == "deadline_miss" and bus.router.pending == 0
    await bus.drain()  # the late response arrives and resolves nothing


# --- 10. identity, related client, runtime ---------------------------------------------------


def test_identity_is_signed_per_message():
    signer = IdentitySigner(SECRET)
    fields = {
        "correlation_id": "c-1",
        "responder_target": f"user:{ALICE}",
        "deadline_us": 1_800_000_000_000_000,
        "query": QUERY,
    }
    agent_id = signer.mint_request(ORG, BOB, **fields)
    org_s, user_s, sig = agent_id.split(":")[1:]
    assert agent_id.startswith("cga:") and (org_s, user_s) == (str(ORG), str(BOB))
    assert len(sig) == 32
    ident = signer.verify_request(agent_id, **fields)
    assert ident is not None and (ident.org_id, ident.user_id) == (ORG, BOB)
    # any field changed, or the id moved onto another request, breaks it
    for field, other in (
        ("correlation_id", "c-2"),
        ("responder_target", f"user:{CARA}"),
        ("deadline_us", fields["deadline_us"] + 1),
        ("query", QUERY + " and salaries"),
    ):
        assert signer.verify_request(agent_id, **{**fields, field: other}) is None, field
    assert signer.verify_request(agent_id.upper(), **fields) is None
    assert signer.verify_request(agent_id.replace(str(ORG), str(OTHER_ORG)), **fields) is None
    assert IdentitySigner(SECRET + "x").verify_request(agent_id, **fields) is None
    # a response id is bound to its correlation id, and is not a request id
    resp = signer.mint_response(ORG, ALICE, "c-1")
    assert signer.verify_response(resp, "c-1") == signer.verify_response(resp, "c-1")
    assert signer.verify_response(resp, "c-1") is not None
    assert signer.verify_response(resp, "c-2") is None
    assert signer.verify_request(resp, **fields) is None
    with pytest.raises(ValueError):
        IdentitySigner("too-short")
    assert parse_target(f"user:{BOB}") == BOB
    assert parse_target(f"user:{str(BOB).upper()}") is None and parse_target("user:x") is None


async def test_the_related_client_sends_the_service_caller_headers(monkeypatch):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"] = str(request.url), dict(request.headers)
        return httpx.Response(200, json={"user_ids": [str(DAN), "junk", str(DAN), str(CARA)]})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        related_mod.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    client = RelatedAttendeesClient("http://core/", "tok")
    ids = await client.related(org_id=ORG, meeting_id=MEETING, user_id=BOB, timeout_s=1)
    assert ids == [DAN, CARA]
    assert seen["url"] == f"http://core/internal/meetings/{MEETING}/related-attendees?user_id={BOB}"
    assert seen["headers"]["x-mytheus-service"] == "agent-runtime-service"
    assert seen["headers"]["x-mytheus-service-token"] == "tok"
    assert seen["headers"]["x-mytheus-org"] == str(ORG)

    monkeypatch.setattr(
        related_mod.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(404)), **kw),
    )
    assert await client.related(org_id=ORG, meeting_id=MEETING, user_id=BOB, timeout_s=1) is None
    assert await client.related(org_id=ORG, meeting_id=MEETING, user_id=BOB, timeout_s=0) is None


def test_the_related_client_is_optional(monkeypatch):
    monkeypatch.delenv("CORE_URL", raising=False)
    monkeypatch.setenv("CORE_CALLER_TOKEN", "x")
    assert related_from_env() is None
    monkeypatch.setenv("CORE_URL", "http://core")
    assert related_from_env() is not None


def test_settings_from_env(monkeypatch):
    for k in (
        "CONTEXT_GAP_A2A",
        "CONTEXT_GAP_DEADLINE_MS",
        "CONTEXT_GAP_GATE_THRESHOLD",
        "CONTEXT_GAP_HAVE_SCORE",
        "CONTEXT_GAP_MAX_REQUESTERS",
    ):
        monkeypatch.delenv(k, raising=False)
    assert ContextGapSettings.from_env() == ContextGapSettings(True, 1500, 0.5, 0.5, 2)
    monkeypatch.setenv("CONTEXT_GAP_DEADLINE_MS", "900")
    monkeypatch.setenv("CONTEXT_GAP_MAX_REQUESTERS", "junk")
    s = ContextGapSettings.from_env()
    assert s.deadline_ms == 900 and s.max_requesters == 2


async def test_registered_in_the_runtime_once_per_meeting_with_one_result_per_requester():
    llm = ScriptedLLM()
    bus = Loopback(
        org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({ALICE: STRONG}), pdp=FakePdp(allow_all())
    )
    sink = Sink()
    router = LiteLLMRouter(budget(), sink, llm)
    deps = ContextGapDeps(
        rag=bus.responder._rag, a2a=bus.client, pep=bus.pep, user_context=bus.user_context
    )
    runtime = AgentRuntime(
        sink=sink,
        user_context=bus.user_context,
        agents=registered_agents(PromptLoader(PROMPTS_DIR), router, context_gap=deps),
    )
    for u in (ALICE, BOB, CARA):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    await runtime.process_window(window())
    gaps = [r for r in sink.results if r["agent"] == "ContextGapAgent"]
    assert sorted(str(r["user_id"]) for r in gaps) == sorted([str(BOB), str(CARA)])
    assert all(r["result"].kind == "context_fill" for r in gaps)
    assert len(llm.gates) == 1
    keys = [k for k in runtime._meeting(ORG, MEETING).agents if k[0] == "ContextGapAgent"]
    assert keys == [("ContextGapAgent", MEETING)]


async def test_a_result_the_registry_cannot_encode_is_dropped_not_crash_looped():
    """agent-results-value without `context_fill` registered: the AvroCodec raises at produce.
    The runtime logs and drops that result; the window still completes."""

    class Failing(Sink):
        async def result(self, **kw: Any) -> None:
            if kw["agent"] == "ContextGapAgent":
                raise ValueError("'context_fill' is not a valid AgentResultKind")
            await super().result(**kw)

    bus = Loopback(
        org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({ALICE: STRONG}), pdp=FakePdp(allow_all())
    )
    router = LiteLLMRouter(budget(), Failing(), ScriptedLLM())
    deps = ContextGapDeps(rag=bus.responder._rag, a2a=bus.client, pep=bus.pep)
    runtime = AgentRuntime(
        sink=Failing(),
        user_context=bus.user_context,
        agents=registered_agents(PromptLoader(PROMPTS_DIR), router, context_gap=deps),
    )
    for u in (ALICE, BOB):
        await runtime.on_room_event(ORG, MEETING, "joined", str(u))
    assert await runtime.process_window(window()) == 0


def test_the_prompts_load_render_and_are_registered():
    loader = PromptLoader(PROMPTS_DIR)
    gate = loader.load("context_gap_gate", loader.resolve("context_gap_gate", ORG))
    assert "Participant 1: x" in gate.render_user(window="Participant 1: x", ground_rules="")
    synth = loader.load("context_gap_agent", loader.resolve("context_gap_agent", ORG))
    rendered = synth.render_user(
        query="q1", window="You: hi", passages="[1] p", company="", ground_rules=""
    )
    assert "q1" in rendered and "You: hi" in rendered and "[1] p" in rendered


async def test_the_emitted_record_matches_the_contract_shape():
    agent, _bus, _sink, _rag = await make(ScriptedLLM())
    bob = (await run(agent))[BOB]
    p = bob.payload
    assert set(p) == {"gap", "collaborations", "fill", "prompt", "model_tier"}
    assert p["gap"] == {"query": QUERY, "query_class": "decision_lineage"}
    assert p["prompt"] == "context_gap_agent@v2" and p["model_tier"] == "extract"
    for c in p["collaborations"]:
        assert set(c) == {
            "id",
            "correlation_id",
            "hop",
            "requester",
            "responder",
            "query",
            "query_class",
            "target_scope",
            "result",
            "success",
            "latency_ms",
        }
        assert c["requester"] == str(BOB) and isinstance(c["latency_ms"], int)
    assert set(p["fill"]) == {
        "id",
        "content",
        "confidence",
        "surfaced",
        "rationale",
        "hop",
        "responder",
        "citations",
    }
    assert all(uuid.UUID(c) for c in bob.source_chunk_ids)
    await asyncio.sleep(0)


async def test_concurrent_misses_share_one_pdp_call():
    class Slow(FakePdp):
        async def authorize(self, *a: Any) -> str | None:
            await asyncio.sleep(0.02)
            return await super().authorize(*a)

    pdp = Slow(allow_all())
    pep = Pep(fakeredis.aioredis.FakeRedis(), pdp)
    got = await asyncio.gather(
        *(pep.authorize(ORG, "member", "meeting_context", "org") for _ in range(12))
    )
    assert all(got) and len(pdp.calls) == 1


async def test_the_pdp_client_speaks_the_identity_authorize_rpc():
    import grpc

    from libs.shared import user_context as uc

    seen: list[Any] = []

    async def authorize(request: Any, context: Any) -> Any:
        seen.append((request, dict(context.invocation_metadata())))
        allowed = request.grantee_role == "admin"
        return uc.AuthorizeResponse(decision="allow" if allowed else "deny")

    handler = grpc.method_handlers_generic_handler(
        "mytheus.identity.v1.IdentityService",
        {
            "Authorize": grpc.unary_unary_rpc_method_handler(
                authorize,
                request_deserializer=uc.AuthorizeRequest.FromString,
                response_serializer=uc.AuthorizeResponse.SerializeToString,
            )
        },
    )
    server = grpc.aio.server()
    server.add_generic_rpc_handlers((handler,))
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        client = uc.PdpClient(f"127.0.0.1:{port}")
        assert await client.authorize(ORG, "admin", "action_status", "org") == "allow"
        assert await client.authorize(ORG, "member", "action_status", "org") == "deny"
        req, metadata = seen[0]
        assert (req.org_id, req.grantee_role, req.query_class, req.target_scope) == (
            str(ORG),
            "admin",
            "action_status",
            "org",
        )
        assert metadata["org-id"] == str(ORG)
        await client.close()
    finally:
        await server.stop(None)
    # nobody listening (or an identity service without Authorize yet): None, i.e. deny
    down = uc.PdpClient(f"127.0.0.1:{port}", timeout_s=0.3)
    assert await down.authorize(ORG, "admin", "action_status", "org") is None
    await down.close()


# --- review fixes: query_class enforced, per-request signatures, replay, DLQ, revoke race ---


MIXED = [
    rag_passage("Transcript: we talked pricing.", 0.95, kind="transcript"),
    rag_passage("Decision: enterprise tier chosen.", 0.9, kind="decision"),
    rag_passage("Action: Bob sends the order form.", 0.85, kind="action"),
    rag_passage("Document: vendor pricing sheet.", 0.8, kind="document"),
]


@pytest.mark.parametrize(
    ("query", "kinds"),
    [
        ("why we chose the enterprise tier", {"decision"}),
        ("who owns the order form task", {"action"}),
        ("analytics vendor onboarding discussion", {"transcript", "document"}),
    ],
)
async def test_the_responder_returns_only_the_kinds_the_query_class_allows(query: str, kinds):
    bus = Loopback(
        org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({ALICE: MIXED}), pdp=FakePdp(allow_all())
    )
    assert await bus.responder.handle(request_value(bus, query=query)) == "answered"
    [resp] = bus.producer.on("agent-responses")
    returned = {p["citation"]["kind"] for p in json.loads(resp["result"])["passages"]}
    assert returned == kinds


async def test_a_class_with_nothing_of_its_kind_is_no_answer():
    only_transcript = [rag_passage("We talked pricing.", 0.95, kind="transcript")]
    bus = Loopback(
        org_contexts={ORG: CONTEXTS},
        rag=ScriptedRag({ALICE: only_transcript}),
        pdp=FakePdp(allow_all()),
    )
    assert await bus.responder.handle(request_value(bus)) == "no_answer"  # decision query


def test_the_requester_drops_off_class_passages_from_an_answer():
    from services.agent_runtime_service.a2a.pep import allowed_kinds

    body = answer_json(MIXED)
    assert {p.citation["kind"] for p in parse_answer(body)} == {"transcript", "decision", "action"}
    assert [p.citation["kind"] for p in parse_answer(body, allowed_kinds("decision_lineage"))] == [
        "decision"
    ]
    assert parse_answer(body, allowed_kinds("not-a-class")) == []


async def test_the_requesters_own_rag_hop_is_bounded_by_the_class_too():
    off_class = [rag_passage("Transcript: we talked pricing.", 0.95, kind="transcript")]
    llm = ScriptedLLM()
    agent, _bus, _sink, _rag = await make(
        llm, by_user={u: off_class for u in CONTEXTS}, grants=set()
    )
    out = await run(agent)  # a decision query: transcript passages are neither "have" nor fill
    assert out and all(hops(r)[-1] == ("rag", "rag", "rag_miss") for r in out.values())
    assert llm.fills == []


async def test_a_tampered_request_fails_its_signature():
    bus = responder_bus()
    for field, other in (
        ("query", "board compensation for the exec team"),
        ("responder_target", f"user:{CARA}"),
        ("deadline", None),
        ("correlation_id", str(uuid.uuid4())),
    ):
        value = request_value(bus)
        value[field] = value["deadline"] + 1 if field == "deadline" else other
        assert await bus.responder.handle(value) == "dlq_bad_signature", field
    assert bus.producer.on("agent-responses") == [] and bus.pdp.calls == []


async def test_a_deadline_more_than_5s_ahead_is_refused():
    bus = responder_bus()
    value = request_value(bus, deadline_s=time.time() + 30)
    assert await bus.responder.handle(value) == "dlq_bad_deadline"
    [dlq] = bus.producer.on(DLQ_TOPIC)
    assert dlq["failure_class"] == "BAD_DEADLINE"
    assert bus.producer.on("agent-responses") == []
    assert bus.responder._rag.calls == []  # type: ignore[union-attr]


async def test_an_exact_replay_is_served_once():
    from services.agent_runtime_service.a2a.bus import dedupe_key

    bus = responder_bus()
    value = request_value(bus)
    assert await bus.responder.handle(value) == "answered"
    assert await bus.responder.handle(dict(value)) == "duplicate"
    assert len(bus.producer.on("agent-responses")) == 1
    assert len(bus.responder._rag.calls) == 1  # type: ignore[union-attr]
    key = dedupe_key(ORG, value["correlation_id"])
    assert key == f"nexus:{ORG}:dedupe:agent-requests:agent-runtime-a2a:{value['correlation_id']}"
    assert 55 <= await bus.redis.ttl(key) <= 60


async def test_without_the_dedupe_store_nothing_is_served():
    class Down:
        async def set(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError

        async def get(self, *_a: Any) -> Any:
            raise ConnectionError

    bus = responder_bus()
    bus.responder._redis = Down()
    assert await bus.responder.handle(request_value(bus)) == "dedupe_unavailable"
    assert bus.producer.on("agent-responses") == []


async def test_a_response_signed_for_another_request_is_ignored():
    bus = Loopback(
        org_contexts={ORG: CONTEXTS}, rag=ScriptedRag({ALICE: STRONG}), pdp=FakePdp(allow_all())
    )
    router = bus.router
    fut = router.expect("c1", bus.signer.mint_response(ORG, ALICE, "c1"))
    stolen = bus.signer.mint_response(ORG, ALICE, "c0")  # a real id from another response
    assert router.resolve({"correlation_id": "c1", "responder_agent_id": stolen}) is False
    assert not fut.done()


async def test_dead_letters_never_flush_and_bad_signatures_are_rate_limited():
    bus = responder_bus()
    bus.responder._bad_signatures._per_s = 5
    outcomes = [
        await bus.responder.handle(request_value(bus, agent_id="cga:junk"), raw=b"x")
        for _ in range(50)
    ]
    dlq = bus.producer.on(DLQ_TOPIC)
    assert 5 <= len(dlq) <= 10  # one second, or a second boundary crossed once
    assert outcomes.count("dropped_bad_signature") == 50 - len(dlq)
    assert bus.producer.flushes == 0  # nothing on the event loop ever waits on the broker


def test_the_rate_limit_resets_every_second():
    from services.agent_runtime_service.a2a.bus import RateLimit

    now = [100.0]
    limit = RateLimit(2, clock=lambda: now[0])
    assert [limit.allow() for _ in range(4)] == [True, True, False, False]
    now[0] = 101.2
    assert [limit.allow() for _ in range(3)] == [True, True, False]


class VersionBumpingPdp(FakePdp):
    """A PDP whose answer races a grant change: identity-tenant-service INCRs the version (and
    may delete the whole `a2a:grant:*` family, version included) while the call is in flight."""

    def __init__(self, other_client: Any, *, delete_after: bool = False) -> None:
        super().__init__(allow_all())
        self.other = other_client
        self.delete_after = delete_after

    async def authorize(self, org_id: uuid.UUID, role: str, qc: str, scope: str) -> str | None:
        answer = await super().authorize(org_id, role, qc, scope)
        await self.other.incr(f"nexus:{org_id}:a2a:grant:version")
        if self.delete_after:
            await self.other.delete(f"nexus:{org_id}:a2a:grant:version")
        return answer


@pytest.mark.parametrize("delete_after", [False, True])
async def test_an_answer_racing_a_grant_change_is_returned_but_not_cached(delete_after: bool):
    server = fakeredis.FakeServer()
    mine = fakeredis.aioredis.FakeRedis(server=server)
    backend = fakeredis.aioredis.FakeRedis(server=server)
    pdp = VersionBumpingPdp(backend, delete_after=delete_after)
    pep = Pep(mine, pdp)
    assert await pep.authorize(ORG, "member", "meeting_context", "org") is True
    assert await mine.get(grant_key(ORG, "member", "meeting_context", "org")) is None
    await pep.authorize(ORG, "member", "meeting_context", "org")
    assert len(pdp.calls) == 2  # nothing stale was cached, so the PDP is asked again


async def test_an_unchanged_version_caches_and_a_missing_one_reads_as_zero():
    server = fakeredis.FakeServer()
    mine = fakeredis.aioredis.FakeRedis(server=server)
    pdp = FakePdp(allow_all())
    pep = Pep(mine, pdp)
    assert await mine.get(f"nexus:{ORG}:a2a:grant:version") is None
    assert await pep.authorize(ORG, "admin", "action_status", "department") is True
    assert await mine.get(grant_key(ORG, "admin", "action_status", "department")) == b"allow"
    await mine.set(f"nexus:{OTHER_ORG}:a2a:grant:version", 7)
    assert await pep.authorize(OTHER_ORG, "admin", "action_status", "department") is True
    assert await mine.get(grant_key(OTHER_ORG, "admin", "action_status", "department")) == b"allow"


def test_the_a2a_emitter_logs_delivery_errors_instead_of_buffering_them():
    from services.agent_runtime_service.emitter import Emitter
    from tests.a2a_helpers import JsonCodec

    class FailingProducer:
        def produce(self, topic: str, key: Any = None, value: bytes = b"", on_delivery: Any = None):
            on_delivery("broker down", None)

        def poll(self, _t: float) -> int:
            return 0

        def flush(self, _t: float = 0) -> int:
            return 0

    seen: list[str] = []
    emitter = Emitter(FailingProducer(), JsonCodec(), on_error=seen.append)
    for _ in range(3):
        asyncio.run(
            emitter.agent_response(
                org_id=ORG,
                correlation_id="c",
                responder_agent_id="a",
                result="{}",
                success=False,
                latency_ms=1,
            )
        )
    emitter.dead_letter(
        topic="agent-requests",
        dlq_topic=DLQ_TOPIC,
        consumer_group="g",
        partition=0,
        offset=0,
        org_id=None,
        key_org=None,
        idempotency_key="",
        failure_class="BAD_SIGNATURE",
        error="x",
        raw_value=b"",
    )
    assert len(seen) == 4 and emitter._errors == []
    emitter.flush()  # nothing buffered, nothing raised
