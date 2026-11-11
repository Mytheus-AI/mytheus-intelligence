"""Room signals (`room_signals.py`, `room_signals_rules.py`): every model call and the core-api
route faked, no paid call.

1. The rules, one by one: cut off, spoken over, a hand never called, a listen-for topic that
   came up while someone said nothing, one nudge per person, the nearest decision, the agreement
   tally and strength, an open thread at the end, never-record windows, a tolerant source.
2. The writer: labels only reach the model (never names, ids or emails), the tiers and efforts,
   quotes verified verbatim, the answer mapped back to user ids, and the payload checked
   against the contract (scratchpad a3-room-signals-contract.md, section 3).
3. Fallbacks: budget blocked or a model failure leaves `partial` with the rules' answers.
4. Never record: nothing from a window naming a topic reaches a model or the payload.
5. The loop: `ended` only, one record per generation, flushed, claimed; retries, then an empty
   partial record; the source client's caller-token scheme; the Avro encoding.
"""

from __future__ import annotations

import io
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import fastavro
import httpx
import pytest

from libs.shared.budget import BudgetGate
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, OrgContextUnavailable
from libs.shared.prompts import PromptLoader
from libs.shared.router import LiteLLMRouter
from services.agent_runtime_service import room_signals_rules as rules
from services.agent_runtime_service.consumer import PROMPTS_DIR
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.minutes import generation_of
from services.agent_runtime_service.room_signals import (
    NIL,
    HttpRoomSignalsSource,
    RoomSignalsLoop,
    RoomSignalsWriter,
    SourceUnavailable,
    claim_key,
    consumer_config,
    parse_last_word,
)
from services.agent_runtime_service.room_signals_rules import (
    AgendaItem,
    Candidate,
    RoomSignal,
    SourceDecision,
    Turn,
)

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
HOST, PRIYA, RAJ, LEE, SAM = (uuid.uuid4() for _ in range(5))
D1 = uuid.uuid4()
ITEM_PRICING, ITEM_MIGRATION = uuid.uuid4(), uuid.uuid4()
SIG_HAND = uuid.uuid4()
START = datetime(2026, 11, 11, 15, 0, tzinfo=UTC)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# --- the scenario -----------------------------------------------------------------------------
# Labels by first speaking: HOST P1, PRIYA P2, RAJ P3, LEE P4, then SAM (agent) P5.
TURNS = [
    (HOST, 0, 4_000, "Let's start with pricing for the enterprise tier."),
    (PRIYA, 5_000, 12_000, "I think we should raise enterprise pricing by ten percent."),
    (RAJ, 13_000, 15_000, "I just want to say"),
    (HOST, 15_200, 19_000, "Great, so we raise pricing by ten percent then."),
    (PRIYA, 20_000, 22_000, "Yes, agreed."),
    (
        PRIYA,
        250_000,
        256_000,
        "We should move the billing tables first and freeze writes for a day.",
    ),
    (HOST, 265_000, 270_000, "I'd rather run both systems in parallel for a week."),
    (RAJ, 280_000, 283_500, "The migration risk is that the old billing tables"),
    (HOST, 284_000, 288_000, "Sorry, we're out of time, let's pick this up next week."),
    (LEE, 300_000, 302_000, "Thanks everyone, bye."),
]


def source_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "meeting_id": str(MEETING),
        "title": "Pricing and migration",
        "confidential": False,
        "host_user_id": str(HOST),
        "previous_meeting_id": None,
        "scheduled_start": START.isoformat(),
        "scheduled_end": (START + timedelta(minutes=30)).isoformat(),
        "room_started_at": START.isoformat(),
        "ended_at": (START + timedelta(minutes=5, seconds=3)).isoformat(),
        "call_ms": 303_000,
        "participants": [
            {
                "user_id": str(HOST),
                "name": "Hana Host",
                "attendance": "live",
                "role_in_meeting": "host",
                "listen_for": [],
            },
            {
                "user_id": str(PRIYA),
                "name": "Priya Patel",
                "attendance": "live",
                "role_in_meeting": "attendee",
                "listen_for": [],
            },
            {
                "user_id": str(RAJ),
                "name": "Raj Rao",
                "attendance": "live",
                "role_in_meeting": "attendee",
                "listen_for": ["migration"],
            },
            {
                "user_id": str(LEE),
                "name": "Lee Lin",
                "attendance": "live",
                "role_in_meeting": "attendee",
                "listen_for": ["pricing", "hiring"],
            },
            {
                "user_id": str(SAM),
                "name": "Sam Su",
                "attendance": "agent",
                "role_in_meeting": "attendee",
                "listen_for": ["pricing"],
            },
        ],
        "speakers": {},
        "turns": [
            {"speaker_id": str(u), "start_ms": s, "end_ms": e, "text": t} for u, s, e, t in TURNS
        ],
        "turns_truncated": False,
        "decisions": [
            {
                "id": str(D1),
                "content": "Raise enterprise pricing by ten percent",
                "status": "proposed",
                "owner_user_id": str(HOST),
                "quote": "so we raise pricing by ten percent",
                "quote_speaker_id": str(HOST),
                "quote_start_ms": 15_200,
                "quote_end_ms": 19_000,
            },
        ],
        "agenda": [
            {
                "id": str(ITEM_PRICING),
                "title": "Enterprise pricing",
                "status": "covered",
                "started_ms": 0,
                "ended_ms": 240_000,
            },
            {
                "id": str(ITEM_MIGRATION),
                "title": "Migration plan",
                "status": "in_progress",
                "started_ms": 240_000,
                "ended_ms": None,
            },
        ],
        "room_signals": [],
    }
    body.update(over)
    return body


LAST_WORD = {
    "signals": [
        {"ref": "S1", "topic": "pricing for the enterprise tier", "keep": True},
        {"ref": "S2", "topic": "the migration risk", "keep": True},
    ]
}
STANCES = {
    "stances": [
        {
            "participant": "Participant 1",
            "stance": "clear_yes",
            "quote": "so we raise pricing by ten percent then",
        },
        {"participant": "Participant 2", "stance": "clear_yes", "quote": "Yes, agreed."},
        {"participant": "Participant 3", "stance": "hedged", "quote": "I just want to say"},
    ]
}
RESUME = {
    "open": True,
    "topic": "the migration risk",
    "positions": [
        {"participant": "Participant 2", "quote": "move the billing tables first"},
        {"participant": "Participant 1", "quote": "run both systems in parallel for a week"},
    ],
    "cut_off": {
        "participant": "Participant 3",
        "quote": "The migration risk is that the old billing tables",
    },
}


class FakeSource:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    async def source(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> Any:
        assert org_id == ORG and meeting_id == MEETING
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def prompt_of(kw: dict[str, Any]) -> str:
    system = kw["messages"][0]["content"][0]["text"]
    if system.startswith("You help people"):
        return "last_word"
    if system.startswith("You read how much"):
        return "consensus_read"
    assert system.startswith("A meeting has just ended")
    return "resume_point"


def user_text(kw: dict[str, Any]) -> str:
    content = kw["messages"][1]["content"]
    if isinstance(content, list):
        return "".join(part["text"] for part in content)
    return str(content)


class LLM:
    def __init__(self, **answers: Any) -> None:
        self.answers = {"last_word": LAST_WORD, "consensus_read": STANCES, "resume_point": RESUME}
        self.answers.update(answers)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        answer = self.answers[prompt_of(kw)]
        if isinstance(answer, Exception):
            raise answer
        return {
            "choices": [
                {"message": {"content": answer if isinstance(answer, str) else json.dumps(answer)}}
            ],
            "usage": {"prompt_tokens": 600, "completion_tokens": 80},
        }

    def of(self, name: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if prompt_of(c) == name]


class Usage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.calls.append(kw)


def budget(*, cap: int = 10**7, consumed: int = 0) -> BudgetGate:
    async def refresh(_org: uuid.UUID) -> dict[str, Any]:
        return {
            "tokens_consumed": consumed,
            "token_quota": 10**7,
            "hard_cap": cap,
            "enforcement": "enforce",
        }

    return BudgetGate(fakeredis.aioredis.FakeRedis(), refresh)


def writer(
    llm: LLM,
    body: dict[str, Any] | None = None,
    *,
    org: CompanyContext | None = None,
    gate: BudgetGate | None = None,
    usage: Usage | None = None,
    source: Any = None,
) -> RoomSignalsWriter:
    async def org_context(_org: uuid.UUID) -> CompanyContext:
        return org or CompanyContext()

    router = LiteLLMRouter(gate or budget(), usage or Usage(), completion_fn=llm)
    return RoomSignalsWriter(
        PromptLoader(PROMPTS_DIR),
        router,
        source=source or FakeSource(rules.parse_source(body or source_body())),
        org_context=org_context,
    )


# --- the contract (section 3), checked on every payload a test produces --------------------------

TOP = {"generation", "status", "prompt_versions", "last_word", "consensus", "resume_point"}


def _uuid(v: Any, where: str, nullable: bool = False) -> None:
    if v is None and nullable:
        return
    assert isinstance(v, str) and UUID_RE.match(v), f"{where}: {v!r}"


def _int(v: Any, where: str, nullable: bool = True) -> None:
    if v is None and nullable:
        return
    assert isinstance(v, int) and not isinstance(v, bool), f"{where}: {v!r}"


def _text(v: Any, limit: int, where: str, nullable: bool = False) -> None:
    if v is None and nullable:
        return
    assert isinstance(v, str) and v.strip() and len(v) <= limit, f"{where}: {v!r}"


def validate(payload: dict[str, Any]) -> None:
    assert set(payload) == TOP, sorted(payload)
    assert isinstance(payload["generation"], str) and payload["generation"]
    assert payload["status"] in ("ready", "partial")
    pv = payload["prompt_versions"]
    assert set(pv) == {"last_word", "consensus", "resume_point"}
    for key, prefix in (
        ("last_word", "last_word@v"),
        ("consensus", "consensus_read@v"),
        ("resume_point", "resume_point@v"),
    ):
        assert pv[key] is None or pv[key].startswith(prefix)
    lw = payload["last_word"]
    assert isinstance(lw, list) and len(lw) <= 50
    assert len({e["user_id"] for e in lw}) == len(lw), "one nudge per person"
    for e in lw:
        assert set(e) == {
            "user_id",
            "kind",
            "topic",
            "quote",
            "at_ms",
            "end_ms",
            "decision_id",
            "signal_ids",
        }
        _uuid(e["user_id"], "user_id")
        assert e["kind"] in ("spoken_over", "hand_not_called", "cut_off", "topic_silent")
        _text(e["topic"], 80, "topic")
        assert not re.search(r"participant\s*\d", e["topic"], re.I)
        _text(e["quote"], 300, "quote", nullable=True)
        _int(e["at_ms"], "at_ms")
        _int(e["end_ms"], "end_ms")
        _uuid(e["decision_id"], "decision_id", nullable=True)
        assert isinstance(e["signal_ids"], list)
        for s in e["signal_ids"]:
            _uuid(s, "signal_ids[]")
    cs = payload["consensus"]
    assert isinstance(cs, list) and len(cs) <= 50
    for c in cs:
        assert set(c) == {
            "decision_id",
            "strength",
            "participants",
            "yes",
            "hedged",
            "silent",
            "objected",
            "talked_over",
            "ask",
        }
        _uuid(c["decision_id"], "decision_id")
        assert c["strength"] in ("strong", "mixed", "weak", "unclear")
        counts = [c[k] for k in ("yes", "hedged", "silent", "objected", "talked_over")]
        assert all(isinstance(n, int) and n >= 0 for n in counts)
        assert sum(counts) == c["participants"]
        assert isinstance(c["ask"], list) and len(c["ask"]) <= 12
        for a in c["ask"]:
            assert set(a) == {"user_id", "reason"}
            _uuid(a["user_id"], "ask.user_id")
            assert a["reason"] in ("hedged", "silent", "talked_over")
    rp = payload["resume_point"]
    if rp is not None:
        assert set(rp) == {"reason", "topic", "agenda_item_id", "positions", "cut_off"}
        assert rp["reason"] in ("time_ran_out", "agenda_unfinished", "cut_off")
        _text(rp["topic"], 120, "resume.topic")
        _uuid(rp["agenda_item_id"], "agenda_item_id", nullable=True)
        assert isinstance(rp["positions"], list) and len(rp["positions"]) <= 2
        for p in rp["positions"]:
            assert set(p) == {"user_id", "quote", "start_ms", "end_ms"}
            _uuid(p["user_id"], "position.user_id", nullable=True)
            _text(p["quote"], 300, "position.quote")
            _int(p["start_ms"], "start_ms", nullable=False)
            _int(p["end_ms"], "end_ms", nullable=False)
        if rp["cut_off"] is not None:
            co = rp["cut_off"]
            assert set(co) == {"user_id", "quote", "start_ms", "end_ms"}
            _uuid(co["user_id"], "cut_off.user_id")
            _text(co["quote"], 300, "cut_off.quote")
    json.dumps(payload)  # serialisable as payload_json


# --- 1. rules ---------------------------------------------------------------------------------


def T(user: uuid.UUID | None, s: int, e: int, text: str) -> Turn:
    return Turn(str(user), s, e, text, user)


@pytest.mark.parametrize(
    ("text", "trails"),
    [
        ("I just want to say", True),
        ("One more thing", True),
        ("we could do it but", True),
        ("and then the cache...", True),
        ("the old billing tables", True),  # short, unpunctuated
        ("Yeah", False),  # a backchannel
        ("ok sure", False),
        ("That works for me.", False),
        ("Can we ship on Friday?", False),
        ("this is an unpunctuated sentence that keeps going on and on for far too long", False),
    ],
)
def test_trails_off(text: str, trails: bool) -> None:
    assert rules.trails_off(text) is trails


def test_cut_off_needs_someone_to_come_in_and_the_person_to_stay_quiet() -> None:
    base = [T(RAJ, 0, 2_000, "I just want to say"), T(HOST, 2_500, 6_000, "Right, moving on.")]
    [c] = rules.cut_offs(base)
    assert (c.user_id, c.kind, c.at_ms, c.quote) == (RAJ, "cut_off", 0, "I just want to say")
    # the next speaker came in after a pause: no cut-off
    assert rules.cut_offs([base[0], T(HOST, 3_000, 6_000, "Right, moving on.")]) == []
    # already talking and kept going over it: a cut-off
    assert rules.cut_offs(
        [T(HOST, 0, 9_000, "Long point here."), T(RAJ, 1_000, 2_000, "but the cache")]
    )
    # the person got their point in right after
    later = T(RAJ, 7_000, 12_000, "what I wanted to say is we need a rollback plan for the cache")
    assert rules.cut_offs([*base, later]) == []
    # nobody came in at all
    assert rules.cut_offs([base[0]]) == []


def test_spoken_over_is_confirmed_by_overlap_or_the_room() -> None:
    turns = [T(HOST, 0, 10_000, "Long explanation about the budget.")]
    sig = RoomSignal(uuid.uuid4(), "spoken_over", PRIYA, 3_000, 4_500, {})
    [c] = rules.spoken_over([sig], turns)
    assert (c.user_id, c.kind, c.signal_ids) == (PRIYA, "spoken_over", [sig.id])
    alone = RoomSignal(uuid.uuid4(), "spoken_over", PRIYA, 20_000, 21_000, {})
    assert rules.spoken_over([alone], turns) == []
    said = RoomSignal(
        uuid.uuid4(), "spoken_over", PRIYA, 20_000, 21_000, {"over_user_id": str(HOST)}
    )
    assert len(rules.spoken_over([said], turns)) == 1
    got_in = [
        *turns,
        T(
            PRIYA,
            12_000,
            20_000,
            "so as I was saying the budget needs another review because the numbers "
            "for the second quarter do not add up at all",
        ),
    ]
    assert rules.spoken_over([sig], got_in) == []


def test_a_hand_counts_unless_they_spoke_anyway() -> None:
    sig = RoomSignal(
        SIG_HAND,
        "hand_not_called",
        LEE,
        10_000,
        60_000,
        {"outcome": "still_raised_at_end", "waited_ms": 50_000},
    )
    [c] = rules.hands_not_called([sig], [])
    assert c.detail == {"outcome": "still_raised_at_end"} and c.signal_ids == [SIG_HAND]
    spoke = [T(LEE, 70_000, 80_000, " ".join(["word"] * 15))]
    assert rules.hands_not_called([sig], spoke) == []


def test_topic_silence_needs_two_mentions_presence_and_quiet() -> None:
    lee = rules.Participant(LEE, listen_for=("pricing",))
    turns = [
        T(HOST, 0, 3_000, "Pricing is first."),
        T(PRIYA, 4_000, 8_000, "Enterprise pricing should go up."),
    ]
    [c] = rules.topic_silences([lee], turns, NeverRecordPolicy())
    assert (c.kind, c.topic, c.at_ms, c.end_ms) == ("topic_silent", "pricing", 0, 8_000)
    assert rules.topic_silences([lee], turns[:1], NeverRecordPolicy()) == []
    talked = [*turns, T(LEE, 9_000, 12_000, "I think pricing should stay flat this year")]
    assert rules.topic_silences([lee], talked, NeverRecordPolicy()) == []
    secret = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("pricing",)))
    assert rules.topic_silences([lee], turns, secret) == []


def test_presence_excludes_agents_and_people_who_never_showed() -> None:
    src = rules.parse_source(source_body())
    assert src is not None
    assert {p.user_id for p in rules.present_people(src)} == {HOST, PRIYA, RAJ, LEE}
    body = source_body()
    body["participants"][3]["present"] = False  # core-api says Lee was not in the call
    src2 = rules.parse_source(body)
    assert src2 is not None and LEE not in {p.user_id for p in rules.present_people(src2)}


def test_one_nudge_per_person_by_priority_then_earliest() -> None:
    cands = [
        Candidate(RAJ, "topic_silent", 1_000, 2_000, topic="x"),
        Candidate(RAJ, "hand_not_called", 5_000, 6_000),
        Candidate(RAJ, "cut_off", 90_000, 91_000),
        Candidate(RAJ, "cut_off", 50_000, 51_000),
        Candidate(LEE, "topic_silent", 3_000, 4_000, topic="y"),
    ]
    out = rules.pick_one_per_user(cands)
    assert [(c.user_id, c.kind, c.at_ms) for c in out] == [
        (LEE, "topic_silent", 3_000),
        (RAJ, "cut_off", 50_000),
    ]


def test_nearest_decision_within_three_minutes() -> None:
    a = SourceDecision(uuid.uuid4(), "A", "proposed", quote_start_ms=10_000)
    b = SourceDecision(uuid.uuid4(), "B", "confirmed", quote_start_ms=100_000)
    none = SourceDecision(uuid.uuid4(), "C", "confirmed")
    assert rules.nearest_decision(60_000, [a, b, none]) == b.id
    assert rules.nearest_decision(400_000, [a, b]) is None
    assert rules.nearest_decision(None, [a]) is None


@pytest.mark.parametrize(
    ("n", "yes", "objected", "over", "want"),
    [
        (1, 1, 0, 0, "unclear"),
        (3, 3, 0, 0, "strong"),
        (3, 2, 0, 0, "mixed"),  # 0.667 rounds below 0.67: not strong, not weak
        (6, 5, 0, 1, "strong"),
        (5, 4, 1, 0, "mixed"),
        (5, 3, 1, 0, "weak"),
        (4, 2, 0, 0, "mixed"),
        (4, 1, 0, 0, "weak"),
        (5, 3, 0, 1, "weak"),
    ],
)
def test_strength(n: int, yes: int, objected: int, over: int, want: str) -> None:
    assert rules.strength(n, yes, objected, over) == want


def test_tally_counts_everyone_once_and_never_asks_the_owner() -> None:
    d = SourceDecision(D1, "x", "proposed", owner_user_id=HOST, quote_start_ms=0)
    frame = rules.ConsensusFrame(
        d, 0, 1, [HOST, PRIYA, RAJ, LEE], [], [HOST, PRIYA, RAJ], [LEE], [RAJ]
    )
    got = rules.tally(frame, {HOST: "hedged", PRIYA: "clear_yes", RAJ: "clear_yes"})
    assert (got["yes"], got["hedged"], got["silent"], got["talked_over"]) == (2, 1, 1, 0)
    assert got["strength"] == "mixed"
    assert got["ask"] == [{"user_id": str(LEE), "reason": "silent"}]  # the hedging owner is not
    got2 = rules.tally(frame, {PRIYA: "clear_yes", RAJ: "hedged"})
    assert got2["talked_over"] == 1 and got2["strength"] == "weak"
    assert {a["reason"] for a in got2["ask"]} == {"talked_over", "silent"}
    lone = rules.ConsensusFrame(d, 0, 1, [PRIYA], [], [], [PRIYA], [])
    assert rules.tally(lone, {})["ask"] == []  # unclear: nobody is asked


def test_open_thread_reasons() -> None:
    src = rules.parse_source(source_body())
    assert src is not None
    turns = src.turns
    cut = rules.cut_offs(turns)
    thread = rules.open_thread(src, turns, cut)
    assert thread is not None and thread.reason == "cut_off"
    assert thread.agenda_item is not None and thread.agenda_item.id == ITEM_MIGRATION
    assert thread.cut_off is not None and thread.cut_off.at_ms == 280_000
    # no cut-off: the running agenda item
    assert rules.open_thread(src, turns, []).reason == "agenda_unfinished"  # type: ignore[union-attr]
    # nothing open: everything covered, ended early
    closed = source_body(
        agenda=[{"id": str(ITEM_PRICING), "title": "Pricing", "status": "covered"}]
    )
    src2 = rules.parse_source(closed)
    assert src2 is not None and rules.open_thread(src2, src2.turns, []) is None
    # pending items once 70% of the scheduled time was used
    pend = source_body(
        agenda=[{"id": str(ITEM_PRICING), "title": "Pricing", "status": "pending"}],
        call_ms=25 * 60_000,
    )
    src3 = rules.parse_source(pend)
    assert src3 is not None
    assert rules.open_thread(src3, src3.turns, []).reason == "agenda_unfinished"  # type: ignore[union-attr]
    early = source_body(agenda=[{"id": str(ITEM_PRICING), "title": "Pricing", "status": "pending"}])
    src4 = rules.parse_source(early)
    assert src4 is not None and rules.open_thread(src4, src4.turns, []) is None


def test_time_ran_out_needs_real_talk_at_the_end() -> None:
    talk = [
        (HOST, 0, 5_000, "Opening remarks for everybody."),
        (PRIYA, 1_700_000, 1_740_000, " ".join(["data"] * 40) + "."),
        (RAJ, 1_745_000, 1_790_000, " ".join(["schema"] * 30) + "."),
    ]
    body = source_body(
        agenda=[],
        turns=[
            {"speaker_id": str(u), "start_ms": s, "end_ms": e, "text": t} for u, s, e, t in talk
        ],
        ended_at=(START + timedelta(minutes=30)).isoformat(),
        call_ms=1_800_000,
    )
    src = rules.parse_source(body)
    assert src is not None
    thread = rules.open_thread(src, src.turns, [])
    assert thread is not None and thread.reason == "time_ran_out"
    quiet_end = dict(body, turns=body["turns"][:2])
    src2 = rules.parse_source(quiet_end)
    assert src2 is not None and rules.open_thread(src2, src2.turns, []) is None


def test_quiet_drops_the_never_record_window_and_the_next() -> None:
    body = source_body(
        turns=[
            {"speaker_id": str(HOST), "start_ms": 0, "end_ms": 3_000, "text": "Budget first."},
            {
                "speaker_id": str(PRIYA),
                "start_ms": 31_000,
                "end_ms": 33_000,
                "text": "About the layoffs in March.",
            },
            {
                "speaker_id": str(RAJ),
                "start_ms": 61_000,
                "end_ms": 63_000,
                "text": "Those reductions hit sales.",
            },
            {
                "speaker_id": str(HOST),
                "start_ms": 95_000,
                "end_ms": 97_000,
                "text": "Back to the budget.",
            },
        ],
        room_signals=[
            {
                "id": str(uuid.uuid4()),
                "kind": "spoken_over",
                "user_id": str(LEE),
                "at_ms": 62_000,
                "end_ms": 63_000,
                "detail": {},
            },
        ],
    )
    src = rules.parse_source(body)
    assert src is not None
    policy = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("layoffs",)))
    turns, signals = rules.quiet(src, policy)
    assert [t.start_ms for t in turns] == [0, 95_000]
    assert signals == []


def test_parse_source_is_tolerant() -> None:
    body = source_body(
        participants=[{"user_id": "nope"}, {"user_id": str(HOST), "listen_for": ["x", 3, ""]}],
        turns=[
            {"speaker_id": str(HOST), "start_ms": "0", "end_ms": 1, "text": "x"},
            {"speaker_id": "spk1", "start_ms": 5, "end_ms": 9, "text": "hi there"},
        ],
        speakers={"spk1": str(PRIYA)},
        decisions=[{"id": "bad"}, {"id": str(D1), "content": ""}],
        agenda=[{"id": str(ITEM_PRICING), "title": ""}],
        room_signals=[
            {"id": str(uuid.uuid4()), "kind": "wave", "user_id": str(HOST), "at_ms": 1, "end_ms": 2}
        ],
    )
    src = rules.parse_source(body)
    assert src is not None
    assert [p.listen_for for p in src.participants] == [("x",)]
    assert [(t.user_id, t.text) for t in src.turns] == [(PRIYA, "hi there")]
    assert src.decisions == [] and src.agenda == [] and src.room_signals == []
    assert rules.parse_source({"meeting_id": "x"}) is None and rules.parse_source([]) is None


# --- 2. the writer ----------------------------------------------------------------------------


async def test_the_whole_reading() -> None:
    llm, usage = LLM(), Usage()
    analysis = await writer(llm, usage=usage).write(ORG, MEETING, "gen1")
    assert analysis is not None
    p = analysis.payload
    validate(p)
    assert p["status"] == "ready" and p["generation"] == "gen1"
    assert p["prompt_versions"] == {
        "last_word": "last_word@v2",
        "consensus": "consensus_read@v1",
        "resume_point": "resume_point@v1",
    }
    by = {e["user_id"]: e for e in p["last_word"]}
    assert set(by) == {str(RAJ), str(LEE)}  # Sam sent his agent: never nudged
    raj, lee = by[str(RAJ)], by[str(LEE)]
    assert (raj["kind"], raj["topic"], raj["quote"], raj["at_ms"], raj["decision_id"]) == (
        "cut_off",
        "pricing for the enterprise tier",
        "I just want to say",
        13_000,
        str(D1),
    )
    assert (lee["kind"], lee["topic"], lee["quote"]) == ("topic_silent", "pricing", None)
    [c] = p["consensus"]
    assert c == {
        "decision_id": str(D1),
        "strength": "weak",
        "participants": 4,
        "yes": 2,
        "hedged": 0,
        "silent": 1,
        "objected": 0,
        "talked_over": 1,
        "ask": [
            {"user_id": str(RAJ), "reason": "talked_over"},
            {"user_id": str(LEE), "reason": "silent"},
        ],
    }
    rp = p["resume_point"]
    assert rp["reason"] == "cut_off" and rp["topic"] == "the migration risk"
    assert rp["agenda_item_id"] == str(ITEM_MIGRATION)
    assert [(x["user_id"], x["quote"], x["start_ms"]) for x in rp["positions"]] == [
        (str(PRIYA), "move the billing tables first", 250_000),
        (str(HOST), "run both systems in parallel for a week", 265_000),
    ]
    assert rp["cut_off"] == {
        "user_id": str(RAJ),
        "quote": "The migration risk is that the old billing tables",
        "start_ms": 280_000,
        "end_ms": 283_500,
    }
    assert analysis.result().kind == "room_signals"
    assert analysis.result().window_id == "room-signals:gen1"
    # every call metered, none for a person
    assert len(usage.calls) == len(llm.calls) == 3
    assert {u["user_id"] for u in usage.calls} == {None}


async def test_the_model_sees_labels_only_at_the_right_tier() -> None:
    llm = LLM()
    body = source_body()
    body["turns"][1]["text"] += " Mail priya@acme.com"
    await writer(llm, body).write(ORG, MEETING, "g")
    for call in llm.calls:
        text = user_text(call)
        for name in ("Hana", "Priya", "Raj", "Lee", "Sam"):
            assert name not in text
        for uid in (HOST, PRIYA, RAJ, LEE, SAM):
            assert str(uid) not in text
        assert "priya@acme.com" not in text
    [lw] = llm.of("last_word")
    assert lw["model"] == "anthropic/claude-haiku-5-5" and "output_config" not in lw
    assert "S1. Participant 3 was cut off mid-sentence at [00:13]" in user_text(lw)
    for name in ("consensus_read", "resume_point"):
        [call] = llm.of(name)
        assert call["model"] == "anthropic/claude-sonnet-5-5"
        assert call["output_config"] == {"effort": "low"}
    consensus = user_text(llm.of("consensus_read")[0])
    assert "Classify: Participant 1, Participant 2, Participant 3" in consensus
    assert "Decision: Raise enterprise pricing by ten percent" in consensus


async def test_a_stance_without_its_own_words_does_not_count() -> None:
    made_up = {
        "stances": [
            {"participant": "Participant 1", "stance": "clear_yes", "quote": "absolutely love it"},
            {"participant": "Participant 2", "stance": "clear_yes", "quote": "Yes, agreed."},
            {"participant": "Participant 2", "stance": "objected", "quote": "Yes, agreed."},
            {"participant": "Participant 9", "stance": "clear_yes", "quote": "x"},
            {"participant": "Participant 3", "stance": "maybe", "quote": None},
        ]
    }
    analysis = await writer(LLM(consensus_read=made_up)).write(ORG, MEETING, "g")
    assert analysis is not None
    [c] = analysis.payload["consensus"]
    assert (c["yes"], c["silent"], c["talked_over"]) == (1, 2, 1)
    assert c["strength"] == "weak"


async def test_last_word_keep_false_and_bad_topics() -> None:
    answer = {
        "signals": [
            {"ref": "S1", "topic": "Participant 2's idea", "keep": True},
            {"ref": "S2", "keep": False},
        ]
    }
    analysis = await writer(LLM(last_word=answer)).write(ORG, MEETING, "g")
    assert analysis is not None
    raj = next(e for e in analysis.payload["last_word"] if e["user_id"] == str(RAJ))
    assert raj["topic"] == "Enterprise pricing"  # a label is never a topic: the agenda item
    dropped = {
        "signals": [
            {"ref": "S1", "topic": "pricing", "keep": False},
            {"ref": "S2", "topic": "the migration risk", "keep": True},
        ]
    }
    analysis2 = await writer(LLM(last_word=dropped)).write(ORG, MEETING, "g")
    assert analysis2 is not None
    raj2 = next(e for e in analysis2.payload["last_word"] if e["user_id"] == str(RAJ))
    assert (raj2["at_ms"], raj2["topic"]) == (280_000, "the migration risk")


def test_parse_last_word() -> None:
    text = (
        "```json\n"
        + json.dumps(
            {
                "signals": [
                    {"ref": "s2", "topic": "x", "keep": True},
                    {"ref": "S2", "topic": "dup"},
                    {"ref": "S9", "topic": "out of range"},
                    {"ref": "Q1"},
                    {"ref": "S1", "topic": "mail me a@b.co"},
                ]
            }
        )
        + "\n```"
    )
    assert parse_last_word(text, 2) == {1: ("x", True), 0: (None, True)}
    assert parse_last_word("nope", 2) == {}


async def test_resume_closed_unverified_and_no_thread() -> None:
    closed = {"open": False, "topic": None, "positions": [], "cut_off": None}
    analysis = await writer(LLM(resume_point=closed)).write(ORG, MEETING, "g")
    assert analysis is not None
    # an agenda item still running at the end is open whatever the model says
    rp = analysis.payload["resume_point"]
    assert rp is not None and rp["topic"] == "Migration plan" and rp["positions"] == []
    assert rp["cut_off"]["user_id"] == str(RAJ)  # the rule's cut-off stands in
    body = source_body()
    body["agenda"][1]["status"] = "covered"
    analysis2 = await writer(LLM(resume_point=closed), body).write(ORG, MEETING, "g")
    assert analysis2 is not None and analysis2.payload["resume_point"] is None
    loose = dict(RESUME, positions=[{"participant": "Participant 2", "quote": "never said this"}])
    analysis3 = await writer(LLM(resume_point=loose)).write(ORG, MEETING, "g")
    assert analysis3 is not None
    [pos] = analysis3.payload["resume_point"]["positions"]
    assert pos["quote"] == TURNS[5][3]  # their own last line, not the invented words
    first = source_body()["turns"]
    calm = source_body(turns=[first[0], first[1], first[3], first[4]], agenda=[])
    llm = LLM()
    analysis4 = await writer(llm, calm).write(ORG, MEETING, "g")
    assert analysis4 is not None and analysis4.payload["resume_point"] is None
    assert llm.of("resume_point") == []
    validate(analysis4.payload)


# --- 3. fallbacks -----------------------------------------------------------------------------


async def test_budget_blocked_falls_back_to_the_rules() -> None:
    llm = LLM()
    analysis = await writer(llm, gate=budget(cap=10, consumed=11)).write(ORG, MEETING, "g")
    assert analysis is not None and llm.calls == []
    p = analysis.payload
    validate(p)
    assert p["status"] == "partial"
    raj = next(e for e in p["last_word"] if e["user_id"] == str(RAJ))
    assert raj["topic"] == "Enterprise pricing"  # the agenda item running at 00:13
    assert p["consensus"] == []  # never ask people on a half reading
    rp = p["resume_point"]
    assert rp["topic"] == "Migration plan" and rp["reason"] == "cut_off"
    assert [x["user_id"] for x in rp["positions"]] == [str(RAJ), str(HOST)]
    assert rp["cut_off"]["quote"] == "The migration risk is that the old billing tables"


async def test_one_failed_step_leaves_the_others() -> None:
    llm = LLM(consensus_read=RuntimeError("provider down"))
    analysis = await writer(llm).write(ORG, MEETING, "g")
    assert analysis is not None
    p = analysis.payload
    validate(p)
    assert p["status"] == "partial" and p["consensus"] == []
    assert p["last_word"] and p["resume_point"]["topic"] == "the migration risk"
    garbled = await writer(LLM(last_word="no json here")).write(ORG, MEETING, "g")
    assert garbled is not None and garbled.payload["status"] == "partial"


async def test_no_transcript_still_reads_the_room() -> None:
    body = source_body(
        turns=[],
        agenda=[],
        decisions=[],
        room_signals=[
            {
                "id": str(SIG_HAND),
                "kind": "hand_not_called",
                "user_id": str(LEE),
                "at_ms": 10_000,
                "end_ms": 90_000,
                "detail": {"outcome": "still_raised_at_end", "waited_ms": 80_000},
            }
        ],
    )
    llm = LLM(last_word={"signals": [{"ref": "S1", "topic": "the agenda", "keep": True}]})
    analysis = await writer(llm, body).write(ORG, MEETING, "g")
    assert analysis is not None
    validate(analysis.payload)
    [e] = analysis.payload["last_word"]
    assert (e["user_id"], e["kind"], e["signal_ids"]) == (
        str(LEE),
        "hand_not_called",
        [str(SIG_HAND)],
    )
    assert analysis.payload["consensus"] == [] and analysis.payload["resume_point"] is None


# --- 4. never record --------------------------------------------------------------------------


async def test_never_record_reaches_no_model_and_no_payload() -> None:
    org = CompanyContext(never_record_topics=("migration",))
    leaky = {"signals": [{"ref": "S1", "topic": "the migration budget", "keep": True}]}
    llm = LLM(last_word=leaky)
    analysis = await writer(llm, org=org).write(ORG, MEETING, "g")
    assert analysis is not None
    for call in llm.calls:
        user = user_text(call)
        assert "old billing tables" not in user  # the windows naming "migration" never left
        assert "Migration plan" not in user  # nor an agenda title naming it
        assert "Never record" in user
    text = json.dumps(analysis.payload).lower()
    assert "migration" not in text
    validate(analysis.payload)


# --- 5. the loop ------------------------------------------------------------------------------


class Sink:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.flushes = 0

    async def agent_result(self, **kw: Any) -> None:
        self.results.append(kw)

    def flush(self, timeout_s: float = 10.0) -> None:
        self.flushes += 1


def ended(event_type: str = "ended", key: str = "meeting:x:ended:t1") -> dict[str, Any]:
    return {
        "header": {"org_id": str(ORG), "idempotency_key": key},
        "meeting_id": str(MEETING),
        "event_type": event_type,
        "livekit_room": None,
    }


async def _no_sleep(_s: float) -> None:
    return None


async def test_loop_emits_once_per_generation_and_claims() -> None:
    llm, sink, redis = LLM(), Sink(), fakeredis.aioredis.FakeRedis()
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    loop = RoomSignalsLoop(writer(llm), sink, claims=redis, sleep=sleep, settle_s=20)
    assert await loop.handle(ended("updated")) == 0
    assert await loop.handle(ended()) == 1
    gen = generation_of("meeting:x:ended:t1", MEETING)
    assert slept == [20] and sink.flushes == 1
    [r] = sink.results
    assert (r["agent"], r["user_id"], r["window_id"]) == (
        "MeetingAgent",
        NIL,
        f"room-signals:{gen}",
    )
    assert r["result"].kind == "room_signals"
    validate(r["result"].payload)
    assert await redis.exists(claim_key(ORG, MEETING, gen))
    assert claim_key(ORG, MEETING, gen).startswith(f"nexus:{ORG}:room-signals-done:")
    calls = len(llm.calls)
    assert await loop.handle(ended()) == 0 and len(llm.calls) == calls
    assert await loop.handle(ended(key="meeting:x:ended:t2")) == 1


async def test_loop_retries_then_sends_an_empty_partial_record() -> None:
    sink = Sink()
    flaky = FakeSource(SourceUnavailable("core-api down"))
    loop = RoomSignalsLoop(writer(LLM(), source=flaky), sink, sleep=_no_sleep, attempts=3)
    assert await loop.handle(ended()) == 1
    assert flaky.calls == 3
    payload = sink.results[0]["result"].payload
    validate(payload)
    assert payload["status"] == "partial" and payload["last_word"] == []

    async def no_brain(_org: uuid.UUID) -> CompanyContext:
        raise OrgContextUnavailable(str(ORG))

    w = writer(LLM())
    w.org_context = no_brain
    sink2 = Sink()
    assert await RoomSignalsLoop(w, sink2, sleep=_no_sleep, attempts=2).handle(ended()) == 1
    assert sink2.results[0]["result"].payload["status"] == "partial"


async def test_an_unknown_meeting_writes_nothing() -> None:
    sink = Sink()
    loop = RoomSignalsLoop(writer(LLM(), source=FakeSource(None)), sink, sleep=_no_sleep)
    assert await loop.handle(ended()) == 0 and sink.results == [] and sink.flushes == 0


def test_consumer_config_is_its_own_group_from_the_end() -> None:
    cfg = consumer_config("b:9092")
    assert cfg["group.id"] == "agent-runtime-room-signals"
    assert cfg["auto.offset.reset"] == "latest" and cfg["enable.auto.commit"] is False


async def test_source_client_speaks_the_caller_token_scheme() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == f"/internal/meetings/{MEETING}/room-signals-source":
            return httpx.Response(200, json=source_body())
        return httpx.Response(404)

    client = HttpRoomSignalsSource("http://core/", "tok", transport=httpx.MockTransport(handler))
    got = await client.source(ORG, MEETING)
    assert got is not None and got.meeting_id == MEETING and len(got.turns) == len(TURNS)
    h = seen[0].headers
    assert (h["X-Mytheus-Service"], h["X-Mytheus-Service-Token"], h["X-Mytheus-Org"]) == (
        "agent-runtime-service",
        "tok",
        str(ORG),
    )
    assert await client.source(ORG, uuid.uuid4()) is None

    def busy(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(409)

    with pytest.raises(SourceUnavailable):
        await HttpRoomSignalsSource("http://core", "t", transport=httpx.MockTransport(busy)).source(
            ORG, MEETING
        )


async def test_record_encodes_against_agent_results_with_the_room_signals_kind() -> None:
    events = Path(__file__).resolve().parents[2] / "a3-events"
    schema_path = events / "schemas" / "agent-results" / "v1.avsc"
    if not schema_path.exists() or "room_signals" not in schema_path.read_text():
        pytest.skip("the a3-events worktree with the room_signals kind is not beside this one")
    named: dict[str, Any] = {}
    fastavro.parse_schema(
        json.loads((events / "schemas" / "_envelope" / "EventEnvelopeHeader.avsc").read_text()),
        named_schemas=named,
    )
    schema = fastavro.parse_schema(json.loads(schema_path.read_text()), named_schemas=named)

    class Codec:
        def serialize_for(self, _topic: str, value: dict[str, Any]) -> bytes:
            buf = io.BytesIO()
            fastavro.schemaless_writer(buf, schema, value)
            return buf.getvalue()

    class Producer:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def produce(self, _topic: str, key: str, value: bytes, on_delivery: Any) -> None:
            self.sent.append(value)

        def poll(self, _t: float) -> None:
            return None

        def flush(self, _t: float) -> int:
            return 0

    producer = Producer()
    loop = RoomSignalsLoop(writer(LLM()), Emitter(producer, Codec()), sleep=_no_sleep)
    assert await loop.handle(ended()) == 1
    back = fastavro.schemaless_reader(io.BytesIO(producer.sent[0]), schema, schema)
    assert (back["agent"], back["kind"]) == ("MeetingAgent", "room_signals")
    assert back["header"]["idempotency_key"] == (
        f"agentresult:{MEETING}:{back['window_id']}:{NIL}:MeetingAgent"
    )
    validate(json.loads(back["payload_json"]))


def test_fallback_topic_and_agenda_clock() -> None:
    agenda = [
        AgendaItem(ITEM_PRICING, "Pricing", "covered", 0, 60_000),
        AgendaItem(ITEM_MIGRATION, "Migration", "in_progress", 60_000, None),
    ]
    c = Candidate(RAJ, "cut_off", 70_000, 71_000)
    assert rules.fallback_topic(c, agenda, NeverRecordPolicy()) == "Migration"
    secret = NeverRecordPolicy.from_org(CompanyContext(never_record_topics=("migration",)))
    assert rules.fallback_topic(c, agenda, secret) == rules.NO_TOPIC
    assert (
        rules.fallback_topic(Candidate(RAJ, "cut_off", None, None), agenda, NeverRecordPolicy())
        == rules.NO_TOPIC
    )
