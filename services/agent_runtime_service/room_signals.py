"""Room signals: after a meeting ends, the MeetingAgent reads who did not get to speak, how real
the agreement on each decision was, and whether the meeting stopped mid-thread (a3: "the last
word", "fake-consensus detector", "pick up where we stopped").

Trigger. `meeting-events` `ended` (core-api's outbox), its own group
`agent-runtime-room-signals` beside the minutes loop. A new group starts at the end of the
topic, so the first deploy analyses no historic meeting.

Input. core-api's internal `GET /internal/meetings/{id}/room-signals-source` (the agent-runtime
caller token, the tenant in `X-Mytheus-Org`): participants with their listen-for topics, the
finals, the decisions with their moments, the agenda with its clock, and the room's own signals
(spoken over, a hand never called) that realtime-gateway recorded and core-api filed. This tier
never reads Postgres.

Per meeting end (`RoomSignalsWriter.write`):
1. Never record. The company brain is read strictly; every 30 s window naming a never-record
   topic, and the one after, is removed (turns and room signals) before any rule or model sees
   it, the prompts carry the "Never record" line, and any topic or quote that names such a
   topic is dropped before it leaves the tier.
2. Rules (`room_signals_rules.py`, no model): the "didn't get in" moments (cut off, spoken
   over, hand never called, a listen-for topic that came up while they said nothing), one per
   person; each decision's window (who was there, who said nothing, who was talked over);
   whether the meeting may have ended mid-thread.
3. Model, only for what words alone cannot say, every call through the LiteLLM router
   (budget-gated, metered, PII-redacted), speakers as "Participant N":
   - `last_word@vN` (triage): one batched call naming the topic of each moment, and dropping
     a moment whose point was clearly made after all;
   - `consensus_read@vN` (extract, effort low): per decision, each speaker's stance (clear
     yes, hedged, objected, no stance) with a quote that must be found verbatim in their own
     words, else the stance does not count;
   - `resume_point@vN` (extract, effort low): whether a thread was open, its topic, the last
     two positions and who was cut off, quotes verified the same way.
   A blocked or failed step leaves `status: partial` and falls back to the rules: a moment's
   topic is the agenda item running then (else "what the room was discussing"); consensus is
   skipped; a resume point is the unfinished agenda item with the last two speakers' turns.
4. Output. One `agent-results` record, agent `MeetingAgent`, kind `room_signals`, user the nil
   uuid, `window_id = room-signals:{generation}` (one generation per `ended` event, so a
   replay republishes the same key and core-api's inbox dedupes it).

Once flushed, `nexus:{org_id}:room-signals-done:{meeting_id}:{generation}` is set (7 days), so
a replayed `ended` does not bill the calls again. (`nexus:{org_id}:room-signals:{meeting_id}` is
realtime-gateway's list of live signals; this key is deliberately different.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import httpx

from libs.shared.json_answer import json_object
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.router import (
    EXTRACT_MAX_TOKENS,
    EXTRACTION_EFFORT,
    BudgetBlocked,
    LiteLLMRouter,
)
from libs.shared.state import AgentResult
from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service import room_signals_rules as rules
from services.agent_runtime_service.agents.company import render_company, render_ground_rules
from services.agent_runtime_service.minutes import ResultSink, generation_of
from services.agent_runtime_service.room_signals_rules import (
    Candidate,
    ConsensusFrame,
    OpenThread,
    RoomSignalsSource,
    Turn,
)

logger = logging.getLogger(__name__)

AGENT = "MeetingAgent"
KIND = "room_signals"
GROUP = "agent-runtime-room-signals"
TOPIC = "meeting-events"
NIL = UUID(int=0)
SERVICE_NAME = "agent-runtime-service"
LAST_WORD = "last_word"
CONSENSUS = "consensus_read"
RESUME = "resume_point"
CLAIM_TTL_S = 7 * 24 * 3600
TRIAGE_MAX_TOKENS = 1024
MAX_MODEL_SIGNALS = 30  # moments sent to last_word in one call
CONTEXT_BEFORE_MS = 90_000
CONTEXT_AFTER_MS = 30_000
MAX_DECISION_CALLS = 20
MAX_CONTEXT_CHARS = 60_000
_LABEL_IN = re.compile(r"\bparticipant\s*#?\s*\d+\b", re.IGNORECASE)


# --- the source client ------------------------------------------------------------------------


class SourceUnavailable(Exception):
    """core-api could not answer now (unreachable, 5xx, 409 not ended yet): retry later."""


class RoomSignalsSourceClient(Protocol):
    async def source(self, org_id: UUID, meeting_id: UUID) -> RoomSignalsSource | None: ...


class HttpRoomSignalsSource:
    """`None` for a meeting core-api does not know (404, another tenant); `SourceUnavailable`
    for anything worth retrying."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    async def source(self, org_id: UUID, meeting_id: UUID) -> RoomSignalsSource | None:
        path = f"/internal/meetings/{meeting_id}/room-signals-source"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(
                    f"{self._url}{path}",
                    headers={
                        "X-Mytheus-Service": SERVICE_NAME,
                        "X-Mytheus-Service-Token": self._token,
                        "X-Mytheus-Org": str(org_id),
                    },
                )
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"{path}: {exc}") from None
        if res.status_code == 404:
            return None
        if res.status_code != 200:
            raise SourceUnavailable(f"{path} answered {res.status_code}")
        try:
            return rules.parse_source(res.json())
        except ValueError:
            return None


def source_from_env() -> HttpRoomSignalsSource | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpRoomSignalsSource(url, token)


# --- labels -----------------------------------------------------------------------------------


@dataclass
class Labels:
    """user id <-> "Participant N" for one meeting."""

    of: dict[UUID | str, str]

    @classmethod
    def build(cls, source: RoomSignalsSource) -> Labels:
        return cls(rules.assign_labels(source))

    def label(self, who: UUID | str | None) -> str:
        return self.of.get(who, "Someone") if who is not None else "Someone"

    def user(self, raw: object) -> UUID | None:
        if not isinstance(raw, str):
            return None
        wanted = " ".join(raw.split()).lower()
        for who, lab in self.of.items():
            if isinstance(who, UUID) and lab.lower() == wanted:
                return who
        return None


def _parse(text: str, key: str) -> dict[str, Any] | None:
    obj = json_object(text, key)
    return obj if isinstance(obj, dict) else None


# --- rendering for the model ------------------------------------------------------------------

_WHAT = {
    "cut_off": "was cut off mid-sentence",
    "spoken_over": "tried to come in and was talked over",
    "hand_not_called": "raised their hand and was never called on",
}


def render_signals(
    moments: list[Candidate],
    turns: list[Turn],
    source: RoomSignalsSource,
    labels: Labels,
    policy: NeverRecordPolicy,
) -> str:
    parts = [f"Meeting: {source.title}", "", "Moments:"]
    for n, c in enumerate(moments, start=1):
        at = c.at_ms or 0
        end = c.end_ms if c.end_ms is not None else at
        item = rules.agenda_at(c.at_ms, source.agenda)
        item_title = item.title if item is not None and not policy.hit(item.title) else None
        near = [
            t for t in turns if rules.overlaps(t, at - CONTEXT_BEFORE_MS, end + CONTEXT_AFTER_MS)
        ]
        parts += [
            "",
            f"S{n}. {labels.label(c.user_id)} {_WHAT[c.kind]} at [{rules.clock(at)}].",
            f"Agenda item then: {item_title or '(unknown)'}",
            f"Their words then: {c.quote or '(none)'}",
            "Lines around it:",
            rules.render_lines(near, labels.of)[: MAX_CONTEXT_CHARS // max(1, len(moments))]
            or "(no transcript)",
        ]
    return "\n".join(parts) + "\n"


def render_decision(frame: ConsensusFrame, source: RoomSignalsSource, labels: Labels) -> str:
    classify = [labels.label(u) for u in frame.speakers]
    return (
        "\n".join(
            [
                f"Meeting: {source.title}",
                "",
                f"Decision: {frame.decision.content}",
                f"Made at about [{rules.clock(frame.decision.quote_start_ms or 0)}]"
                + (f': "{frame.decision.quote}"' if frame.decision.quote else ""),
                "",
                "Classify: " + ", ".join(classify),
                "",
                "Transcript around it:",
                rules.render_lines(frame.turns, labels.of)[:MAX_CONTEXT_CHARS],
            ]
        )
        + "\n"
    )


_WHY = {
    "cut_off": "someone was cut off mid-explanation near the end",
    "agenda_unfinished": "an agenda item was still open when the meeting ended",
    "time_ran_out": "the meeting reached its scheduled end while people were still talking",
}


def render_thread(
    thread: OpenThread, source: RoomSignalsSource, labels: Labels, policy: NeverRecordPolicy
) -> str:
    cut = thread.cut_off
    item = thread.agenda_item
    item_title = item.title if item is not None and not policy.hit(item.title) else None
    lines = [
        f"Meeting: {source.title}",
        f"Why it may have ended open: {_WHY[thread.reason]}",
        f"Unfinished agenda item: {item_title or '(none)'}",
        (
            f"Possibly cut off: {labels.label(cut.user_id)} at [{rules.clock(cut.at_ms or 0)}]: "
            f'"{cut.quote}"'
            if cut is not None
            else "Possibly cut off: (nobody)"
        ),
        "",
        "The last minutes of the transcript:",
        rules.render_lines(thread.tail, labels.of)[-MAX_CONTEXT_CHARS:] or "(no transcript)",
    ]
    return "\n".join(lines) + "\n"


# --- the writer -------------------------------------------------------------------------------

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


@dataclass
class Analysis:
    """What one meeting end produces: the payload, and whether any model step fell back."""

    payload: dict[str, Any]

    def result(self) -> AgentResult:
        return AgentResult(
            kind=KIND,
            content="",
            relevance_score=1.0,
            confidence=1.0,
            source_chunk_ids=[],
            payload=self.payload,
            window_id=f"room-signals:{self.payload['generation']}",
        )


@dataclass
class _Run:
    org_id: UUID
    source: RoomSignalsSource
    turns: list[Turn]
    labels: Labels
    policy: NeverRecordPolicy
    company: str
    ground_rules: str
    partial: bool = False
    versions: dict[str, str | None] = field(
        default_factory=lambda: {"last_word": None, "consensus": None, "resume_point": None}
    )

    def safe(self, value: str | None) -> str | None:
        """None when the text names a never-record topic."""
        if value is None:
            return None
        return None if self.policy.hit(value) else value


def empty_payload(generation: str, status: str = "ready") -> dict[str, Any]:
    return {
        "generation": generation,
        "status": status,
        "prompt_versions": {"last_word": None, "consensus": None, "resume_point": None},
        "last_word": [],
        "consensus": [],
        "resume_point": None,
    }


class RoomSignalsWriter:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        source: RoomSignalsSourceClient,
        org_context: OrgContextFn | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.loader = loader
        self.router = router
        self.source = source
        self.org_context = org_context
        self._sem = asyncio.Semaphore(max_concurrency)

    def _prompt(self, key: str, org_id: UUID) -> Prompt:
        return self.loader.load(key, self.loader.resolve(key, org_id))

    async def _call(self, run: _Run, prompt: Prompt, tier: str, body_key: str, body: str) -> str:
        values = {"company": run.company, "ground_rules": run.ground_rules, body_key: body}
        async with self._sem:
            out = await self.router.complete(
                tier="triage" if tier == "triage" else "extract",
                system=prompt.system,
                org_id=run.org_id,
                user_id=None,
                max_tokens=TRIAGE_MAX_TOKENS if tier == "triage" else EXTRACT_MAX_TOKENS,
                prompt_ref=prompt.ref,
                effort=None if tier == "triage" else EXTRACTION_EFFORT,
                **prompt.user_parts(**values),
            )
        return out.text

    async def write(self, org_id: UUID, meeting_id: UUID, generation: str) -> Analysis | None:
        """The payload for one meeting end, or None for a meeting core-api does not know.
        Raises `SourceUnavailable` or `OrgContextUnavailable` when the run should be retried."""
        source = await self.source.source(org_id, meeting_id)
        if source is None:
            logger.info("room signals: meeting %s unknown to core-api", meeting_id)
            return None
        org = await self.org_context(org_id) if self.org_context is not None else None
        policy = NeverRecordPolicy.from_org(org)
        turns, signals = rules.quiet(source, policy)
        # Rules and labels run on the quiet meeting only.
        # A title or decision naming a never-record topic never reaches a model either.
        title = "Untitled meeting" if policy.hit(source.title) else source.title
        decisions = [d for d in source.decisions if not policy.hits(d.content, d.quote)]
        quiet = RoomSignalsSource(
            **{
                **source.__dict__,
                "title": title,
                "turns": turns,
                "room_signals": signals,
                "decisions": decisions,
            }
        )
        run = _Run(
            org_id=org_id,
            source=quiet,
            turns=turns,
            labels=Labels.build(quiet),
            policy=policy,
            company=render_company(org) if org is not None else "",
            ground_rules=render_ground_rules(org) if org is not None else "",
        )
        cut = rules.cut_offs(turns)
        over = rules.spoken_over(signals, turns)
        hands = rules.hands_not_called(signals, turns)
        silent = rules.topic_silences(rules.present_people(quiet), turns, policy)
        last_word, consensus, resume = await asyncio.gather(
            self._last_word(run, [*cut, *over, *hands], silent),
            self._consensus(run, rules.consensus_frames(quiet, turns, signals, cut, over)),
            self._resume(run, rules.open_thread(quiet, turns, cut)),
        )
        payload = {
            "generation": generation,
            "status": "partial" if run.partial else "ready",
            "prompt_versions": dict(run.versions),
            "last_word": last_word,
            "consensus": consensus,
            "resume_point": resume,
        }
        return Analysis(payload)

    # --- the last word ----------------------------------------------------------------------

    async def _last_word(
        self, run: _Run, moments: list[Candidate], silent: list[Candidate]
    ) -> list[dict[str, Any]]:
        moments = sorted(moments, key=lambda c: (c.at_ms or 0, str(c.user_id)))
        # The best moment per person first: the model names the moments that can be sent.
        best = rules.pick_one_per_user(moments)
        chosen = {id(c) for c in best}
        rest = [c for c in moments if id(c) not in chosen]
        to_name = (best + rest)[:MAX_MODEL_SIGNALS]
        named: dict[int, tuple[str | None, bool]] = {}
        if to_name:
            prompt = self._prompt(LAST_WORD, run.org_id)
            run.versions["last_word"] = prompt.ref
            try:
                text = await self._call(
                    run,
                    prompt,
                    "triage",
                    "signals",
                    render_signals(to_name, run.turns, run.source, run.labels, run.policy),
                )
                named = parse_last_word(text, len(to_name))
                if not named:
                    raise ValueError("last_word answer unparseable")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the rules' topics stand in
                self._fell_back(run, LAST_WORD, exc)
                named = {}
        kept: list[Candidate] = []
        for i, c in enumerate(to_name):
            topic, keep = named.get(i, (None, True))
            if not keep:
                continue
            c.topic = run.safe(rules.clip(topic, rules.MAX_TOPIC)) if topic else None
            if c.topic is None:
                c.topic = rules.fallback_topic(c, run.source.agenda, run.policy)
            kept.append(c)
        out: list[dict[str, Any]] = []
        for c in rules.pick_one_per_user([*kept, *silent]):
            topic = run.safe(c.topic)
            if topic is None:
                continue
            quote = rules.clip(run.safe(c.quote), rules.MAX_QUOTE)
            out.append(
                {
                    "user_id": str(c.user_id),
                    "kind": c.kind,
                    "topic": topic,
                    "quote": quote,
                    "at_ms": c.at_ms,
                    "end_ms": c.end_ms,
                    "decision_id": _opt(rules.nearest_decision(c.at_ms, run.source.decisions)),
                    "signal_ids": [str(s) for s in c.signal_ids],
                }
            )
        return out[: rules.MAX_LAST_WORD]

    # --- consensus --------------------------------------------------------------------------

    async def _consensus(self, run: _Run, frames: list[ConsensusFrame]) -> list[dict[str, Any]]:
        if not frames:
            return []
        prompt = self._prompt(CONSENSUS, run.org_id)

        async def read(frame: ConsensusFrame) -> dict[str, Any] | None:
            if not frame.speakers:
                return rules.tally(frame, {})  # nobody spoke around it: no model needed
            text = await self._call(
                run,
                prompt,
                "extract",
                "decision_block",
                render_decision(frame, run.source, run.labels),
            )
            stances = parse_stances(text, frame, run.labels, run.policy)
            if stances is None:
                raise ValueError("consensus_read answer unparseable")
            return rules.tally(frame, stances)

        frames = frames[:MAX_DECISION_CALLS]
        run.versions["consensus"] = prompt.ref if any(f.speakers for f in frames) else None
        answers = await asyncio.gather(*(read(f) for f in frames), return_exceptions=True)
        failed = [a for a in answers if isinstance(a, BaseException)]
        for a in failed:
            if isinstance(a, asyncio.CancelledError):
                raise a
        if failed:
            # Without a stance read, a yes cannot be told from a hedge: skip consensus for this
            # meeting rather than ask people on a half reading.
            self._fell_back(run, CONSENSUS, failed[0])
            return []
        return [a for a in answers if isinstance(a, dict)]

    # --- resume point -----------------------------------------------------------------------

    async def _resume(self, run: _Run, thread: OpenThread | None) -> dict[str, Any] | None:
        if thread is None:
            return None
        prompt = self._prompt(RESUME, run.org_id)
        run.versions["resume_point"] = prompt.ref
        try:
            text = await self._call(
                run,
                prompt,
                "extract",
                "thread",
                render_thread(thread, run.source, run.labels, run.policy),
            )
            answer = _parse(text, "open")
            if answer is None or not isinstance(answer.get("open"), bool):
                raise ValueError("resume_point answer unparseable")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the rules' resume point stands in
            self._fell_back(run, RESUME, exc)
            return fallback_resume(run, thread)
        return parse_resume(answer, thread, run)

    def _fell_back(self, run: _Run, step: str, exc: BaseException) -> None:
        run.partial = True
        if isinstance(exc, BudgetBlocked):
            logger.info("room signals %s: budget blocked for %s", step, run.source.meeting_id)
        else:
            logger.warning("room signals %s failed for %s: %r", step, run.source.meeting_id, exc)


def _opt(value: UUID | None) -> str | None:
    return str(value) if value is not None else None


# --- parsing the answers ----------------------------------------------------------------------


def parse_last_word(text: str, count: int) -> dict[int, tuple[str | None, bool]]:
    """index -> (topic, keep) for the refs S1..S{count}; {} when not the JSON asked for."""
    raw = _parse(text, "signals")
    items = raw.get("signals") if raw else None
    if not isinstance(items, list):
        return {}
    out: dict[int, tuple[str | None, bool]] = {}
    for e in items:
        if not isinstance(e, dict) or not isinstance(e.get("ref"), str):
            continue
        ref = e["ref"].strip().upper()
        if not ref.startswith("S") or not ref[1:].isdigit():
            continue
        i = int(ref[1:]) - 1
        if not 0 <= i < count or i in out:
            continue
        topic = e.get("topic") if isinstance(e.get("topic"), str) else None
        if topic is not None and (_LABEL_IN.search(topic) or "@" in topic):
            topic = None  # a label or an email is never a topic
        out[i] = (topic, e.get("keep") is not False)
    return out


def parse_stances(
    text: str, frame: ConsensusFrame, labels: Labels, policy: NeverRecordPolicy
) -> dict[UUID, str] | None:
    """user id -> stance for the frame's speakers. A stance other than no_stance counts only
    when its quote is found verbatim in that person's own words in the window (and names no
    never-record topic); otherwise it is no_stance. None when not the JSON asked for."""
    raw = _parse(text, "stances")
    items = raw.get("stances") if raw else None
    if not isinstance(items, list):
        return None
    speakers = set(frame.speakers)
    out: dict[UUID, str] = {}
    for e in items:
        if not isinstance(e, dict):
            continue
        uid = labels.user(e.get("participant"))
        if uid is None or uid not in speakers or uid in out:
            continue
        stance = e.get("stance") if e.get("stance") in rules.STANCES else "no_stance"
        if stance != "no_stance":
            quote = e.get("quote")
            turn = rules.find_quote(quote, frame.turns, uid)
            if turn is None or policy.hit(str(quote)):
                stance = "no_stance"
        out[uid] = str(stance)
    return out


def _verified(raw: object, run: _Run, turns: list[Turn]) -> tuple[UUID | None, Turn] | None:
    """A {participant, quote} pair from the model as (user, the turn holding the quote). An
    unverifiable quote falls back to that participant's last turn in the tail."""
    if not isinstance(raw, dict):
        return None
    uid = run.labels.user(raw.get("participant"))
    if uid is None:
        return None
    turn = rules.find_quote(raw.get("quote"), turns, uid)
    if turn is None:
        own = [t for t in turns if t.user_id == uid]
        if not own:
            return None
        turn = own[-1]
    return uid, turn


def _quote_for(raw: object, turn: Turn) -> str | None:
    """The model's quote when it is verbatim in the turn, else the turn's own text."""
    if isinstance(raw, dict) and isinstance(raw.get("quote"), str):
        q = " ".join(str(raw["quote"]).split())
        if q and rules.norm(q) in rules.norm(turn.text):
            return rules.clip(q, rules.MAX_QUOTE)
    return rules.clip(turn.text, rules.MAX_QUOTE)


def _cut_dict(c: Candidate) -> dict[str, Any] | None:
    if c.quote is None or c.at_ms is None:
        return None
    return {
        "user_id": str(c.user_id),
        "quote": rules.clip(c.quote, rules.MAX_QUOTE),
        "start_ms": c.at_ms,
        "end_ms": c.end_ms if c.end_ms is not None else c.at_ms,
    }


def parse_resume(answer: dict[str, Any], thread: OpenThread, run: _Run) -> dict[str, Any] | None:
    # An agenda item still running when the call ended is open whatever the model says.
    keep_anyway = thread.agenda_item is not None and thread.agenda_item.status == "in_progress"
    if answer.get("open") is not True and not keep_anyway:
        return None
    raw_topic = answer.get("topic") if isinstance(answer.get("topic"), str) else None
    if raw_topic and _LABEL_IN.search(raw_topic):
        raw_topic = None
    topic = rules.clip(raw_topic, rules.MAX_RESUME_TOPIC)
    if topic is None and thread.agenda_item is not None:
        topic = rules.clip(thread.agenda_item.title, rules.MAX_RESUME_TOPIC)
    topic = run.safe(topic)
    if topic is None:
        return None  # nothing nameable (or it names a never-record topic): no resume point
    tail = thread.tail
    positions: list[dict[str, Any]] = []
    for raw in answer.get("positions") or [] if isinstance(answer.get("positions"), list) else []:
        found = _verified(raw, run, tail)
        if found is None:
            continue
        uid, turn = found
        quote = run.safe(_quote_for(raw, turn))
        if quote is None:
            continue
        positions.append(
            {"user_id": str(uid), "quote": quote, "start_ms": turn.start_ms, "end_ms": turn.end_ms}
        )
    positions = positions[-rules.MAX_POSITIONS :]
    cut_off: dict[str, Any] | None = None
    found = _verified(answer.get("cut_off"), run, tail)
    if found is not None:
        uid, turn = found
        quote = run.safe(_quote_for(answer.get("cut_off"), turn))
        if quote is not None:
            cut_off = {
                "user_id": str(uid),
                "quote": quote,
                "start_ms": turn.start_ms,
                "end_ms": turn.end_ms,
            }
    if cut_off is None and thread.cut_off is not None and thread.reason == "cut_off":
        cut_off = _cut_dict(thread.cut_off)
        if cut_off is not None and run.safe(cut_off["quote"]) is None:
            cut_off = None
    return {
        "reason": thread.reason,
        "topic": topic,
        "agenda_item_id": str(thread.agenda_item.id) if thread.agenda_item else None,
        "positions": positions,
        "cut_off": cut_off,
    }


def fallback_resume(run: _Run, thread: OpenThread) -> dict[str, Any] | None:
    """No model answer: the unfinished agenda item (else a neutral topic), the last two
    speakers' turns, and the rule's cut-off."""
    topic = run.safe(thread.agenda_item.title) if thread.agenda_item is not None else None
    if topic is None and thread.agenda_item is None:
        topic = "where you left off"
    if topic is None:
        return None
    positions = []
    for t in rules.fallback_positions(thread.tail):
        quote = run.safe(rules.clip(t.text, rules.MAX_QUOTE))
        if quote is None:
            continue
        positions.append(
            {
                "user_id": _opt(t.user_id),
                "quote": quote,
                "start_ms": t.start_ms,
                "end_ms": t.end_ms,
            }
        )
    cut = _cut_dict(thread.cut_off) if thread.cut_off is not None else None
    if cut is not None and run.safe(cut["quote"]) is None:
        cut = None
    return {
        "reason": thread.reason,
        "topic": rules.clip(topic, rules.MAX_RESUME_TOPIC),
        "agenda_item_id": str(thread.agenda_item.id) if thread.agenda_item else None,
        "positions": positions,
        "cut_off": cut,
    }


# --- the loop ---------------------------------------------------------------------------------


def claim_key(org_id: UUID, meeting_id: UUID, generation: str) -> str:
    return f"nexus:{org_id}:room-signals-done:{meeting_id}:{generation}"


@dataclass
class EndedMeeting:
    org_id: UUID
    meeting_id: UUID
    generation: str


def parse_ended(value: dict[str, Any]) -> EndedMeeting | None:
    if str(value.get("event_type")) != "ended":
        return None
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    try:
        meeting_id = UUID(str(value.get("meeting_id")))
    except ValueError:
        return None
    key = str(header.get("idempotency_key") or "")
    if org_id is None or not key:
        return None
    return EndedMeeting(org_id, meeting_id, generation_of(key, meeting_id))


class RoomSignalsLoop:
    """One decoded `meeting-events` record in, one `room_signals` record out, flushed.

    `settle_s`: the last finals land a few seconds after the end. `attempts`: a retryable
    failure is retried with backoff; after the last attempt an empty `partial` record is emitted
    so the after page stops waiting (nothing is nudged on a reading that never happened)."""

    def __init__(
        self,
        writer: RoomSignalsWriter,
        sink: ResultSink,
        *,
        claims: Any = None,  # redis.asyncio: `exists` and `set(..., ex=)`
        settle_s: float = 20.0,
        attempts: int = 3,
        backoff_s: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.writer = writer
        self.sink = sink
        self.claims = claims
        self.settle_s = settle_s
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self._sleep = sleep

    async def _claimed(self, key: str) -> bool:
        if self.claims is None:
            return False
        try:
            return bool(await self.claims.exists(key))
        except Exception:  # noqa: BLE001 - no claim store: write (core-api dedupes the key)
            return False

    async def handle(self, value: dict[str, Any]) -> int:
        ended = parse_ended(value)
        if ended is None:
            return 0
        key = claim_key(ended.org_id, ended.meeting_id, ended.generation)
        if await self._claimed(key):
            return 0
        if self.settle_s > 0:
            await self._sleep(self.settle_s)
        analysis: Analysis | None = None
        known = True
        for attempt in range(self.attempts):
            try:
                analysis = await self.writer.write(ended.org_id, ended.meeting_id, ended.generation)
                known = analysis is not None
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then given up loudly
                logger.warning(
                    "room signals for %s failed (attempt %d): %r",
                    ended.meeting_id,
                    attempt + 1,
                    exc,
                )
                if attempt + 1 < self.attempts:
                    await self._sleep(self.backoff_s * (2**attempt))
        if not known:
            return 0  # core-api does not know the meeting: nothing to file
        if analysis is None:
            logger.error("room signals for %s given up; an empty record is sent", ended.meeting_id)
            analysis = Analysis(empty_payload(ended.generation, "partial"))
        await self.sink.agent_result(
            org_id=ended.org_id,
            meeting_id=ended.meeting_id,
            window_id=f"room-signals:{ended.generation}",
            user_id=NIL,
            agent=AGENT,
            result=analysis.result(),
        )
        self.sink.flush()  # raises EmitError: the caller must not commit
        if self.claims is not None:
            try:
                await self.claims.set(key, "1", ex=CLAIM_TTL_S)
            except Exception:  # noqa: BLE001 - a replay re-bills at worst; core-api dedupes
                logger.warning("room signals claim not stored for %s", ended.meeting_id)
        return 1


async def run_room_signals_loop(
    consumer: Any, codec: Any, loop: RoomSignalsLoop, stop: asyncio.Event
) -> None:  # the confluent-kafka harness, like the minutes loop's
    from confluent_kafka import TopicPartition

    while not stop.is_set():
        msg = await asyncio.to_thread(consumer.poll, 1.0)
        if msg is None or msg.error():
            continue
        topic, partition, offset, raw = msg.topic(), msg.partition(), msg.offset(), msg.value()
        if topic is None or partition is None or offset is None:
            continue
        if raw is not None:
            try:
                value = codec.deserialize_for(topic, raw)
            except Exception:  # noqa: BLE001
                logger.exception("undecodable message on %s@%s; skipped", topic, offset)
            else:
                await loop.handle(value)  # EmitError propagates: no commit
        await asyncio.to_thread(
            consumer.commit,
            offsets=[TopicPartition(topic, partition, offset + 1)],
            asynchronous=False,
        )


def consumer_config(brokers: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": GROUP,
        "enable.auto.commit": False,
        # A new group starts at the end: no historic meeting is analysed (or billed).
        "auto.offset.reset": "latest",
        "max.poll.interval.ms": 900_000,
    }
