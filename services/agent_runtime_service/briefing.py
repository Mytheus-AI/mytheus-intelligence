"""A briefing from your agent ("Beyond the meeting", Layer 1): the person's own agent writes
about 90 seconds of speech for them, after a meeting or when they ask.

Hand-off (core-api owns `agent_briefings`; this tier writes no Postgres):
1. core-api commits a `pending` briefing and PUBLISHes its id on `nexus:{org_id}:briefings:ring`.
   `BriefingLoop` listens on `nexus:*:briefings:ring` (the tenant comes from the channel name,
   never from the message).
2. `POST {CORE_URL}/internal/briefings/{id}/claim` (the agent-runtime caller token, the tenant
   in `X-Mytheus-Org`) answers the source: only that person's records and their own minutes, no
   transcript. 409: someone else took it, or it is finished; nothing to do.
3. `BriefingWriter.write`:
   - Never record. The company brain is read strictly; every source item that names a
     never-record topic is dropped before the model sees it, the prompt carries the "Never
     record" line, and every segment the model writes that names one is dropped after.
   - One model call, prompt `briefing@vN` from the registry, extract tier (Sonnet 5.5) through
     the LiteLLM router: budget gated, metered, PII redacted. The person's agent persona (name,
     personality sentence, their own wording instructions in a delimited block) is the voice;
     the item refs (C1, W1, P1, Q1) become "open the moment" links here, never in the model.
   - Language. The script is written in English (the evaluated prompt), then each segment is
     translated with the live-caption translator's prompt (`caption_translate@vN`, triage tier,
     `translation.py`'s languages): the same infrastructure the room uses. A failed translation
     falls back to the English briefing, said in the result's `language`.
   - Budget blocked: `unavailable` / `budget`. A model failure or an unparseable answer:
     `unavailable` / `model_error`.
4. `POST {CORE_URL}/internal/briefings/{id}/result` with the segments; core-api validates them
   again (links to anything the person may not open are dropped), stores, voices (when its TTS
   is connected) and notifies.

Off by default only when `CORE_URL` / `CORE_CALLER_TOKEN` are missing, or `BRIEFING_AGENT=0`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import httpx

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import (
    EXTRACT_MAX_TOKENS,
    SYNTHESIS_EFFORT,
    BudgetBlocked,
    LiteLLMRouter,
)
from libs.shared.tenancy import parse_org_id
from services.agent_runtime_service.agents.company import render_company, render_ground_rules
from services.agent_runtime_service.agents.personal_listener_agent import PERSONALITY_VOICE
from services.agent_runtime_service.translation import LANGUAGES as CAPTION_LANGUAGES
from services.agent_runtime_service.translation import PROMPT_KEY as TRANSLATE_KEY
from services.agent_runtime_service.translation import PROMPTS_DIR as TRANSLATE_PROMPTS
from services.agent_runtime_service.translation import max_tokens_for, parse_translations

logger = logging.getLogger(__name__)

PROMPT_KEY = "briefing"
SERVICE_NAME = "agent-runtime-service"
RING_PATTERN = "nexus:*:briefings:ring"
KINDS = ("opening", "changed", "waiting", "promise", "question", "closing")
MAX_SEGMENTS = 12
MAX_SEGMENT_CHARS = 420
MAX_INSTRUCTIONS = 500
INSTRUCTIONS_TAG = "wording_instructions"
_TAG = re.compile(rf"</?\s*{INSTRUCTIONS_TAG}\s*>", re.IGNORECASE)
_REF = re.compile(r"^\s*([CWPQ])(\d+)\s*$", re.IGNORECASE)
_LABEL = re.compile(r"\b[CWPQ]\d+\b")
_CHANNEL = re.compile(r"^nexus:([0-9a-fA-F-]{36}):briefings:ring$")


# --- the source -------------------------------------------------------------------------------


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _str(raw: object, limit: int = 300) -> str:
    value = " ".join(raw.split()) if isinstance(raw, str) else ""
    return value[:limit]


@dataclass(frozen=True)
class Item:
    text: str
    id: UUID | None = None
    meeting_id: UUID | None = None
    meeting_title: str | None = None
    status: str | None = None
    who: str | None = None
    due: str | None = None
    at_ms: int | None = None
    carried_to: str | None = None


@dataclass
class Source:
    briefing_id: UUID
    user_id: UUID
    length: str
    target_words: int
    language: str
    first_name: str
    agent_name: str
    personality: str
    instructions: str | None
    meeting_id: UUID | None = None
    meeting_title: str | None = None
    summary: str = ""
    minutes_decisions: list[str] = field(default_factory=list)
    minutes_open: list[str] = field(default_factory=list)
    changed: list[Item] = field(default_factory=list)
    waiting: list[Item] = field(default_factory=list)
    promises: list[Item] = field(default_factory=list)
    questions: list[Item] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


def _items(raw: object) -> list[Item]:
    out: list[Item] = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        text = _str(r.get("text"))
        if not text:
            continue
        at = r.get("at_ms")
        out.append(
            Item(
                text=text,
                id=_uuid(r.get("id")),
                meeting_id=_uuid(r.get("meeting_id")),
                meeting_title=_str(r.get("meeting_title"), 200) or None,
                status=_str(r.get("status"), 120) or None,
                who=_str(r.get("who"), 120) or None,
                due=_str(r.get("due"), 60) or None,
                at_ms=at if isinstance(at, int) and not isinstance(at, bool) else None,
                carried_to=_str(r.get("carried_to"), 160) or None,
            )
        )
    return out[:8]


def parse_source(body: object) -> Source | None:
    """Tolerant: a malformed entry is dropped; no briefing or user id is None."""
    if not isinstance(body, dict):
        return None
    briefing_id, user_id = _uuid(body.get("briefing_id")), _uuid(body.get("user_id"))
    if briefing_id is None or user_id is None:
        return None
    person = body.get("person") if isinstance(body.get("person"), dict) else {}
    agent = body.get("agent") if isinstance(body.get("agent"), dict) else {}
    meeting = body.get("meeting") if isinstance(body.get("meeting"), dict) else None
    minutes = body.get("minutes") if isinstance(body.get("minutes"), dict) else None
    assert isinstance(person, dict) and isinstance(agent, dict)
    first = _str(person.get("first_name"), 60) or "there"
    target = body.get("target_words")
    return Source(
        briefing_id=briefing_id,
        user_id=user_id,
        length="short" if body.get("length") == "short" else "standard",
        target_words=target if isinstance(target, int) and 40 <= target <= 400 else 220,
        language=str(body.get("language") or "en"),
        first_name=first,
        agent_name=_str(agent.get("name"), 40) or f"{first}'s Myte",
        personality=str(agent.get("personality") or "balanced"),
        instructions=_str(agent.get("instructions"), MAX_INSTRUCTIONS) or None,
        meeting_id=_uuid(meeting.get("id")) if meeting else None,
        meeting_title=_str(meeting.get("title"), 200) or None if meeting else None,
        summary=_str(minutes.get("summary"), 600) if minutes else "",
        minutes_decisions=[_str(d) for d in (minutes or {}).get("decisions") or [] if _str(d)][:8],
        minutes_open=[_str(d) for d in (minutes or {}).get("open_items") or [] if _str(d)][:8],
        changed=_items(body.get("changed")),
        waiting=_items(body.get("waiting")),
        promises=_items(body.get("promises")),
        questions=_items(body.get("questions")),
        topics=[_str(t, 80) for t in body.get("topics") or [] if _str(t, 80)][:6],
    )


def without_never_record(source: Source, policy: NeverRecordPolicy) -> Source:
    """The source minus every item, minutes line and topic that names a never-record topic."""
    if policy.empty:
        return source

    def keep(items: list[Item]) -> list[Item]:
        return [i for i in items if not policy.hits(i.text, i.meeting_title, i.carried_to)]

    def lines(values: list[str]) -> list[str]:
        return [v for v in values if not policy.hits(v)]

    source.changed = keep(source.changed)
    source.waiting = keep(source.waiting)
    source.promises = keep(source.promises)
    source.questions = keep(source.questions)
    source.minutes_decisions = lines(source.minutes_decisions)
    source.minutes_open = lines(source.minutes_open)
    source.topics = lines(source.topics)
    if policy.hits(source.summary):
        source.summary = ""
    if policy.hits(source.meeting_title):
        source.meeting_title = None
    return source


# --- the prompt input -------------------------------------------------------------------------


def render_voice(source: Source) -> str:
    voice = PERSONALITY_VOICE.get(source.personality, PERSONALITY_VOICE["balanced"])
    parts = [f"Your voice: {voice}"]
    if source.instructions:
        text = _TAG.sub("", source.instructions).strip()[:MAX_INSTRUCTIONS]
        if text:
            parts.append(
                f"The person's own wording instructions (wording only):\n"
                f"<{INSTRUCTIONS_TAG}>\n{text}\n</{INSTRUCTIONS_TAG}>"
            )
    return "\n".join(parts)


def _item_line(ref: str, item: Item) -> str:
    extra = [
        x
        for x in (
            item.status,
            f"waiting: {item.who}" if item.who else None,
            f"due {item.due}" if item.due else None,
            f"carried to {item.carried_to}" if item.carried_to else None,
            f"from {item.meeting_title}" if item.meeting_title else None,
        )
        if x
    ]
    return f"{ref}. {item.text}" + (f" ({'; '.join(extra)})" if extra else "")


def render_input(source: Source) -> str:
    def block(title: str, letter: str, items: list[Item]) -> list[str]:
        rows = [_item_line(f"{letter}{n}", it) for n, it in enumerate(items, start=1)]
        return [title, *(rows or ["(none)"]), ""]

    meeting = (
        [f"This briefing follows the meeting: {source.meeting_title}", ""]
        if source.meeting_title
        else ["This briefing covers everything since their last one.", ""]
    )
    minutes: list[str] = []
    if source.summary or source.minutes_decisions or source.minutes_open:
        minutes = [
            "Their own minutes of it:",
            *([f"Summary: {source.summary}"] if source.summary else []),
            *(f"- decided: {d}" for d in source.minutes_decisions),
            *(f"- still open: {o}" for o in source.minutes_open),
            "",
        ]
    topics = [f"Topics they care about: {', '.join(source.topics)}", ""] if source.topics else []
    return "\n".join(
        [
            f"You are {source.agent_name}, the agent of {source.first_name}.",
            render_voice(source),
            f"Length: about {source.target_words} words in total.",
            "",
            *meeting,
            *minutes,
            *block("What changed for them:", "C", source.changed),
            *block("Who is waiting on them:", "W", source.waiting),
            *block("Their open promises:", "P", source.promises),
            *block("Questions left open:", "Q", source.questions),
            *topics,
        ]
    )


# --- the answer -------------------------------------------------------------------------------


def _json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _link(source: Source, ref: object) -> dict[str, Any] | None:
    m = _REF.match(ref) if isinstance(ref, str) else None
    if m is None:
        return None
    letter, n = m.group(1).upper(), int(m.group(2))
    pool = {
        "C": source.changed,
        "W": source.waiting,
        "P": source.promises,
        "Q": source.questions,
    }[letter]
    if not 1 <= n <= len(pool):
        return None
    item = pool[n - 1]
    if item.meeting_id is None:
        return None
    if letter == "P":
        kind = "promise"
    elif letter == "Q":
        kind = "question"
    elif letter == "W":
        kind = "action"
    else:
        # A change is a decision unless it is a task that became theirs.
        kind = "action" if item.status == "now yours" else "decision"
    return {
        "type": kind,
        "meeting_id": str(item.meeting_id),
        "id": str(item.id) if item.id and kind != "question" else None,
        "at_ms": item.at_ms,
    }


def parse_segments(
    text: str, source: Source, policy: NeverRecordPolicy
) -> list[dict[str, Any]] | None:
    """The model's answer as result segments, or None when it is not the JSON asked for. Each
    text is clipped, any stray "C1"-style label removed, and a segment naming a never-record
    topic dropped."""
    raw = _json_object(text)
    if raw is None or not isinstance(raw.get("segments"), list):
        return None
    out: list[dict[str, Any]] = []
    for s in raw["segments"]:
        if not isinstance(s, dict):
            continue
        body = _LABEL.sub("", _str(s.get("text"), MAX_SEGMENT_CHARS * 2))
        body = " ".join(body.replace(" ,", ",").replace(" .", ".").split())[:MAX_SEGMENT_CHARS]
        if not body or policy.hit(body):
            continue
        kind = s.get("kind") if s.get("kind") in KINDS else "changed"
        out.append({"kind": kind, "text": body, "link": _link(source, s.get("ref"))})
        if len(out) == MAX_SEGMENTS:
            break
    return out


# --- core-api ---------------------------------------------------------------------------------


class Unavailable(Exception):
    """core-api could not answer now: the briefing stays pending and core-api re-rings it."""


class BriefingApi(Protocol):
    async def claim(self, org_id: UUID, briefing_id: UUID) -> dict[str, Any] | None: ...

    async def result(self, org_id: UUID, briefing_id: UUID, body: dict[str, Any]) -> None: ...


class HttpBriefingApi:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self._transport = transport

    def _headers(self, org_id: UUID) -> dict[str, str]:
        return {
            "X-Mytheus-Service": SERVICE_NAME,
            "X-Mytheus-Service-Token": self._token,
            "X-Mytheus-Org": str(org_id),
        }

    async def _post(self, org_id: UUID, path: str, body: dict[str, Any] | None) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                return await c.post(f"{self._url}{path}", json=body, headers=self._headers(org_id))
        except httpx.HTTPError as exc:
            raise Unavailable(f"{path}: {type(exc).__name__}") from None

    async def claim(self, org_id: UUID, briefing_id: UUID) -> dict[str, Any] | None:
        res = await self._post(org_id, f"/internal/briefings/{briefing_id}/claim", None)
        if res.status_code in (404, 409):
            return None
        if res.status_code != 200:
            raise Unavailable(f"claim answered {res.status_code}")
        try:
            body = res.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    async def result(self, org_id: UUID, briefing_id: UUID, body: dict[str, Any]) -> None:
        res = await self._post(org_id, f"/internal/briefings/{briefing_id}/result", body)
        if res.status_code >= 500:
            raise Unavailable(f"result answered {res.status_code}")
        if res.status_code != 200:
            logger.warning("briefing %s result refused: %s", briefing_id, res.status_code)


def api_from_env() -> HttpBriefingApi | None:
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token:
        return None
    return HttpBriefingApi(url, token)


# --- the writer -------------------------------------------------------------------------------

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


class BriefingWriter:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        org_context: OrgContextFn | None = None,
        translate_loader: PromptLoader | None = None,
    ) -> None:
        self.loader = loader
        self.router = router
        self.org_context = org_context
        self.translate_loader = translate_loader or PromptLoader(TRANSLATE_PROMPTS)

    async def write(self, org_id: UUID, body: dict[str, Any]) -> dict[str, Any]:
        """The result body for core-api. Raises `OrgContextUnavailable` (via the strict brain
        read) when the run should be retried rather than answered."""
        prompt = self.loader.load(PROMPT_KEY, self.loader.resolve(PROMPT_KEY, org_id))
        source = parse_source(body)
        if source is None:
            return {"status": "unavailable", "reason": "model_error", "prompt_version": prompt.ref}
        org = await self.org_context(org_id) if self.org_context is not None else None
        policy = NeverRecordPolicy.from_org(org)
        source = without_never_record(source, policy)
        values = {
            "company": render_company(org) if org is not None else "",
            "ground_rules": render_ground_rules(org) if org is not None else "",
            "briefing_input": render_input(source),
        }
        try:
            out = await self.router.complete(
                tier="extract",
                system=prompt.system,
                org_id=org_id,
                user_id=source.user_id,
                max_tokens=EXTRACT_MAX_TOKENS,
                prompt_ref=prompt.ref,
                effort=SYNTHESIS_EFFORT,
                **prompt.user_parts(**values),
            )
        except BudgetBlocked:
            return {"status": "unavailable", "reason": "budget", "prompt_version": prompt.ref}
        except Exception:  # noqa: BLE001 - said as unavailable; the person can ask again
            logger.exception("briefing call failed for %s", source.briefing_id)
            return {"status": "unavailable", "reason": "model_error", "prompt_version": prompt.ref}
        segments = parse_segments(out.text, source, policy)
        if segments is None:
            logger.warning("briefing answer unparseable for %s", source.briefing_id)
            return {"status": "unavailable", "reason": "model_error", "prompt_version": prompt.ref}
        language = "en"
        if source.language != "en" and source.language in CAPTION_LANGUAGES and segments:
            translated = await self.translate(org_id, source.user_id, segments, source.language)
            if translated is not None:
                segments, language = translated, source.language
        return {
            "status": "ready",
            "prompt_version": prompt.ref,
            "language": language,
            "segments": segments,
        }

    async def translate(
        self, org_id: UUID, user_id: UUID, segments: list[dict[str, Any]], language: str
    ) -> list[dict[str, Any]] | None:
        """Each segment through the live-caption translator (one triage call per segment, in
        parallel). None when any fails: the briefing then stays in English rather than mixed."""
        prompt = self.translate_loader.load(
            TRANSLATE_KEY, self.translate_loader.resolve(TRANSLATE_KEY, org_id)
        )
        wanted = f"{language} ({CAPTION_LANGUAGES[language]})"

        async def one(text: str) -> str | None:
            try:
                out = await self.router.complete(
                    tier="triage",
                    system=prompt.system,
                    org_id=org_id,
                    user_id=user_id,
                    max_tokens=max_tokens_for(text, 1),
                    prompt_ref=prompt.ref,
                    **prompt.user_parts(languages=wanted, text=text),
                )
            except Exception:  # noqa: BLE001 - the whole briefing falls back to English
                return None
            got = parse_translations(out.text, [language])
            return got.get(language) if got else None

        results = await asyncio.gather(*(one(s["text"]) for s in segments))
        if any(r is None for r in results):
            logger.warning("briefing translation to %s incomplete; kept English", language)
            return None
        return [
            {**s, "text": str(r)[:MAX_SEGMENT_CHARS]}
            for s, r in zip(segments, results, strict=True)
        ]


# --- the loop ---------------------------------------------------------------------------------


def parse_ring(channel: object, data: object) -> tuple[UUID, UUID] | None:
    """(org_id, briefing_id) from a doorbell message; the tenant is the channel's, never the
    message's."""
    name = channel.decode() if isinstance(channel, bytes | bytearray) else str(channel or "")
    m = _CHANNEL.match(name)
    org_id = parse_org_id(m.group(1)) if m else None
    raw = data.decode() if isinstance(data, bytes | bytearray) else data
    briefing_id = _uuid(raw)
    if org_id is None or briefing_id is None:
        return None
    return org_id, briefing_id


class BriefingLoop:
    """The testable core: one ring in, one briefing written and posted back. Rings for a
    briefing already in flight here are ignored; core-api's claim decides between instances."""

    def __init__(
        self, writer: BriefingWriter, api: BriefingApi, *, max_concurrency: int = 4
    ) -> None:
        self.writer = writer
        self.api = api
        self._sem = asyncio.Semaphore(max_concurrency)
        self._inflight: set[UUID] = set()
        self._tasks: set[asyncio.Task[bool]] = set()

    def ring(self, org_id: UUID, briefing_id: UUID) -> asyncio.Task[bool] | None:
        if briefing_id in self._inflight:
            return None
        self._inflight.add(briefing_id)
        task = asyncio.create_task(self.handle(org_id, briefing_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def handle(self, org_id: UUID, briefing_id: UUID) -> bool:
        """True when a result was posted."""
        try:
            async with self._sem:
                source = await self.api.claim(org_id, briefing_id)
                if source is None:
                    return False
                body = await self.writer.write(org_id, source)
                await self.api.result(org_id, briefing_id, body)
                return True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - left composing; core-api re-rings it
            logger.warning("briefing %s not written now", briefing_id, exc_info=True)
            return False
        finally:
            self._inflight.discard(briefing_id)


async def run_briefing_loop(redis: Any, loop: BriefingLoop, stop: asyncio.Event) -> None:
    """Listen on every tenant's doorbell (redis.asyncio pub/sub) until `stop`."""
    pubsub = redis.pubsub()
    await pubsub.psubscribe(RING_PATTERN)
    try:
        while not stop.is_set():
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not msg:
                continue
            parsed = parse_ring(msg.get("channel"), msg.get("data"))
            if parsed is None:
                logger.warning("briefing doorbell: unreadable message ignored")
                continue
            loop.ring(*parsed)
    finally:
        await pubsub.punsubscribe(RING_PATTERN)
        await pubsub.close()


def enabled() -> bool:
    return os.environ.get("BRIEFING_AGENT", "1") != "0"
