"""The live watch ("In every meeting at once", Beyond the meeting layer 3).

A person can mark a meeting "my agent listens" (attendance `agent`). While that meeting is live,
their agent watches every window and interrupts them only when something needs them now:

- a decision touching their listen-for topics, their projects or something they own;
- their name, team or project comes up with a question for them;
- a promise is being made on their behalf.

Per closed window (the runtime calls `ListenWatch.observe` only after its off-the-record and
never-record checks, so a window nobody may record never reaches the watch):

1. **Listeners.** core-api's `GET /internal/meetings/{id}/listeners` (cached `ttl_s`, 30 s by
   default; a failure is "no listeners" and is not cached). A confidential meeting has none
   (core-api answers [] and the watch also skips a `confidential: true` body). A listener in the
   runtime's live roster joined after all and is skipped; so is one core-api marks not
   `interruptible` (paused, quiet hours, focus block, throttled) or whose `next_allowed_at` is
   still ahead: no model is called for someone who could not be interrupted anyway.
2. **Pre-filter** (no model). The window must name them (full or first name), one of their
   teams or projects, a listen-for topic, or something they own. Promise cues ("X will",
   "X can take", "on behalf of X", "I'll ask X") are recorded as the hit's reason.
3. **Spend limit** (Redis, `nexus:{org_id}:listen-watch:{meeting_id}:{user_id}`): at most one
   model call per `min_gap_s` (60 s) and `max_calls` (40) per meeting per listener, and none
   while core-api's last answer said `next_allowed_at` is still ahead.
4. **One triage call** (`interrupt_gate@vN`, Haiku) through the router, so the tenant budget
   gates it (`BudgetBlocked` is a silent skip) and PII is redacted. The listener's name is never
   sent: it becomes "[the absent person]" in the window and the meeting title, and the other
   speakers keep the meeting-wide "Participant N" labels. Their teams, projects, listen-for
   topics and owned work are sent (work data, not identity).
5. **Post.** An answer with `interrupt: true`, a known reason and `confidence >=
   min_confidence` (0.7) goes to core-api's `POST /internal/meetings/{id}/interruptions` with
   the idempotency key `{window_id}:{user_id}`. core-api decides whether it is delivered, held or
   suppressed (throttle, quiet hours, focus blocks, the person in the room, attendance changed);
   its `next_allowed_at` pauses this listener's model calls until then. Before posting, every
   text passes the window's `Postprocess` (labels become names; a never-record topic drops the
   text, and the whole interruption when it is the headline).

The agent tier still writes no Postgres: core-api is the one writer of the interruption, as it
is of everything else; the watch only proposes. Nothing here changes the runtime when the watch
is not wired (`watch=None`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

import httpx
from prometheus_client import Counter

from libs.shared.json_answer import json_object
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import TranscriptWindow
from services.agent_runtime_service.names import Postprocess, speaker_labels

logger = logging.getLogger(__name__)

SERVICE_NAME = "agent-runtime-service"
PROMPT_KEY = "interrupt_gate"
ABSENT = "[the absent person]"
REASONS = ("decision", "question", "promise")
DEFAULT_TIMEOUT_S = 2.0
MAX_KEY = 200
MAX_HEADLINE = 160
MAX_DETAIL = 400
MAX_TOPIC = 120
MAX_QUOTE = 300
MAX_LIST = 20
MAX_ITEM = 200
WATCH_TTL_S = 6 * 3600  # the spend-limit key outlives any meeting

WATCH_TOTAL = Counter(
    "mytheus_listen_watch_total",
    "Live watch outcomes per listener and window: prefilter_miss, limited, budget_blocked, "
    "model_error, no_interrupt, low_confidence, never_record, posted, post_failed.",
    ["outcome"],
)


def _count(outcome: str) -> None:
    try:
        WATCH_TOTAL.labels(outcome).inc()
    except Exception:  # noqa: BLE001 - telemetry never breaks the watch
        pass


# --- core-api: listeners and interruptions ---------------------------------------------------


@dataclass(frozen=True)
class Listener:
    user_id: UUID
    name: str = ""
    first_name: str = ""
    teams: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    listen_for: tuple[str, ...] = ()
    owns: tuple[str, ...] = ()
    interruptible: bool = True
    quiet_reason: str | None = None
    next_allowed_at: datetime | None = None


@dataclass(frozen=True)
class Listeners:
    meeting_id: UUID
    title: str = ""
    confidential: bool = True  # absent: treated as confidential (fail closed)
    status: str = ""
    listeners: tuple[Listener, ...] = ()


@dataclass(frozen=True)
class Interruption:
    user_id: UUID
    key: str
    reason: str
    headline: str
    confidence: float
    detail: str | None = None
    topic: str | None = None
    quote: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    prompt: str | None = None

    def body(self) -> dict[str, Any]:
        return {
            "user_id": str(self.user_id),
            "key": self.key,
            "reason": self.reason,
            "headline": self.headline,
            "detail": self.detail,
            "topic": self.topic,
            "quote": self.quote,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": round(self.confidence, 3),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class PostAnswer:
    id: UUID | None
    status: str
    reason: str | None = None
    next_allowed_at: datetime | None = None


class WatchSource(Protocol):
    async def listeners(self, org_id: UUID, meeting_id: UUID) -> Listeners | None: ...

    async def interrupt(
        self, org_id: UUID, meeting_id: UUID, interruption: Interruption
    ) -> PostAnswer | None: ...


def _uuid(raw: object) -> UUID | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


def _when(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _flat(raw: object, limit: int) -> str:
    if not isinstance(raw, str):
        return ""
    text = " ".join(raw.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _strings(raw: object) -> tuple[str, ...]:
    out: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        text = _flat(value, MAX_ITEM)
        if text and text.lower() not in (o.lower() for o in out):
            out.append(text)
    return tuple(out[:MAX_LIST])


def parse_listener(raw: object) -> Listener | None:
    if not isinstance(raw, dict):
        return None
    user_id = _uuid(raw.get("user_id"))
    if user_id is None:
        return None
    name = _flat(raw.get("name"), 80)
    first = _flat(raw.get("first_name"), 40) or (name.split()[0] if name.split() else "")
    interruptible = raw.get("interruptible")
    quiet = raw.get("quiet_reason")
    return Listener(
        user_id=user_id,
        name=name,
        first_name=first,
        teams=_strings(raw.get("teams")),
        projects=_strings(raw.get("projects")),
        listen_for=_strings(raw.get("listen_for")),
        owns=_strings(raw.get("owns")),
        interruptible=interruptible is True,  # fail closed: absent or odd is "not now"
        quiet_reason=quiet if isinstance(quiet, str) and quiet else None,
        next_allowed_at=_when(raw.get("next_allowed_at")),
    )


def parse_listeners(body: object) -> Listeners | None:
    if not isinstance(body, dict):
        return None
    meeting_id = _uuid(body.get("meeting_id"))
    if meeting_id is None:
        return None
    confidential = body.get("confidential")
    found = [parse_listener(r) for r in body.get("listeners") or [] if isinstance(r, dict)]
    return Listeners(
        meeting_id=meeting_id,
        title=_flat(body.get("title"), 200),
        confidential=confidential is not False,
        status=_flat(body.get("status"), 32),
        listeners=tuple(x for x in found if x is not None),
    )


def parse_answer(body: object) -> PostAnswer | None:
    if not isinstance(body, dict) or not isinstance(body.get("status"), str):
        return None
    reason = body.get("reason")
    return PostAnswer(
        id=_uuid(body.get("id")),
        status=str(body["status"]),
        reason=reason if isinstance(reason, str) and reason else None,
        next_allowed_at=_when(body.get("next_allowed_at")),
    )


class HttpWatchClient:
    """core-api's two internal routes, over the service-caller scheme the agenda route uses."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
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

    async def listeners(self, org_id: UUID, meeting_id: UUID) -> Listeners | None:
        path = f"/internal/meetings/{meeting_id}/listeners"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.get(f"{self._url}{path}", headers=self._headers(org_id))
        except httpx.HTTPError as exc:
            logger.warning("core-api %s unreachable: %s (no watch this window)", path, exc)
            return None
        if res.status_code != 200:
            logger.info("core-api %s answered %s (no watch this window)", path, res.status_code)
            return None
        try:
            return parse_listeners(res.json())
        except ValueError:
            return None

    async def interrupt(
        self, org_id: UUID, meeting_id: UUID, interruption: Interruption
    ) -> PostAnswer | None:
        path = f"/internal/meetings/{meeting_id}/interruptions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                res = await c.post(
                    f"{self._url}{path}", headers=self._headers(org_id), json=interruption.body()
                )
        except httpx.HTTPError as exc:
            logger.warning("core-api %s unreachable: %s (interruption lost)", path, exc)
            return None
        if res.status_code != 200:
            logger.info("core-api %s answered %s", path, res.status_code)
            return None
        try:
            return parse_answer(res.json())
        except ValueError:
            return None


class CachedWatchSource:
    """One listeners read per (org, meeting) per `ttl_s`; a failure (None) is not cached.
    `interrupt` is passed through, and drops the meeting's cached listeners so the next window
    sees core-api's new throttle state."""

    def __init__(
        self,
        source: WatchSource,
        *,
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._ttl = ttl_s
        self._clock = clock
        self._cache: dict[tuple[UUID, UUID], tuple[float, Listeners]] = {}

    async def listeners(self, org_id: UUID, meeting_id: UUID) -> Listeners | None:
        key = (org_id, meeting_id)
        hit = self._cache.get(key)
        now = self._clock()
        if hit is not None and now < hit[0]:
            return hit[1]
        try:
            found = await self._source.listeners(org_id, meeting_id)
        except Exception:  # noqa: BLE001 - any failure is "no listeners this time"
            logger.warning("listeners read failed meeting=%s", meeting_id)
            found = None
        if found is None:
            self._cache.pop(key, None)
            return None
        self._cache[key] = (now + self._ttl, found)
        return found

    async def interrupt(
        self, org_id: UUID, meeting_id: UUID, interruption: Interruption
    ) -> PostAnswer | None:
        self._cache.pop((org_id, meeting_id), None)
        return await self._source.interrupt(org_id, meeting_id, interruption)


# --- the deterministic pre-filter ------------------------------------------------------------

_STOP = frozenset(
    "the a an and or of for to in on at by with from this that our your their is are be "
    "new next q1 q2 q3 q4 team project plan work".split()
)
_PROMISE_AFTER = r"(?:'ll|\s+will|\s+can(?:\s+take|\s+own|\s+do|\s+handle)?|\s+is\s+going\s+to|\s+(?:would|could)\s+take|\s+(?:owns|takes))\b"  # noqa: E501
_PROMISE_BEFORE = (
    r"\b(?:on\s+behalf\s+of|ask|tell|have|get|assign(?:ed)?(?:\s+it)?\s+to|give\s+it\s+to)\s+"
)


def _phrase(text: str) -> re.Pattern[str] | None:
    words = [re.escape(w) for w in re.findall(r"[\w']+", text.lower())]
    if not words:
        return None
    return re.compile(r"(?<![\w])" + r"[\s\-]+".join(words) + r"(?:s|es|'s)?(?![\w])", re.I)


def _owned_words(item: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][\w\-]{3,}", item.lower()) if w not in _STOP]


@dataclass(frozen=True)
class Hit:
    kind: str  # name | team | project | topic | owns
    term: str
    promise: bool = False


def _names(listener: Listener) -> list[str]:
    out: list[str] = []
    for n in (listener.name, listener.first_name):
        n = n.strip()
        if len(n) >= 2 and n.lower() not in (o.lower() for o in out):
            out.append(n)
    return out


def prefilter(text: str, listener: Listener) -> list[Hit]:
    """What in `text` points at this listener; [] means no model call."""
    hits: list[Hit] = []
    anchors = [("name", n) for n in _names(listener)]
    anchors += [("team", t) for t in listener.teams] + [("project", p) for p in listener.projects]
    for kind, term in anchors:
        pattern = _phrase(term)
        if pattern is None or not pattern.search(text):
            continue
        body = pattern.pattern
        promise = bool(
            re.search(body + r"(?:\s+(?:team|folks|squad|side))?" + _PROMISE_AFTER, text, re.I)
            or re.search(_PROMISE_BEFORE + body, text, re.I)
        )
        hits.append(Hit(kind, term, promise))
    for topic in listener.listen_for:
        pattern = _phrase(topic)
        if pattern is not None and pattern.search(text):
            hits.append(Hit("topic", topic))
    lowered = text.lower()
    for item in listener.owns:
        pattern = _phrase(item)
        words = _owned_words(item)
        if (pattern is not None and pattern.search(text)) or (
            len(words) >= 2 and sum(1 for w in set(words) if w in lowered) >= 2
        ):
            hits.append(Hit("owns", item))
    return hits


# --- what the model sees ---------------------------------------------------------------------


def _name_pattern(name: str) -> re.Pattern[str] | None:
    words = [re.escape(w) for w in re.findall(r"[\w']+", name.lower()) if w != "'"]
    if not words:
        return None
    return re.compile(r"(?<![\w])" + r"[\s\-]+".join(words) + r"(?P<pos>'s)?(?![\w])", re.I)


def redact_name(text: str, listener: Listener) -> str:
    """The listener's full name, then first name, become `ABSENT` (a possessive is kept)."""
    for n in sorted(_names(listener), key=len, reverse=True):
        pattern = _name_pattern(n)
        if pattern is not None:
            text = pattern.sub(lambda m: ABSENT + (m.group("pos") or ""), text)
    return text


def render_window(window: TranscriptWindow, listener: Listener) -> str:
    labels = speaker_labels(window)
    return "\n".join(
        f"{labels[t.speaker_id]}: {redact_name(' '.join(t.text.split()), listener)}"
        for t in window.turns
    )


def render_profile(listener: Listener) -> str:
    def line(label: str, values: Iterable[str]) -> str:
        items = [redact_name(v, listener) for v in values]
        return f"{label}: {'; '.join(items) if items else '(none)'}"

    return "\n".join(
        [
            line("Teams", listener.teams),
            line("Projects", listener.projects),
            line("Listens for", listener.listen_for),
            line("Owns", listener.owns),
        ]
    )


@dataclass(frozen=True)
class Verdict:
    interrupt: bool
    reason: str = ""
    confidence: float = 0.0
    headline: str = ""
    detail: str = ""
    topic: str = ""
    quote: str = ""
    parsed: bool = True


def parse_verdict(text: str) -> Verdict:
    """The model's JSON. Unreadable, or `interrupt` true without a known reason and a headline,
    is no interruption (`parsed=False` for the unreadable case)."""
    obj = json_object(text, "interrupt")
    if obj is None or not isinstance(obj.get("interrupt"), bool):
        return Verdict(False, parsed=False)
    if obj["interrupt"] is not True:
        return Verdict(False)
    reason = str(obj.get("reason") or "").strip().lower()
    headline = _flat(obj.get("headline"), MAX_HEADLINE).rstrip(" .")
    raw = obj.get("confidence")
    if reason not in REASONS or not headline or isinstance(raw, bool):
        return Verdict(False, parsed=False)
    try:
        confidence = max(0.0, min(1.0, float(raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return Verdict(False, parsed=False)
    return Verdict(
        True,
        reason=reason,
        confidence=confidence,
        headline=headline,
        detail=_flat(obj.get("detail"), MAX_DETAIL),
        topic=_flat(obj.get("topic"), MAX_TOPIC),
        quote=_flat(obj.get("quote"), MAX_QUOTE),
    )


# --- the spend limit ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchSettings:
    min_confidence: float = 0.7
    min_gap_s: float = 60.0
    max_calls: int = 40
    ttl_s: float = 30.0

    @classmethod
    def from_env(cls) -> WatchSettings:
        def num(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(
            min_confidence=max(0.0, min(1.0, num("LISTEN_WATCH_MIN_CONFIDENCE", 0.7))),
            min_gap_s=max(0.0, num("LISTEN_WATCH_MIN_GAP_S", 60.0)),
            max_calls=max(0, int(num("LISTEN_WATCH_MAX_CALLS", 40))),
            ttl_s=max(1.0, num("LISTEN_WATCH_CACHE_S", 30.0)),
        )


def watch_key(org_id: UUID, meeting_id: UUID, user_id: UUID) -> str:
    return f"nexus:{org_id}:listen-watch:{meeting_id}:{user_id}"


def _f(raw: object) -> float:
    try:
        return float(raw.decode() if isinstance(raw, bytes) else raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError):
        return 0.0


class SpendLimit:
    """Per (meeting, listener): fields `last` (epoch s of the last model call), `calls`, `until`
    (epoch s before which no call is made, from core-api's `next_allowed_at`). One asyncio task
    runs a meeting's windows, so the read-then-write needs no lock."""

    def __init__(self, redis: Any, settings: WatchSettings) -> None:
        self._redis = redis
        self._settings = settings

    async def take(self, org_id: UUID, meeting_id: UUID, user_id: UUID, now: float) -> bool:
        key = watch_key(org_id, meeting_id, user_id)
        raw = await self._redis.hgetall(key)
        state = {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}
        if now < _f(state.get("until")):
            return False
        if int(_f(state.get("calls"))) >= self._settings.max_calls:
            return False
        last = state.get("last")
        if last is not None and now - _f(last) < self._settings.min_gap_s:
            return False
        await self._redis.hset(key, mapping={"last": str(now)})
        await self._redis.hincrby(key, "calls", 1)
        await self._redis.expire(key, WATCH_TTL_S)
        return True

    async def hold(self, org_id: UUID, meeting_id: UUID, user_id: UUID, until: float) -> None:
        key = watch_key(org_id, meeting_id, user_id)
        await self._redis.hset(key, mapping={"until": str(until)})
        await self._redis.expire(key, WATCH_TTL_S)


# --- the watch ---------------------------------------------------------------------------------


@dataclass
class _Run:
    """What one listener's pass did (tests and logs)."""

    user_id: UUID
    outcome: str
    hits: list[Hit] = field(default_factory=list)
    answer: PostAnswer | None = None


class ListenWatch:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        *,
        source: WatchSource,
        redis: Any,
        settings: WatchSettings | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._loader = loader
        self._router = router
        self._source = source
        self.settings = settings or WatchSettings()
        self._limit = SpendLimit(redis, self.settings)
        self._clock = clock
        self.last_runs: list[_Run] = []

    async def observe(
        self,
        window: TranscriptWindow,
        *,
        roster: Iterable[UUID] = (),
        post: Postprocess | None = None,
    ) -> int:
        """Watch one window for every listener; returns the number of interruptions posted.
        Never raises (a failed watch must not fail the window)."""
        try:
            return await self._observe(window, set(roster), post)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("live watch failed on window %s; skipped", window.window_id)
            return 0

    async def _observe(
        self, window: TranscriptWindow, roster: set[UUID], post: Postprocess | None
    ) -> int:
        self.last_runs = []
        if not window.turns:
            return 0
        found = await self._source.listeners(window.org_id, window.meeting_id)
        if found is None or found.confidential or found.status == "ended":
            return 0
        now = self._clock()
        listeners = [
            x
            for x in found.listeners
            if x.user_id not in roster
            and x.interruptible
            and (x.next_allowed_at is None or x.next_allowed_at.timestamp() <= now)
        ]
        if not listeners:
            return 0
        runs = await asyncio.gather(
            *(self._one(window, found.title, x, post, now) for x in listeners)
        )
        self.last_runs = list(runs)
        return sum(1 for r in runs if r.outcome == "posted")

    async def _one(
        self,
        window: TranscriptWindow,
        title: str,
        listener: Listener,
        post: Postprocess | None,
        now: float,
    ) -> _Run:
        text = "\n".join(t.text for t in window.turns)
        hits = prefilter(text, listener)
        if not hits:
            return self._done(_Run(listener.user_id, "prefilter_miss"))
        org, meeting = window.org_id, window.meeting_id
        if not await self._limit.take(org, meeting, listener.user_id, now):
            return self._done(_Run(listener.user_id, "limited", hits))
        version = self._loader.resolve(PROMPT_KEY, org)
        prompt = self._loader.load(PROMPT_KEY, version)
        try:
            out = await self._router.complete(
                tier="triage",
                system=prompt.system,
                **prompt.user_parts(
                    profile=render_profile(listener),
                    meeting=redact_name(title, listener) or "Untitled meeting",
                    window=render_window(window, listener),
                ),
                org_id=org,
                user_id=listener.user_id,
                max_tokens=400,
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            return self._done(_Run(listener.user_id, "budget_blocked", hits))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a provider error skips this listener's window
            logger.exception("interrupt gate failed for window %s", window.window_id)
            return self._done(_Run(listener.user_id, "model_error", hits))
        verdict = parse_verdict(out.text)
        if not verdict.interrupt:
            return self._done(_Run(listener.user_id, "no_interrupt", hits))
        if verdict.confidence < self.settings.min_confidence:
            return self._done(_Run(listener.user_id, "low_confidence", hits))
        interruption = self._interruption(window, listener, verdict, prompt.ref, post)
        if interruption is None:
            return self._done(_Run(listener.user_id, "never_record", hits))
        answer = await self._source.interrupt(org, meeting, interruption)
        if answer is None:
            return self._done(_Run(listener.user_id, "post_failed", hits))
        if answer.next_allowed_at is not None:
            await self._limit.hold(
                org, meeting, listener.user_id, answer.next_allowed_at.timestamp()
            )
        return self._done(_Run(listener.user_id, "posted", hits, answer))

    @staticmethod
    def _done(run: _Run) -> _Run:
        _count(run.outcome)
        return run

    @staticmethod
    def _interruption(
        window: TranscriptWindow,
        listener: Listener,
        verdict: Verdict,
        prompt_ref: str,
        post: Postprocess | None,
    ) -> Interruption | None:
        """The verdict as core-api receives it: the absent-person token back to "you" (headline,
        detail) or their first name (quote, topic), labels to names, never-record applied."""
        me = listener.first_name or listener.name or "you"

        def clean(raw: str, limit: int, who: str) -> str | None:
            if not raw:
                return None

            def back(m: re.Match[str]) -> str:
                if m.group(1):
                    return "your" if who == "you" else f"{who}'s"
                return who

            text = re.sub(re.escape(ABSENT) + r"('s)?", back, raw)
            if post is not None:
                lined = post.line(text)
                if lined is None:
                    return None
                text = lined
            return _flat(text, limit) or None

        headline = clean(verdict.headline, MAX_HEADLINE, "you")
        if headline is None:
            return None
        headline = headline[:1].upper() + headline[1:]
        return Interruption(
            user_id=listener.user_id,
            key=f"{window.window_id}:{listener.user_id}"[:MAX_KEY],
            reason=verdict.reason,
            headline=headline,
            detail=clean(verdict.detail, MAX_DETAIL, "you"),
            topic=clean(verdict.topic, MAX_TOPIC, me),
            quote=clean(verdict.quote, MAX_QUOTE, me),
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            confidence=verdict.confidence,
            prompt=prompt_ref,
        )


def watch_from_env(loader: PromptLoader, router: LiteLLMRouter, redis: Any) -> ListenWatch | None:
    """On when `CORE_URL` and `CORE_CALLER_TOKEN` are set and `LISTEN_WATCH` is not "0"."""
    url = os.environ.get("CORE_URL", "").strip()
    token = os.environ.get("CORE_CALLER_TOKEN", "").strip()
    if not url or not token or os.environ.get("LISTEN_WATCH", "1").strip() == "0":
        return None
    settings = WatchSettings.from_env()
    return ListenWatch(
        loader,
        router,
        source=CachedWatchSource(HttpWatchClient(url, token), ttl_s=settings.ttl_s),
        redis=redis,
        settings=settings,
    )
