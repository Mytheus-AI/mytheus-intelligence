"""Live caption translation (polish program, wave 3): a viewer in the room reads the captions in
the language they chose. Run beside the window loop (`consumer.run`, `CAPTION_TRANSLATION`,
default on, "0" disables) or alone: `uv run python -m services.agent_runtime_service.translation`.

Input: `transcript-chunks` finals, consumer group `agent-runtime-translation`, offsets from
`latest` with auto-commit (translations are throwaway: a restart never translates history).

Languages: realtime-gateway keeps the Redis hash `nexus:{org_id}:caption-langs:{meeting_id}`
(field = socket id, value = a lowercase ISO 639-1 code). The languages wanted for a meeting are
its distinct values within `LANGUAGES`; anything else is ignored.

Per final, cheapest first, a chunk is skipped (it stays in its original language in the room,
nothing is emitted) when:
1. its header `occurred_at` is more than 30 s old when its turn comes;
2. nobody in the meeting asked for a language;
3. the meeting is off the record (`libs/shared/off_record.py`; an unreadable flag counts as off);
4. it is never-record: its 30-second window is marked (`index_guard.mark_key`, set by the
   indexers' `IndexGuard`) or the org's never-record topics (the company brain,
   `NeverRecordPolicy`) are named in it. An unreadable mark counts as marked.
Otherwise ONE triage-tier (Haiku) call, `caption_translate@v1`, budget-gated and metered by
`LiteLLMRouter` like every model call (user_id None: the org pays), translates the line into
every wanted language at once and answers `{"es": "...", "fr": "..."}`.

Output: one `agent-stream` record per language (`Emitter.caption_translation`: kind
`caption_translation`, the nil user, `window_id` `caption:{chunk_id}`, seq 0, done, status
`translated` with the text). A language missing from the answer gets nothing. `BudgetBlocked`
gives `budget_denied`, anything else that went wrong (an error, the company brain unreadable,
an unparseable answer) gives `unavailable`, both with an empty delta and at most once per
meeting and language a minute (`SET NX EX` on
`nexus:{org_id}:caption-translate:notice:{meeting_id}:{language}`).

Concurrency: at most `MAX_IN_FLIGHT` chunks are worked on at once, at most `MAX_PENDING` wait;
beyond that a chunk is dropped (it would be stale before its turn). The poll loop never waits
on a model call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from libs.shared.index_guard import WINDOW_MS, mark_key
from libs.shared.metrics import record_caption_translation, record_off_record_skip
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.off_record import is_off_record
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.tenancy import parse_org_id

logger = logging.getLogger(__name__)

GROUP = "agent-runtime-translation"
TOPIC = "transcript-chunks"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "libs" / "shared" / "prompt_templates"
PROMPT_KEY = "caption_translate"
LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
}
STALE_S = 30.0
NOTICE_TTL_S = 60
MAX_IN_FLIGHT = 8
MAX_PENDING = 64
MAX_CHARS = 1000  # a live final is a sentence or two; longer is not translated
MIN_MAX_TOKENS, MAX_MAX_TOKENS = 256, 8192

OrgContextFn = Callable[[UUID], Awaitable[CompanyContext]]


def langs_key(org_id: UUID, meeting_id: UUID) -> str:
    return f"nexus:{org_id}:caption-langs:{meeting_id}"


def notice_key(org_id: UUID, meeting_id: UUID, language: str) -> str:
    return f"nexus:{org_id}:caption-translate:notice:{meeting_id}:{language}"


def max_tokens_for(text: str, languages: int) -> int:
    """Room for every language's line (about one token per 1.3 source characters, generous for
    scripts like Hindi or Japanese) plus the JSON around it, bounded."""
    per_language = len(text) * 3 // 4 + 32
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, 64 + languages * per_language))


class CaptionSink(Protocol):
    async def caption_translation(
        self,
        *,
        org_id: UUID,
        meeting_id: UUID,
        chunk_id: str,
        language: str,
        status: str,
        text: str,
    ) -> None: ...


@dataclass(frozen=True)
class Caption:
    org_id: UUID
    meeting_id: UUID
    chunk_id: str
    start_ms: int
    text: str
    occurred_at: float | None  # epoch seconds, from the header


def _epoch(raw: object) -> float | None:
    """The header's `occurred_at`: a datetime from the deserializer, or timestamp-micros."""
    if isinstance(raw, datetime):
        return (raw if raw.tzinfo else raw.replace(tzinfo=UTC)).timestamp()
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw / 1_000_000
    return None


def parse_caption(value: dict[str, Any]) -> Caption | None:
    header = value.get("header") or {}
    org_id = parse_org_id(header.get("org_id"))
    try:
        meeting_id = UUID(str(value.get("meeting_id")))
        chunk_id = str(UUID(str(value.get("chunk_id"))))
        start_ms = int(value.get("start_ms", 0))
    except (TypeError, ValueError):
        return None
    if org_id is None or not value.get("is_final", True):
        return None
    return Caption(
        org_id=org_id,
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        start_ms=start_ms,
        text=str(value.get("text") or ""),
        occurred_at=_epoch(header.get("occurred_at")),
    )


async def requested_languages(redis: Any, org_id: UUID, meeting_id: UUID) -> list[str]:
    """The distinct known languages viewers of the meeting chose; none when unreadable."""
    try:
        raw = await redis.hvals(langs_key(org_id, meeting_id))
    except Exception:  # noqa: BLE001 - nobody is served a translation this time
        logger.warning("caption languages unreadable meeting=%s", meeting_id, exc_info=True)
        return []
    found = set()
    for v in raw or []:
        code = (v.decode() if isinstance(v, bytes | bytearray) else str(v)).strip().lower()
        if code in LANGUAGES:
            found.add(code)
    return sorted(found)


def parse_translations(text: str, languages: list[str]) -> dict[str, str] | None:
    """`{"es": "...", ...}` -> the wanted languages that hold a non-empty string; None when the
    answer is not a JSON object."""
    body = text.strip().strip("`").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for lang in languages:
        value = data.get(lang)
        if isinstance(value, str) and value.strip():
            out[lang] = value.strip()
    return out


class Translator:
    def __init__(
        self,
        *,
        router: Any,
        redis: Any,
        sink: CaptionSink,
        org_context: OrgContextFn,
        loader: PromptLoader | None = None,
        clock: Callable[[], float] = time.time,
        max_in_flight: int = MAX_IN_FLIGHT,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self._router = router
        self._redis = redis
        self._sink = sink
        self._org = org_context
        self._loader = loader or PromptLoader(PROMPTS_DIR)
        self._clock = clock
        self._sem = asyncio.Semaphore(max_in_flight)
        self._max_pending = max_pending
        self._tasks: set[asyncio.Task[str]] = set()

    def submit(self, value: dict[str, Any]) -> bool:
        """Queue one decoded `transcript-chunks` value without waiting; False when dropped."""
        if len(self._tasks) >= self._max_pending:
            record_caption_translation("dropped")
            return False
        task = asyncio.ensure_future(self._guarded(value))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _guarded(self, value: dict[str, Any]) -> str:
        try:
            return await self.handle(value)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one caption must never stop the loop
            logger.exception("caption translation failed")
            return "error"

    async def handle(self, value: dict[str, Any]) -> str:
        """Translate one final (or skip it). Returns the outcome, also counted in metrics."""
        caption = parse_caption(value)
        if caption is None or not caption.text.strip():
            return "ignored"
        async with self._sem:
            outcome = await self._handle(caption)
        record_caption_translation(outcome)
        return outcome

    def _stale(self, caption: Caption) -> bool:
        return caption.occurred_at is None or self._clock() - caption.occurred_at > STALE_S

    async def _handle(self, c: Caption) -> str:
        if self._stale(c):
            return "stale"
        languages = await requested_languages(self._redis, c.org_id, c.meeting_id)
        if not languages:
            return "no_languages"
        if await is_off_record(self._redis, c.org_id, c.meeting_id):
            record_off_record_skip("caption")
            return "off_record"
        if await self._window_marked(c):
            return "never_record"
        try:
            policy = NeverRecordPolicy.from_org(await self._org(c.org_id))
        except Exception:  # noqa: BLE001 - not vetted, so not sent to a model
            logger.warning("company brain unreadable org=%s; caption not translated", c.org_id)
            await self._notices(c, languages, "unavailable")
            return "unavailable"
        if policy.hit(c.text) is not None:
            return "never_record"
        if len(c.text) > MAX_CHARS:
            return "too_long"
        return await self._translate(c, languages)

    async def _window_marked(self, c: Caption) -> bool:
        key = mark_key(c.org_id, c.meeting_id, max(0, c.start_ms) // WINDOW_MS)
        try:
            return bool(await self._redis.exists(key))
        except Exception:  # noqa: BLE001 - unknown: treated as never-record
            logger.warning("never-record mark unreadable meeting=%s", c.meeting_id)
            return True

    async def _translate(self, c: Caption, languages: list[str]) -> str:
        from libs.shared.router import BudgetBlocked

        prompt = self._loader.load(PROMPT_KEY, self._loader.resolve(PROMPT_KEY, c.org_id))
        listed = ", ".join(f"{code} ({LANGUAGES[code]})" for code in languages)
        try:
            out = await self._router.complete(
                tier="triage",
                system=prompt.system,
                **prompt.user_parts(languages=listed, text=c.text),
                org_id=c.org_id,
                user_id=None,
                max_tokens=max_tokens_for(c.text, len(languages)),
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            await self._notices(c, languages, "budget_denied")
            return "budget_denied"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the caption stays in its original language
            logger.warning("caption translation call failed org=%s", c.org_id, exc_info=True)
            await self._notices(c, languages, "unavailable")
            return "unavailable"
        found = parse_translations(out.text, languages)
        if found is None:
            logger.warning("%s answer unparseable org=%s", prompt.ref, c.org_id)
            await self._notices(c, languages, "unavailable")
            return "unavailable"
        for lang, translated in found.items():
            await self._emit(c, lang, "translated", translated)
        return "translated" if found else "missing"

    async def _notices(self, c: Caption, languages: list[str], status: str) -> None:
        """`status` with an empty delta, at most once per meeting and language a minute."""
        for lang in languages:
            try:
                first = await self._redis.set(
                    notice_key(c.org_id, c.meeting_id, lang), status, nx=True, ex=NOTICE_TTL_S
                )
            except Exception:  # noqa: BLE001 - no notice this time
                logger.warning("caption notice throttle unreadable meeting=%s", c.meeting_id)
                continue
            if first:
                await self._emit(c, lang, status, "")

    async def _emit(self, c: Caption, language: str, status: str, text: str) -> None:
        try:
            await self._sink.caption_translation(
                org_id=c.org_id,
                meeting_id=c.meeting_id,
                chunk_id=c.chunk_id,
                language=language,
                status=status,
                text=text,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - throwaway: losing one beats stopping the loop
            logger.exception("could not emit caption translation chunk=%s", c.chunk_id)


def enabled() -> bool:
    return os.environ.get("CAPTION_TRANSLATION", "1").strip() != "0"


def consumer_config(brokers: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": brokers,
        "group.id": GROUP,
        "enable.auto.commit": True,
        "auto.offset.reset": "latest",
    }


async def run_translation_loop(
    consumer: Any, codec: Any, translator: Translator, stop: asyncio.Event
) -> None:  # pragma: no cover - the confluent-kafka harness
    try:
        while not stop.is_set():
            msg = await asyncio.to_thread(consumer.poll, 1.0)
            if msg is None or msg.error() or msg.value() is None:
                continue
            try:
                value = codec.deserialize_for(TOPIC, msg.value())
            except Exception:  # noqa: BLE001
                logger.exception("undecodable transcript-chunks message; not translated")
                continue
            translator.submit(value)
    finally:
        await translator.drain()


def start_translation(
    *,
    brokers: str,
    codec: Any,
    redis: Any,
    router: Any,
    org_context: OrgContextFn,
    stop: asyncio.Event,
) -> tuple[asyncio.Task[None], Any]:  # pragma: no cover - the confluent-kafka harness
    """The loop as a background task beside the window loop: (task, consumer to close). Its
    records go through their own producer that logs delivery errors, so a lost translation
    never fails a window commit."""
    from confluent_kafka import Consumer, Producer

    from services.agent_runtime_service.emitter import Emitter

    consumer = Consumer(consumer_config(brokers))
    consumer.subscribe([TOPIC])
    sink = Emitter(
        Producer({"bootstrap.servers": brokers, "acks": "all", "linger.ms": 0}),
        codec,
        on_error=lambda err: logger.warning("caption translation delivery failed: %s", err),
    )
    translator = Translator(router=router, redis=redis, sink=sink, org_context=org_context)
    return asyncio.create_task(run_translation_loop(consumer, codec, translator, stop)), consumer


async def run() -> None:  # pragma: no cover - the confluent-kafka harness
    from confluent_kafka import Producer
    from redis.asyncio import from_url

    from libs.shared.avro_codec import AvroCodec
    from libs.shared.budget import BillingRefresher, BudgetGate
    from libs.shared.llm_cache import ResponseCache
    from libs.shared.metrics import serve_metrics, set_service
    from libs.shared.org_context import OrgContextClient
    from libs.shared.pii import EgressRedactor
    from libs.shared.router import LiteLLMRouter
    from services.agent_runtime_service.emitter import Emitter

    logging.basicConfig(level=logging.INFO)
    set_service(GROUP)
    serve_metrics()
    brokers = os.environ.get("REDPANDA_BROKERS", "localhost:9092")
    codec = AvroCodec(os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8085"))
    redis = from_url(os.environ.get("REDIS_URL", "redis://localhost:6390"))
    usage = Emitter(
        Producer({"bootstrap.servers": brokers, "acks": "all", "enable.idempotence": True}),
        codec,
        on_error=lambda err: logger.error("usage-events delivery failed: %s", err),
    )
    router = LiteLLMRouter(
        BudgetGate(redis, BillingRefresher()),
        usage,
        redactor=EgressRedactor(redis),
        cache=ResponseCache(redis),
    )
    org = OrgContextClient()
    stop = asyncio.Event()
    task, consumer = start_translation(
        brokers=brokers,
        codec=codec,
        redis=redis,
        router=router,
        org_context=org.get_strict,
        stop=stop,
    )
    try:
        await task
    finally:
        stop.set()
        consumer.close()
        await org.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run())
