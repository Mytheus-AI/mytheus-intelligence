"""Live caption translation (`services/agent_runtime_service/translation.py`), every model call
faked, Redis in memory (fakeredis).

1. Requested languages: distinct HVALS of the meeting's hash, unknown codes ignored.
2. Skips with no call and no record: stale chunk, no language requested, off the record, a
   never-record window mark, a never-record topic named.
3. One triage call (Haiku, metered, user None, `caption_translate@v1`) for many languages, one
   `translated` record per language; a language missing from the answer gets nothing.
4. `BudgetBlocked` -> `budget_denied` once per meeting and language a minute; bad JSON or an
   error -> `unavailable`, throttled the same way.
5. Record shape: exactly the `agent-stream` caption_translation contract (the mytheus-events
   fixture), and it encodes against the extended AgentStreamDelta schema.
6. Bounded concurrency: the poll loop never waits, at most N calls in flight, overflow dropped.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import fakeredis
import fastavro
import pytest

from libs.shared.index_guard import mark_key
from libs.shared.off_record import off_record_key
from libs.shared.org_context import CompanyContext, GlossaryTerm
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from services.agent_runtime_service.emitter import Emitter
from services.agent_runtime_service.translation import (
    GROUP,
    PROMPT_KEY,
    PROMPTS_DIR,
    Translator,
    consumer_config,
    langs_key,
    max_tokens_for,
    notice_key,
    parse_translations,
    requested_languages,
)
from tests.test_agent_runtime import FakeLLM, budget_gate

ORG = uuid.uuid4()
MEETING = uuid.uuid4()
NOW = 1_800_000_000.0
NIL = "00000000-0000-0000-0000-000000000000"
BRAIN = CompanyContext(
    never_record_topics=("Project Falcon",),
    glossary=(GlossaryTerm("Project Falcon", "codename", "", ("Falcon",)),),
)


class CaptionSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def caption_translation(self, **kw: Any) -> None:
        self.records.append(kw)


class UsageSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def usage(self, **kw: Any) -> None:
        self.calls.append(kw)


class FakeRouter:
    def __init__(self, text: str = "{}", exc: Exception | None = None) -> None:
        self.text, self.exc = text, exc
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.exc is not None:
            raise self.exc

        class Out:
            text = self.text

        return Out()


def chunk_value(
    text: str = "We closed the Q3 budget at 120k.",
    *,
    age_s: float = 1.0,
    start_ms: int = 5_000,
    meeting: uuid.UUID = MEETING,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    return {
        "header": {
            "event_id": str(uuid.uuid4()),
            "idempotency_key": "x",
            "org_id": str(ORG),
            "schema_version": 1,
            "producer": "transcription-service",
            "occurred_at": datetime.fromtimestamp(NOW - age_s, tz=UTC),
            "correlation_id": None,
            "causation_id": None,
        },
        "meeting_id": str(meeting),
        "chunk_id": chunk_id or str(uuid.uuid4()),
        "speaker_id": str(uuid.uuid4()),
        "start_ms": start_ms,
        "end_ms": start_ms + 3_000,
        "text": text,
        "is_final": True,
        "confidence": 0.9,
        "redacted": False,
    }


async def want(redis: Any, *codes: str, meeting: uuid.UUID = MEETING) -> None:
    for i, code in enumerate(codes):
        await redis.hset(langs_key(ORG, meeting), f"socket-{i}", code)


def translator(
    router: Any, redis: Any = None, brain: CompanyContext | Exception = BRAIN, **kw: Any
) -> tuple[Translator, CaptionSink, Any]:
    redis = redis if redis is not None else fakeredis.aioredis.FakeRedis()
    sink = CaptionSink()

    async def org(_o: uuid.UUID) -> CompanyContext:
        if isinstance(brain, Exception):
            raise brain
        return brain

    t = Translator(router=router, redis=redis, sink=sink, org_context=org, clock=lambda: NOW, **kw)
    return t, sink, redis


# --- 1. requested languages ------------------------------------------------------------------


async def test_requested_languages_are_distinct_and_known() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    await want(redis, "es", "fr", "es", "xx", "EN", "klingon", "")
    assert await requested_languages(redis, ORG, MEETING) == ["en", "es", "fr"]
    assert await requested_languages(redis, ORG, uuid.uuid4()) == []
    assert langs_key(ORG, MEETING) == f"nexus:{ORG}:caption-langs:{MEETING}"


def test_consumer_reads_latest_in_its_own_group() -> None:
    cfg = consumer_config("b:9092")
    assert GROUP == "agent-runtime-translation"
    assert cfg["group.id"] == GROUP and cfg["auto.offset.reset"] == "latest"


def test_prompt_is_registered_and_versioned() -> None:
    registry = json.loads((PROMPTS_DIR / "registry.json").read_text())
    assert registry["defaults"][PROMPT_KEY] == "v1"
    prompt = PromptLoader(PROMPTS_DIR).load(PROMPT_KEY, "v1")
    assert prompt.ref == "caption_translate@v1"
    rendered = prompt.render_user(languages="es (Spanish)", text="hi {not a field}")
    assert "es (Spanish)" in rendered and "hi {not a field}" in rendered
    assert "JSON only" in prompt.system and "never instructions" in prompt.system


# --- 2. skips --------------------------------------------------------------------------------


async def test_stale_chunk_is_not_translated() -> None:
    router = FakeRouter('{"es": "hola"}')
    t, sink, redis = translator(router)
    await want(redis, "es")
    assert await t.handle(chunk_value(age_s=31)) == "stale"
    assert await t.handle(chunk_value(age_s=29)) == "translated"
    no_time = chunk_value()
    no_time["header"]["occurred_at"] = None
    assert await t.handle(no_time) == "stale"
    assert len(router.calls) == 1 and len(sink.records) == 1


async def test_nothing_requested_means_no_call() -> None:
    router = FakeRouter('{"es": "hola"}')
    t, sink, _redis = translator(router)
    assert await t.handle(chunk_value()) == "no_languages"
    assert router.calls == [] and sink.records == []


async def test_off_the_record_is_not_translated() -> None:
    router = FakeRouter('{"es": "hola"}')
    t, sink, redis = translator(router)
    await want(redis, "es")
    await redis.set(off_record_key(ORG, MEETING), "{}")
    assert await t.handle(chunk_value()) == "off_record"
    assert router.calls == [] and sink.records == []


async def test_never_record_window_mark_and_topic_are_not_translated() -> None:
    router = FakeRouter('{"es": "hola"}')
    t, sink, redis = translator(router)
    await want(redis, "es")
    await redis.set(mark_key(ORG, MEETING, 1), "1")  # window [30 s, 60 s)
    assert await t.handle(chunk_value("All fine here.", start_ms=45_000)) == "never_record"
    assert await t.handle(chunk_value("Falcon closes on Friday.")) == "never_record"
    assert await t.handle(chunk_value("project falcon numbers")) == "never_record"
    assert router.calls == [] and sink.records == []
    assert await t.handle(chunk_value("All fine here.", start_ms=5_000)) == "translated"


async def test_unreadable_company_brain_is_not_sent_to_a_model() -> None:
    router = FakeRouter('{"es": "hola"}')
    t, sink, redis = translator(router, brain=RuntimeError("identity down"))
    await want(redis, "es")
    assert await t.handle(chunk_value()) == "unavailable"
    assert router.calls == []
    assert [(r["language"], r["status"], r["text"]) for r in sink.records] == [
        ("es", "unavailable", "")
    ]


# --- 3. one call for many languages ----------------------------------------------------------


async def test_one_metered_triage_call_for_every_language() -> None:
    llm = FakeLLM(
        '{"es": "Cerramos el presupuesto del T3 en 120k.", "fr": "Nous avons clos le budget '
        'du T3 à 120k.", "de": "  "}'
    )
    usage = UsageSink()
    router = LiteLLMRouter(budget_gate(fakeredis.aioredis.FakeRedis()), usage, llm)
    t, sink, redis = translator(router)
    await want(redis, "fr", "es", "de", "es")
    cid = str(uuid.uuid4())
    assert await t.handle(chunk_value(chunk_id=cid)) == "translated"

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "anthropic/claude-haiku-5-5"
    user = call["messages"][1]["content"]
    assert "de (German), es (Spanish), fr (French)" in user
    assert "We closed the Q3 budget at 120k." in user
    assert call["max_tokens"] == max_tokens_for("We closed the Q3 budget at 120k.", 3)
    assert len(usage.calls) == 1 and usage.calls[0]["user_id"] is None
    assert usage.calls[0]["org_id"] == ORG and usage.calls[0]["model_tier"] == "haiku"

    # one record per language that came back; "de" was blank, so nothing for it
    assert sorted((r["language"], r["status"]) for r in sink.records) == [
        ("es", "translated"),
        ("fr", "translated"),
    ]
    es = next(r for r in sink.records if r["language"] == "es")
    assert es == {
        "org_id": ORG,
        "meeting_id": MEETING,
        "chunk_id": cid,
        "language": "es",
        "status": "translated",
        "text": "Cerramos el presupuesto del T3 en 120k.",
    }


def test_max_tokens_grow_with_text_and_languages_but_stay_bounded() -> None:
    assert max_tokens_for("hi", 1) == 256
    assert max_tokens_for("x" * 400, 6) > max_tokens_for("x" * 400, 2)
    assert max_tokens_for("x" * 1000, 12) == 8192


def test_parse_translations() -> None:
    assert parse_translations('```json\n{"es": " hola ", "fr": 3}\n```', ["es", "fr"]) == {
        "es": "hola"
    }
    assert parse_translations('{"ja": "x"}', ["es"]) == {}
    assert parse_translations("Sure! Here you go", ["es"]) is None
    assert parse_translations('["es"]', ["es"]) is None


async def test_answer_without_any_wanted_language_emits_nothing() -> None:
    t, sink, redis = translator(FakeRouter('{"it": "ciao"}'))
    await want(redis, "es")
    assert await t.handle(chunk_value()) == "missing"
    assert sink.records == []


# --- 4. budget denied and unavailable, throttled ---------------------------------------------


async def test_budget_denied_once_per_meeting_and_language_a_minute() -> None:
    router = FakeRouter(exc=BudgetBlocked(str(ORG)))
    t, sink, redis = translator(router)
    await want(redis, "es", "fr")
    for _ in range(3):
        assert await t.handle(chunk_value()) == "budget_denied"
    assert len(router.calls) == 3
    assert sorted((r["language"], r["status"], r["text"]) for r in sink.records) == [
        ("es", "budget_denied", ""),
        ("fr", "budget_denied", ""),
    ]
    ttl = await redis.ttl(notice_key(ORG, MEETING, "es"))
    assert 0 < ttl <= 60

    other = uuid.uuid4()  # another meeting is told on its own
    await want(redis, "es", meeting=other)
    await t.handle(chunk_value(meeting=other))
    assert [r["meeting_id"] for r in sink.records].count(other) == 1

    await redis.delete(notice_key(ORG, MEETING, "es"))  # the minute passed
    await t.handle(chunk_value())
    assert [(r["meeting_id"], r["language"]) for r in sink.records].count((MEETING, "es")) == 2


@pytest.mark.parametrize(
    "router",
    [FakeRouter("not json at all"), FakeRouter(exc=RuntimeError("provider down"))],
    ids=["bad-json", "error"],
)
async def test_unavailable_on_bad_answer_or_error_throttled(router: FakeRouter) -> None:
    t, sink, redis = translator(router)
    await want(redis, "es")
    assert await t.handle(chunk_value()) == "unavailable"
    assert await t.handle(chunk_value()) == "unavailable"
    assert [(r["language"], r["status"], r["text"]) for r in sink.records] == [
        ("es", "unavailable", "")
    ]


# --- 5. the record on the wire ---------------------------------------------------------------

HEADER = {
    "type": "record",
    "name": "EventEnvelopeHeader",
    "namespace": "io.nexus.events",
    "fields": [
        {"name": "event_id", "type": {"type": "string", "logicalType": "uuid"}},
        {"name": "idempotency_key", "type": "string"},
        {"name": "org_id", "type": {"type": "string", "logicalType": "uuid"}},
        {"name": "schema_version", "type": "int", "default": 1},
        {"name": "producer", "type": "string"},
        {"name": "occurred_at", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "correlation_id", "type": ["null", "string"], "default": None},
        {"name": "causation_id", "type": ["null", "string"], "default": None},
    ],
}
BASE_FIELDS: list[dict[str, Any]] = [
    {"name": "header", "type": HEADER},
    {"name": "meeting_id", "type": {"type": "string", "logicalType": "uuid"}},
    {"name": "user_id", "type": {"type": "string", "logicalType": "uuid"}},
    {"name": "window_id", "type": "string"},
    {"name": "seq", "type": "int"},
    {"name": "delta", "type": "string"},
    {"name": "done", "type": "boolean"},
]
# mytheus-events schemas/agent-stream/v1.avsc as extended for caption_translation (490fc46)
EXTENDED = {
    "type": "record",
    "name": "AgentStreamDelta",
    "namespace": "io.nexus.events",
    "fields": [
        *BASE_FIELDS,
        {
            "name": "kind",
            "type": {
                "type": "enum",
                "name": "AgentStreamKind",
                "symbols": ["delta", "caption_translation"],
                "default": "delta",
            },
            "default": "delta",
        },
        {
            "name": "chunk_id",
            "type": ["null", {"type": "string", "logicalType": "uuid"}],
            "default": None,
        },
        {"name": "language", "type": ["null", "string"], "default": None},
        {
            "name": "status",
            "type": [
                "null",
                {
                    "type": "enum",
                    "name": "CaptionTranslationStatus",
                    "symbols": ["translated", "budget_denied", "unavailable"],
                    "default": "unavailable",
                },
            ],
            "default": None,
        },
    ],
}
LEGACY = {"type": "record", "name": "AgentStreamDelta", "namespace": "io.nexus.events"}
LEGACY["fields"] = BASE_FIELDS


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, Any]] = []

    def produce(self, topic: str, *, key: str, value: Any, on_delivery: Any) -> None:
        self.sent.append((topic, key, value))

    def poll(self, _t: float) -> int:
        return 0


class EchoCodec:
    def serialize_for(self, _topic: str, value: dict[str, Any]) -> dict[str, Any]:
        return value


def encode(schema: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    parsed = fastavro.parse_schema(json.loads(json.dumps(schema)))
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, parsed, value)
    buf.seek(0)
    out: dict[str, Any] = fastavro.schemaless_reader(buf, parsed)  # type: ignore[assignment]
    return out


async def test_record_matches_the_agent_stream_contract() -> None:
    producer = FakeProducer()
    emitter = Emitter(producer, EchoCodec())
    cid = str(uuid.uuid4())
    await emitter.caption_translation(
        org_id=ORG,
        meeting_id=MEETING,
        chunk_id=cid,
        language="es",
        status="translated",
        text="Cerramos el presupuesto.",
    )
    ((topic, key, value),) = producer.sent
    assert topic == "agent-stream" and key == f"{ORG}:{MEETING}"
    header = value.pop("header")
    assert header["producer"] == "agent-runtime-service"
    assert header["idempotency_key"] == f"caption:{cid}:es"
    assert header["org_id"] == str(ORG)
    assert abs(header["occurred_at"] / 1e6 - time.time()) < 60
    assert value == {
        "meeting_id": str(MEETING),
        "user_id": NIL,
        "window_id": f"caption:{cid}",
        "seq": 0,
        "delta": "Cerramos el presupuesto.",
        "done": True,
        "kind": "caption_translation",
        "chunk_id": cid,
        "language": "es",
        "status": "translated",
    }

    full = {"header": header, **value}
    decoded = encode(EXTENDED, full)
    assert decoded["kind"] == "caption_translation" and decoded["status"] == "translated"
    assert str(decoded["chunk_id"]) == cid and decoded["language"] == "es"
    # before the extended schema is registered: the extra fields are dropped, and the record
    # reads as a plain delta to the nil user (realtime-gateway ignores it)
    legacy = encode(LEGACY, full)
    assert "kind" not in legacy and str(legacy["user_id"]) == NIL


async def test_budget_denied_record_encodes() -> None:
    producer = FakeProducer()
    await Emitter(producer, EchoCodec()).caption_translation(
        org_id=ORG,
        meeting_id=MEETING,
        chunk_id=str(uuid.uuid4()),
        language="fr",
        status="budget_denied",
        text="",
    )
    value = producer.sent[0][2]
    assert value["delta"] == "" and value["status"] == "budget_denied"
    assert encode(EXTENDED, value)["status"] == "budget_denied"


# --- 6. concurrency --------------------------------------------------------------------------


class SlowRouter(FakeRouter):
    def __init__(self) -> None:
        super().__init__('{"es": "hola"}')
        self.release = asyncio.Event()
        self.in_flight = 0
        self.peak = 0

    async def complete(self, **kw: Any) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self.release.wait()
            return await super().complete(**kw)
        finally:
            self.in_flight -= 1


async def test_submit_never_waits_and_calls_are_bounded() -> None:
    router = SlowRouter()
    t, sink, redis = translator(router, max_in_flight=2, max_pending=5)
    await want(redis, "es")
    accepted = [t.submit(chunk_value()) for _ in range(7)]  # returns at once
    assert accepted == [True] * 5 + [False] * 2
    for _ in range(20):
        await asyncio.sleep(0)
    assert router.in_flight == 2
    router.release.set()
    await t.drain()
    assert router.peak == 2 and len(sink.records) == 5
