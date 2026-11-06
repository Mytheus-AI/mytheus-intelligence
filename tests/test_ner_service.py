"""ner-service: span resolution, regex validators, auth, size limit, readiness (stub model)."""

from __future__ import annotations

import logging
import uuid

import httpx

from services.ner_service import detect as detect_mod
from services.ner_service.detect import create_ner_app
from services.ner_service.regex_detector import detect_regex, iban_ok, luhn_ok
from services.ner_service.spans import Span, resolve_spans
from tests.conftest import auth_headers


class StubModel:
    name = "stub"

    def __init__(self, spans: list[Span] | None = None, fail_load: bool = False) -> None:
        self._spans = spans or []
        self._fail = fail_load
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        if self._fail:
            raise RuntimeError("no weights")

    def detect(self, text: str) -> list[Span]:
        return list(self._spans)


def _client(model: StubModel) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_ner_app(model)), base_url="http://t"
    )


H = lambda: auth_headers(uuid.uuid4())  # noqa: E731


# --- spans -----------------------------------------------------------------------------------
def test_same_type_overlap_is_unioned():
    out = resolve_spans([Span("PERSON", 0, 5), Span("PERSON", 3, 9)], 20)
    assert [(s.start, s.end) for s in out] == [(0, 9)]


def test_cross_type_overlap_higher_priority_wins():
    out = resolve_spans([Span("PERSON", 0, 10), Span("EMAIL", 2, 8)], 20)
    assert [s.type for s in out] == ["EMAIL"]


def test_invalid_bounds_dropped_and_output_sorted():
    out = resolve_spans(
        [Span("ORG", 10, 12), Span("LOC", 0, 3), Span("ORG", 5, 5), Span("ORG", 18, 30)], 20
    )
    assert [(s.type, s.start) for s in out] == [("LOC", 0), ("ORG", 10)]


def test_order_independent():
    a = [Span("PERSON", 0, 4), Span("PERSON", 3, 8), Span("EMAIL", 10, 20)]
    assert resolve_spans(a, 30) == resolve_spans(list(reversed(a)), 30)


# --- regex -----------------------------------------------------------------------------------
def test_unicode_offsets_are_codepoints():
    text = "Zoë 😀 écrit à zoe@acme.io"
    (sp,) = [s for s in detect_regex(text) if s.type == "EMAIL"]
    assert text[sp.start : sp.end] == "zoe@acme.io"


def test_validators():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB82 WEST 1234 5698 7654 33")


def test_card_beats_phone_and_ssn_detected():
    text = "card 4111 1111 1111 1111 ssn 123-45-6789"
    out = resolve_spans(detect_regex(text), len(text))
    assert [(s.type, text[s.start : s.end]) for s in out] == [
        ("CARD", "4111 1111 1111 1111"),
        ("GOVID", "123-45-6789"),
    ]


def test_phone():
    text = "call +1 415-555-0132 now"
    (sp,) = detect_regex(text)
    assert sp.type == "PHONE" and text[sp.start : sp.end] == "+1 415-555-0132"


# --- endpoint --------------------------------------------------------------------------------
async def test_detect_merges_model_and_regex_without_echoing_text(caplog):
    text = "Zoë Müller 😀 <zoe@acme.io>"
    model = StubModel([Span("PERSON", 0, 10, 0.9, "model")])
    caplog.set_level(logging.INFO)
    async with _client(model) as c:
        r = await c.post("/ner/detect", json={"text": text}, headers=H())
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "model" and body["partial"] is False
    got = [(s["type"], text[s["start"] : s["end"]]) for s in body["spans"]]
    assert got == [("PERSON", "Zoë Müller"), ("EMAIL", "zoe@acme.io")]
    assert "Zoë" not in r.text and "acme" not in r.text
    assert "acme" not in caplog.text and "Zoë" not in caplog.text


async def test_requires_auth():
    async with _client(StubModel()) as c:
        assert (await c.post("/ner/detect", json={"text": "x"})).status_code == 401
        r = await c.post(
            "/ner/detect", json={"text": "x"}, headers={"Authorization": "Bearer junk"}
        )
        assert r.status_code in (401, 503)  # junk token: 401 (503 only if JWKS unreachable)


async def test_no_org_claim_is_401():
    async with _client(StubModel()) as c:
        r = await c.post("/ner/detect", json={"text": "x"}, headers=auth_headers(None))
    assert r.status_code == 401


async def test_size_limit(monkeypatch):
    monkeypatch.setenv("NER_MAX_CHARS", "10")
    async with _client(StubModel()) as c:
        assert (
            await c.post("/ner/detect", json={"text": "a" * 11}, headers=H())
        ).status_code == 413
        assert (
            await c.post("/ner/detect", json={"text": "a" * 10}, headers=H())
        ).status_code == 200
        big = await c.post("/ner/detect", json={"text": "a" * 5000}, headers=H())
        assert big.status_code == 413


async def test_model_unavailable_is_503_and_fallback_is_opt_in():
    m = StubModel(fail_load=True)
    async with _client(m) as c:
        assert (await c.get("/readyz")).status_code == 503
        r = await c.post("/ner/detect", json={"text": "mail a@b.co"}, headers=H())
        assert r.status_code == 503
        r = await c.post(
            "/ner/detect", json={"text": "mail a@b.co", "allow_regex_fallback": True}, headers=H()
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "regex_only" and body["partial"] is True
        assert [s["type"] for s in body["spans"]] == ["EMAIL"]
    assert m.loads == 1  # failed load is not retried on every request


async def test_readyz_ok_when_loaded_and_empty_text():
    async with _client(StubModel()) as c:
        assert (await c.get("/readyz")).status_code == 200
        r = await c.post("/ner/detect", json={"text": ""}, headers=H())
        assert r.status_code == 200 and r.json()["spans"] == []


async def test_full_app_healthz_ok_even_when_not_ready(monkeypatch):
    from services.ner_service.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/healthz")).json()["service"] == "ner-service"


def test_module_exports():
    assert detect_mod.DEFAULT_MAX_CHARS == 100_000
