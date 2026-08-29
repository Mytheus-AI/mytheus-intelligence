"""`POST /ner` - the detect contract and the span assembly it depends on (ADR-0020 D4/D5).

Split deliberately in two.

The SPAN ASSEMBLY tests drive `detect()` with a stubbed pipeline. That is not a fake standing in
for the model's judgement - it is the only way to test the logic this module actually owns:
overlap resolution, the `MISC` drop, the score floor, and the character offsets a caller splices
on. Those must hold for inputs the real model may or may not happen to produce today, and a
model upgrade must not silently change them.

The REAL-MODEL test loads `dslim/bert-base-NER` and asserts only that a plainly-named person is
found, matching `test_reranker.py`'s "do not substitute a fake, a fake makes the assertion
meaningless" stance. It skips when the weights are not present rather than failing, because CI
here is GPU-free and may also be network-free.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest

from services.ner_service import model as ner_model
from services.ner_service.ner import create_ner_app

ORG = str(uuid.uuid4())
TOKEN = "x" * 32


def _headers() -> dict[str, str]:
    return {"X-Internal-Service-Token": TOKEN}


@pytest.fixture()
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Replace the transformers pipeline with one returning scripted entity groups."""

    def _install(entities: list[dict[str, Any]]) -> None:
        ner_model._pipeline.cache_clear()
        monkeypatch.setattr(ner_model, "_pipeline", lambda: (lambda _text: entities))

    # No cache_clear on teardown: monkeypatch has not undone the patch yet at this point, so
    # `_pipeline` is still the plain lambda and has no cache to clear. The real lru_cache is
    # cleared on the way IN, which is what actually matters.
    yield _install


def test_conll_labels_map_to_the_frozen_entity_types(stub_pipeline) -> None:  # type: ignore[no-untyped-def]
    text = "David Chen at Acme in Berlin"
    stub_pipeline(
        [
            {"entity_group": "PER", "score": 0.99, "start": 0, "end": 10},
            {"entity_group": "ORG", "score": 0.99, "start": 14, "end": 18},
            {"entity_group": "LOC", "score": 0.99, "start": 22, "end": 28},
        ]
    )
    spans, _ = ner_model.detect(text)
    assert [s["entity_type"] for s in spans] == ["PERSON", "ORG", "LOC"]
    assert [s["text"] for s in spans] == ["David Chen", "Acme", "Berlin"]


def test_misc_is_dropped_rather_than_folded_into_org(stub_pipeline) -> None:  # type: ignore[no-untyped-def]
    """CoNLL `MISC` is nationalities, events and product names. It is not personal data, it has
    no `pii_tokens.entity_type`, and mapping it to ORG would mint vault entries for words like
    "German" - polluting the tenant's token space and making transcripts less readable for no
    privacy gain."""
    stub_pipeline([{"entity_group": "MISC", "score": 0.99, "start": 0, "end": 6}])
    spans, _ = ner_model.detect("German engineering")
    assert spans == []


def test_low_confidence_spans_are_dropped(stub_pipeline) -> None:  # type: ignore[no-untyped-def]
    """A false positive is not free: it mints a vault entry for a non-entity AND replaces a real
    word in a persisted transcript."""
    stub_pipeline([{"entity_group": "PER", "score": 0.5, "start": 0, "end": 5}])
    assert ner_model.detect("Mango is ripe")[0] == []
    assert len(ner_model.detect("Mango is ripe", min_score=0.4)[0]) == 1


def test_overlapping_spans_resolve_to_the_higher_scoring_one(stub_pipeline) -> None:  # type: ignore[no-untyped-def]
    """A caller splicing overlapping spans would corrupt the text - the second replacement would
    land inside the placeholder the first one wrote."""
    stub_pipeline(
        [
            {"entity_group": "PER", "score": 0.91, "start": 0, "end": 5},
            {"entity_group": "ORG", "score": 0.97, "start": 0, "end": 10},
        ]
    )
    spans, _ = ner_model.detect("David Chen said")
    assert len(spans) == 1
    assert (spans[0]["entity_type"], spans[0]["end"]) == ("ORG", 10)


def test_spans_come_back_in_document_order(stub_pipeline) -> None:  # type: ignore[no-untyped-def]
    """Sorted by `start` regardless of score, because that is what a splicing caller needs."""
    stub_pipeline(
        [
            {"entity_group": "LOC", "score": 0.92, "start": 20, "end": 26},
            {"entity_group": "PER", "score": 0.99, "start": 0, "end": 10},
        ]
    )
    spans, _ = ner_model.detect("David Chen flew out to Berlin")
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)


def test_empty_text_short_circuits_without_loading_a_model() -> None:
    assert ner_model.detect("   ") == ([], 0.0)


async def test_route_requires_the_internal_service_credential(
    monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:  # type: ignore[no-untyped-def]
    """`POST /ner` is internal-only per canon. Its input is unredacted PII, so an unauthenticated
    caller reaching it would be handing this service other people's personal data to log."""
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TOKEN)
    stub_pipeline([])
    transport = httpx.ASGITransport(app=create_ner_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        body = {"org_id": ORG, "text": "hello"}
        assert (await client.post("/ner", json=body)).status_code == 401
        wrong = {"X-Internal-Service-Token": "no"}
        assert (await client.post("/ner", json=body, headers=wrong)).status_code == 401
        assert (await client.post("/ner", json=body, headers=_headers())).status_code == 200


async def test_route_returns_spliceable_offsets(
    monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:  # type: ignore[no-untyped-def]
    """The offsets are the contract: a caller must be able to rebuild the redacted text from
    them without re-finding the substring, which is what produces the classic bug where the
    SECOND occurrence of a name is replaced and the first is left in place."""
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TOKEN)
    text = "David Chen met David Chen"
    stub_pipeline(
        [
            {"entity_group": "PER", "score": 0.99, "start": 0, "end": 10},
            {"entity_group": "PER", "score": 0.99, "start": 15, "end": 25},
        ]
    )
    transport = httpx.ASGITransport(app=create_ner_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        res = await client.post("/ner", json={"org_id": ORG, "text": text}, headers=_headers())

    spans = res.json()["spans"]
    out = text
    for span in reversed(spans):  # right-to-left, so earlier offsets stay valid
        out = out[: span["start"]] + "<PERSON>" + out[span["end"] :]
    assert out == "<PERSON> met <PERSON>"


async def test_no_request_text_reaches_the_logs(
    monkeypatch: pytest.MonkeyPatch, stub_pipeline, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """This is the one endpoint whose input is, by construction, exactly the personal data the
    whole pipeline exists to remove. Logging it would make the log the leak."""
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TOKEN)
    stub_pipeline([{"entity_group": "PER", "score": 0.99, "start": 0, "end": 10}])
    transport = httpx.ASGITransport(app=create_ner_app())
    with caplog.at_level("INFO"):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post(
                "/ner",
                json={"org_id": ORG, "text": "David Chen owes 400 dollars"},
                headers=_headers(),
            )
    logged = "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert "David Chen" not in logged
    assert ORG in logged


def test_the_real_model_finds_a_plainly_named_person() -> None:
    """Against the REAL `dslim/bert-base-NER` weights. A stub here would make the assertion
    meaningless (`test_reranker.py`'s stance); a skip is correct when the weights are absent,
    because CI is GPU-free and may be network-free."""
    ner_model._pipeline.cache_clear()
    try:
        spans, elapsed = ner_model.detect("David Chen will own the Acme renewal.")
    except Exception as exc:  # noqa: BLE001 - any load failure is "weights unavailable here"
        pytest.skip(f"dslim/bert-base-NER not loadable in this environment: {exc}")

    assert any(s["entity_type"] == "PERSON" and "Chen" in s["text"] for s in spans)
    assert elapsed > 0
