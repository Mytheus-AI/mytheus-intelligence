"""Reranker/embedding warm-floor + preload wiring. No infra, no model download, no Ray."""

from __future__ import annotations

import asyncio

import pytest

import services.embedding_service.main as emb_main
import services.reranker_service.main as rr_main
import services.reranker_service.model as rr_model
from services.reranker_service import serve as rr_serve


def test_min_replicas_default_is_one(monkeypatch):
    monkeypatch.delenv("RERANKER_MIN_REPLICAS", raising=False)
    assert rr_serve.min_replicas() == 1


@pytest.mark.parametrize(("raw", "want"), [("0", 1), ("-3", 1), ("2", 2)])
def test_min_replicas_never_below_one(monkeypatch, raw, want):
    monkeypatch.setenv("RERANKER_MIN_REPLICAS", raw)
    assert rr_serve.min_replicas() == want


def test_max_replicas_never_below_min(monkeypatch):
    monkeypatch.setenv("RERANKER_MIN_REPLICAS", "5")
    monkeypatch.setenv("RERANKER_MAX_REPLICAS", "2")
    assert rr_serve.max_replicas() == 5


def test_warmup_runs_a_real_rerank_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rr_model, "rerank_batch", lambda q, p, top_k=10: calls.append((q, p, top_k)) or ([], 0.0)
    )
    assert rr_model.warmup() >= 0
    assert len(calls) == 1 and len(calls[0][1]) >= 1


@pytest.mark.parametrize(
    ("mod", "flag"), [(rr_main, "RERANKER_PRELOAD"), (emb_main, "EMBEDDING_PRELOAD")]
)
def test_startup_preloads_model_unless_opted_out(monkeypatch, mod, flag):
    called = []
    monkeypatch.setattr(mod, "warmup", lambda: called.append(1) or 0.0)
    monkeypatch.delenv(flag, raising=False)
    asyncio.run(mod._preload_model())
    assert called == [1]
    monkeypatch.setenv(flag, "0")
    asyncio.run(mod._preload_model())
    assert called == [1]


def test_startup_handlers_registered():
    for mod in (rr_main, emb_main):
        assert mod._preload_model in mod.app.router.on_startup
