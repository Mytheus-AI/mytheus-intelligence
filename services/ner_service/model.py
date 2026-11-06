"""Model interface, lazy spaCy implementation, and the load-state holder.

The spaCy dependency lives in the optional `ner` extra (`pip install .[ner]`) and is imported
only inside `SpacyNerModel.load()`, so this module and the whole service import cleanly without
it (readiness then reports not-ready and `/ner` returns 503 unless degraded mode is
requested).

PRIVACY: nothing in here logs or stores `text`. Exceptions are logged by class name only because
library errors can embed input fragments.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Protocol

from services.ner_service.spans import Span

logger = logging.getLogger("ner_service")

# spaCy label -> our entity type. Anything else is ignored.
_SPACY_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
}
_RETRY_AFTER_S = 30.0


class NerModel(Protocol):
    name: str

    def load(self) -> None:
        """Load weights. Raise on failure. Must be idempotent."""

    def detect(self, text: str) -> list[Span]:
        """Return model spans (offsets = code points). Called only after `load()` succeeded."""


class SpacyNerModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("NER_SPACY_MODEL", "en_core_web_sm")
        self.name = f"spacy:{self.model_name}"
        self._nlp: object | None = None

    def load(self) -> None:
        if self._nlp is not None:
            return
        import spacy  # optional extra; ModuleNotFoundError -> not ready

        self._nlp = spacy.load(self.model_name, disable=["lemmatizer", "parser"])

    def detect(self, text: str) -> list[Span]:
        nlp = self._nlp
        if nlp is None:
            raise RuntimeError("model not loaded")
        doc = nlp(text)  # type: ignore[operator]
        return [
            Span(_SPACY_LABELS[e.label_], e.start_char, e.end_char, score=0.85, source="model")
            for e in doc.ents
            if e.label_ in _SPACY_LABELS
        ]


class ModelHolder:
    """Thread-safe lazy loader. A failed load is retried at most every `_RETRY_AFTER_S`."""

    def __init__(self, model: NerModel) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._loaded = False
        self._last_failure = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            if self._last_failure and time.monotonic() - self._last_failure < _RETRY_AFTER_S:
                return False
            try:
                self.model.load()
            except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
                self._last_failure = time.monotonic()
                logger.warning("ner model load failed", extra={"error_type": type(exc).__name__})
                return False
            self._loaded = True
            return True
