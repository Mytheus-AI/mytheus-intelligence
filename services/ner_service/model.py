"""The lightweight NER model behind the PII redaction pass. Loaded once per process.

[Compliance §5.2](../../../mytheus-docs/12-compliance-and-privacy.md) asks for "a lightweight
NER model run on the dedicated inference tier - never on the API tier", and
[ADR-0020](../../../mytheus-docs/adr/0020-pii-token-vault-and-ner-placement.md) D4/D5 turned
that into this service. `dslim/bert-base-NER` is that model: a BERT-base token classifier
fine-tuned on CoNLL-2003, whose four labels (`PER`, `ORG`, `LOC`, `MISC`) are close to the exact
set the deterministic layer cannot reach.

WHY A DETECTOR AND NOT A REDACTOR (ADR-0020 D5)

This module returns SPANS. It does not rewrite text, it does not mint tokens, and it never sees
the vault. Each redaction boundary keeps its own pass and assembles the result itself, because
canon already places the "PII redaction pass" responsibility on `ingestion-service` for the
document path and on `transcription-service` for the transcript path. What moved to this
service is the model, not the pass. A `redact()` here would quietly become a fourth boundary.

`MISC` IS DROPPED, DELIBERATELY

CoNLL's `MISC` covers nationalities, event names, product names and adjectival forms of
countries. It is not personal data in the sense §5.2 means, and `pii_tokens.entity_type` has no
value for it (ADR-0020 D3 froze the eight). Mapping it onto `ORG` to avoid throwing it away
would be worse than dropping it: it would mint vault entries for words like "German", pollute
the tenant's token space, and make redacted transcripts less readable for no privacy gain.

CROSS-TENANT ISOLATION

`detect()` runs the model over exactly ONE call's own text, in one `pipeline()` invocation.
There is no queue, cache, or `@serve.batch`-style fan-in anywhere in this module or `serve.py`
that could merge two callers' text into one forward pass - the same property, for the same
reason, that `reranker_service/model.py` documents at length. Here the stakes are higher than
there: the input to this model IS the unredacted PII, so a batch that mixed two tenants' text
would put one tenant's raw names in the same tensor as another's, and an index-alignment bug
would return them to the wrong caller. If a future task wants GPU throughput batching, it must
partition by `org_id` and must not redistribute results by anything looser than same-request
identity. That design is NOT implemented here.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

MODEL_NAME = "dslim/bert-base-NER"

# CoNLL-2003 label -> ADR-0020 D3 `entity_type`. `MISC` is absent on purpose; see the docstring.
_LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}

# Below this the model is guessing. A false positive here is not free: it mints a vault entry
# for a non-entity and replaces a real word in a persisted transcript, so the transcript reads
# worse AND the tenant's token space fills with noise. 0.9 is deliberately conservative,
# because the deterministic layer already covers every high-confidence structured identifier
# and this pass exists only for the unstructured remainder.
MIN_SCORE = 0.9


def gpu_requested() -> bool:
    return os.environ.get("NER_GPU", "0") == "1"


@lru_cache(maxsize=1)
def _pipeline() -> Any:
    """The token-classification pipeline, built once.

    `aggregation_strategy="simple"` is required rather than cosmetic: without it the pipeline
    returns WORDPIECE tokens, so "David Chen" comes back as `Da`, `##vid`, `Chen` with separate
    scores and no span covering the whole name. A redaction pass splicing those back would
    leave fragments of the name in the text, which is a redaction that does not redact.
    """
    from transformers import pipeline

    device = 0 if gpu_requested() else -1  # -1 is transformers' CPU sentinel
    return pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="simple",
        device=device,
    )


def detect(text: str, min_score: float = MIN_SCORE) -> tuple[list[dict[str, Any]], float]:
    """Real spans from a real forward pass over THIS CALL's text only.

    Returns `(spans, wall_clock_seconds)`. Each span is `{entity_type, start, end, text,
    score}`, with `start`/`end` as character offsets into the input so the caller can splice
    without re-finding the substring - re-finding is what produces the classic bug where the
    SECOND occurrence of a name gets replaced and the first is left in place.

    Spans are returned sorted by `start` and non-overlapping, keeping the higher-scoring span
    when the model emits two that overlap. A caller replacing overlapping spans right-to-left
    would otherwise corrupt the text.
    """
    if not text.strip():
        return [], 0.0

    t0 = time.perf_counter()
    raw = _pipeline()(text)
    elapsed = time.perf_counter() - t0

    # Narrowed into typed tuples immediately rather than carried as `dict[str, Any]`. The
    # pipeline's output is untyped, and comparing untyped offsets is precisely where an
    # off-by-one in the overlap check below would hide - so the untyped boundary is crossed
    # once, here, and everything after it works on real ints and floats.
    candidates: list[tuple[str, int, int, float]] = []
    for ent in raw:
        entity_type = _LABEL_MAP.get(str(ent.get("entity_group", "")))
        score = float(ent.get("score", 0.0))
        if entity_type is None or score < min_score:
            continue
        candidates.append((entity_type, int(ent["start"]), int(ent["end"]), score))

    # Highest score first, so the winner of an overlap is the more confident span; then emit in
    # document order, which is what a splicing caller needs.
    kept: list[tuple[str, int, int, float]] = []
    for entity_type, start, end, score in sorted(candidates, key=lambda c: (-c[3], c[1])):
        if any(start < k_end and k_start < end for _, k_start, k_end, _ in kept):
            continue
        kept.append((entity_type, start, end, score))
    kept.sort(key=lambda c: c[1])

    spans: list[dict[str, Any]] = [
        {
            "entity_type": entity_type,
            "start": start,
            "end": end,
            "text": text[start:end],
            "score": score,
        }
        for entity_type, start, end, score in kept
    ]
    return spans, elapsed
