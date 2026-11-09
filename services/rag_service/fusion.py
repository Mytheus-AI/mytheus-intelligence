"""Stage 3a: Reciprocal Rank Fusion, `RRF(d) = sum over lists of 1 / (k + rank)`, k = 60
(mytheus-docs doc 09 section 3.3). Ranks start at 1. Ties keep first-seen order."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(
    ranked_lists: Sequence[Sequence[str]], *, k: int = RRF_K, top: int = 50
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, key in enumerate(ranked, start=1):
            if key in seen:  # a list counts each document once, at its best rank
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top]
