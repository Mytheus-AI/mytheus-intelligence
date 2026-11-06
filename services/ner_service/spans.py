"""Entity spans and deterministic merge/resolve logic.

Offsets are Python `str` indices, i.e. Unicode CODE POINT offsets into the request `text`
(`text[start:end]` is the entity). They are NOT UTF-8 byte offsets and NOT UTF-16 units; callers
in other runtimes (JS, Go) must convert.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Higher wins when two DIFFERENT-type spans overlap. Validated structured identifiers beat
# statistical model output (a model may tag a card number as a PERSON/ORG fragment).
TYPE_PRIORITY: dict[str, int] = {
    "CARD": 100,
    "IBAN": 95,
    "GOVID": 90,
    "EMAIL": 85,
    "PHONE": 70,
    "PERSON": 40,
    "ORG": 30,
    "LOC": 20,
}


@dataclass(frozen=True)
class Span:
    type: str
    start: int
    end: int
    score: float = 1.0
    source: str = "regex"  # "regex" | "model" | "regex+model"

    @property
    def length(self) -> int:
        return self.end - self.start


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def _rank(s: Span) -> tuple[int, int, float, int]:
    # Sort key, best first when sorted ascending: priority desc, longer desc, score desc, earlier.
    return (-TYPE_PRIORITY.get(s.type, 0), -s.length, -s.score, s.start)


def resolve_spans(spans: list[Span], text_len: int) -> list[Span]:
    """Return a non-overlapping, start-sorted span list.

    - Spans with invalid bounds (empty, negative, past `text_len`) are dropped.
    - Overlapping spans of the SAME type are unioned (a model and a regex both seeing an email).
    - Overlapping spans of DIFFERENT types: the higher `TYPE_PRIORITY` wins, then longer, then
      higher score; losers are dropped entirely (never split).
    Deterministic for any input order.
    """
    valid = [s for s in spans if 0 <= s.start < s.end <= text_len]
    accepted: list[Span] = []
    for cand in sorted(valid, key=_rank):
        hits = [a for a in accepted if _overlaps(a, cand)]
        if not hits:
            accepted.append(cand)
            continue
        if all(h.type == cand.type for h in hits) and len(hits) == 1:
            h = hits[0]
            merged = replace(
                h,
                start=min(h.start, cand.start),
                end=max(h.end, cand.end),
                score=max(h.score, cand.score),
                source=h.source if h.source == cand.source else "regex+model",
            )
            others = [a for a in accepted if a is not h]
            if not any(_overlaps(merged, o) for o in others):
                accepted = [*others, merged]
        # else: lost to a higher-ranked span of another type -> dropped
    return sorted(accepted, key=lambda s: (s.start, s.end))
