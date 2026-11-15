"""A participant-scoped retrieval stand-in over the synthetic org (`org.py`).

`SyntheticRag.retrieve(org_id, user_id, query)` returns, like rag-service under ADR-0033 D1, only
passages from sources that user can read (`Source.readers`), each with a citation of the
source's kind. Scoring is lexical: the idf-weighted share of the query's content words found in
the passage (light suffix stemming), so 1.0 means every query word matched. This replaces the
hybrid retrieval + cross-encoder rerank of production; its scores are on the same 0..1 scale
the ContextGapAgent's `have_score` threshold (0.5) reads, but it is not that ranker (stated as a
threat to validity in the README).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from libs.shared.rag_client import RagPassage
from research.redteam.org import SOURCES, USERS, Source, uid

_WORD = re.compile(r"[a-z0-9][a-z0-9'._-]*")
STOP = frozenset(
    """a an the of to in on for and or with from by at is are was were be been it its this that
    these those what which who whom when where why how did do does we our us you your they them
    their i me my last time earlier previous meeting meetings about again remind exactly
    happened happen per as discussed any some there here then than so up out into over just
    still also please quick one get got re can could would should will shall going decide
    decided decision decisions""".split()
)
TOP_K = 10
MIN_SCORE = 0.05


def stem(word: str) -> str:
    word = word.strip("'._-")
    for suffix in ("ing", "ions", "ion", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def terms(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


@dataclass
class Indexed:
    source: Source
    n: int
    text: str
    terms: frozenset[str]


@dataclass
class SyntheticRag:
    extra: tuple[Source, ...] = ()
    as_user_override: dict[UUID, UUID] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._docs = [
            Indexed(s, n, text, frozenset(terms(text)))
            for s in SOURCES + self.extra
            for n, text in enumerate(s.passages)
        ]
        df: dict[str, int] = {}
        for d in self._docs:
            for t in d.terms:
                df[t] = df.get(t, 0) + 1
        total = len(self._docs)
        self._idf = {t: math.log(1 + total / c) for t, c in df.items()}
        self._default_idf = math.log(1 + total)
        self._by_id = {u.id: k for k, u in USERS.items()}

    def score(self, query_terms: list[str], doc: Indexed) -> float:
        if not query_terms:
            return 0.0
        # Query words that occur nowhere in the corpus cannot discriminate between passages;
        # like a semantic ranker, they do not lower every score.
        q = {t for t in query_terms if t in self._idf}
        weight = sum(self._idf[t] for t in q)
        hit = sum(self._idf[t] for t in q if t in doc.terms)
        return round(hit / weight, 4) if weight else 0.0

    def visible(self, user_key: str) -> list[Indexed]:
        return [d for d in self._docs if user_key in d.source.readers]

    async def retrieve(
        self, *, org_id: UUID, user_id: UUID, query: str, meeting_id: UUID | None = None
    ) -> list[RagPassage]:
        user_key = self._by_id.get(user_id)
        self.calls.append({"user": user_key, "query": query})
        if user_key is None:
            return []
        q = terms(query)
        scored = [(self.score(q, d), d) for d in self.visible(user_key)]
        scored = [(s, d) for s, d in scored if s >= MIN_SCORE]
        scored.sort(key=lambda x: (-x[0], x[1].source.key, x[1].n))
        return [passage(d, s, i + 1) for i, (s, d) in enumerate(scored[:TOP_K])]


def passage(d: Indexed, score: float, n: int) -> RagPassage:
    s = d.source
    pid = s.passage_id(d.n)
    ids: dict[str, Any] = {
        "transcript": {"chunk_id": pid},
        "document": {"document_id": pid},
        "decision": {"decision_id": pid},
        "action": {"action_id": pid},
    }[s.kind]
    return RagPassage(
        n=n,
        text=d.text,
        score=score,
        kind=s.kind,  # type: ignore[arg-type]
        meeting_id=s.meeting_id if s.kind != "document" else None,
        **ids,
    )


def source_of_citation(citation: dict[str, Any], extra: tuple[Source, ...] = ()) -> str | None:
    """The source key a stored citation points at (for the leak oracle)."""
    ident = next(
        (
            citation.get(k)
            for k in ("chunk_id", "document_id", "decision_id", "action_id")
            if k in citation
        ),
        None,
    )
    if ident is None:
        return None
    for s in SOURCES + extra:
        for n in range(len(s.passages)):
            if str(s.passage_id(n)) == str(ident):
                return s.key
    return None


__all__ = ["SyntheticRag", "source_of_citation", "terms", "uid"]
