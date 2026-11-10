"""The org's never-record topics as a deterministic text policy (mytheus-docs ADR-0039 C2).

The prompts tell every model call about the "Never record" line, which is an instruction, not a
guarantee. This module is the guarantee for the cases a machine can see: a topic, or a glossary
alias of it, said outright. It is shared by the agent runtime (every agent's output passes
through it before it leaves the tier, `services/agent_runtime_service/names.py`) and by the
indexing consumers (`libs/shared/index_guard.py`: embedding-service and search-service never
index such a window).

Matching is case-insensitive on word boundaries, with a plural tolerance ("layoff" matches
"layoffs" and the other way round), over the topic itself and the aliases of a glossary term
whose name or alias IS the topic ("Project Falcon" with alias "Falcon" -> both). That is a
`hit`.

`ambiguous` is the cheap pre-filter for the paid check: no topic is said outright, but a
significant word of one is (at least 4 letters, not a stop word). The indexers send only those
to a triage-tier classifier; everything else is decided here for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from libs.shared.org_context import EMPTY, CompanyContext

_STOP = frozenset(
    {
        "about",
        "after",
        "also",
        "from",
        "have",
        "into",
        "just",
        "like",
        "more",
        "much",
        "only",
        "other",
        "over",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "very",
        "what",
        "when",
        "where",
        "which",
        "with",
        "your",
        "team",
        "work",
        "plan",
        "plans",
        "project",
        "meeting",
        "company",
        "people",
        "topic",
        "topics",
        "discussion",
        "discussions",
    }
)
_WORD = re.compile(r"[a-z0-9][a-z0-9'&-]*", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _word_pattern(word: str) -> str:
    """One word with a plural tolerance: `layoff` <-> `layoffs`, `salary` <-> `salaries`,
    `process` -> `processes`."""
    lower = word.lower()
    forms = {f"{re.escape(lower)}(?:s|es)?"}
    if lower.endswith("ies") and len(lower) > 4:
        forms.add(re.escape(lower[:-3] + "y"))
    elif lower.endswith("y") and len(lower) > 2:
        forms.add(re.escape(lower[:-1] + "ies"))
    if lower.endswith("s") and len(lower) > 3:
        forms.add(re.escape(lower[:-1]))
    return "(?:" + "|".join(sorted(forms, key=len, reverse=True)) + ")"


def phrase_pattern(text: str) -> re.Pattern[str] | None:
    words = _norm(text).split()
    if not words:
        return None
    body = r"\s+".join(_word_pattern(w) for w in words)
    return re.compile(rf"(?<![\w]){body}(?![\w])", re.IGNORECASE)


def topic_terms(topic: str, org: CompanyContext) -> tuple[str, ...]:
    """The topic plus every name and alias of a glossary term that is (or is aliased as) it."""
    wanted = _norm(topic)
    out = [topic]
    for term in org.glossary:
        names = [term.term, *term.aliases]
        if any(_norm(n) == wanted for n in names):
            out.extend(n for n in names if _norm(n) not in {_norm(o) for o in out})
    return tuple(out)


@dataclass(frozen=True)
class NeverRecordPolicy:
    """`(topic, terms)` per never-record topic of one org. Empty for an org that set none."""

    topics: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @classmethod
    def from_org(cls, org: CompanyContext | None) -> NeverRecordPolicy:
        org = org if org is not None else EMPTY
        return cls(
            tuple((t, topic_terms(t, org)) for t in org.never_record_topics if t and t.strip())
        )

    @property
    def empty(self) -> bool:
        return not self.topics

    @cached_property
    def _patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        out: list[tuple[str, re.Pattern[str]]] = []
        for topic, terms in self.topics:
            for term in terms:
                p = phrase_pattern(term)
                if p is not None:
                    out.append((topic, p))
        return tuple(out)

    @cached_property
    def _words(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        out: list[tuple[str, re.Pattern[str]]] = []
        for topic, terms in self.topics:
            seen: set[str] = set()
            for term in terms:
                for word in _WORD.findall(term.lower()):
                    if len(word) < 4 or word in _STOP or word in seen:
                        continue
                    seen.add(word)
                    p = phrase_pattern(word)
                    if p is not None:
                        out.append((topic, p))
        return tuple(out)

    def hit(self, text: str | None) -> str | None:
        """The never-record topic `text` names outright (itself or an alias), else None."""
        if not text or self.empty:
            return None
        for topic, pattern in self._patterns:
            if pattern.search(text):
                return topic
        return None

    def hits(self, *texts: str | None) -> bool:
        return any(self.hit(t) is not None for t in texts)

    def ambiguous(self, text: str | None) -> str | None:
        """No topic said outright, but a significant word of one is: worth a classifier."""
        if not text or self.empty or self.hit(text) is not None:
            return None
        for topic, pattern in self._words:
            if pattern.search(text):
                return topic
        return None
