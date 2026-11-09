"""Stage 1, query expansion (mytheus-docs doc 09 section 3.1, ADR-0033 D2).

One Haiku call returns up to two query variants AND the filters the question implies: a date
range, a person, a source, and topics from the role-topics taxonomy (which seed the graph arm).
Parsing is defensive: anything malformed degrades to "no variants, no filters", never to an
error, because expansion only ever widens recall.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol
from uuid import UUID

from libs.shared.prompts import PromptLoader
from services.rag_service.domain import Filters, Person

SOURCES = frozenset({"transcripts", "documents", "decisions"})
MAX_VARIANTS = 2


class Llm(Protocol):
    async def complete(
        self,
        *,
        tier: Any,
        system: str,
        user: str,
        org_id: UUID,
        user_id: UUID | None,
        max_tokens: int,
        prompt_ref: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class Expansion:
    variants: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)
    topics: list[str] = field(default_factory=list)
    person_text: str | None = None


def should_expand(query: str) -> bool:
    """A query of three words or fewer is a lookup (a name, a codename, a ticket id): the raw
    query is the best query, and an LLM call would only add latency."""
    return len(query.split()) > 3


def _date(raw: Any) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def resolve_person(name: str | None, people: Sequence[Person]) -> tuple[UUID, ...]:
    """Match a name from the question to org members by email local part. `users` holds no
    display name, so "Priya" matches priya@..., priya.nair@...; more than three matches is too
    ambiguous to filter on."""
    if not name:
        return ()
    wanted = [t for t in re.split(r"[\s.]+", name.lower()) if len(t) > 1]
    if not wanted:
        return ()
    hits: list[UUID] = []
    for p in people:
        tokens = set(re.split(r"[._+\-]+", p.email.split("@", 1)[0].lower()))
        if wanted[0] in tokens or "".join(wanted) in tokens:
            hits.append(p.id)
    return tuple(hits) if 0 < len(hits) <= 3 else ()


def parse_expansion(
    raw: str, query: str, people: Sequence[Person], known_topics: Sequence[str]
) -> Expansion:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return Expansion()
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Expansion()
    if not isinstance(body, dict):
        return Expansion()
    variants = [
        v.strip()
        for v in body.get("variants") or []
        if isinstance(v, str) and v.strip() and v.strip().lower() != query.strip().lower()
    ][:MAX_VARIANTS]
    person_text = body.get("person") if isinstance(body.get("person"), str) else None
    source = body.get("source") if body.get("source") in SOURCES else None
    topics_known = set(known_topics)
    topics = [t for t in body.get("topics") or [] if isinstance(t, str) and t in topics_known]
    date_from, date_to = _date(body.get("date_from")), _date(body.get("date_to"))
    if date_from and date_to and date_from > date_to:
        date_from, date_to = date_to, date_from
    return Expansion(
        variants=variants,
        filters=Filters(
            date_from=date_from,
            date_to=date_to,
            person_ids=resolve_person(person_text, people),
            source=source,
        ),
        topics=topics[:3],
        person_text=person_text,
    )


class Expander:
    PROMPT_KEY = "rag_expand"

    def __init__(self, llm: Llm, prompts: PromptLoader, known_topics: Sequence[str]) -> None:
        self._llm = llm
        self._prompts = prompts
        self._topics = list(known_topics)

    async def expand(
        self,
        query: str,
        *,
        org_id: UUID,
        user_id: UUID,
        people: Sequence[Person],
        today: date,
    ) -> Expansion:
        version = self._prompts.resolve(self.PROMPT_KEY, org_id)
        prompt = self._prompts.load(self.PROMPT_KEY, version)
        completion = await self._llm.complete(
            tier="triage",
            system=prompt.system,
            user=prompt.render_user(
                today=today.isoformat(),
                weekday=today.strftime("%A"),
                topics=", ".join(self._topics),
                query=query,
            ),
            org_id=org_id,
            user_id=user_id,
            # 200 truncated real Haiku output mid-JSON (finish_reason=length, D7 live run).
            max_tokens=1000,
            prompt_ref=prompt.ref,
        )
        return parse_expansion(completion.text, query, people, self._topics)
