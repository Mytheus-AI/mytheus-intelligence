"""The grounded answer (mytheus-docs ADR-0033 D6).

Sonnet writes a short answer from the numbered passages. Before it is returned, every sentence
is checked: it must carry at least one `[n]` marker, and every marker must point at a returned
passage. A sentence that fails is removed. If nothing survives, or the model says NOT_FOUND,
the answer is the fixed abstention. The answer is returned whole, after the check, and not
streamed: a streamed sentence could reach the user before the check removes it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

from libs.shared.prompts import PromptLoader
from services.rag_service.contracts import Answer, AnswerPassage
from services.rag_service.expand import Llm

ABSTAIN_TEXT = "I couldn't find this in your meetings or documents."
NOT_FOUND = "NOT_FOUND"
_MARKER = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKER_AFTER_STOP = re.compile(r"([.!?])\s*((?:\[\d+\])+)")


def abstain() -> Answer:
    return Answer(text=ABSTAIN_TEXT, abstained=True, cited=[])


def validate(raw: str, passage_count: int) -> Answer:
    text = raw.strip()
    if not text or text.upper().startswith(NOT_FOUND):
        return abstain()
    # "Friday. [1]" and "Friday [1]." mean the same thing: move a trailing marker inside.
    text = _MARKER_AFTER_STOP.sub(r" \2\1", text)
    kept: list[str] = []
    cited: set[int] = set()
    for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
        if not sentence or not re.search(r"[A-Za-z0-9]", _MARKER.sub("", sentence)):
            continue
        markers = [int(m) for m in _MARKER.findall(sentence)]
        if not markers or any(not 1 <= m <= passage_count for m in markers):
            continue
        kept.append(sentence)
        cited.update(markers)
    if not kept:
        return abstain()
    return Answer(text=" ".join(kept), abstained=False, cited=sorted(cited))


def render_passages(passages: Sequence[AnswerPassage]) -> str:
    lines: list[str] = []
    for p in passages:
        where = []
        if p.meeting_title:
            where.append(f'meeting "{p.meeting_title}"')
        if p.occurred_at:
            where.append(p.occurred_at.date().isoformat())
        if p.citation.kind in ("decision", "action"):
            where.append(f"confirmed {p.citation.kind}")
        head = f"[{p.n}]" + (f" ({', '.join(where)})" if where else "")
        body = " ".join(x for x in (p.context_before, p.text, p.context_after) if x)
        lines.append(f"{head} {body}")
    return "\n".join(lines)


class AnswerWriter:
    PROMPT_KEY = "rag_answer"

    def __init__(self, llm: Llm, prompts: PromptLoader) -> None:
        self._llm = llm
        self._prompts = prompts

    async def write(
        self, query: str, passages: Sequence[AnswerPassage], *, org_id: UUID, user_id: UUID
    ) -> Answer:
        if not passages:
            return abstain()
        prompt = self._prompts.load(self.PROMPT_KEY, self._prompts.resolve(self.PROMPT_KEY, org_id))
        completion = await self._llm.complete(
            tier="extract",
            system=prompt.system,
            user=prompt.render_user(query=query, passages=render_passages(passages)),
            org_id=org_id,
            user_id=user_id,
            max_tokens=1500,
            prompt_ref=prompt.ref,
        )
        return validate(completion.text, len(passages))
