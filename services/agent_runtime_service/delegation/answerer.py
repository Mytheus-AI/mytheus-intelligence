"""`DelegatedAnswerer`: an absent person's agent answers questions put to them, ONLY from what they
pre-approved for this meeting (polish wave 3, "Delegated answers").

Used by `ContextGapAgent` (the peer path: the answer is another person's agent answering) once
per closed window, before its own gap pass. Per window, for each delegate (a participant who
sent their agent and pre-approved statements, `client.py`):

1. **Detector** (no model call): the last turn, not spoken by the delegate, that names them
   (first name, full name, or "{name}'s agent") and reads as a question (a "?" or an opening
   interrogative). A question already answered in this meeting (normalised text) is skipped,
   and at most `MAX_ANSWERS_PER_DELEGATE` answers are given per meeting.
2. **One strict model call** (triage tier, budget-gated by the router: `BudgetBlocked` means no
   answer and the agent is marked blocked), prompt `delegated_answer@vN`: the person's first
   name, their numbered statements and the question. The model may only say which statement
   answers it, or that none does. It writes the answer from that statement alone.
3. **Fail closed.** A statement number outside the list, an empty answer, or anything
   unparseable is "not covered". Not covered -> the fixed refusal
   `"{First} didn't pre-approve an answer to that."`, written here, never by the model.
   `addressed: false` (the turn was not really a question for them) -> nothing.

Output: one `context_fill` AgentResult per LIVE participant (the whole room sees what the agent
said), each with a `peer` collaboration entry (responder `user:{delegate}`, result `answered` or
`no_answer`), a surfaced fill (`hop: peer`) that realtime-gateway shows as a context card, and a
`delegated` block core-api logs once (same `id` on every copy):
`{id, delegator_user_id, statement_id, statement_text, question, answer, covered, asked_by}`.
The fill cites no passage on purpose: it is grounded in the person's own statement, which is
named in the rationale and the log, not in retrieved context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext
from libs.shared.prompts import Prompt, PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from libs.shared.state import AgentResult, TranscriptWindow, Turn
from services.agent_runtime_service.a2a import collab
from services.agent_runtime_service.a2a.identity import responder_target
from services.agent_runtime_service.agents.candidates import CANDIDATE_NS
from services.agent_runtime_service.delegation.client import Delegate, DelegationSource

logger = logging.getLogger(__name__)

PROMPT_KEY = "delegated_answer"
QUERY_CLASS = "delegated_answer"
CONFIDENCE = 0.95  # deterministic grounding: a pre-approved statement or a fixed refusal
MAX_ANSWERS_PER_DELEGATE = 20
MAX_QUESTION_CHARS = 500
MAX_ANSWER_CHARS = 300
MAX_RECIPIENTS = 25

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_INTERROGATIVE = re.compile(
    r"^\s*(?:so |and |but |hey |ok |okay )?(?:what|which|who|whom|whose|when|where|why|how|"
    r"is|are|was|were|do|does|did|can|could|would|will|should|shall|may|might|has|have|had)\b",
    re.IGNORECASE,
)


def refusal(delegate: Delegate) -> str:
    return f"{delegate.first_name} didn't pre-approve an answer to that."


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def names_delegate(text: str, delegate: Delegate) -> bool:
    first = re.escape(delegate.first_name.lower())
    if not first:
        return False
    return re.search(rf"\b{first}(?:'s)?\b", text.lower()) is not None


def is_question(text: str) -> bool:
    return "?" in text or _INTERROGATIVE.match(text) is not None


def find_question(window: TranscriptWindow, delegate: Delegate) -> Turn | None:
    """The last turn, not the delegate's own, that asks them something (no model call)."""
    hits = [
        t
        for t in window.turns
        if t.speaker_id != str(delegate.user_id)
        and names_delegate(t.text, delegate)
        and is_question(t.text)
    ]
    return hits[-1] if hits else None


def render_statements(delegate: Delegate) -> str:
    return "\n".join(f"[{i + 1}] {s.text}" for i, s in enumerate(delegate.statements))


def parse_verdict(text: str, delegate: Delegate) -> tuple[bool, int | None, str]:
    """(addressed, statement index or None, answer). Fails closed to "not covered"."""
    body = _FENCE.sub("", text.strip()).strip()
    obj: dict[str, Any] | None = None
    for candidate in (body, *(m.group(0) for m in [_OBJECT.search(body)] if m)):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            obj = data
            break
    if obj is None:
        return True, None, ""
    addressed = obj.get("addressed") is not False
    raw = obj.get("statement")
    index: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= len(delegate.statements):
        index = raw - 1
    answer = obj.get("answer")
    answer = " ".join(answer.split())[:MAX_ANSWER_CHARS] if isinstance(answer, str) else ""
    if index is None or not answer:
        return addressed, None, ""
    return addressed, index, answer


@dataclass
class _MeetingState:
    asked: set[str] = field(default_factory=set)
    answers: dict[UUID, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DelegatedAnswer:
    delegate: Delegate
    question: str
    asked_by: UUID | None
    statement_index: int | None
    answer: str

    @property
    def covered(self) -> bool:
        return self.statement_index is not None


class DelegatedAnswerer:
    def __init__(
        self,
        loader: PromptLoader,
        router: LiteLLMRouter,
        source: DelegationSource,
        org_id: UUID,
    ) -> None:
        self._router = router
        self._source = source
        self.prompt: Prompt = loader.load(PROMPT_KEY, loader.resolve(PROMPT_KEY, org_id))
        self._state = _MeetingState()
        self.blocked = False

    async def answers(
        self, window: TranscriptWindow, org: CompanyContext | None = None
    ) -> list[DelegatedAnswer]:
        try:
            delegates = await self._source.delegates(window.org_id, window.meeting_id)
        except Exception:  # noqa: BLE001 - no statements this window: nobody answers
            logger.warning("delegates unavailable meeting=%s", window.meeting_id)
            return []
        never = NeverRecordPolicy.from_org(org) if org is not None else None
        out: list[DelegatedAnswer] = []
        for delegate in delegates:
            turn = find_question(window, delegate)
            if turn is None:
                continue
            question = " ".join(turn.text.split())[:MAX_QUESTION_CHARS]
            key = f"{delegate.user_id}:{_norm(question)}"
            if key in self._state.asked:
                continue
            if self._state.answers.get(delegate.user_id, 0) >= MAX_ANSWERS_PER_DELEGATE:
                continue
            if never is not None and never.hits(question):
                continue
            self._state.asked.add(key)
            verdict = await self._ask(window, delegate, question)
            if verdict is None:
                return out  # budget blocked: no more model calls this window
            addressed, index, answer = verdict
            if not addressed:
                continue
            self._state.answers[delegate.user_id] = self._state.answers.get(delegate.user_id, 0) + 1
            out.append(
                DelegatedAnswer(
                    delegate=delegate,
                    question=question,
                    asked_by=_uuid(turn.speaker_id),
                    statement_index=index,
                    answer=answer if index is not None else refusal(delegate),
                )
            )
            break  # one answer per window: see `results`
        return out

    async def _ask(
        self, window: TranscriptWindow, delegate: Delegate, question: str
    ) -> tuple[bool, int | None, str] | None:
        try:
            out = await self._router.complete(
                tier="triage",
                system=self.prompt.system,
                **self.prompt.user_parts(
                    name=delegate.first_name,
                    statements=render_statements(delegate),
                    question=question,
                ),
                org_id=window.org_id,
                user_id=delegate.user_id,  # metered to the person whose agent answers
                max_tokens=200,
                prompt_ref=self.prompt.ref,
            )
        except BudgetBlocked:
            self.blocked = True
            return None
        return parse_verdict(out.text, delegate)

    def results(
        self, window: TranscriptWindow, recipients: list[UUID], answers: list[DelegatedAnswer]
    ) -> list[tuple[UUID, AgentResult]]:
        """One context_fill per live participant for the window's first answer. A participant
        gets at most one result per window (its idempotency key is per window and user), so a
        second delegate asked in the same 30 seconds is answered in a later window only if
        asked again; in practice one question to one absent person is what a window holds."""
        if not answers:
            return []
        a = answers[0]
        delegated_id = str(
            uuid5(CANDIDATE_NS, f"delegated:{window.window_id}:{a.delegate.user_id}")
        )
        statement = (
            a.delegate.statements[a.statement_index] if a.statement_index is not None else None
        )
        block = {
            "id": delegated_id,
            "delegator_user_id": str(a.delegate.user_id),
            "statement_id": str(statement.id) if statement else None,
            "statement_text": statement.text if statement else None,
            "question": a.question,
            "answer": a.answer,
            "covered": a.covered,
            "asked_by": str(a.asked_by) if a.asked_by else None,
        }
        responder = responder_target(a.delegate.user_id)
        content = f"{a.delegate.first_name}'s Myte: {a.answer}"
        rationale = (
            f'Pre-approved by {a.delegate.first_name} before the meeting: "{statement.text}"'
            if statement
            else f"{a.delegate.first_name} pre-approved answers only on other topics."
        )
        out: list[tuple[UUID, AgentResult]] = []
        for requester in [r for r in recipients if r != a.delegate.user_id][:MAX_RECIPIENTS]:
            entries = collab.number(
                [
                    collab.entry(
                        hop="peer",
                        requester=requester,
                        responder=responder,
                        query=a.question,
                        query_class=QUERY_CLASS,
                        target_scope=None,
                        result="answered" if a.covered else "no_answer",
                        success=a.covered,
                        latency_ms=0,
                    )
                ],
                window.window_id,
                requester,
            )
            fill = {
                "id": entries[0]["id"],
                "content": content,
                "confidence": CONFIDENCE,
                "surfaced": True,
                "rationale": rationale,
                "hop": "peer",
                "responder": responder,
                "citations": [],
            }
            payload = collab.payload(
                query=a.question,
                query_class=QUERY_CLASS,
                collaborations=entries,
                fill=fill,
                model_tier="triage",
                prompt=self.prompt.ref,
            )
            payload["delegated"] = block
            out.append(
                (
                    requester,
                    AgentResult(
                        kind=collab.KIND,
                        content=content,
                        relevance_score=1.0,
                        confidence=CONFIDENCE,
                        source_chunk_ids=window.chunk_ids,
                        payload=payload,
                    ),
                )
            )
        return out


def _uuid(raw: str) -> UUID | None:
    try:
        return UUID(raw)
    except ValueError:
        return None
