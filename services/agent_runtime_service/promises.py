"""Promise extraction (promise ledger, "Beyond the meeting" Layer 2): a person's own agent reads
one message that person wrote (Slack, or an email they sent) and writes down the promises THEY
made in it, for the person to confirm or dismiss.

`promise_extract@vN`, triage tier (Haiku: one short message, a narrow extraction), through the
shared `LiteLLMRouter`, so every call is budget-gated (`BudgetBlocked` -> `budget_denied`, no
model call), metered (`usage-events`), PII-redacted on egress and cached by prompt ref. The
company's never-record topics are enforced twice, as everywhere in this tier: a message that
names one outright is never sent (`never_record`), and an extracted promise that names one is
dropped. The org context is read strictly: when the company brain cannot be read, the call
fails (the caller retries) rather than guessing that nothing is off limits.

What the model answers is checked, never trusted:
- confidence: the canon gate, only `>= SURFACE_AT` reaches the person (no worklog here);
- quote: must appear in the message (whitespace and case aside), else it is dropped, so a
  source chip never shows words nobody wrote;
- due_date: a real date from the day before the message to a year after it, else null;
- recipient: a number from the people the message mentions (emails never go to the model: the
  number maps back to the email here), else a short free-text label.

This tier writes nothing: the result goes back to the caller (integration-service), which hands
it to core-api, the only writer of `commitments`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Protocol
from uuid import UUID

from libs.shared.json_answer import json_object
from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import EMPTY, CompanyContext
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter

logger = logging.getLogger(__name__)

PROMPT_KEY = "promise_extract"
TIER: Literal["triage"] = "triage"
MAX_TOKENS = 1024
SURFACE_AT = 0.7
MAX_PROMISES = 5
MAX_TEXT = 4000
MAX_PEOPLE = 10
Status = Literal["ok", "budget_denied", "never_record"]


class OrgContexts(Protocol):
    async def get_strict(self, org_id: UUID) -> CompanyContext: ...


@dataclass(frozen=True)
class Person:
    label: str
    email: str | None = None


@dataclass(frozen=True)
class ExtractRequest:
    channel: Literal["slack", "email"]
    text: str
    reference_date: date
    author_name: str | None = None
    people: tuple[Person, ...] = ()
    context: str | None = None


@dataclass(frozen=True)
class ExtractedPromise:
    content: str
    due_hint: str | None
    due_date: date | None
    recipient_label: str | None
    recipient_email: str | None
    quote: str | None
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "due_hint": self.due_hint,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "recipient_label": self.recipient_label,
            "recipient_email": self.recipient_email,
            "quote": self.quote,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ExtractResult:
    status: Status
    prompt: str | None
    promises: tuple[ExtractedPromise, ...]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _clip(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _date(value: Any, ref: date) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        found = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
    if found < ref - timedelta(days=1) or found > ref + timedelta(days=366):
        return None
    return found


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(0.0, min(1.0, float(value)))


def people_block(people: tuple[Person, ...]) -> str:
    if not people:
        return "(nobody)"
    return "\n".join(f"{i}. {p.label}" for i, p in enumerate(people, start=1))


def parse(
    text: str, req: ExtractRequest, policy: NeverRecordPolicy
) -> tuple[ExtractedPromise, ...] | None:
    """The model's answer, checked. None when it is not the expected JSON (a parse failure)."""
    obj = json_object(text, "promises")
    if obj is None or not isinstance(obj.get("promises"), list):
        return None
    message = _norm(req.text)
    out: list[ExtractedPromise] = []
    for raw in obj["promises"]:
        if not isinstance(raw, dict):
            continue
        content = _clip(raw.get("content"), 500)
        confidence = _confidence(raw.get("confidence"))
        if content is None or confidence is None or confidence < SURFACE_AT:
            continue
        quote = _clip(raw.get("quote"), 1000)
        if quote is not None and _norm(quote) not in message:
            quote = None  # grounding: only words the author wrote
        if policy.hits(content, quote):
            continue
        recipient_label: str | None = None
        recipient_email: str | None = None
        ref = raw.get("recipient")
        if isinstance(ref, int) and not isinstance(ref, bool) and 1 <= ref <= len(req.people):
            person = req.people[ref - 1]
            recipient_label, recipient_email = person.label, person.email
        else:
            recipient_label = _clip(raw.get("recipient_label"), 200)
        out.append(
            ExtractedPromise(
                content=content,
                due_hint=_clip(raw.get("due_hint"), 100),
                due_date=_date(raw.get("due_date"), req.reference_date),
                recipient_label=recipient_label,
                recipient_email=recipient_email,
                quote=quote,
                confidence=round(confidence, 3),
            )
        )
        if len(out) == MAX_PROMISES:
            break
    return tuple(out)


class PromiseExtractor:
    def __init__(
        self,
        router: LiteLLMRouter,
        prompts: PromptLoader,
        org_contexts: OrgContexts | None = None,
    ) -> None:
        self._router = router
        self._prompts = prompts
        self._orgs = org_contexts

    async def extract(self, org_id: UUID, req: ExtractRequest) -> ExtractResult:
        org = await self._orgs.get_strict(org_id) if self._orgs is not None else EMPTY
        policy = NeverRecordPolicy.from_org(org)
        text = req.text[:MAX_TEXT]
        if policy.hit(text) or policy.hit(req.context):
            return ExtractResult("never_record", None, ())
        version = self._prompts.resolve(PROMPT_KEY, org_id)
        prompt = self._prompts.load(PROMPT_KEY, version)
        people = req.people[:MAX_PEOPLE]
        user = prompt.render_user(
            author=req.author_name or "the author",
            channel="Slack" if req.channel == "slack" else "email",
            weekday=req.reference_date.strftime("%A"),
            reference_date=req.reference_date.isoformat(),
            people=people_block(people),
            never_record=(
                ", ".join(org.never_record_topics) if org.never_record_topics else "(none)"
            ),
            context=(
                f"Earlier in the thread (context only, never a promise by the author):\n<<<\n"
                f"{req.context[:2000]}\n>>>\n\n"
                if req.context
                else ""
            ),
            text=text,
        )
        try:
            completion = await self._router.complete(
                tier=TIER,
                system=prompt.system,
                user=user,
                org_id=org_id,
                user_id=None,
                max_tokens=MAX_TOKENS,
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            return ExtractResult("budget_denied", prompt.ref, ())
        found = parse(
            completion.text,
            ExtractRequest(
                channel=req.channel,
                text=text,
                reference_date=req.reference_date,
                author_name=req.author_name,
                people=people,
                context=req.context,
            ),
            policy,
        )
        if found is None:
            logger.warning("%s answer did not parse; no promises written", prompt.ref)
            return ExtractResult("ok", prompt.ref, ())
        return ExtractResult("ok", prompt.ref, found)
