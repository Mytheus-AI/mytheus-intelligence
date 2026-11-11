"""The decision-conflict check (A1 "Decisions that follow you"): which confirmed decisions does a
piece of text from anywhere (a Slack message, an email draft, a doc paragraph, a ticket, a draft
Claude is writing) contradict or reopen?

`POST /internal/decisions/check` on rag-service (`main.py`), called only by core-api, which owns
the user route (`POST /api/v1/decisions/check`), the Slack path, dismissals and challenges. The
check runs AS the person whose text it is (`X-Mytheus-User`): retrieval is participant-scoped
(ADR-0033 D1), so nobody is ever told about a decision from a meeting they did not attend.

Pipeline, cheapest exit first:

1. Too short (fewer than `MIN_WORDS` words): `skipped / too_short`, nothing called.
2. Never record (ADR-0039 C2): the org's never-record topics come from the company brain with
   `get_strict`; unreadable -> `NeverRecordUnavailable` (503: text we cannot screen is never
   checked). A topic named in the text -> `skipped / never_record`, no retrieval, no model.
3. Hybrid retrieval over the org's confirmed decisions, two arms in parallel, fused with RRF
   (k=60) to `MAX_CANDIDATES`:
   - the `RAGService` pipeline with `source="decisions"`, `generate=false` and expansion off on
     `text[:1000]`: the vector arm over `decisions__{org_id}`, the graph arm, the rerank;
   - the lexical arm, core-api's FTS over confirmed decisions (`CoreDecisionLexical`).
   A candidate naming a never-record topic is dropped. None left -> `ok / no_candidates`.
4. ONE extract-tier judge call through the router (budget-gated, metered to the person,
   PII-redacted on egress), prompt `decision_conflict_check@vN`. `BudgetBlocked` ->
   `degraded / budget` with no conflicts; a throttled tenant is answered by the triage tier and
   marked `degraded / budget` too.
5. Only confidence >= `MIN_CONFIDENCE` (0.7, the house "surface" band) survives, best first, at
   most `max_conflicts`, one per decision. `quote` must be in the text (case and whitespace
   normalised); otherwise the text's sentence with the most words in common with the decision
   stands in. URLs are scrubbed from `quote` and `why`; a conflict whose quote, why or decision
   names a never-record topic is dropped.

Nothing is stored here: the AI tier never writes Postgres. core-api records challenges.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from libs.shared.never_record import NeverRecordPolicy
from libs.shared.org_context import CompanyContext, OrgContextUnavailable
from libs.shared.prompts import PromptLoader
from libs.shared.router import EXTRACTION_EFFORT, BudgetBlocked
from services.rag_service.contracts import RagQueryRequest, RagResponse
from services.rag_service.domain import Caller, LexicalDecision
from services.rag_service.fusion import rrf

logger = logging.getLogger(__name__)

PROMPT_KEY = "decision_conflict_check"
MIN_WORDS = 4
MIN_CONFIDENCE = 0.7
MAX_CANDIDATES = 8
LEXICAL_LIMIT = 10
RETRIEVAL_CHARS = 1000
MAX_QUOTE = 300
MAX_WHY = 240
MAX_CANDIDATE_TEXT = 600
JUDGE_MAX_TOKENS = 2000

Status = Literal["ok", "skipped", "degraded"]
Reason = Literal["too_short", "never_record", "budget", "no_candidates"]
Relation = Literal["contradicts", "reopens"]
Source = Literal["slack", "extension", "mcp", "app"]


# --- the wire contract -----------------------------------------------------------------------


class CheckContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place_label: str | None = Field(default=None, max_length=120)
    channel: str | None = Field(default=None, max_length=120)
    topic: str | None = Field(default=None, max_length=200)
    author: str | None = Field(default=None, max_length=120)


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    source: Source = "app"
    context: CheckContext = Field(default_factory=CheckContext)
    max_conflicts: int = Field(default=3, ge=1, le=5)


class Conflict(BaseModel):
    decision_id: UUID
    meeting_id: UUID | None = None
    decision_content: str
    meeting_title: str | None = None
    decided_at: str | None = None  # YYYY-MM-DD
    quote: str
    relation: Relation
    confidence: float
    why: str


class CheckResponse(BaseModel):
    status: Status
    reason: Reason | None = None
    candidates: int = 0
    prompt: str | None = None
    conflicts: list[Conflict] = Field(default_factory=list)


# --- ports ---------------------------------------------------------------------------------


class DecisionRetriever(Protocol):
    """The `RAGService` (or a fake): `query(caller, req)`."""

    async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse: ...


class LexicalDecisions(Protocol):
    async def search(
        self, org_id: UUID, user_id: UUID, query: str, limit: int = 10
    ) -> list[LexicalDecision]: ...


class OrgContexts(Protocol):
    async def get_strict(self, org_id: UUID) -> CompanyContext: ...


class Judge(Protocol):
    """The LiteLLM router (budget gate, metering, PII redaction) or a test double."""

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
        effort: Any = None,
        cached_prefix: str = "",
    ) -> Any: ...


class NeverRecordUnavailable(RuntimeError):
    """The org's never-record list could not be read: the check must not run (503)."""


# --- helpers -------------------------------------------------------------------------------

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = frozenset(
    "a an and are as at be but by for from has have i in is it of on or our so that the this "
    "to was we will with you your not do does should would can could".split()
)


@dataclass
class _Candidate:
    decision_id: UUID
    content: str
    rationale: str | None = None
    meeting_id: UUID | None = None
    meeting_title: str | None = None
    decided_at: str | None = None


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def scrub(text: str, limit: int) -> str:
    out = " ".join(_URL.sub("", text).split())
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _day(raw: Any) -> str | None:
    if raw is None:
        return None
    value = raw.isoformat() if hasattr(raw, "isoformat") else str(raw)
    value = value.strip()
    return value[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", value) else None


def split_decision_text(text: str) -> tuple[str, str | None]:
    """The records route renders a decision as `content Rationale: rationale`."""
    content, sep, rationale = text.partition(" Rationale: ")
    return content.strip(), (rationale.strip() or None) if sep else None


def find_quote(text: str, quote: str, decision: str) -> str:
    """`quote` exactly as it appears in `text` when it does (case and whitespace normalised);
    otherwise the sentence of `text` sharing the most content words with `decision`."""
    q = _norm(quote.strip().strip('"').strip("“”"))
    if q:
        flat = " ".join(text.split())
        at = flat.lower().find(q)
        if at >= 0:
            return flat[at : at + len(q)]
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()] or [text.strip()]
    target = {w for w in words(decision) if w not in _STOP}

    def overlap(s: str) -> int:
        return len({w for w in words(s) if w not in _STOP} & target)

    return " ".join(max(sentences, key=overlap).split())


def parse_judge(raw: str, n: int) -> list[tuple[int, Relation, float, str, str]]:
    """(0-based candidate, relation, confidence, quote, why), best per candidate. Anything
    malformed is skipped, never guessed."""
    body = raw.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body[body.find("{") :] if "{" in body else body
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        obj = json.loads(body[start : end + 1])
    except ValueError:
        return []
    items = obj.get("conflicts") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    best: dict[int, tuple[int, Relation, float, str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index, conf = item.get("candidate"), item.get("confidence")
        relation = item.get("relation")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= n:
            continue
        if isinstance(conf, bool) or not isinstance(conf, int | float):
            continue
        if relation not in ("contradicts", "reopens"):
            continue
        quote, why = item.get("quote"), item.get("why")
        entry = (
            index - 1,
            relation,
            max(0.0, min(1.0, float(conf))),
            quote if isinstance(quote, str) else "",
            why if isinstance(why, str) else "",
        )
        if index - 1 not in best or entry[2] > best[index - 1][2]:
            best[index - 1] = entry
    return sorted(best.values(), key=lambda e: -e[2])


def render_context(source: str, ctx: CheckContext) -> str:
    where = {
        "slack": "a Slack message",
        "extension": "a draft in the person's browser",
        "mcp": "a draft an AI assistant is writing",
        "app": "text written in Mytheus",
    }.get(source, "a piece of writing")
    parts = [where]
    if ctx.place_label:
        parts.append(f"place: {scrub(ctx.place_label, 120)}")
    if ctx.channel:
        parts.append(f"channel: {scrub(ctx.channel, 120)}")
    if ctx.topic:
        parts.append(f"topic: {scrub(ctx.topic, 200)}")
    return "; ".join(parts)


def render_candidates(candidates: Sequence[_Candidate]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        meta = [m for m in (c.meeting_title and f'meeting "{c.meeting_title}"', c.decided_at) if m]
        head = f"[{i}]" + (f" ({', '.join(meta)})" if meta else "")
        body = c.content + (f" Why: {c.rationale}" if c.rationale else "")
        lines.append(f"{head} {scrub(body, MAX_CANDIDATE_TEXT)}")
    return "\n".join(lines)


# --- the service ---------------------------------------------------------------------------


class DecisionCheckService:
    def __init__(
        self,
        *,
        retriever: DecisionRetriever,
        lexical: LexicalDecisions | None,
        org_context: OrgContexts,
        judge: Judge,
        prompts: PromptLoader,
    ) -> None:
        self._retriever = retriever
        self._lexical = lexical
        self._org = org_context
        self._judge = judge
        self._prompts = prompts

    async def check(self, caller: Caller, req: CheckRequest) -> CheckResponse:
        assert caller.user_id is not None, "the check always runs as one person"
        text = req.text.strip()
        if len(words(text)) < MIN_WORDS:
            return CheckResponse(status="skipped", reason="too_short")
        try:
            org = await self._org.get_strict(caller.org_id)
        except OrgContextUnavailable as exc:
            raise NeverRecordUnavailable(str(caller.org_id)) from exc
        never = NeverRecordPolicy.from_org(org)
        if never.hits(text, req.context.topic, req.context.channel, req.context.place_label):
            return CheckResponse(status="skipped", reason="never_record")

        candidates = [
            c
            for c in await self._retrieve(caller, text)
            if not never.hits(c.content, c.rationale, c.meeting_title)
        ]
        if not candidates:
            return CheckResponse(status="ok", reason="no_candidates")

        prompt = self._prompts.load(PROMPT_KEY, self._prompts.resolve(PROMPT_KEY, caller.org_id))
        try:
            out = await self._judge.complete(
                tier="extract",
                system=prompt.system,
                **prompt.user_parts(
                    context=render_context(req.source, req.context),
                    text=text,
                    candidates=render_candidates(candidates),
                ),
                org_id=caller.org_id,
                user_id=caller.user_id,  # the person asked: metered to them
                max_tokens=JUDGE_MAX_TOKENS,
                effort=EXTRACTION_EFFORT,
                prompt_ref=prompt.ref,
            )
        except BudgetBlocked:
            return CheckResponse(
                status="degraded", reason="budget", candidates=len(candidates), prompt=prompt.ref
            )

        conflicts: list[Conflict] = []
        for index, relation, confidence, quote, why in parse_judge(out.text, len(candidates)):
            if confidence < MIN_CONFIDENCE:
                continue
            c = candidates[index]
            exact = scrub(find_quote(text, quote, c.content), MAX_QUOTE)
            reason = scrub(why, MAX_WHY) or "This goes against a decision the team confirmed."
            if not exact or never.hits(exact, reason):
                continue
            conflicts.append(
                Conflict(
                    decision_id=c.decision_id,
                    meeting_id=c.meeting_id,
                    decision_content=c.content,
                    meeting_title=c.meeting_title,
                    decided_at=c.decided_at,
                    quote=exact,
                    relation=relation,
                    confidence=round(confidence, 3),
                    why=reason,
                )
            )
            if len(conflicts) >= req.max_conflicts:
                break
        throttled = getattr(out, "tier", "extract") == "triage"
        return CheckResponse(
            status="degraded" if throttled else "ok",
            reason="budget" if throttled else None,
            candidates=len(candidates),
            prompt=prompt.ref,
            conflicts=conflicts,
        )

    async def _retrieve(self, caller: Caller, text: str) -> list[_Candidate]:
        assert caller.user_id is not None
        query = text[:RETRIEVAL_CHARS]
        vector_task = self._vector(caller, query)
        lexical_task = self._lexical_arm(caller, query)
        vector, lexical = await asyncio.gather(vector_task, lexical_task)
        pool: dict[str, _Candidate] = {}
        for c in [*lexical, *vector]:  # lexical first: its fields are the canonical row
            key = str(c.decision_id)
            if key not in pool:
                pool[key] = c
            else:
                kept = pool[key]
                pool[key] = replace(
                    kept,
                    meeting_id=kept.meeting_id or c.meeting_id,
                    meeting_title=kept.meeting_title or c.meeting_title,
                    decided_at=kept.decided_at or c.decided_at,
                    rationale=kept.rationale or c.rationale,
                )
        fused = rrf(
            [[str(c.decision_id) for c in vector], [str(c.decision_id) for c in lexical]],
            top=MAX_CANDIDATES,
        )
        return [pool[k] for k, _ in fused if k in pool]

    async def _vector(self, caller: Caller, query: str) -> list[_Candidate]:
        """The RAG pipeline, decisions only: vector + graph arms and the rerank."""
        try:
            res = await self._retriever.query(
                caller, RagQueryRequest(query=query, generate=False, source="decisions")
            )
        except Exception:  # noqa: BLE001 - one arm down: the lexical arm still answers
            logger.warning("decision check: the vector arm failed")
            return []
        out: list[_Candidate] = []
        for p in res.answer_passages:
            if p.citation.kind != "decision" or p.citation.decision_id is None:
                continue
            content, rationale = split_decision_text(p.text)
            out.append(
                _Candidate(
                    decision_id=p.citation.decision_id,
                    content=content,
                    rationale=rationale,
                    meeting_id=p.citation.meeting_id,
                    meeting_title=p.meeting_title,
                    decided_at=_day(p.occurred_at),
                )
            )
        return out

    async def _lexical_arm(self, caller: Caller, query: str) -> list[_Candidate]:
        if self._lexical is None:
            return []
        assert caller.user_id is not None
        try:
            hits = await self._lexical.search(caller.org_id, caller.user_id, query, LEXICAL_LIMIT)
        except Exception:  # noqa: BLE001
            logger.warning("decision check: the lexical arm failed")
            return []
        return [
            _Candidate(
                decision_id=h.id,
                content=h.content,
                rationale=h.rationale,
                meeting_id=h.meeting_id,
                meeting_title=h.meeting_title,
                decided_at=_day(h.decided_at),
            )
            for h in hits
        ]
