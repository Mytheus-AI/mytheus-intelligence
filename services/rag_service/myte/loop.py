"""The Ask Myte loop (mytheus-docs ADR-0047 D2-D5, D9, D10).

One answer:
1. The tool catalogue: bff's MCP tools (cached five minutes per process) minus the writes, plus
   `propose_action`, `show_card` and `search_help` (`tools.py`).
2. The system prompt `myte_answer@vN` (versioned, tenant-pinnable, `prompts/`), and its
   per-call context block (the persona, today in the person's time zone, scope, page).
3. Up to 8 model rounds on tier `extract` (Sonnet 5.5, no thinking: `between_tools`)
   through `LiteLLMRouter.converse()`: budget gate, redaction, caching and one metering record
   per round. Text deltas stream live. A round's tool calls run concurrently, as the person,
   within 45 s of tool time per answer; each is a `step` (running, then done or failed) and
   every item it returns is a numbered `source`. The last round, or a round after the tool
   time ran out, is asked to answer without tools.
4. One triage call (Haiku 5.5): three follow-up questions and, when asked, a title.
5. `done {abstained, degraded, cited}`.

`abstained` is decided from what happened, not from a model flag: no registered source is cited
in the text and nothing was proposed or shown. `degraded` means the answer is less than it
could have been: a lookup failed or timed out, the tool time or the rounds ran out, the budget
throttled the answer to triage, or the answer stopped at `max_tokens`.

Errors end the stream with one `error` event (and no `done`): `budget` (the workspace is at its
hard cap), `session` (bff refused the person's token), `not_connected` (no model access),
`unavailable` (bff or the model provider could not be reached). Whatever streamed before an
error stays with the person (the UI keeps it and offers Retry).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from libs.shared.json_answer import json_object
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter, ToolCall
from services.rag_service.domain import Caller
from services.rag_service.myte.contracts import ErrorReason, MyteRequest
from services.rag_service.myte.mcp import (
    McpClient,
    McpSessionError,
    McpUnavailable,
    ToolCatalog,
    bff_url,
)
from services.rag_service.myte.sources import Draft, SourceRegistry, map_output, remap_citations
from services.rag_service.myte.tools import (
    CARD,
    HELP,
    PROPOSE,
    model_tools,
    schema_properties,
    search_help,
    step_label,
)

logger = logging.getLogger(__name__)

ANSWER_PROMPT = "myte_answer"
FOLLOWUPS_PROMPT = "myte_followups"

ERROR_MESSAGES: dict[str, str] = {
    "budget": (
        "Your workspace reached its AI limit for now. Answers come back when the limit resets, "
        "or when an admin raises it."
    ),
    "not_connected": (
        "Ask Myte isn't connected on this workspace yet. An admin can turn it on in Workspace "
        "settings."
    ),
    "unavailable": "I can't reach Mytheus right now. Try again in a moment.",
    "session": "Your session ended. Refresh the page and ask again.",
    "invalid": "That question could not be read. Try again.",
}
_FORBIDDEN = "Your account can't use Ask Myte in this workspace. An admin can change your role."
_OUT_OF_ROUNDS = "I ran out of steps before I could finish that. Try a narrower question."
_EMPTY = "I couldn't put an answer together that time. Try asking again."
_PROPOSED = (
    "Shown to the person as a suggestion card with Confirm, Edit and Not now. Nothing is saved "
    "until they confirm. Tell them in one short sentence what you suggested; do not say it is "
    "done."
)
_CARD_DETAIL_KINDS = frozenset({"meeting", "decision", "capacity"})
_CARD_DETAIL_MAX = 32_000  # bytes of JSON


@dataclass(frozen=True)
class Event:
    name: str
    data: dict[str, Any]


class MyteError(Exception):
    def __init__(self, reason: ErrorReason, message: str | None = None) -> None:
        super().__init__(reason)
        self.reason: ErrorReason = reason
        self.message = message or ERROR_MESSAGES[reason]


@dataclass(frozen=True)
class LoopConfig:
    max_rounds: int = 8
    tool_budget_s: float = 45.0
    max_tokens: int = 2048
    # None = no thinking (`between_tools`): owner's call, thinking costs too many tokens.
    effort: str | None = None
    followup_max_tokens: int = 400
    max_proposals: int = 3
    max_cards: int = 3


Emit = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class _Answer:
    emit: Emit
    deadline: float
    registry: SourceRegistry = field(default_factory=SourceRegistry)
    streamed: list[str] = field(default_factory=list)
    round_has_text: bool = False
    steps: int = 0
    degraded: bool = False
    out_of_time: bool = False
    session_lost: McpSessionError | None = None
    proposals: list[dict[str, Any]] = field(default_factory=list)
    cards: int = 0

    @property
    def text(self) -> str:
        return "".join(self.streamed)

    async def delta(self, piece: str) -> None:
        if not self.round_has_text:
            self.round_has_text = True
            if self.streamed and not self.text.endswith("\n"):
                piece = "\n\n" + piece
        self.streamed.append(piece)
        await self.emit("delta", {"text": piece})

    def next_step(self) -> str:
        self.steps += 1
        return f"step-{self.steps}"

    def known_meeting_ids(self) -> set[str]:
        return {s.draft.meeting_id for s in self.registry.sources if s.draft.meeting_id}


class MyteLoop:
    def __init__(
        self,
        router: LiteLLMRouter,
        prompts: PromptLoader,
        http: httpx.AsyncClient,
        catalog: ToolCatalog | None = None,
        base_url: str | None = None,
        config: LoopConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._router = router
        self._prompts = prompts
        self._http = http
        self._catalog = catalog or ToolCatalog()
        self._base_url = base_url or bff_url()
        self._config = config or LoopConfig()
        self._clock = clock or (lambda: datetime.now(UTC))

    async def stream(
        self, req: MyteRequest, caller: Caller, authorization: str
    ) -> AsyncIterator[Event]:
        """The answer as events. Closing the iterator (the client went away) cancels the work."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def emit(name: str, data: dict[str, Any]) -> None:
            await queue.put(Event(name, data))

        task = asyncio.create_task(self._guarded(req, caller, authorization, emit, queue))
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _guarded(
        self,
        req: MyteRequest,
        caller: Caller,
        authorization: str,
        emit: Emit,
        queue: asyncio.Queue[Event | None],
    ) -> None:
        try:
            await self._run(req, caller, authorization, emit)
        except MyteError as exc:
            await emit("error", {"reason": exc.reason, "message": exc.message})
        except BudgetBlocked:
            await emit("error", {"reason": "budget", "message": ERROR_MESSAGES["budget"]})
        except McpSessionError as exc:
            message = _FORBIDDEN if exc.status == 403 else ERROR_MESSAGES["session"]
            await emit("error", {"reason": "session", "message": message})
        except McpUnavailable as exc:
            logger.warning("myte: bff unavailable org=%s: %s", caller.org_id, exc)
            await emit("error", {"reason": "unavailable", "message": ERROR_MESSAGES["unavailable"]})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason: ErrorReason = (
                "not_connected"
                if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")
                else "unavailable"
            )
            logger.exception("myte: answer failed org=%s (%s)", caller.org_id, reason)
            await emit("error", {"reason": reason, "message": ERROR_MESSAGES[reason]})
        finally:
            await queue.put(None)

    # --- the answer ------------------------------------------------------------------------

    async def _run(self, req: MyteRequest, caller: Caller, authorization: str, emit: Emit) -> None:
        cfg = self._config
        state = _Answer(emit=emit, deadline=time.monotonic() + cfg.tool_budget_s)
        mcp = McpClient(self._http, authorization, self._base_url)
        tools = model_tools(await self._catalog.get(mcp))
        prompt = self._prompts.load(
            ANSWER_PROMPT, self._prompts.resolve(ANSWER_PROMPT, caller.org_id)
        )
        context = prompt.render_user(**self._context(req))
        messages = _messages(req)
        effort: Any = cfg.effort
        for round_no in range(cfg.max_rounds):
            last = round_no == cfg.max_rounds - 1 or state.out_of_time
            state.round_has_text = False
            result = await self._router.converse(
                tier="extract",
                system=prompt.system,
                system_context=context,
                messages=messages,
                tools=tools,
                tool_choice="none" if last else None,
                org_id=caller.org_id,
                user_id=caller.user_id,
                max_tokens=cfg.max_tokens,
                effort=effort,
                on_delta=state.delta,
                prompt_ref=prompt.ref,
            )
            if result.tier != "extract" or result.truncated:
                state.degraded = True
            if not result.tool_calls:
                break
            if last:
                # Asked for no tools and still called one: answer with what there is.
                state.degraded = True
                if not state.text.strip():
                    await state.delta(_OUT_OF_ROUNDS)
                break
            messages.append(result.assistant_message())
            # Every lookup of the round shows as running at once, then they run concurrently.
            plans = [await self._start(call, state, tools, req) for call in result.tool_calls]
            outputs = await asyncio.gather(
                *(
                    self._tool(call, plan, state, mcp, tools, req)
                    for call, plan in zip(result.tool_calls, plans, strict=True)
                )
            )
            if state.session_lost is not None:
                raise state.session_lost
            messages.extend(
                {"role": "tool", "tool_call_id": call.id, "content": out}
                for call, out in zip(result.tool_calls, outputs, strict=True)
            )
        if not state.text.strip():
            state.degraded = True
            await state.delta(_EMPTY)
        answer = state.text
        cited = state.registry.cited(answer)
        await self._followups(req, answer, caller, emit)
        await emit(
            "done",
            {
                "abstained": not cited and not state.proposals and state.cards == 0,
                "degraded": state.degraded,
                "cited": cited,
            },
        )

    def _context(self, req: MyteRequest) -> dict[str, str]:
        tz_name = req.timezone or ""
        try:
            tz: Any = ZoneInfo(tz_name) if tz_name else UTC
        except (ZoneInfoNotFoundError, ValueError):
            tz, tz_name = UTC, ""
        now = self._clock().astimezone(tz)
        today = f"{now:%A}, {now:%B} {now.day}, {now.year}, {now:%H:%M}"
        persona = req.persona
        scope = req.scope
        if scope.kind == "meeting" and scope.meeting_id:
            title = (
                f' ("{req.page.title}")'
                if req.page.kind == "meeting" and req.page.id == scope.meeting_id and req.page.title
                else (f' ("{scope.label.strip()}")' if scope.label and scope.label.strip() else "")
            )
            scope_line = (
                f"Scope: one meeting, meeting_id={scope.meeting_id}{title}. Questions are about "
                "this meeting unless they say otherwise: start with get_meeting_outcome and pass "
                "meeting_id to every tool that takes it."
            )
        elif scope.kind == "person" and scope.person_id:
            who = (scope.label or "").strip()
            named = (
                f'"{who}" (person_id={scope.person_id})' if who else f"person_id={scope.person_id}"
            )
            scope_line = (
                f"Scope: one person, {named}. Questions are about this person: he, she, they, "
                f"his, her, their and \"this person\" mean {who or 'them'}, so never ask who is "
                "meant. To answer: find_people with their name for role, team and department; "
                "search_meetings and ask_meetings with their name for what they said, decided "
                "or took on in meetings the asker shared with them; get_decision_record for "
                "decisions they are part of. You cannot see their private tasks or promises: "
                "answer from what shared meetings show and say so in one short clause."
            )
        elif scope.kind == "as_of" and scope.as_of:
            scope_line = (
                f"Scope: as of {scope.as_of.isoformat()}. Answer as things stood at the end of "
                f"that date: prefer company_as_of with date={scope.as_of.isoformat()}, and say "
                "that it is as of then."
            )
        else:
            scope_line = "Scope: everything they can see."
        page = req.page
        if page.kind:
            named = f' for "{page.title}"' if page.title else ""
            ident = f" ({page.kind}_id={page.id})" if page.id else ""
            page_line = (
                f"They are on the {page.kind} page{named}{ident}. "
                f'"This {page.kind}" means that one.'
            )
        else:
            page_line = "They are not on a specific page."
        return {
            "name": persona.name.strip() or "Myte",
            "personality": persona.personality,
            "instructions": " ".join((persona.instructions or "").split()) or "none",
            "today": today,
            "timezone": tz_name or "UTC; their own time zone is unknown",
            "scope": scope_line,
            "page": page_line,
        }

    # --- tools -----------------------------------------------------------------------------

    def _scoped_args(
        self, name: str, args: dict[str, Any], req: MyteRequest, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        props = schema_properties(tools, name)
        out = dict(args)
        scope = req.scope
        if scope.kind == "meeting" and scope.meeting_id and "meeting_id" in props:
            out.setdefault("meeting_id", str(scope.meeting_id))
        if scope.kind == "as_of" and scope.as_of and name == "company_as_of":
            out.setdefault("date", scope.as_of.isoformat())
        if req.timezone and "timezone" in props:
            out.setdefault("timezone", req.timezone)
        return out

    async def _start(
        self, call: ToolCall, state: _Answer, tools: list[dict[str, Any]], req: MyteRequest
    ) -> tuple[dict[str, Any], str, str] | None:
        """A lookup's arguments and its `step`, announced as running. None for the pseudo-tools
        (they look nothing up) and for arguments that did not parse."""
        if call.parse_error or call.name in (PROPOSE, CARD):
            return None
        args = self._scoped_args(call.name, call.arguments, req, tools)
        step = state.next_step()
        label = step_label(call.name, args)
        await state.emit("step", {"id": step, "label": label, "state": "running"})
        return args, step, label

    async def _tool(
        self,
        call: ToolCall,
        plan: tuple[dict[str, Any], str, str] | None,
        state: _Answer,
        mcp: McpClient,
        tools: list[dict[str, Any]],
        req: MyteRequest,
    ) -> str:
        if call.parse_error:
            return "Error: the arguments were not valid JSON. Call the tool again with an object."
        if call.name == PROPOSE:
            return await self._propose(call.arguments, state, req)
        if call.name == CARD:
            return await self._card(call.arguments, state)
        assert plan is not None
        args, step, label = plan

        async def failed(message: str, *, degrade: bool = True) -> str:
            state.degraded = state.degraded or degrade
            await state.emit("step", {"id": step, "label": label, "state": "failed"})
            return f"Error: {message}"

        header = ""
        if call.name == HELP:
            articles = search_help(str(args.get("query") or ""))
            drafts = [
                Draft(
                    kind="document",
                    title=f"Help: {a['title']}",
                    text=a["text"],
                    ref_id=a["id"],
                    raw=a,
                )
                for a in articles
            ]
            text = "" if drafts else "Nothing in the Mytheus help matches that."
        elif not _has_tool(tools, call.name):
            return await failed(f"there is no tool called {call.name}.", degrade=False)
        else:
            remaining = state.deadline - time.monotonic()
            if remaining <= 0:
                state.out_of_time = True
                return await failed(
                    "out of time for lookups. Answer with what you have and say what you could "
                    "not check."
                )
            try:
                out = await asyncio.wait_for(
                    mcp.call_tool(call.name, args, remaining), timeout=remaining
                )
            except TimeoutError:
                state.out_of_time = True
                return await failed(
                    "this lookup took too long. Answer with what you have and say what you "
                    "could not check."
                )
            except McpSessionError as exc:
                state.session_lost = exc
                return await failed("the session ended.")
            except McpUnavailable as exc:
                logger.warning("myte: %s unavailable: %s", call.name, exc)
                return await failed(
                    "Mytheus did not answer this lookup. Say what you could not check."
                )
            if out.is_error:
                return await failed(out.text or "the lookup failed.", degrade=False)
            drafts = map_output(call.name, args, out.structured, out.text)
            text = out.text
            if call.name == "ask_meetings" and isinstance(out.structured, dict):
                header = str(out.structured.get("answer") or "")
        lines: list[str] = []
        aliases: dict[int, int] = {}
        for draft in drafts:
            source, send = state.registry.add(draft)
            if source is None:
                lines.append("(more results were left out: this answer has enough sources)")
                break
            if send:
                await state.emit("source", source.event())
            if draft.alias is not None:
                aliases[draft.alias] = source.n
            lines.append(source.line())
        await state.emit("step", {"id": step, "label": label, "state": "done"})
        if header:
            header = "Summary from these sources: " + remap_citations(header, aliases)
        if lines:
            return "\n".join(([header] if header else []) + lines)
        return text.strip() or "No results."

    async def _propose(self, args: dict[str, Any], state: _Answer, req: MyteRequest) -> str:
        kind = args.get("kind")
        if kind not in ("task", "promise", "attendance", "agenda_item"):
            return "Error: kind must be task, promise, attendance or agenda_item."
        title = " ".join(str(args.get("title") or "").split())[:200]
        if not title:
            return "Error: a title is required."
        proposal: dict[str, Any] = {"id": str(uuid.uuid4()), "kind": kind, "title": title}
        if args.get("due_date"):
            try:
                proposal["due_date"] = date.fromisoformat(str(args["due_date"])[:10]).isoformat()
            except ValueError:
                return "Error: due_date must be a date like 2026-11-20."
        if args.get("meeting_id"):
            try:
                meeting = str(uuid.UUID(str(args["meeting_id"])))
            except ValueError:
                return "Error: meeting_id must be the id of a meeting from your sources."
            allowed = state.known_meeting_ids() | {
                str(m)
                for m in (req.scope.meeting_id, req.page.id if req.page.kind == "meeting" else None)
                if m
            }
            if meeting not in allowed:
                return "Error: use a meeting_id from your sources; look the meeting up first."
            proposal["meeting_id"] = meeting
        if kind in ("attendance", "agenda_item") and "meeting_id" not in proposal:
            return f"Error: {kind} needs the meeting_id of the meeting; look it up first."
        if kind == "attendance":
            if args.get("choice") not in ("attend", "agent", "skip"):
                return "Error: attendance needs choice: attend, agent or skip."
            proposal["choice"] = args["choice"]
        note = " ".join(str(args.get("note") or "").split())[:500]
        if note:
            proposal["note"] = note
        same = {k: v for k, v in proposal.items() if k != "id"}
        if any({k: v for k, v in p.items() if k != "id"} == same for p in state.proposals):
            return "Already shown to the person."
        if len(state.proposals) >= self._config.max_proposals:
            return "Error: you have suggested enough changes in this answer."
        state.proposals.append(proposal)
        await state.emit("proposal", proposal)
        return _PROPOSED

    async def _card(self, args: dict[str, Any], state: _Answer) -> str:
        kind = args.get("kind")
        if kind not in ("meeting", "decision", "tasks", "promises", "capacity"):
            return "Error: kind must be meeting, decision, tasks, promises or capacity."
        raw = args.get("source_ns")
        ns: list[int] = []
        for n in raw if isinstance(raw, list) else []:
            if isinstance(n, int) and state.registry.get(n) is not None and n not in ns:
                ns.append(n)
        ns = ns[:20]
        if not ns:
            return "Error: none of those numbers are sources in this answer."
        if state.cards >= self._config.max_cards:
            return "Error: you have shown enough cards in this answer."
        sources = [s for n in ns if (s := state.registry.get(n)) is not None]
        detail = sources[0].draft.raw if kind in _CARD_DETAIL_KINDS else None
        if detail is not None and len(json.dumps(detail, default=str)) > _CARD_DETAIL_MAX:
            detail = None
        state.cards += 1
        await state.emit(
            "card",
            {
                "kind": kind,
                "source_ns": ns,
                "data": {"items": [s.event() for s in sources], "detail": detail},
            },
        )
        return "Card shown beside your answer. Keep your text short."

    # --- follow-ups and title ----------------------------------------------------------------

    async def _followups(self, req: MyteRequest, answer: str, caller: Caller, emit: Emit) -> None:
        obj: dict[str, Any] | None = None
        try:
            prompt = self._prompts.load(
                FOLLOWUPS_PROMPT, self._prompts.resolve(FOLLOWUPS_PROMPT, caller.org_id)
            )
            out = await self._router.complete(
                tier="triage",
                system=prompt.system,
                user=prompt.render_user(
                    want_title="yes" if req.want_title else "no",
                    question=req.question,
                    answer=answer[:4000],
                ),
                org_id=caller.org_id,
                user_id=caller.user_id,
                max_tokens=self._config.followup_max_tokens,
                prompt_ref=prompt.ref,
            )
            obj = json_object(out.text, "followups")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Follow-ups are a nicety: a failure (budget, provider) never fails the answer.
            logger.warning("myte: follow-ups failed org=%s", caller.org_id, exc_info=True)
        items: list[str] = []
        if obj is not None and isinstance(obj.get("followups"), list):
            items = [" ".join(str(x).split())[:120] for x in obj["followups"]]
            items = [x for x in items if x][:3]
        if len(items) == 3:
            await emit("followups", {"items": items})
        if req.want_title:
            title = _clean_title(obj.get("title") if obj else None) or _title_from(req.question)
            await emit("title", {"text": title})


def _has_tool(tools: list[dict[str, Any]], name: str) -> bool:
    return any((t.get("function") or {}).get("name") == name for t in tools)


def _messages(req: MyteRequest) -> list[dict[str, Any]]:
    """History (user/myte) + the question, as alternating user/assistant turns starting with
    the user (consecutive turns of one role are merged)."""
    turns: list[dict[str, Any]] = []
    for m in [*req.history, None]:
        role, content = (
            ("user", req.question)
            if m is None
            else (
                "user" if m.role == "user" else "assistant",
                m.content,
            )
        )
        if not content.strip() or (not turns and role == "assistant"):
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def _clean_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    title = " ".join(value.split()).strip("\"'“” ").rstrip(".")
    if len(title) > 60:
        title = title[:60].rsplit(" ", 1)[0].rstrip(",;:-")
    return title


def _title_from(question: str) -> str:
    title = " ".join(question.split()).rstrip("?.!")
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "..."
    return title or "New question"
