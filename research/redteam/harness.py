"""Run the red-team corpus through the REAL agent code under each condition.

ContextGapAgent instances run end to end: the real detector, gate prompt, `classify_query`,
PEP (with a grant table standing in for the PDP), signed AgentRequests over the real `Emitter`
on an in-memory bus (`tests/a2a_helpers.Loopback`), the real `Responder` (re-authorisation,
dedupe, retrieval, class-to-kind filter, passage caps), the related-meeting hop, and the real
synthesis prompt through the real `LiteLLMRouter`. Only retrieval (`rag.SyntheticRag`, ACL
enforced), the PDP (a grant set), the broker and the model transport (`llm.BatchLLM`) are
stand-ins.

DecisionAgent / ActionAgent windows (T4) and ProactiveAgent conflict checks (T3 suppression) go
through those agents' real prompts and parsers.

Every record is one (instance, condition, run): what each requester was shown, the fill, the
citations, the collaboration entries, and the oracle verdicts.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID

import fakeredis

from libs.shared.prompts import PromptLoader
from libs.shared.router import EXTRACT_MAX_TOKENS, SYNTHESIS_EFFORT, LiteLLMRouter
from libs.shared.state import AgentState, TranscriptWindow, Turn
from research.redteam import org
from research.redteam.corpus import ConflictCase, Instance
from research.redteam.llm import RUN, BatchLLM
from research.redteam.rag import SyntheticRag, source_of_citation
from services.agent_runtime_service.a2a import pep as pep_mod
from services.agent_runtime_service.agents.action_agent import ActionAgent
from services.agent_runtime_service.agents.context_gap_agent import (
    ContextGapAgent,
    ContextGapDeps,
    ContextGapSettings,
    Winner,
)
from services.agent_runtime_service.agents.decision_agent import DecisionAgent
from services.agent_runtime_service.agents.proactive_agent import (
    CONFLICT_KEY,
    parse_conflicts,
)
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.a2a_helpers import FakePdp, FakeRelated, Loopback, budget

ALL_KINDS = frozenset({"transcript", "document", "decision", "action"})
ROLES = ("owner", "admin", "member")
CLASSES = ("meeting_context", "decision_lineage", "action_status")
SCOPES = ("self", "same_team", "department", "org")


def template(scope: str, classes: tuple[str, ...] = CLASSES) -> set[tuple[str, str, str]]:
    """identity-tenant-service `POST /agent-authz-scopes/template`: allow every role x class at
    `scope` and every narrower scope."""
    upto = SCOPES[: SCOPES.index(scope) + 1]
    return {(r, q, s) for r in ROLES for q in classes for s in upto}


@dataclass(frozen=True)
class Condition:
    name: str
    grants: frozenset[tuple[str, str, str]] = frozenset(template("org"))
    a2a: bool = True
    class_filter: bool = True
    as_requester: bool = False
    env: tuple[tuple[str, str], ...] = ()  # code flags (the fixes), set while the condition runs
    pins: tuple[tuple[str, str], ...] = ()  # prompt version pins for the org (spotlight off)
    note: str = ""


BASELINES: dict[str, Condition] = {
    "B0": Condition(
        "B0",
        grants=frozenset(template("org")),
        class_filter=False,
        note="no PEP: every grant, no class-to-kind filter",
    ),
    "B1": Condition("B1", note="origin/main, tenant adopted the platform template at org scope"),
    "B1t": Condition(
        "B1t",
        grants=frozenset(template("same_team")),
        note="origin/main, template adopted at same_team scope (narrowest useful)",
    ),
    "B2": Condition("B2", as_requester=True, note="responder retrieves AS THE REQUESTER"),
    "B3": Condition("B3", a2a=False, note="no A2A: requester's own RAG only"),
}


@contextmanager
def applied(cond: Condition) -> Iterator[None]:
    saved_env = {k: os.environ.get(k) for k, _ in cond.env}
    saved_kinds = dict(pep_mod.CLASS_KINDS)
    try:
        for k, v in cond.env:
            os.environ[k] = v
        if not cond.class_filter:
            for k in list(pep_mod.CLASS_KINDS):
                pep_mod.CLASS_KINDS[k] = ALL_KINDS
        yield
    finally:
        pep_mod.CLASS_KINDS.clear()
        pep_mod.CLASS_KINDS.update(saved_kinds)
        for k, old in saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


async def drain(loop: Loopback) -> None:
    """`Loopback.drain` spins without yielding when every task is done but its discard
    callback has not run yet; yield first, and between rounds."""
    await asyncio.sleep(0)
    while loop.tasks:
        await asyncio.gather(*list(loop.tasks), return_exceptions=True)
        await asyncio.sleep(0)


class NoSink:
    def __init__(self) -> None:
        self.units = 0
        self.calls = 0

    async def usage(self, **kw: Any) -> None:
        self.units += int(kw.get("tokens") or 0)
        self.calls += 1


def planted_sources(inst: Instance) -> tuple[org.Source, ...]:
    return tuple(
        org.Source(
            key=p.key,
            kind=p.kind,
            readers=frozenset(p.readers),
            passages=(p.text,),
            meeting=p.meeting or (f"planted-{p.key}" if p.kind != "document" else None),
            author=p.author,
        )
        for p in inst.planted
    )


def window_of(inst: Instance, meeting: UUID, index: int = 0) -> TranscriptWindow:
    turns = []
    for n, (who, text) in enumerate(inst.lines):
        turns.append(
            Turn(
                speaker_id=str(org.USERS[who].id),
                text=text,
                start_ms=n * 5_000,
                end_ms=n * 5_000 + 4_500,
                chunk_ids=[str(org.uid("chunk", inst.id, str(n)))],
            )
        )
    start = index * 30_000
    return TranscriptWindow(org.ORG_ID, meeting, f"{meeting}:{start}", start, start + 30_000, turns)


class RecordingGapAgent(ContextGapAgent):
    """ContextGapAgent that also records what each requester's synthesis was shown."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.shown: dict[UUID, Winner] = {}

    async def _synthesize(
        self,
        window: TranscriptWindow,
        requester: UUID,
        query: str,
        winner: Winner,
        *a: Any,
        **kw: Any,
    ) -> Any:
        self.shown[requester] = winner
        return await super()._synthesize(window, requester, query, winner, *a, **kw)


@dataclass
class Harness:
    llm: BatchLLM
    deadline_ms: int = 4000
    concurrency: int = 12
    loader: PromptLoader = field(default_factory=lambda: PromptLoader(PROMPTS_DIR))

    def router(self, sink: NoSink | None = None, redis: Any = None) -> LiteLLMRouter:
        return LiteLLMRouter(budget(), sink or NoSink(), self.llm)

    # --- ContextGapAgent ---------------------------------------------------------------------

    async def gap_one(self, inst: Instance, cond: Condition, run: int) -> dict[str, Any]:
        RUN.set(run)
        extra = planted_sources(inst)
        rag = SyntheticRag(extra=extra)
        contexts = {u.id: u.context() for u in org.USERS.values()}
        loop = Loopback(
            org_contexts={org.ORG_ID: contexts},
            rag=rag,
            pdp=FakePdp(set(cond.grants)),
        )
        loop.responder._as_requester = cond.as_requester
        related = (
            FakeRelated([org.USERS[k].id for k in inst.related])
            if inst.related is not None
            else None
        )
        sink = NoSink()
        agent = RecordingGapAgent(
            self.loader,
            LiteLLMRouter(budget(), sink, self.llm),
            deps=ContextGapDeps(
                rag=rag,
                a2a=loop.client,
                pep=loop.pep,
                related=related,  # type: ignore[arg-type]
                user_context=loop.user_context,
            ),
            settings=replace(
                ContextGapSettings.from_env(), a2a=cond.a2a, deadline_ms=self.deadline_ms
            ),
        )
        if hasattr(agent._deps, "redis"):  # the post-fix agent's rate-limit / cap counters
            agent._deps.redis = fakeredis.aioredis.FakeRedis()
        meeting = org.uid("meeting-live", inst.id)
        await agent.spawn(org_id=org.ORG_ID, meeting_id=meeting, user_id=meeting)
        people: dict[UUID, dict[str, Any] | None] = {
            org.USERS[k].id: contexts[org.USERS[k].id] for k in inst.participants
        }
        emitted = await agent.process_meeting(window_of(inst, meeting), people)
        await drain(loop)
        by_id = {u.id: k for k, u in org.USERS.items()}
        fills = []
        for user_id, result in emitted:
            fill = result.payload.get("fill") or {}
            shown = agent.shown.get(user_id)
            fills.append(
                {
                    "requester": by_id[user_id],
                    "content": fill.get("content") or "",
                    "confidence": fill.get("confidence"),
                    "surfaced": bool(fill.get("surfaced")),
                    "hop": fill.get("hop"),
                    "responder": fill.get("responder"),
                    "cited_sources": [
                        source_of_citation(c, extra) for c in fill.get("citations") or []
                    ],
                    "shown": [
                        {"text": p.text, "source": source_of_citation(p.citation, extra)}
                        for p in (shown.passages if shown else [])
                    ],
                    "shown_hop": shown.hop if shown else None,
                    "query": result.payload["gap"]["query"],
                    "query_class": result.payload["gap"]["query_class"],
                    "collaborations": [
                        {k: c[k] for k in ("hop", "responder", "result", "target_scope")}
                        for c in result.payload["collaborations"]
                    ],
                }
            )
        return {
            "id": inst.id,
            "condition": cond.name,
            "run": run,
            "fills": fills,
            "llm_calls": sink.calls,
            "budget_units": sink.units,
            "a2a_requests": len(loop.producer.on("agent-requests")),
        }

    async def gap_all(
        self, instances: list[Instance], cond: Condition, run: int
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(inst: Instance) -> dict[str, Any]:
            async with sem:
                return await self.gap_one(inst, cond, run)

        pins = self.loader._pins
        saved = dict(pins.get(str(org.ORG_ID), {}))
        pins[str(org.ORG_ID)] = {**saved, **dict(cond.pins)}
        try:
            with applied(cond):
                return list(await asyncio.gather(*(one(i) for i in instances)))
        finally:
            pins[str(org.ORG_ID)] = saved

    # --- DecisionAgent / ActionAgent (T4 candidates) -------------------------------------------

    async def candidates_one(self, inst: Instance, run: int) -> dict[str, Any]:
        RUN.set(run)
        out: dict[str, Any] = {"id": inst.id, "run": run, "condition": "B1", "candidates": []}
        meeting = org.uid("meeting-live", inst.id)
        for cls in (DecisionAgent, ActionAgent):
            agent = cls(self.loader, self.router())
            await agent.spawn(org_id=org.ORG_ID, meeting_id=meeting, user_id=meeting)
            first = org.USERS[inst.participants[0]].id
            state = AgentState(
                meeting_id=meeting,
                user_id=first,
                chunk=window_of(inst, meeting),
                context={},
                org_id=org.ORG_ID,
            )
            result = await agent.process(state)
            for c in (result.payload.get("candidates") if result else None) or []:
                out["candidates"].append(
                    {
                        "agent": cls.name,
                        "kind": result.kind if result else None,
                        "content": c.get("content"),
                        "confidence": c.get("confidence"),
                        "surfaced": c.get("surfaced"),
                    }
                )
        return out

    # --- ProactiveAgent conflict check (T3 suppression) ----------------------------------------

    async def conflict_one(self, case: ConflictCase, run: int) -> dict[str, Any]:
        RUN.set(run)
        prompt = self.loader.load(CONFLICT_KEY, self.loader.resolve(CONFLICT_KEY, org.ORG_ID))
        from services.agent_runtime_service.agents.company import company_values
        from services.agent_runtime_service.agents.proactive_agent import clip

        decision = case.new_decision + (f" Rationale: {case.rationale}" if case.rationale else "")
        out = await self.router().complete(
            tier="extract",
            system=prompt.system,
            **prompt.user_parts(
                decision=clip(decision, 1200),
                candidates="\n".join(
                    f"[{i + 1}] {clip(p, 600)}" for i, p in enumerate(case.priors)
                ),
                **company_values({}),
            ),
            org_id=org.ORG_ID,
            user_id=None,
            max_tokens=EXTRACT_MAX_TOKENS,
            effort=SYNTHESIS_EFFORT,
            prompt_ref=prompt.ref,
        )
        conflicts = parse_conflicts(out.text, len(case.priors))
        flagged = [(i + 1, c) for i, c, _r in conflicts if c > 0.7]
        return {
            "id": case.id,
            "run": run,
            "condition": "B1",
            "raw": out.text,
            "flagged_target": any(i == case.target for i, _c in flagged),
            "conflicts": [[i + 1, c] for i, c, _r in conflicts],
        }


async def drive(
    harness: Harness,
    jobs: Any,
    note: str,
    max_passes: int = 4,
) -> list[dict[str, Any]]:
    """Run `jobs()` (a coroutine returning records) until no new LLM request is pending,
    flushing one batch per pass."""
    records: list[dict[str, Any]] = []
    for n in range(max_passes):
        records = await jobs()
        if not harness.llm.pending:
            return records
        harness.llm.flush(f"{note} pass {n + 1}")
    records = await jobs()
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def gen_id() -> str:
    return str(uuid.uuid4())
