"""rag-service's decision-conflict check (`decision_check.py`, `POST /internal/decisions/check`).

No infra and no paid model: retrieval arms, the company brain and the judge are faked, and the
real `LiteLLMRouter` runs with a scripted completion where the router itself is under test
(budget gate, metering to the person).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import fakeredis
import httpx
import pytest
from starlette.testclient import TestClient

from libs.shared.budget import BudgetGate
from libs.shared.org_context import EMPTY, CompanyContext, OrgContextUnavailable
from libs.shared.prompts import PromptLoader
from libs.shared.router import BudgetBlocked, LiteLLMRouter
from services.rag_service import main as rag_main
from services.rag_service.clients import CoreDecisionLexical, ScopeUnavailable
from services.rag_service.contracts import AnswerPassage, Citation, RagQueryRequest, RagResponse
from services.rag_service.decision_check import (
    MIN_CONFIDENCE,
    CheckContext,
    CheckRequest,
    CheckResponse,
    DecisionCheckService,
    NeverRecordUnavailable,
    find_quote,
    parse_judge,
)
from services.rag_service.domain import Caller, LexicalDecision
from services.rag_service.main import PROMPTS_DIR

ORG, USER = uuid.uuid4(), uuid.uuid4()
M_PRICING, M_INFRA = uuid.uuid4(), uuid.uuid4()
D_DISCOUNT, D_DB, D_LAUNCH = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
CALLER = Caller(org_id=ORG, user_id=USER)
SECRET, CORE_SECRET, BFF_SECRET = "a" * 40, "c" * 40, "b" * 40

DRAFT = (
    "Hi Sam, thanks for the call. Happy to offer Acme 30% off the annual plan if they sign "
    "this week. Let me know if that works."
)


def passage(n: int, decision: uuid.UUID, meeting: uuid.UUID, text: str, title: str) -> Any:
    return AnswerPassage(
        n=n,
        text=text,
        rerank_score=0.8,
        citation=Citation(kind="decision", decision_id=decision, meeting_id=meeting),
        meeting_title=title,
        occurred_at=datetime(2026, 10, 3, 15, tzinfo=UTC),
    )


class FakeRetriever:
    def __init__(self, passages: list[AnswerPassage] | None = None, fail: bool = False) -> None:
        self.passages = passages if passages is not None else []
        self.fail = fail
        self.calls: list[tuple[Caller, RagQueryRequest]] = []

    async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse:
        self.calls.append((caller, req))
        if self.fail:
            raise RuntimeError("vector arm down")
        return RagResponse(answer_passages=self.passages)


class FakeLexical:
    def __init__(self, hits: list[LexicalDecision] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[uuid.UUID, uuid.UUID, str, int]] = []

    async def search(
        self, org_id: uuid.UUID, user_id: uuid.UUID, query: str, limit: int = 10
    ) -> list[LexicalDecision]:
        self.calls.append((org_id, user_id, query, limit))
        return self.hits


class FakeOrg:
    def __init__(self, ctx: CompanyContext = EMPTY, down: bool = False) -> None:
        self.ctx, self.down = ctx, down

    async def get_strict(self, org_id: uuid.UUID) -> CompanyContext:
        if self.down:
            raise OrgContextUnavailable(str(org_id))
        return self.ctx


class Out:
    def __init__(self, text: str, tier: str = "extract") -> None:
        self.text, self.tier = text, tier


class FakeJudge:
    def __init__(self, reply: str = '{"conflicts": []}', *, blocked: bool = False, tier="extract"):
        self.reply, self.blocked, self.tier = reply, blocked, tier
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> Out:
        self.calls.append(kw)
        if self.blocked:
            raise BudgetBlocked(str(kw["org_id"]))
        return Out(self.reply, self.tier)


DISCOUNT = passage(
    1,
    D_DISCOUNT,
    M_PRICING,
    "No discounts above 20% without finance sign-off. Rationale: Margins on annual plans.",
    "Pricing review",
)
DB = passage(2, D_DB, M_INFRA, "Use Postgres for the events store.", "Infra sync")
LEX_LAUNCH = LexicalDecision(
    id=D_LAUNCH,
    meeting_id=M_PRICING,
    content="Annual billing launches on 15 November.",
    meeting_title="Pricing review",
    decided_at="2026-10-03T15:00:00+00:00",
)
LEX_DISCOUNT = LexicalDecision(
    id=D_DISCOUNT,
    meeting_id=M_PRICING,
    content="No discounts above 20% without finance sign-off.",
    rationale="Margins on annual plans.",
    meeting_title="Pricing review",
    decided_at="2026-10-03T15:00:00+00:00",
)


def build(
    *,
    passages: list[AnswerPassage] | None = None,
    lexical: list[LexicalDecision] | None = None,
    judge: Any = None,
    org: FakeOrg | None = None,
    retriever: FakeRetriever | None = None,
) -> tuple[DecisionCheckService, dict[str, Any]]:
    parts = {
        "retriever": retriever or FakeRetriever([DISCOUNT, DB] if passages is None else passages),
        "lexical": FakeLexical([LEX_DISCOUNT, LEX_LAUNCH] if lexical is None else lexical),
        "judge": judge or FakeJudge(),
        "org": org or FakeOrg(),
    }
    svc = DecisionCheckService(
        retriever=parts["retriever"],
        lexical=parts["lexical"],
        org_context=parts["org"],
        judge=parts["judge"],
        prompts=PromptLoader(PROMPTS_DIR),
    )
    return svc, parts


def judged(*entries: dict[str, Any]) -> str:
    return json.dumps({"conflicts": list(entries)})


def candidate_number(user: str, content: str) -> int:
    for line in user.splitlines():
        if content in line and line.startswith("["):
            return int(line[1 : line.index("]")])
    raise AssertionError(f"{content!r} was not shown to the judge")


# --- the pipeline --------------------------------------------------------------------------


async def test_a_contradiction_is_found_with_the_decision_quote_and_why():
    class Judge(FakeJudge):
        async def complete(self, **kw: Any) -> Out:
            self.calls.append(kw)
            n = candidate_number(kw["user"], "No discounts above 20%")
            return Out(
                judged(
                    {
                        "candidate": n,
                        "relation": "contradicts",
                        "confidence": 0.93,
                        "quote": "happy to offer acme 30% off the annual plan",
                        "why": "The pricing review capped discounts at 20% without finance.",
                    }
                )
            )

    svc, parts = build(judge=Judge())
    res = await svc.check(CALLER, CheckRequest(text=DRAFT, source="extension"))
    assert res.status == "ok" and res.reason is None and res.prompt == "decision_conflict_check@v1"
    assert res.candidates == 3  # discount (both arms, fused once), db, launch
    [c] = res.conflicts
    assert c.decision_id == D_DISCOUNT and c.meeting_id == M_PRICING
    assert c.decision_content == "No discounts above 20% without finance sign-off."
    assert c.meeting_title == "Pricing review" and c.decided_at == "2026-10-03"
    assert c.relation == "contradicts" and c.confidence == 0.93
    # Case-normalised match, returned as the writing spelled it.
    assert c.quote == "Happy to offer Acme 30% off the annual plan"
    assert "20%" in c.why


async def test_retrieval_runs_as_the_person_on_both_arms_and_fuses_them():
    svc, parts = build()
    await svc.check(CALLER, CheckRequest(text=DRAFT))
    [(caller, req)] = parts["retriever"].calls
    assert caller == CALLER
    assert req.source == "decisions" and req.generate is False and req.query == DRAFT
    [(org, user, query, limit)] = parts["lexical"].calls
    assert (org, user, query, limit) == (ORG, USER, DRAFT, 10)
    user_msg = parts["judge"].calls[0]["user"]
    # The decision both arms found ranks first; each decision is shown once.
    assert user_msg.count("No discounts above 20%") == 1
    assert candidate_number(user_msg, "No discounts above 20%") == 1
    assert parts["judge"].calls[0]["user_id"] == USER
    assert parts["judge"].calls[0]["tier"] == "extract"


async def test_only_the_first_1000_characters_are_retrieved_on():
    svc, parts = build()
    long = ("We should rethink the pricing tiers entirely. " * 60)[:3900]
    await svc.check(CALLER, CheckRequest(text=long))
    assert len(parts["retriever"].calls[0][1].query) == 1000
    assert long.strip() in parts["judge"].calls[0]["user"]  # the judge reads it all


async def test_action_passages_are_never_candidates():
    action = AnswerPassage(
        n=3,
        text="Sam to send the Acme quote",
        rerank_score=0.9,
        citation=Citation(kind="action", action_id=uuid.uuid4(), meeting_id=M_PRICING),
    )
    svc, parts = build(passages=[action], lexical=[])
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert res.status == "ok" and res.reason == "no_candidates"
    assert parts["judge"].calls == []


async def test_too_short_text_calls_nothing():
    svc, parts = build()
    res = await svc.check(CALLER, CheckRequest(text="ok thanks!"))
    assert (res.status, res.reason) == ("skipped", "too_short")
    assert parts["retriever"].calls == [] and parts["judge"].calls == []


async def test_never_record_text_is_not_retrieved_on_or_judged():
    org = FakeOrg(CompanyContext(never_record_topics=("layoffs",)))
    svc, parts = build(org=org)
    res = await svc.check(CALLER, CheckRequest(text="Draft plan for the Q1 layoffs in support"))
    assert (res.status, res.reason) == ("skipped", "never_record")
    assert parts["retriever"].calls == [] and parts["lexical"].calls == []
    assert parts["judge"].calls == []


async def test_never_record_in_the_context_also_skips():
    org = FakeOrg(CompanyContext(never_record_topics=("Project Falcon",)))
    svc, parts = build(org=org)
    res = await svc.check(
        CALLER,
        CheckRequest(text=DRAFT, context=CheckContext(channel="project falcon")),
    )
    assert res.reason == "never_record" and parts["judge"].calls == []


async def test_a_candidate_naming_a_never_record_topic_never_reaches_the_judge():
    secret = LexicalDecision(
        id=uuid.uuid4(), meeting_id=M_PRICING, content="Layoffs will not touch sales."
    )
    org = FakeOrg(CompanyContext(never_record_topics=("layoffs",)))
    svc, parts = build(org=org, lexical=[secret], passages=[DB])
    await svc.check(CALLER, CheckRequest(text="We should move the events store to MySQL now"))
    assert "Layoffs" not in parts["judge"].calls[0]["user"]


async def test_an_unreadable_never_record_list_fails_closed():
    svc, parts = build(org=FakeOrg(down=True))
    with pytest.raises(NeverRecordUnavailable):
        await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert parts["retriever"].calls == []


async def test_no_candidates_means_no_model_call():
    svc, parts = build(passages=[], lexical=[])
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert (res.status, res.reason, res.conflicts) == ("ok", "no_candidates", [])
    assert parts["judge"].calls == []


async def test_one_arm_down_still_checks_with_the_other():
    svc, parts = build(retriever=FakeRetriever(fail=True))
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert res.candidates == 2 and len(parts["judge"].calls) == 1


async def test_budget_blocked_is_degraded_and_empty():
    svc, _ = build(judge=FakeJudge(blocked=True))
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert (res.status, res.reason, res.conflicts) == ("degraded", "budget", [])


async def test_budget_blocked_through_the_real_router():
    async def no_budget(_org: uuid.UUID) -> Any:
        return None  # no budget mirror and no refresh: the gate blocks

    called: list[Any] = []

    async def completion(**kw: Any) -> Any:
        called.append(kw)
        return {"choices": [{"message": {"content": "{}"}}]}

    class Sink:
        async def usage(self, **kw: Any) -> None: ...

    router = LiteLLMRouter(
        BudgetGate(fakeredis.aioredis.FakeRedis(), no_budget), Sink(), completion
    )
    svc, _ = build(judge=router)
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert (res.status, res.reason) == ("degraded", "budget") and called == []


async def test_a_throttled_tenant_is_answered_but_marked_degraded():
    reply = judged(
        {"candidate": 1, "relation": "contradicts", "confidence": 0.9, "quote": "30% off"}
    )
    svc, _ = build(judge=FakeJudge(reply, tier="triage"))
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert (res.status, res.reason) == ("degraded", "budget") and len(res.conflicts) == 1


async def test_low_confidence_is_dropped_and_max_conflicts_holds():
    reply = judged(
        {"candidate": 1, "relation": "contradicts", "confidence": 0.95, "quote": "30% off"},
        {"candidate": 2, "relation": "reopens", "confidence": MIN_CONFIDENCE - 0.01, "quote": "x"},
        {"candidate": 3, "relation": "reopens", "confidence": 0.8, "quote": "this week"},
    )
    svc, _ = build(judge=FakeJudge(reply))
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert [c.confidence for c in res.conflicts] == [0.95, 0.8]
    res = await svc.check(CALLER, CheckRequest(text=DRAFT, max_conflicts=1))
    assert [c.confidence for c in res.conflicts] == [0.95]


async def test_an_invented_quote_falls_back_to_the_closest_sentence():
    reply = judged(
        {
            "candidate": 1,
            "relation": "contradicts",
            "confidence": 0.9,
            "quote": "we will give them a big discount",
            "why": "See https://evil.example.com/x for why.",
        }
    )
    svc, _ = build(judge=FakeJudge(reply))
    [c] = (await svc.check(CALLER, CheckRequest(text=DRAFT))).conflicts
    assert c.quote == "Happy to offer Acme 30% off the annual plan if they sign this week."
    assert "http" not in c.why and "evil" not in c.why


async def test_a_conflict_whose_quote_names_a_never_record_topic_is_dropped():
    org = FakeOrg(CompanyContext(never_record_topics=("Falcon",)))
    reply = judged(
        {"candidate": 1, "relation": "reopens", "confidence": 0.9, "quote": "x", "why": "Falcon"}
    )
    svc, _ = build(judge=FakeJudge(reply), org=org)
    res = await svc.check(CALLER, CheckRequest(text=DRAFT))
    assert res.conflicts == []


async def test_malformed_judge_output_is_no_conflict_not_a_guess():
    for raw in ("not json", '{"conflicts": "x"}', '{"conflicts": [{"candidate": 9}]}', ""):
        svc, _ = build(judge=FakeJudge(raw))
        res = await svc.check(CALLER, CheckRequest(text=DRAFT))
        assert res.status == "ok" and res.conflicts == []


def test_parse_judge_keeps_the_best_per_candidate_and_reads_fences():
    raw = (
        "```json\n"
        + judged(
            {"candidate": 1, "relation": "reopens", "confidence": 0.75, "quote": "a"},
            {"candidate": 1, "relation": "contradicts", "confidence": 0.9, "quote": "b"},
            {"candidate": True, "relation": "contradicts", "confidence": 0.9},
            {"candidate": 2, "relation": "agrees", "confidence": 0.9},
            {"candidate": 2, "relation": "reopens", "confidence": "high"},
        )
        + "\n```"
    )
    assert parse_judge(raw, 2) == [(0, "contradicts", 0.9, "b", "")]


def test_find_quote_prefers_the_verbatim_span():
    text = "First line.\nWe  should   use MySQL instead. Thanks!"
    assert find_quote(text, "we should use mysql", "Use Postgres") == "We should use MySQL"
    assert find_quote(text, "nothing like this", "Use MySQL? no, Postgres") == (
        "We should use MySQL instead."
    )


async def test_the_prompt_renders_the_context_and_numbered_candidates():
    svc, parts = build()
    await svc.check(
        CALLER,
        CheckRequest(
            text=DRAFT,
            source="slack",
            context=CheckContext(channel="#pricing", topic="Acme renewal"),
        ),
    )
    call = parts["judge"].calls[0]
    assert call["prompt_ref"] == "decision_conflict_check@v1"
    assert "a Slack message; channel: #pricing; topic: Acme renewal" in call["user"]
    assert '[1] (meeting "Pricing review", 2026-10-03) No discounts above 20%' in call["user"]
    assert "Why: Margins on annual plans." in call["user"]
    assert "contradicts" in call["system"]


# --- the lexical client --------------------------------------------------------------------


async def test_the_lexical_client_sends_tenant_and_person_and_tolerates_failure():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": str(D_DB),
                        "meeting_id": str(M_INFRA),
                        "content": "Use Postgres.",
                        "rationale": None,
                        "meeting_title": "Infra",
                        "decided_at": "2026-10-01T10:00:00Z",
                        "rank": 0.4,
                    },
                    {"id": "not-a-uuid", "content": "x"},
                ]
            },
        )

    client = CoreDecisionLexical("http://core", "t" * 40, transport=httpx.MockTransport(handler))
    hits = await client.search(ORG, USER, "q" * 1500, 5)
    assert [h.id for h in hits] == [D_DB]
    req = seen[0]
    assert req.url.path == "/internal/retrieval/decisions/search"
    assert req.headers["x-mytheus-org"] == str(ORG)
    assert req.headers["x-mytheus-service"] == "rag-service"
    body = json.loads(req.content)
    assert body == {"user_id": str(USER), "query": "q" * 1000, "limit": 5}

    down = CoreDecisionLexical(
        "http://core", "t" * 40, transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert await down.search(ORG, USER, "q") == []


# --- the HTTP route ------------------------------------------------------------------------


class RecordingCheck:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Caller, CheckRequest]] = []
        self.error = error

    async def check(self, caller: Caller, req: CheckRequest) -> Any:
        self.calls.append((caller, req))
        if self.error:
            raise self.error
        return CheckResponse(status="ok", reason="no_candidates")


@pytest.fixture()
def tokens(monkeypatch):
    monkeypatch.setenv(
        rag_main.CALLER_TOKENS_ENV,
        f"agent-runtime-service={SECRET},bff={BFF_SECRET},core-api={CORE_SECRET}",
    )


def headers(service: str = "core-api", secret: str = CORE_SECRET, user: Any = USER) -> dict:
    h = {
        "X-Mytheus-Service": service,
        "X-Mytheus-Service-Token": secret,
        "X-Mytheus-Org": str(ORG),
    }
    if user is not None:
        h["X-Mytheus-User"] = str(user)
    return h


BODY = {"text": DRAFT, "source": "slack", "context": {"channel": "#pricing"}}


def test_core_api_checks_as_the_named_person(tokens):
    svc = RecordingCheck()
    client = TestClient(rag_main.build_app(check=svc))  # type: ignore[arg-type]
    r = client.post("/internal/decisions/check", json=BODY, headers=headers())
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
    caller, req = svc.calls[0]
    assert caller == Caller(org_id=ORG, user_id=USER)
    assert req.source == "slack" and req.context.channel == "#pricing"


@pytest.mark.parametrize(
    "h",
    [
        headers(service="bff", secret=BFF_SECRET),
        headers(service="agent-runtime-service", secret=SECRET),
        headers(secret=SECRET),
        headers(user=None),
        headers(user="nope"),
        {},
    ],
)
def test_only_core_api_may_check(tokens, h):
    svc = RecordingCheck()
    client = TestClient(rag_main.build_app(check=svc))  # type: ignore[arg-type]
    assert client.post("/internal/decisions/check", json=BODY, headers=h).status_code == 401
    assert svc.calls == []


def test_core_api_may_not_query_rag(tokens):
    class Rag:
        calls = 0

        async def query(self, caller: Caller, req: RagQueryRequest) -> RagResponse:
            Rag.calls += 1
            return RagResponse(answer_passages=[])

    client = TestClient(rag_main.build_app(Rag()))  # type: ignore[arg-type]
    r = client.post("/api/rag/query", json={"query": "x"}, headers=headers())
    assert r.status_code == 401 and Rag.calls == 0
    ok = client.post("/api/rag/query", json={"query": "x"}, headers=headers("bff", BFF_SECRET))
    assert ok.status_code == 200


def test_unwired_check_is_503(tokens):
    client = TestClient(rag_main.build_app())
    r = client.post("/internal/decisions/check", json=BODY, headers=headers())
    assert r.status_code == 503


@pytest.mark.parametrize("error", [ScopeUnavailable("down"), NeverRecordUnavailable("x")])
def test_unscreenable_or_unscoped_is_503(tokens, error):
    client = TestClient(rag_main.build_app(check=RecordingCheck(error)))  # type: ignore[arg-type]
    r = client.post("/internal/decisions/check", json=BODY, headers=headers())
    assert r.status_code == 503


def test_the_body_is_validated(tokens):
    client = TestClient(rag_main.build_app(check=RecordingCheck()))  # type: ignore[arg-type]
    for body in (
        {"text": ""},
        {"text": "x" * 4001},
        {"text": DRAFT, "source": "email"},
        {"text": DRAFT, "max_conflicts": 9},
        {"text": DRAFT, "org_id": str(uuid.uuid4())},
    ):
        assert (
            client.post("/internal/decisions/check", json=body, headers=headers()).status_code
            == 422
        )
