"""The real downstream clients behind `ports.py`.

rag-service calls everything as itself: `X-Mytheus-Service: rag-service`, its secret
(`RAG_SERVICE_TOKEN`, listed as `rag-service=<secret>` in each callee's caller-tokens env), and
the tenant in `X-Mytheus-Org` (gRPC metadata on the FTS hop). It has already authenticated the
asker, and an agent asking for a participant has no user token to forward.

Graceful degradation is per arm, never silent about tenancy: an unreachable arm returns no
candidates and is logged (the other arms still answer), but the scope call failing fails the
request, because without scope nothing may be retrieved at all.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import grpc
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from libs.shared import rag_proto
from libs.shared.qdrant import (
    LOGICAL_DECISIONS,
    LOGICAL_DOCUMENT_CHUNKS,
    LOGICAL_TRANSCRIPT_CHUNKS,
    physical_collection,
)
from services.rag_service.domain import (
    Caller,
    Candidate,
    MeetingInfo,
    Person,
    Scope,
    action_key,
    decision_key,
    document_key,
    transcript_key,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "rag-service"


def service_headers(token: str, org_id: UUID) -> dict[str, str]:
    return {
        "X-Mytheus-Service": SERVICE_NAME,
        "X-Mytheus-Service-Token": token,
        "X-Mytheus-Org": str(org_id),
    }


class ScopeUnavailable(RuntimeError):
    """core-api could not answer the scope call: the request must fail, not widen."""


class CoreScopeClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 3.0) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    async def scope(self, caller: Caller) -> Scope | None:
        params = (
            {"user_id": str(caller.user_id)}
            if caller.user_id
            else {"clerk_user_id": caller.clerk_user_id or ""}
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.get(
                    f"{self._url}/internal/retrieval/scope",
                    params=params,
                    headers=service_headers(self._token, caller.org_id),
                )
        except httpx.HTTPError as exc:
            raise ScopeUnavailable("core-api unreachable") from exc
        if res.status_code == 404:
            return None
        if res.status_code != 200:
            raise ScopeUnavailable(f"core-api answered {res.status_code}")
        body = res.json()
        return Scope(
            user_id=UUID(body["user_id"]),
            meetings={
                UUID(m["id"]): MeetingInfo(
                    title=m.get("title"), started_at=datetime.fromisoformat(m["started_at"])
                )
                for m in body["meetings"]
            },
            people=tuple(Person(id=UUID(p["id"]), email=p["email"]) for p in body["people"]),
        )

    async def records(
        self, org_id: UUID, user_id: UUID, decision_ids: Sequence[UUID], action_ids: Sequence[UUID]
    ) -> dict[str, Candidate]:
        if not decision_ids and not action_ids:
            return {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(
                    f"{self._url}/internal/retrieval/records",
                    json={
                        "user_id": str(user_id),
                        "decision_ids": [str(d) for d in decision_ids],
                        "action_ids": [str(a) for a in action_ids],
                    },
                    headers=service_headers(self._token, org_id),
                )
        except httpx.HTTPError:
            logger.warning("rag: decision/action records unavailable")
            return {}
        if res.status_code != 200:
            logger.warning("rag: records answered %s", res.status_code)
            return {}
        body = res.json()
        out: dict[str, Candidate] = {}
        for d in body["decisions"]:
            text = d["content"] + (f" Rationale: {d['rationale']}" if d.get("rationale") else "")
            out[decision_key(d["id"])] = Candidate(
                kind="decision",
                key=decision_key(d["id"]),
                decision_id=UUID(d["id"]),
                meeting_id=UUID(d["meeting_id"]),
                text=text,
            )
        for a in body["actions"]:
            due = f" (due {a['due_date']})" if a.get("due_date") else ""
            out[action_key(a["id"])] = Candidate(
                kind="action",
                key=action_key(a["id"]),
                action_id=UUID(a["id"]),
                meeting_id=UUID(a["meeting_id"]),
                text=a["content"] + due,
            )
        return out


class HttpEmbedder:
    def __init__(self, base_url: str, token: str, timeout_s: float = 5.0) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    async def embed(self, org_id: UUID, queries: Sequence[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            res = await client.post(
                f"{self._url}/embed",
                json={"queries": list(queries)},
                headers=service_headers(self._token, org_id),
            )
        res.raise_for_status()
        vectors: list[list[float]] = res.json()["vectors"]
        return vectors


def _sigmoid_if_logit(score: float) -> float:
    return score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + math.exp(-score))


class HttpReranker:
    def __init__(self, base_url: str, token: str, timeout_s: float = 10.0) -> None:
        self._url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    async def rerank(
        self, org_id: UUID, query: str, passages: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            res = await client.post(
                f"{self._url}/rerank",
                json={"query": query, "passages": list(passages), "top_k": top_k},
                headers=service_headers(self._token, org_id),
            )
        res.raise_for_status()
        return rebind(passages, [(r["passage"], r["score"]) for r in res.json()["results"]])


def rebind(passages: Sequence[str], scored: Sequence[tuple[str, float]]) -> list[tuple[int, float]]:
    """`/rerank` echoes passage text, not an index. Map each back to the first unused index
    with that exact text, so duplicates bind to distinct candidates."""
    free: dict[str, list[int]] = {}
    for i, p in enumerate(passages):
        free.setdefault(p, []).append(i)
    out: list[tuple[int, float]] = []
    for text, score in scored:
        slots = free.get(text)
        if slots:
            out.append((slots.pop(0), _sigmoid_if_logit(float(score))))
    return out


class QdrantVectorSearch:
    """Reads with the READ-ONLY key. Every query is org-scoped twice (collection and payload
    filter), and transcript and decision hits are restricted to the asker's meetings."""

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def _query(self, name: str, vector: list[float], flt: Filter, limit: int) -> list[Any]:
        try:
            res = await self._client.query_points(
                collection_name=name, query=vector, query_filter=flt, limit=limit
            )
        except Exception:  # a tenant with no collection yet, or Qdrant down: this arm is empty
            logger.info("rag: vector search on %s returned nothing", name)
            return []
        return list(res.points)

    async def search(
        self,
        org_id: UUID,
        vector: list[float],
        *,
        meeting_ids: Sequence[UUID],
        speaker_ids: Sequence[str],
        kinds: set[str],
        limit: int,
    ) -> list[Candidate]:
        org = FieldCondition(key="org_id", match=MatchValue(value=str(org_id)))
        meetings = FieldCondition(
            key="meeting_id", match=MatchAny(any=[str(m) for m in meeting_ids])
        )
        hits: list[tuple[float, Candidate]] = []
        if "transcript" in kinds and meeting_ids:
            must: list[Any] = [org, meetings]
            if speaker_ids:
                must.append(FieldCondition(key="speaker", match=MatchAny(any=list(speaker_ids))))
            for p in await self._query(
                physical_collection(LOGICAL_TRANSCRIPT_CHUNKS, org_id),
                vector,
                Filter(must=must),
                limit,
            ):
                pl = p.payload or {}
                hits.append(
                    (
                        p.score,
                        Candidate(
                            kind="transcript",
                            key=transcript_key(pl["chunk_id"]),
                            chunk_id=UUID(pl["chunk_id"]),
                            meeting_id=UUID(pl["meeting_id"]),
                            speaker=pl.get("speaker"),
                            start_ms=pl.get("start_ms"),
                        ),
                    )
                )
        if "document" in kinds:
            for p in await self._query(
                physical_collection(LOGICAL_DOCUMENT_CHUNKS, org_id),
                vector,
                Filter(must=[org]),
                limit,
            ):
                pl = p.payload or {}
                hits.append(
                    (
                        p.score,
                        Candidate(
                            kind="document",
                            key=document_key(pl["document_id"], pl["element_id"]),
                            document_id=UUID(pl["document_id"]),
                            element_id=pl["element_id"],
                            element_type=pl.get("element_type"),
                        ),
                    )
                )
        if "decision" in kinds and meeting_ids:
            status = FieldCondition(key="status", match=MatchAny(any=["confirmed", "superseded"]))
            for p in await self._query(
                physical_collection(LOGICAL_DECISIONS, org_id),
                vector,
                Filter(must=[org, meetings, status]),
                limit,
            ):
                pl = p.payload or {}
                hits.append(
                    (
                        p.score,
                        Candidate(
                            kind="decision",
                            key=decision_key(pl["decision_id"]),
                            decision_id=UUID(pl["decision_id"]),
                            meeting_id=UUID(pl["meeting_id"]),
                        ),
                    )
                )
        # One ranked list per arm call; a chunk split into several points counts once.
        hits.sort(key=lambda h: h[0], reverse=True)
        seen: set[str] = set()
        out: list[Candidate] = []
        for _, c in hits:
            if c.key not in seen:
                seen.add(c.key)
                out.append(c)
        return out[:limit]


class GrpcLexicalSearch:
    def __init__(self, target: str, token: str, timeout_s: float = 3.0) -> None:
        self._channel = grpc.aio.insecure_channel(target)  # mesh-internal
        self._token = token
        self._timeout = timeout_s
        self._search = self._channel.unary_unary(
            rag_proto.FTS_SEARCH_METHOD,
            request_serializer=rag_proto.FtsSearchRequest.SerializeToString,
            response_deserializer=rag_proto.FtsSearchResponse.FromString,
        )
        self._passages = self._channel.unary_unary(
            rag_proto.GET_PASSAGES_METHOD,
            request_serializer=rag_proto.GetPassagesRequest.SerializeToString,
            response_deserializer=rag_proto.GetPassagesResponse.FromString,
        )

    def _md(self, org_id: UUID) -> tuple[tuple[str, str], ...]:
        return tuple((k.lower(), v) for k, v in service_headers(self._token, org_id).items())

    async def search(
        self,
        org_id: UUID,
        query: str,
        *,
        meeting_ids: Sequence[UUID],
        speaker_ids: Sequence[str],
        limit: int,
    ) -> list[Candidate]:
        if not meeting_ids:
            return []
        try:
            resp = await self._search(
                rag_proto.FtsSearchRequest(
                    query=query,
                    limit=limit,
                    scoped=True,
                    meeting_ids=[str(m) for m in meeting_ids],
                    speaker_ids=list(speaker_ids),
                    match_any=True,
                ),
                metadata=self._md(org_id),
                timeout=self._timeout,
            )
        except grpc.aio.AioRpcError as exc:
            logger.warning("rag: FTS arm failed: %s", exc.code())
            return []
        return [
            Candidate(
                kind="transcript",
                key=transcript_key(h.chunk_id),
                chunk_id=UUID(h.chunk_id),
                meeting_id=UUID(h.meeting_id),
                speaker=h.speaker_id,
                start_ms=h.start_ms,
                text=h.passage,
            )
            for h in resp.results
        ]

    async def passages(
        self, org_id: UUID, candidates: Sequence[Candidate], neighbors: int
    ) -> dict[str, Candidate]:
        chunk_ids = [str(c.chunk_id) for c in candidates if c.kind == "transcript" and c.chunk_id]
        docs = [
            rag_proto.DocumentRef(document_id=str(c.document_id), element_id=c.element_id)
            for c in candidates
            if c.kind == "document" and c.document_id and c.element_id
        ]
        if not chunk_ids and not docs:
            return {}
        try:
            resp = await self._passages(
                rag_proto.GetPassagesRequest(
                    chunk_ids=chunk_ids, neighbors=neighbors, documents=docs
                ),
                metadata=self._md(org_id),
                timeout=self._timeout,
            )
        except grpc.aio.AioRpcError as exc:
            logger.warning("rag: passage lookup failed: %s", exc.code())
            return {}
        out: dict[str, Candidate] = {}
        for t in resp.transcripts:
            out[transcript_key(t.chunk_id)] = Candidate(
                kind="transcript",
                key=transcript_key(t.chunk_id),
                chunk_id=UUID(t.chunk_id),
                meeting_id=UUID(t.meeting_id),
                speaker=t.speaker_id,
                start_ms=t.start_ms,
                text=t.text,
                before=t.before,
                after=t.after,
            )
        for d in resp.documents:
            out[document_key(d.document_id, d.element_id)] = Candidate(
                kind="document",
                key=document_key(d.document_id, d.element_id),
                document_id=UUID(d.document_id),
                element_id=d.element_id,
                element_type=d.element_type,
                text=d.text,
            )
        return out


class HttpGraphSearch:
    """The graph arm: confirmed decisions linked to a person in the question
    (`decisions_by_person`) or to a known topic (`related_decisions_for_topic`). The graph is
    optional in production (the `kg` profile); unconfigured or down means no graph candidates."""

    def __init__(self, base_url: str | None, token: str, timeout_s: float = 2.0) -> None:
        self._url = base_url.rstrip("/") if base_url else None
        self._token = token
        self._timeout = timeout_s

    async def _template(self, org_id: UUID, name: str, params: dict[str, Any]) -> list[Any]:
        if not self._url:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                res = await client.post(
                    f"{self._url}/graph/query/{name}",
                    json={"params": params},
                    headers=service_headers(self._token, org_id),
                )
        except httpx.HTTPError:
            logger.warning("rag: graph arm unreachable")
            return []
        if res.status_code != 200:
            logger.warning("rag: graph template %s answered %s", name, res.status_code)
            return []
        nodes: list[Any] = res.json()["nodes"]
        return nodes

    async def decisions(
        self, org_id: UUID, *, person_ids: Sequence[UUID], topics: Sequence[str], limit: int
    ) -> list[Candidate]:
        nodes: list[Any] = []
        for pid in person_ids:
            nodes += await self._template(
                org_id, "decisions_by_person", {"user_id": str(pid), "limit": limit}
            )
        for topic in topics:
            nodes += await self._template(
                org_id, "related_decisions_for_topic", {"topic": topic, "limit": limit}
            )
        out: list[Candidate] = []
        seen: set[str] = set()
        for n in nodes:
            if n.get("label") != "Decision" or not n.get("id"):
                continue
            key = decision_key(n["id"])
            if key in seen:
                continue
            seen.add(key)
            meeting = (n.get("properties") or {}).get("meeting_id")
            out.append(
                Candidate(
                    kind="decision",
                    key=key,
                    decision_id=UUID(n["id"]),
                    meeting_id=UUID(meeting) if meeting else None,
                )
            )
        return out[:limit]
