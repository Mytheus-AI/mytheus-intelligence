"""Optional RAG grounding for `QuestionAgent` (mytheus-docs ADR-0034 D9, ADR-0033 D1).

The agent depends on the narrow `Grounder` protocol below, never on `rag-service` directly. The
one production implementation, `RagClientGrounder`, adapts `libs.shared.rag_client` (owned
by the RAG track, mytheus-docs ADR-0033; this module does not edit it).
Rules every implementation follows:

- **Participant-scoped.** Every call carries the participant's `user_id`, the user the agent
  is running for. Never a service identity: retrieval returns only meetings that user attended.
- **Passages only.** `generate: false` (enforced by `HttpRagClient`). Grounding a question needs
  context, not an answer, and a generated answer costs a Sonnet call.
- **Never fails the window.** A timeout or error returns no passages; the agent proposes
  without context.

Off unless `QUESTION_RAG_GROUNDING=1`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from libs.shared.rag_client import HttpRagClient, RagClient, RagPassage

logger = logging.getLogger(__name__)

CITATION_KINDS = frozenset({"transcript", "document", "decision", "action"})
DEFAULT_LIMIT = 4
DEFAULT_TIMEOUT_S = 0.8
MAX_PASSAGE_CHARS = 600


@dataclass(frozen=True)
class Passage:
    text: str
    citation: dict[str, Any] = field(default_factory=dict)  # {"kind": ..., ids per kind}


class Grounder(Protocol):
    async def passages(
        self, *, org_id: UUID, user_id: UUID, query: str, limit: int
    ) -> list[Passage]: ...


def grounding_enabled() -> bool:
    return os.environ.get("QUESTION_RAG_GROUNDING", "").strip().lower() in {"1", "true", "yes"}


def valid_citation(raw: object) -> dict[str, Any] | None:
    """A citation of one of the four ADR-0033 D3 kinds, with string values only; else None."""
    if not isinstance(raw, dict) or raw.get("kind") not in CITATION_KINDS:
        return None
    out = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str | int)}
    return out if "kind" in out else None


async def ground(
    grounder: Grounder | None,
    *,
    org_id: UUID,
    user_id: UUID,
    query: str,
    limit: int = DEFAULT_LIMIT,
    timeout_s: float | None = None,
) -> list[Passage]:
    """Passages for the participant, bounded in time; [] on no grounder, timeout or error."""
    if grounder is None or not query.strip():
        return []
    timeout = timeout_s if timeout_s is not None else _timeout()
    try:
        found = await asyncio.wait_for(
            grounder.passages(org_id=org_id, user_id=user_id, query=query, limit=limit),
            timeout,
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - grounding is best effort, the window still runs
        logger.warning("question grounding failed for user %s; proposing without it", user_id)
        return []
    out = []
    for p in found[:limit]:
        citation = valid_citation(p.citation)
        if citation is not None and p.text.strip():
            out.append(Passage(text=p.text.strip()[:MAX_PASSAGE_CHARS], citation=citation))
    return out


def _timeout() -> float:
    try:
        return float(os.environ.get("QUESTION_RAG_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


class RagClientGrounder:
    """`Grounder` over the RAG track's `libs.shared.rag_client` (`RagClient.retrieve`). The only
    place that knows its shape: if that interface changes, this adapter is what changes.

    No `meeting_id` filter: the window's own meeting is already in the prompt, and grounding is
    for what the participant saw in OTHER meetings. `HttpRagClient` always asks for passages
    only (`generate: false`) and returns [] on any failure."""

    def __init__(self, client: RagClient) -> None:
        self._client = client

    async def passages(
        self, *, org_id: UUID, user_id: UUID, query: str, limit: int
    ) -> list[Passage]:
        found = await self._client.retrieve(org_id=org_id, user_id=user_id, query=query)
        return [Passage(text=p.text, citation=_citation(p)) for p in found[:limit]]


def _citation(p: RagPassage) -> dict[str, Any]:
    """The passage's citation as the stored shape: its kind, the ids it has, and a title."""
    ids = {
        "meeting_id": p.meeting_id,
        "chunk_id": p.chunk_id,
        "document_id": p.document_id,
        "decision_id": p.decision_id,
        "action_id": p.action_id,
    }
    out: dict[str, Any] = {"kind": p.kind}
    out.update({k: str(v) for k, v in ids.items() if v is not None})
    if p.meeting_title:
        out["title"] = p.meeting_title
    return out


def grounder_from_env() -> Grounder | None:
    """The production grounder when `QUESTION_RAG_GROUNDING` is on: `HttpRagClient` against
    `RAG_URL` with this service's `RAG_CALLER_TOKEN` (an unset token means no grounding)."""
    if not grounding_enabled():
        return None
    return RagClientGrounder(HttpRagClient(timeout_s=_timeout()))
