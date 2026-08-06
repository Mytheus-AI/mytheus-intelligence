"""600/60 chunking — plan 3.7 step 4 verbatim: `RecursiveCharacterTextSplitter` from LangChain,
chunk size 600, overlap 60.

Promoted here from `services/embedding_service/splitter.py` (Plan 5-6 task 9,
`ingestion-service`): the plan's own pseudocode for `ingestion-service`'s parse pipeline
(`t9-3.9-ingestion.md` step 2) uses the identical splitter to chunk each parsed document
element before emitting `document-chunks`, and `embedding-service`'s consumer applies the same
600/60 split again as a safety net on whatever it receives on `transcript-chunks` /
`document-chunks` (a no-op on already-short text). Two services needing byte-identical chunking
behavior is exactly the case CLAUDE.md's "prefer extending libs/shared/ ... over duplicating it
per service" calls for, so this is promoted out of `embedding_service/` rather than duplicated
a second time. `services/embedding_service/splitter.py` now re-exports from here unchanged, so
`EmbeddingConsumer` and the existing `test_splitter.py` need no code changes.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def split_600_60(text: str) -> list[str]:
    """Never returns an empty list for non-empty input; a text shorter than CHUNK_SIZE comes
    back as a single-element list (one embed call, not zero)."""
    if not text:
        return []
    return _splitter.split_text(text)
