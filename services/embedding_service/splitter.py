"""600/60 chunking — plan 3.7 step 4 verbatim: `RecursiveCharacterTextSplitter` from LangChain,
chunk size 600, overlap 60. Used for BOTH `transcript-chunks` and `document-chunks` text before
embedding, so a long transcript utterance or document element is split into embedding-sized
pieces rather than truncated or embedded whole."""

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
