"""Re-export of the 600/60 splitter, promoted to `libs/shared/splitter.py` (Plan 5-6 task 9)
because `ingestion-service` needs the byte-identical splitter too. Kept here so
`EmbeddingConsumer`'s existing import and `tests/test_splitter.py` need no changes."""

from __future__ import annotations

from libs.shared.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_600_60

__all__ = ["CHUNK_OVERLAP", "CHUNK_SIZE", "split_600_60"]
