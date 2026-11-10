"""Deterministic text matching shared by the in-meeting ledger (`memory.py`) and the moment
anchoring (`moments.py`): lowercased content words, and their Jaccard similarity. No imports
from the agents package, so both modules can use it without an import cycle."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = frozenset(
    "a an the and or but if then so to of in on at by for with from up out about into over "
    "is are was were be been being am do does did done have has had will would shall should "
    "can could may might must i you he she it we they me him her us them my your his its our "
    "their this that these those there here what which who whom when where why how all any "
    "some no not just also very really okay ok yeah yes um uh like well let's lets i'll we'll "
    "you'll it's that's i'm we're you're don't won't can't".split()
)


def words(text: str) -> set[str]:
    """Lowercased content words, a trailing plural `s` dropped (crude, deterministic)."""
    out = set()
    for w in _WORD.findall(text.lower().replace("’", "'")):
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def similarity(a: str, b: str) -> float:
    """Jaccard over content words: the near-duplicate test of the in-meeting ledger."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
