"""Read one JSON object out of a model answer, tolerantly.

The agents ask for "JSON only", and a model mostly complies, but valid-but-variant answers do
occur: a code fence around the object, a sentence before or after it, or (seen in the first live
eval, `proactive_conflict@v2`) an answer followed by "Correction: ... here it is:" and a second,
corrected object. Slicing from the first "{" to the last "}" turns that last case into invalid
JSON and the answer was lost. This module decodes every top-level object in the text instead and
picks the model's final word: the LAST object that holds the expected key (or the last object at
all when no key is given). Never raises; anything unreadable is None.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_DECODER = json.JSONDecoder()


def strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object in `text`, in order (inner fences and prose skipped)."""
    body = strip_fences(text)
    try:
        whole = json.loads(body)
    except ValueError:
        whole = None
    if isinstance(whole, dict):
        return [whole]
    found: list[dict[str, Any]] = []
    i = body.find("{")
    while i >= 0:
        try:
            value, end = _DECODER.raw_decode(body, i)
        except ValueError:
            i = body.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        i = body.find("{", end)
    return found


def json_object(text: str, key: str | None = None) -> dict[str, Any] | None:
    """The answer's object: the last one holding `key` (any last one without a key)."""
    objects = json_objects(text)
    if key is not None:
        objects = [o for o in objects if key in o]
    return objects[-1] if objects else None
