"""Load Hugging Face models from the local cache first.

With the weights already on disk, the default loader still calls the Hub on every start to check
for updates: measured 6.9 s versus 0.17 s with `local_files_only=True` for BGE-small-en on CPU.
This tries the cache first and only goes online when the files are not there (a first run on a
developer machine). Images bake the weights and set `HF_HUB_OFFLINE=1`, so they never go online.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def load_local_first(factory: Callable[..., T], name: str, **kwargs: Any) -> T:
    try:
        return factory(name, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        # Not in the local cache (huggingface_hub raises OSError for a missing local file).
        return factory(name, **kwargs)
