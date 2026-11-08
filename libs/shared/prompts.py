"""`PromptLoader`: versioned prompt templates, `{prompt_key}@{version}`, tenant pins over the
global default. Plan 7-9 section 3.2 step 4, doc 08 sections 3.3 and 6.2.

Templates are files, reviewable in git: `<prompts_dir>/<prompt_key>@<version>.md`. The
default version per key and any tenant pins live in `<prompts_dir>/registry.json`:

    {"defaults": {"personal_listener": "v1"},
     "pins": {"<org_id>": {"personal_listener": "v2"}}}

A pin is the canary mechanism: pin ~1% of tenants to the candidate, watch, then advance the
default. Rolling back is deleting the pin; nothing is redeployed except this file.

A template is split on a line holding only `---user---`: everything above is the system
prompt (stable, so Anthropic prompt caching applies to it), everything below is the per-call
user message, rendered with `str.format_map`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from uuid import UUID

USER_MARKER = "---user---"


@dataclass(frozen=True)
class Prompt:
    key: str
    version: str
    system: str
    user_template: str

    @property
    def ref(self) -> str:
        return f"{self.key}@{self.version}"

    def render_user(self, **values: Any) -> str:
        return self.user_template.format_map(values)


class PromptLoader:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        registry = json.loads((prompts_dir / "registry.json").read_text(encoding="utf-8"))
        self._defaults: dict[str, str] = registry.get("defaults", {})
        self._pins: dict[str, dict[str, str]] = registry.get("pins", {})

    def resolve(self, prompt_key: str, org_id: UUID) -> str:
        """The version this tenant runs: its pin if it has one, else the global default."""
        pinned = self._pins.get(str(org_id), {}).get(prompt_key)
        if pinned:
            return pinned
        if prompt_key not in self._defaults:
            raise KeyError(f"no default version registered for prompt {prompt_key!r}")
        return self._defaults[prompt_key]

    def load(self, prompt_key: str, version: str) -> Prompt:
        return _read(self._dir, prompt_key, version)


@cache
def _read(prompts_dir: Path, key: str, version: str) -> Prompt:
    # CRLF -> LF: a Windows checkout (core.autocrlf) must not hide the `---user---` line
    raw = (prompts_dir / f"{key}@{version}.md").read_text(encoding="utf-8")
    raw = raw.replace("\r\n", "\n")
    system, sep, user = raw.partition(f"\n{USER_MARKER}\n")
    if not sep:
        raise ValueError(f"{key}@{version} has no '{USER_MARKER}' line")
    return Prompt(key=key, version=version, system=system.strip(), user_template=user.strip())
