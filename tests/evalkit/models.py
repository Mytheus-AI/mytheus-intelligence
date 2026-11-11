"""The completion functions the eval harness hands to `LiteLLMRouter`.

The router calls `completion_fn(model=..., messages=[system, user], max_tokens=..., ...)`. Each
model here identifies the prompt from the system message (sha256 of the system text, over the
repo's prompts and the eval fixtures) and the case from what the harness set on it (`case`, and
`persona` for the PersonalListener), so a recording is keyed `(prompt@version, case)`.

- `SimulatedModel`: answers from `simulate.py` and remembers every answer (the generator writes
  them out as the simulated recordings).
- `ReplayModel`: answers from `tests/eval/recordings/<ref>.json`. A recording made for a
  different prompt text (sha mismatch) or a missing one is an error, collected in `errors` and
  raised by the harness: a changed prompt must be re-recorded live before the gate can pass.
- `LiveModel`: calls the real provider through litellm (`EVAL_LIVE=1` only) and, with
  `record=True`, remembers the answers so the harness can write them as `source: recorded`.

All of them return litellm-shaped responses (a dict, or an async iterator of chunks when the
router streams), with a token count so the router meters and the telemetry counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from libs.shared.agent_metrics import output_ok
from libs.shared.prompts import USER_MARKER
from services.agent_runtime_service.consumer import PROMPTS_DIR
from tests.evalkit import simulate
from tests.evalkit.corpus import FIXTURE_PROMPTS_DIR, RECORDINGS_DIR


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _template_regex(template: str) -> re.Pattern[str]:
    parts = _PLACEHOLDER.split(template)
    # split() alternates literal, name, literal, ...: literals escaped, names as lazy wildcards
    body = "".join(re.escape(p) if i % 2 == 0 else "(?:.*?)" for i, p in enumerate(parts))
    return re.compile(body, re.DOTALL)


@dataclass(frozen=True)
class PromptIndex:
    by_system: dict[str, list[tuple[str, re.Pattern[str]]]]  # sha256(system) -> refs
    file_sha: dict[str, str]  # ref -> sha256(system + user template)

    def ref_for(self, system: str, user: str = "") -> str:
        """The prompt a call was rendered from. Two versions may share a system prompt and differ
        only in the user template: then the template that renders `user` wins."""
        found = self.by_system.get(_sha(system.strip()), [])
        if len(found) > 1:
            fits = [ref for ref, rx in found if rx.fullmatch(user.strip())]
            found = [(ref, rx) for ref, rx in found if ref in fits] or found
        return found[-1][0] if found else "unknown@unknown"

    def sha(self, ref: str) -> str:
        return self.file_sha[ref]


@cache
def prompt_index() -> PromptIndex:
    by_system: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    file_sha: dict[str, str] = {}
    for folder in (PROMPTS_DIR, FIXTURE_PROMPTS_DIR):
        for path in sorted(Path(folder).glob("*@*.md")):
            ref = path.stem
            raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            system, _, user = raw.partition(f"\n{USER_MARKER}\n")
            entry = (ref, _template_regex(user.strip()))
            by_system.setdefault(_sha(system.strip()), []).append(entry)
            file_sha[ref] = _sha(f"{system.strip()}\n{USER_MARKER}\n{user.strip()}")
    return PromptIndex(by_system, file_sha)


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))


async def _chunks(text: str, usage: dict[str, int]) -> AsyncIterator[dict[str, Any]]:
    step = 24
    for i in range(0, len(text), step):
        piece = text[i : i + step]
        yield {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
    yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}], "usage": usage}


@dataclass
class Call:
    ref: str
    key: str
    text: str
    ok: bool | None  # the answer format check (`agent_metrics.output_ok`)


@dataclass
class Model:
    case: str = ""
    persona: str = ""
    calls: list[Call] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def key_for(self, ref: str) -> str:
        if ref.startswith("personal_listener") or ref.startswith("note_quality"):
            return f"{self.case}:{self.persona}"
        return self.case

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        raise NotImplementedError

    async def __call__(self, **kw: Any) -> Any:
        system = kw["messages"][0]["content"][0]["text"]
        user = _user_text(kw["messages"][1]["content"])
        ref = prompt_index().ref_for(system, user)
        key = self.key_for(ref)
        text, usage = await self.answer(ref, key, kw)
        if not usage:
            usage = {
                "prompt_tokens": (len(system) + len(user)) // 4,
                "completion_tokens": max(1, len(text) // 4),
            }
        self.calls.append(Call(ref, key, text, output_ok(ref, text)))
        if kw.get("stream"):
            return _chunks(text, usage)
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": usage,
        }

    def parse_failure_rate(self, prefix: str) -> float:
        checked = [c for c in self.calls if c.ok is not None and c.ref.startswith(prefix)]
        return sum(1 for c in checked if not c.ok) / len(checked) if checked else 0.0


@dataclass
class SimulatedModel(Model):
    recorded: dict[str, dict[str, str]] = field(default_factory=dict)

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        try:
            text = simulate.answer(ref, key)
        except Exception as exc:  # noqa: BLE001 - surfaced by the harness
            self.errors.append(f"simulate {ref} {key}: {exc!r}")
            text = ""
        self.recorded.setdefault(ref, {})[key] = text
        return text, {}


class StaleRecording(RuntimeError):
    pass


@dataclass
class ReplayModel(Model):
    folder: Path = RECORDINGS_DIR
    _files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _recording(self, ref: str) -> dict[str, Any]:
        if ref not in self._files:
            path = self.folder / f"{ref}.json"
            if not path.exists():
                raise StaleRecording(
                    f"no recording for {ref}: a new or changed prompt must be recorded live "
                    "(EVAL_LIVE=1 EVAL_RECORD=1, see tests/evalkit/harness.py)"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("prompt_sha256") != prompt_index().sha(ref):
                raise StaleRecording(
                    f"{ref} changed since it was recorded: re-record it live "
                    "(EVAL_LIVE=1 EVAL_RECORD=1, see tests/evalkit/harness.py)"
                )
            self._files[ref] = data
        return self._files[ref]

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        try:
            entries = self._recording(ref)["entries"]
            if key not in entries:
                raise StaleRecording(f"{ref} has no recorded answer for case {key}")
            return str(entries[key]), {}
        except StaleRecording as exc:
            self.errors.append(str(exc))
            raise

    def sources(self) -> dict[str, str]:
        return {ref: str(d.get("source")) for ref, d in sorted(self._files.items())}


@dataclass
class LiveModel(Model):
    """The real provider. Streaming requests are answered non-streaming and replayed as chunks
    (the agents' stream handling still runs; the answer is the same).

    `shared` (pass one dict to every LiveModel of a run) answers a `(prompt, case)` already
    asked in this run with the same messages from that first answer, as a replay would.
    Without it, a prompt both variants call (the gates, the stakeholder gate) was answered
    twice and the second, non-deterministic answer overwrote the first in the recording, so the
    recorded gate no longer matched the extraction it had let through (the first live run: a
    `personal_listener@v4` case with no recorded answer). It also halves the spend on them.
    The messages are part of the key: the judge asks every note under one empty case key."""

    recorded: dict[str, dict[str, str]] = field(default_factory=dict)
    shared: dict[str, dict[str, str]] = field(default_factory=dict)

    async def answer(self, ref: str, key: str, kw: dict[str, Any]) -> tuple[str, dict[str, int]]:
        import litellm

        asked = f"{key}\x00{_sha(json.dumps(kw.get('messages'), sort_keys=True, default=str))}"
        seen = self.shared.get(ref, {})
        if asked in seen:  # the same prompt, case and message: the first answer, as a replay
            self.recorded.setdefault(ref, {})[key] = seen[asked]
            return seen[asked], {}
        args = {k: v for k, v in kw.items() if k not in ("stream", "stream_options")}
        resp = await litellm.acompletion(**args)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        counts = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        self.recorded.setdefault(ref, {})[key] = text
        self.shared.setdefault(ref, {})[asked] = text
        return text, counts


def write_recordings(recorded: dict[str, dict[str, str]], source: str) -> list[str]:
    """Write (merge) recordings for every ref; returns the refs written."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    index = prompt_index()
    written = []
    for ref, entries in sorted(recorded.items()):
        path = RECORDINGS_DIR / f"{ref}.json"
        old: dict[str, Any] = {}
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("prompt_sha256") != index.sha(ref):
                old = {}
        merged = (
            {**(old.get("entries") or {}), **entries} if source == old.get("source") else entries
        )
        body = {
            "prompt": ref,
            "prompt_sha256": index.sha(ref),
            "source": source,
            "about": "Real model answers recorded by the live eval (EVAL_LIVE=1 EVAL_RECORD=1).",
            "entries": dict(sorted(merged.items())),
        }
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(ref)
    return written
