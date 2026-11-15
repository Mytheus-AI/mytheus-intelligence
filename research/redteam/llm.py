"""A record-and-batch LLM backend for the red-team harness.

The harness runs the REAL agents (`ContextGapAgent`, `DecisionAgent`, ...) through the REAL
`LiteLLMRouter`; only the router's `completion_fn` is replaced by `BatchLLM`. A call is keyed by
sha256 over (run, model, system, user, thinking, effort, max_tokens):

- a key already in the response store (`responses.jsonl`) is answered from it, in the litellm
  dict shape the router reads;
- a miss is recorded as pending and answered with an empty text (which every agent parses as
  "nothing"), so one pass of the harness collects every request it can reach;
- `flush()` sends the pending requests as ONE Message Batch (50% off), waits, appends every
  answer to the store and records the spend in the shared ledger (`budget.py`, stream
  `redteam`), after `guard()` checked the estimate.

The harness repeats passes until a pass adds no pending request (a gate answer unlocks the
synthesis call behind it). `run` is part of the key, so run 2 is a fresh sample of the same
prompt, while two conditions that produce the byte-identical prompt in one run share one sample
(common random numbers; stated in the README).

Model settings (the shared brief): Sonnet at effort `low` (the router asks `medium` for
syntheses; the override is applied here and recorded with every answer), Haiku with thinking
off exactly as the router sends it.

The API key is read from the environment (`ANTHROPIC_API_KEY`, loaded from the intelligence
`.env` by `load_key()`), never printed or written anywhere.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

API = "https://api.anthropic.com/v1"
EFFORT_OVERRIDE = "low"
STREAM = "redteam"
RUN: contextvars.ContextVar[int] = contextvars.ContextVar("redteam_run", default=1)
CHARS_PER_TOKEN = 3.6


def load_key(env_file: Path) -> None:
    """Put ANTHROPIC_API_KEY (and the workspace id) from a dotenv file into os.environ."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in env_file.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID"):
            os.environ.setdefault(name, value.strip().strip('"').strip("'"))


def budget_module(metrics_root: Path) -> Any:
    sys.path.insert(0, str(metrics_root))
    import budget  # the shared ledger lives outside the repository

    return budget


def _bare(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _text_parts(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))


@dataclass
class Request:
    key: str
    run: int
    params: dict[str, Any]
    est_in: int
    est_out: int


@dataclass
class BatchLLM:
    store: Path
    metrics_root: Path
    pending: dict[str, Request] = field(default_factory=dict)
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    offline: bool = False  # never queue: a miss stays empty (replays, CI)

    def __post_init__(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        if self.store.exists():
            for line in self.store.read_text("utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.answers[row["key"]] = row

    # --- the completion_fn the router calls ------------------------------------------------

    def params_of(self, kw: dict[str, Any]) -> dict[str, Any]:
        messages = kw["messages"]
        system = _text_parts(messages[0]["content"])
        user_content = messages[1]["content"]
        model = _bare(str(kw["model"]))
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": int(kw.get("max_tokens") or 1024),
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        user_content
                        if isinstance(user_content, str)
                        else [
                            {
                                k: v
                                for k, v in part.items()
                                if k in ("type", "text", "cache_control")
                            }
                            for part in user_content
                        ]
                    ),
                }
            ],
        }
        if kw.get("thinking"):
            params["thinking"] = kw["thinking"]
        if kw.get("output_config"):
            params["output_config"] = {**kw["output_config"], "effort": EFFORT_OVERRIDE}
        return params

    @staticmethod
    def key_of(run: int, params: dict[str, Any]) -> str:
        blob = json.dumps({"run": run, **params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    async def __call__(self, **kw: Any) -> dict[str, Any]:
        run = RUN.get()
        params = self.params_of(kw)
        key = self.key_of(run, params)
        hit = self.answers.get(key)
        if hit is not None:
            return {
                "choices": [{"message": {"content": hit["text"]}, "finish_reason": hit["stop"]}],
                "usage": {
                    "prompt_tokens": hit["usage"].get("input_tokens", 0),
                    "completion_tokens": hit["usage"].get("output_tokens", 0),
                },
            }
        if not self.offline and key not in self.pending:
            chars = len(json.dumps(params["system"])) + len(json.dumps(params["messages"]))
            haiku = "haiku" in params["model"]
            self.pending[key] = Request(
                key, run, params, int(chars / CHARS_PER_TOKEN), 80 if haiku else 700
            )
        return {
            "choices": [{"message": {"content": ""}, "finish_reason": "pending"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # --- batches ---------------------------------------------------------------------------

    def estimate(self, requests: list[Request]) -> float:
        """Dollars for these requests: the mean measured batch cost per request of the same
        model in the response store, times 1.5, once there are at least 5 answers of that
        model; before that, a pessimistic bound (every input token uncached, 700 output
        tokens for Sonnet). The first estimate of the baseline run used the bound and was
        about five times the real cost (README)."""
        budget = budget_module(self.metrics_root)
        seen: dict[str, list[float]] = {}
        for a in self.answers.values():
            model = budget._norm(a["model"])
            seen.setdefault(model, []).append(budget.cost(a["usage"], model, batch=True))
        total = 0.0
        for r in requests:
            model = budget._norm(r.params["model"])
            history = seen.get(model, [])
            if len(history) >= 5:
                total += 1.5 * sum(history) / len(history)
                continue
            pin, pout, _read, _write = budget.PRICES[model]
            total += (r.est_in * pin + r.est_out * pout) / 1e6 * 0.5
        return total

    def flush(self, note: str, poll_s: float = 20.0, max_wait_s: float = 6 * 3600) -> float:
        """Send every pending request as one batch; returns the dollars recorded."""
        if not self.pending:
            return 0.0
        budget = budget_module(self.metrics_root)
        requests = list(self.pending.values())
        est = self.estimate(requests)
        budget.guard(STREAM, est)  # raises BudgetExceeded: the caller stops that experiment
        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
        body = {"requests": [{"custom_id": r.key[:64], "params": r.params} for r in requests]}
        by_id = {r.key[:64]: r for r in requests}
        with httpx.Client(timeout=120) as client:
            res = client.post(f"{API}/messages/batches", headers=headers, json=body)
            if res.status_code >= 300:
                raise RuntimeError(f"batch create failed {res.status_code}: {res.text[:500]}")
            batch = res.json()
            print(f"[llm] batch {batch['id']} with {len(requests)} requests, est ${est:.3f}")
            start = time.time()
            while batch.get("processing_status") != "ended":
                if time.time() - start > max_wait_s:
                    raise RuntimeError(f"batch {batch['id']} did not end in {max_wait_s}s")
                time.sleep(poll_s)
                batch = client.get(f"{API}/messages/batches/{batch['id']}", headers=headers).json()
            results = client.get(batch["results_url"], headers=headers)
        usd = 0.0
        sums: dict[str, dict[str, int]] = {}
        with self.store.open("a", encoding="utf-8") as out:
            for line in results.text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                req = by_id.get(row["custom_id"])
                if req is None:
                    continue
                result = row["result"]
                if result.get("type") != "succeeded":
                    print(f"[llm] {row['custom_id'][:12]} {result.get('type')}: {result}")
                    continue
                msg = result["message"]
                text = "".join(b.get("text", "") for b in msg["content"] if b["type"] == "text")
                usage = msg.get("usage", {})
                model = budget._norm(msg.get("model", req.params["model"]))
                agg = sums.setdefault(model, {})
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                ):
                    agg[k] = agg.get(k, 0) + int(usage.get(k) or 0)
                stop = {"max_tokens": "length"}.get(msg.get("stop_reason"), "stop")
                answer = {
                    "key": req.key,
                    "run": req.run,
                    "model": msg.get("model", req.params["model"]),
                    "effort": (req.params.get("output_config") or {}).get("effort"),
                    "text": text,
                    "stop": stop,
                    "usage": usage,
                    "batch_id": batch["id"],
                    "ts": time.time(),
                    "system_sha": hashlib.sha256(
                        req.params["system"][0]["text"].encode()
                    ).hexdigest()[:16],
                    "user": _text_parts(req.params["messages"][0]["content"]),
                }
                out.write(json.dumps(answer, ensure_ascii=False) + "\n")
                self.answers[req.key] = answer
                self.pending.pop(req.key, None)
        for model, usage in sums.items():
            usd += budget.record(STREAM, usage, model, batch=True, note=note)
        failed = len(self.pending)
        self.pending.clear()  # failures are retried by the next pass, never silently reused
        print(f"[llm] batch done: ${usd:.4f} recorded, {failed} failed")
        return usd
