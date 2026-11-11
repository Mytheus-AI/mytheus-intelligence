"""Generate the eval corpus and the simulated recordings. Deterministic: fixed seed, no clock.

    uv run python -m tests.evalkit.generate          # write tests/eval/corpus + recordings
    uv run python -m tests.evalkit.generate --check  # exit 1 if the committed files differ

What it writes:
- `corpus/windows.json`: the hand-curated `corpus/core.json` (24 windows) plus 120 generated
  windows (`GENERATED`), with ASR-style noise: fillers and repeated words, mis-heard names,
  windows cut mid-sentence at either end, crosstalk interjections splitting a turn, and
  diarization label churn (one person's later turns under a new speaker label).
- `corpus/persona_pairs.json`: 24 (window, persona A, persona B) pairs for the fact-overlap suite.
- `recordings/<prompt>@<version>.json`: what the model answered per case, replayed offline.
  Only prompts in `SIMULATED_REFS` are ever simulated (by `simulate.py`); a recording whose
  `source` is `recorded` (written by the live record mode) is never overwritten here. So a NEW
  prompt version cannot get a free pass from this script: it needs a live recording, which is
  exactly the gate ("A prompt change must beat the current default before it can roll out").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from typing import Any

from tests.evalkit import banks
from tests.evalkit.corpus import (
    CORPUS_DIR,
    RECORDINGS_DIR,
    SEED,
    corpus_overrides,
    load,
    window_by_id,
    windows,
)

# The prompt versions this script may simulate: the registry defaults when the harness was
# written (2026-11-11) and the deliberately worse fixtures. Anything else needs a live recording.
SIMULATED_REFS = frozenset(
    {
        "decision_gate@v2",
        "decision_agent@v2",
        "action_gate@v3",
        "action_agent@v2",
        "personal_listener_gate@v4",
        "personal_listener@v3",
        "meeting_agent@v2",
        "proactive_conflict@v2",
        "proactive_stakeholder@v2",
        "decision_gate@v0-worse",
        "decision_agent@v0-worse",
        "action_gate@v0-worse",
        "action_agent@v0-worse",
        "personal_listener@v0-worse",
        "meeting_agent@v0-worse",
        "proactive_conflict@v0-worse",
        # rag-service's decision-conflict check (A1). Simulated by its own eval module,
        # tests/test_decision_check_eval.py `simulated_answer`, not by this harness; the live
        # record mode there (EVAL_LIVE=1 EVAL_RECORD=1) replaces it with `source: recorded`.
        "decision_conflict_check@v1",
        # a3 room signals: simulated from the golden cases' ideal answers
        # (tests/evalkit/room_signals.py) until the coordinator's live record run replaces them
        "last_word@v1",
        "consensus_read@v1",
        "resume_point@v1",
        # A briefing from your agent (tests/test_briefing_eval.py): hand-written stand-ins for
        # the golden set, never generated here, until the live record run replaces them
        # (`source: recorded`): EVAL_LIVE=1 EVAL_RECORD=1 ... tests/test_briefing_eval.py -k live.
        "briefing@v1",
        "briefing@v0-worse",
        # Decisions that know why (2026-11-11): simulated until the first live recording
        # (EVAL_LIVE=1 EVAL_RECORD=1 EVAL_SUITES=decision_assumptions,decision_ripple,
        # assumption_check), which then wins over these (a `recorded` file is never rewritten).
        "decision_assumptions@v1",
        "decision_ripple@v1",
        "assumption_check@v1",
        "decision_assumptions@v0-worse",
        "decision_ripple@v0-worse",
        "assumption_check@v0-worse",
    }
)

GENERATED = {
    "decision": 22,
    "action": 22,
    "decision_action": 13,
    "lookalike": 22,
    "question": 12,
    "never_record": 14,
    "smalltalk": 15,
}
PAIRS = (("eng", "design"), ("sales", "finance"), ("eng", "finance"), ("design", "sales"))


# --- windows ----------------------------------------------------------------------------------


class _Builder:
    """One window under construction: turns as [speaker, text], speakers in order."""

    def __init__(self, rng: random.Random, n_speakers: int) -> None:
        self.rng = rng
        self.speakers: list[str] = [f"S{i + 1}" for i in range(n_speakers)]
        self.aliases: dict[str, str | None] = {s: None for s in self.speakers}
        self.turns: list[list[str]] = []
        self.key_turns: set[int] = set()  # turns that carry an item (never cut or churned away)

    def say(self, speaker: str, text: str, key: bool = False) -> int:
        self.turns.append([speaker, text])
        if key:
            self.key_turns.add(len(self.turns) - 1)
        return len(self.turns) - 1

    def other(self, *not_these: str) -> str:
        pool = [s for s in self.speakers if s not in not_these] or self.speakers
        return self.rng.choice(pool)


def _name(rng: random.Random) -> str:
    return rng.choice(sorted(banks.NAMES))


def _decision(b: _Builder, out: dict[str, Any]) -> None:
    rng = b.rng
    d = rng.choice(banks.DECISIONS)
    speaker = rng.choice(b.speakers)
    if rng.random() < 0.6:
        opener = rng.choice(banks.OPENERS)
        if rng.random() < 0.5:
            opener = f"{_name(rng)}, {opener[0].lower()}{opener[1:]}"
        b.say(b.other(speaker), opener)
    b.say(speaker, d["say"], key=True)
    b.say(b.other(speaker), rng.choice(banks.ACKS))
    out["decisions"].append(
        {"content": d["content"], "keywords": d["keywords"], "domain": d["domain"]}
    )


def _action(b: _Builder, out: dict[str, Any]) -> None:
    rng = b.rng
    a = rng.choice(banks.ACTIONS)
    owner = rng.choice(b.speakers)
    if rng.random() < 0.5:
        b.say(b.other(owner), f"{_name(rng)}, can you {a['ask']}?")
        b.say(owner, f"Yeah. {a['say']}", key=True)
    else:
        b.say(b.other(owner), rng.choice(banks.WHO_ASKS))
        b.say(owner, a["say"], key=True)
    if rng.random() < 0.6:
        b.say(b.other(owner), rng.choice(banks.THANKS))
    out["actions"].append(
        {
            "content": a["content"],
            "keywords": a["keywords"],
            "domain": a["domain"],
            "assignee": owner,
            "due_date": a["due"],
        }
    )


def _lookalike(b: _Builder, out: dict[str, Any], kind: str | None = None) -> None:
    rng = b.rng
    kind = kind or rng.choice(["decision", "action"])
    bank = banks.DECISION_LOOKALIKES if kind == "decision" else banks.ACTION_LOOKALIKES
    item = rng.choice(bank)
    speaker = rng.choice(b.speakers)
    b.say(speaker, item["say"])
    b.say(b.other(speaker), rng.choice(banks.LOOKALIKE_REPLIES))
    out["lookalikes"].append(
        {"kind": kind, "content": item["content"], "keywords": item["keywords"]}
    )


def _noise(b: _Builder, kinds: list[str]) -> None:
    rng = b.rng
    if "misheard_name" in kinds:
        named = [i for i, (_, t) in enumerate(b.turns) if any(n in t for n in banks.NAMES)]
        if not named:
            b.turns.insert(0, [b.speakers[0], f"{_name(rng)}, you're up."])
            b.key_turns = {i + 1 for i in b.key_turns}
            named = [0]
        for i in named:
            for name, heard in banks.NAMES.items():
                b.turns[i][1] = b.turns[i][1].replace(name, rng.choice(heard))
    if "crosstalk" in kinds and b.key_turns:
        i = min(b.key_turns)
        speaker, text = b.turns[i]
        cut = text.find(", ")
        if cut > 0:
            first, rest = text[: cut + 1], text[cut + 2 :]
            b.turns[i] = [speaker, first]
            b.turns.insert(i + 1, [b.other(speaker), rng.choice(banks.INTERJECTIONS)])
            b.turns.insert(i + 2, [speaker, rest])
            b.key_turns = {k if k < i else k + 2 for k in b.key_turns} | {i, i + 2}
        else:
            b.turns.insert(i + 1, [b.other(speaker), rng.choice(banks.INTERJECTIONS)])
            b.key_turns = {k if k <= i else k + 1 for k in b.key_turns}
    if "fillers" in kinds:
        for turn in b.turns:
            words = turn[1].split(" ")
            if len(words) > 3 and rng.random() < 0.6:
                at = rng.randint(1, min(3, len(words) - 1))
                words.insert(at, rng.choice(banks.FILLERS))
            if len(words) > 4 and rng.random() < 0.25:
                at = rng.randint(0, len(words) - 2)
                if words[at].isalpha() and len(words[at]) <= 4:
                    words.insert(at, words[at])
            turn[1] = " ".join(words)
    if "partial" in kinds:
        b.turns.insert(0, [b.other(), rng.choice(banks.LEAD_FRAGMENTS)])
        b.key_turns = {k + 1 for k in b.key_turns}
        b.turns.append([b.other(), rng.choice(banks.TAIL_FRAGMENTS)])
    if "label_churn" in kinds:
        spoken: dict[str, list[int]] = {}
        for i, (s, _) in enumerate(b.turns):
            spoken.setdefault(s, []).append(i)
        candidates = sorted(s for s, idx in spoken.items() if len(idx) >= 2)
        if candidates:
            who = rng.choice(candidates)
            new = f"S{len(b.speakers) + 1}"
            b.speakers.append(new)
            b.aliases[new] = who
            for i in spoken[who][1:]:
                b.turns[i][0] = new


def _window(rng: random.Random, category: str, index: int) -> dict[str, Any]:
    b = _Builder(rng, rng.choice([2, 3, 3, 4]))
    out: dict[str, Any] = {"decisions": [], "actions": [], "lookalikes": [], "never_record": None}
    if category == "decision":
        _decision(b, out)
        if rng.random() < 0.3:
            _lookalike(b, out, "decision")
    elif category == "action":
        _action(b, out)
        if rng.random() < 0.3:
            _lookalike(b, out, "action")
    elif category == "decision_action":
        _decision(b, out)
        _action(b, out)
    elif category == "lookalike":
        _lookalike(b, out)
        if rng.random() < 0.4:
            _lookalike(b, out)
    elif category == "question":
        q, reply = rng.choice(banks.QUESTIONS)
        s = rng.choice(b.speakers)
        b.say(s, q)
        b.say(b.other(s), reply)
    elif category == "smalltalk":
        for i, line in enumerate(rng.choice(banks.SMALLTALK)):
            b.say(b.speakers[i % len(b.speakers)], line)
    elif category == "never_record":
        tpl = rng.choice(banks.NEVER)
        canary = rng.choice(tpl["canaries"])
        for i, line in enumerate(tpl["turns"]):
            b.say(b.speakers[i % len(b.speakers)], line.format(canary=canary), key=True)
        out["never_record"] = {"topic": tpl["topic"], "mode": tpl["mode"], "canary": canary}
    kinds = sorted(
        k
        for k, p in (
            ("fillers", 0.7),
            ("misheard_name", 0.35),
            ("partial", 0.3),
            ("crosstalk", 0.3),
            ("label_churn", 0.25),
        )
        if rng.random() < p
    )
    if category == "never_record":
        # never cut or reword the canary away: names, fillers and churn only
        kinds = [k for k in kinds if k not in ("partial", "crosstalk")]
    _noise(b, kinds)
    used = []
    for s, _ in b.turns:
        if s not in used:
            used.append(s)
    return {
        "id": f"w-gen-{index:03d}",
        "source": "generated",
        "category": category,
        "noise": kinds,
        "speakers": {s: b.aliases.get(s) for s in sorted(used)},
        "turns": [{"speaker": s, "text": t} for s, t in b.turns],
        **out,
    }


def build_windows() -> dict[str, Any]:
    rng = random.Random(SEED)
    core = [{**w, "source": "core"} for w in load("core.json")["windows"]]
    generated = []
    n = 0
    for category, count in GENERATED.items():
        for _ in range(count):
            n += 1
            generated.append(_window(rng, category, n))
    return {
        "about": (
            "The eval corpus: corpus/core.json (hand-curated) plus windows generated by "
            "tests/evalkit/generate.py with seed 20261111. Do not edit by hand."
        ),
        "seed": SEED,
        "windows": [*core, *generated],
    }


def build_pairs(corpus: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(SEED + 1)
    rich = [w for w in corpus["windows"] if len(w["decisions"]) + len(w["actions"]) >= 2]
    single = [
        w
        for w in corpus["windows"]
        if len(w["decisions"]) + len(w["actions"]) == 1 and not w["never_record"]
    ]
    chosen = rich[:16] + rng.sample(single, 24 - min(16, len(rich)))
    pairs = []
    for i, w in enumerate(chosen):
        a, b = PAIRS[i % len(PAIRS)]
        pairs.append({"id": f"p-{i + 1:02d}", "window": w["id"], "a": a, "b": b})
    return {
        "about": (
            "Persona pairs: the same window seen by two participants with different roles. Both "
            "notes must capture the same facts (the window's decisions and actions)."
        ),
        "pairs": pairs,
    }


# --- recordings --------------------------------------------------------------------------------


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def simulate_recordings() -> dict[str, dict[str, Any]]:
    """Run every suite in both variants against the simulated model; return the recordings
    (ref -> file body) for the refs this script may simulate."""
    from tests.evalkit.harness import VARIANTS, run_all
    from tests.evalkit.models import SimulatedModel, prompt_index

    model = SimulatedModel()
    for variant in VARIANTS:
        asyncio.run(run_all(model, variant))
    if model.errors:
        raise SystemExit("\n".join(model.errors))
    out: dict[str, dict[str, Any]] = {}
    index = prompt_index()
    for ref, entries in sorted(model.recorded.items()):
        if ref not in SIMULATED_REFS:
            path = RECORDINGS_DIR / f"{ref}.json"
            if path.exists() and json.loads(path.read_text("utf-8")).get("source") == "recorded":
                continue  # a newer default with a live recording: nothing to simulate
            raise SystemExit(
                f"{ref} has no live recording. Run the live record mode (tests/evalkit/harness.py"
                " docstring) instead of simulating a new prompt version."
            )
        out[ref] = {
            "prompt": ref,
            "prompt_sha256": index.sha(ref),
            "source": "simulated",
            "about": (
                "Simulated model answers (tests/evalkit/simulate.py), seed 20261111. Replaced "
                "by real answers when the live record mode runs."
            ),
            "entries": dict(sorted(entries.items())),
        }
    return out


def outputs() -> dict[str, str]:
    """Every generated file: relative path -> content."""
    corpus = build_windows()
    files = {
        "corpus/windows.json": _dump(corpus),
        "corpus/persona_pairs.json": _dump(build_pairs(corpus)),
    }
    # the harness reads these instead of the committed files while it simulates
    corpus_overrides["windows.json"] = corpus
    corpus_overrides["persona_pairs.json"] = json.loads(files["corpus/persona_pairs.json"])
    windows.cache_clear()
    window_by_id.cache_clear()
    for ref, body in simulate_recordings().items():
        path = RECORDINGS_DIR / f"{ref}.json"
        if path.exists() and json.loads(path.read_text("utf-8")).get("source") == "recorded":
            continue  # a live recording wins
        files[f"recordings/{ref}.json"] = _dump(body)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed files differ")
    args = parser.parse_args(argv)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for rel, body in outputs().items():
        path = CORPUS_DIR.parent / rel
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == body:
            continue
        stale.append(rel)
        if not args.check:
            path.write_text(body, encoding="utf-8")
    if args.check and stale:
        print("out of date: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(("up to date" if not stale else "wrote " + ", ".join(stale)) + f" ({len(stale)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
