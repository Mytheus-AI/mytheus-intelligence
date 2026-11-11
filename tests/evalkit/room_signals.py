"""The room signals eval (a3): golden meetings, their source as core-api would send it, the
simulated recordings, and the scoring. Used by `tests/test_room_signals_eval.py`.

Three prompts are measured over one golden set (`tests/eval/room_signals_golden.json`), each
case run through the real writer and router (`RoomSignalsWriter`), so the rules, the parsing,
the quote checks and the never-record filter are scored together with the model's answers:

- `last_word@vN`: **topic_hit_rate** (each expected nudge found for the right person, kind and
  topic keywords) and **nudge_precision** (emitted nudges that were expected);
- `consensus_read@vN`: **stance_accuracy** (decisions whose counts of clear yes / hedged /
  silent / objected / talked over are exactly right), **strength_accuracy**, **ask_accuracy**
  (exactly the right people asked, for the right reason);
- `resume_point@vN`: **resume_open_accuracy**, **resume_topic_hit_rate**,
  **resume_cut_off_accuracy**;
- all: **quote_verbatim_rate** (every quote in the payload is in the transcript) and
  **leak_free** (no forbidden never-record text anywhere; must be 1.0).

Simulated recordings: each golden case carries `simulated`, the ideal answer per prompt.
`write_simulated()` turns them into `tests/eval/recordings/<ref>.json` (`source: simulated`,
the prompt's sha256), which the offline eval replays through `ReplayModel`. The live record mode
(EVAL_LIVE=1 EVAL_RECORD=1, the coordinator's) replaces them with real answers.

    uv run python -m tests.evalkit.room_signals          # rewrite the simulated recordings
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from libs.shared.org_context import CompanyContext
from services.agent_runtime_service.room_signals_rules import (
    RoomSignalsSource,
    norm,
    parse_source,
)
from tests.evalkit.corpus import EVAL_DIR, RECORDINGS_DIR

GOLDEN = EVAL_DIR / "room_signals_golden.json"
REFS = {
    "last_word": "last_word@v2",
    "consensus_read": "consensus_read@v1",
    "resume_point": "resume_point@v1",
}
START = datetime(2026, 11, 11, 15, 0, tzinfo=UTC)
NS = uuid.UUID("6f1d3c52-6a8c-4a0e-9f3b-0a3a5e2d7c11")


def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN.read_text("utf-8"))
    return data


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else int(round(seconds * 1000))


def users_of(case: dict[str, Any]) -> dict[str, uuid.UUID]:
    """label -> a stable synthetic user id (uuid5), so reports and reruns line up."""
    return {p["label"]: uuid.uuid5(NS, f"{case['id']}:{p['label']}") for p in case["participants"]}


def build_source(case: dict[str, Any]) -> tuple[RoomSignalsSource, dict[str, uuid.UUID]]:
    users = users_of(case)
    meeting = uuid.uuid5(NS, case["id"])
    decisions = []
    for i, d in enumerate(case.get("decisions", [])):
        decisions.append(
            {
                "id": str(uuid.uuid5(NS, f"{case['id']}:d{i}")),
                "content": d["content"],
                "status": d.get("status", "proposed"),
                "owner_user_id": str(users[d["owner"]]) if d.get("owner") else None,
                "quote": d.get("quote"),
                "quote_speaker_id": str(users[d["owner"]]) if d.get("owner") else None,
                "quote_start_ms": _ms(d["at"]),
                "quote_end_ms": _ms(d.get("end")),
            }
        )
    body = {
        "meeting_id": str(meeting),
        "title": case["id"].replace("-", " "),
        "confidential": False,
        "host_user_id": str(users[case["participants"][0]["label"]]),
        "scheduled_start": START.isoformat(),
        "scheduled_end": (START + timedelta(minutes=case["scheduled_minutes"])).isoformat(),
        "room_started_at": START.isoformat(),
        "ended_at": (START + timedelta(seconds=case["call_seconds"])).isoformat(),
        "call_ms": case["call_seconds"] * 1000,
        "participants": [
            {
                "user_id": str(users[p["label"]]),
                "name": p["label"].replace("Participant", "Person"),
                "attendance": p.get("attendance", "live"),
                "role_in_meeting": p.get("role", "attendee"),
                "listen_for": p.get("listen_for", []),
            }
            for p in case["participants"]
        ],
        "speakers": {},
        "turns": [
            {
                "speaker_id": str(users[who]),
                "start_ms": _ms(s),
                "end_ms": _ms(e),
                "text": text,
            }
            for who, s, e, text in case["transcript"]
        ],
        "turns_truncated": False,
        "decisions": decisions,
        "agenda": [
            {
                "id": str(uuid.uuid5(NS, f"{case['id']}:a{i}")),
                "title": a["title"],
                "status": a["status"],
                "started_ms": _ms(a.get("started")),
                "ended_ms": _ms(a.get("ended")),
            }
            for i, a in enumerate(case.get("agenda", []))
        ],
        "room_signals": [
            {
                "id": str(uuid.uuid5(NS, f"{case['id']}:s{i}")),
                "kind": s["kind"],
                "user_id": str(users[s["who"]]),
                "at_ms": _ms(s["at"]),
                "end_ms": _ms(s["end"]),
                "detail": s.get("detail", {}),
            }
            for i, s in enumerate(case.get("room_signals", []))
        ],
    }
    source = parse_source(body)
    assert source is not None
    return source, users


def org_context(case: dict[str, Any]) -> CompanyContext:
    return CompanyContext(never_record_topics=tuple(case.get("never_record", [])))


def _keywords(text: str, groups: list[str]) -> bool:
    low = text.lower()
    return all(any(alt in low for alt in g.split("|")) for g in groups)


def _rate(hit: int, total: int) -> float:
    return 1.0 if total == 0 else hit / total


def score(
    runs: list[tuple[dict[str, Any], dict[str, Any], dict[str, uuid.UUID]]],
) -> dict[str, Any]:
    """`runs`: (golden case, the emitted payload, label -> user id)."""
    nudge = [0, 0]
    emitted = [0, 0]
    stance, strength, ask = [0, 0], [0, 0], [0, 0]
    r_open, r_topic, r_cut = [0, 0], [0, 0], [0, 0]
    quotes = [0, 0]
    leak_free = True
    for case, payload, users in runs:
        exp = case["expect"]
        label_of = {str(u): lab for lab, u in users.items()}
        lines = [norm(t[3]) for t in case["transcript"]]
        text = json.dumps(payload).lower()
        if any(f.lower() in text for f in exp.get("forbidden", [])):
            leak_free = False
        got = {label_of.get(e["user_id"]): e for e in payload["last_word"]}
        for n in exp["nudges"]:
            nudge[1] += 1
            e = got.get(n["who"])
            nudge[0] += bool(e and e["kind"] == n["kind"] and _keywords(e["topic"], n["keywords"]))
        expected_who = {n["who"] for n in exp["nudges"]}
        for who in got:
            emitted[1] += 1
            emitted[0] += who in expected_who
        for want, c in zip(exp["consensus"], payload["consensus"] + [None] * 9, strict=False):
            stance[1] += 1
            strength[1] += 1
            ask[1] += 1
            if c is None:
                continue
            counts = {k: c[k] for k in ("yes", "hedged", "silent", "objected", "talked_over")}
            stance[0] += counts == want["counts"]
            strength[0] += c["strength"] == want["strength"]
            asked = {label_of.get(a["user_id"]): a["reason"] for a in c["ask"]}
            ask[0] += asked == want["ask"]
        if len(payload["consensus"]) > len(exp["consensus"]):
            stance[1] += len(payload["consensus"]) - len(exp["consensus"])
        rp, want_rp = payload["resume_point"], exp["resume"]
        r_open[1] += 1
        r_open[0] += (rp is not None) == want_rp["open"]
        if want_rp["open"]:
            r_topic[1] += 1
            r_cut[1] += 1
            if rp is not None:
                r_topic[0] += _keywords(rp["topic"], want_rp["keywords"]) and (
                    rp["reason"] == want_rp["reason"]
                )
                who = label_of.get(rp["cut_off"]["user_id"]) if rp["cut_off"] else None
                r_cut[0] += who == want_rp["cut_off"]
        found = [e["quote"] for e in payload["last_word"] if e["quote"]]
        if rp is not None:
            found += [p["quote"] for p in rp["positions"]]
            found += [rp["cut_off"]["quote"]] if rp["cut_off"] else []
        for q in found:
            quotes[1] += 1
            quotes[0] += any(norm(q.rstrip("…")) in line for line in lines)
    return {
        "topic_hit_rate": _rate(*nudge),
        "nudge_precision": _rate(*emitted),
        "stance_accuracy": _rate(*stance),
        "strength_accuracy": _rate(*strength),
        "ask_accuracy": _rate(*ask),
        "resume_open_accuracy": _rate(*r_open),
        "resume_topic_hit_rate": _rate(*r_topic),
        "resume_cut_off_accuracy": _rate(*r_cut),
        "quote_verbatim_rate": _rate(*quotes),
        "leak_free": 1.0 if leak_free else 0.0,
    }


def simulated_recordings() -> dict[str, dict[str, Any]]:
    """ref -> recording body, from each golden case's `simulated` answers."""
    from tests.evalkit.models import prompt_index

    index = prompt_index()
    out: dict[str, dict[str, Any]] = {}
    for key, ref in REFS.items():
        entries = {
            case["id"]: json.dumps(case["simulated"][key], ensure_ascii=False)
            for case in golden()["cases"]
            if key in case.get("simulated", {})
        }
        out[ref] = {
            "prompt": ref,
            "prompt_sha256": index.sha(ref),
            "source": "simulated",
            "about": (
                "Simulated model answers: the golden cases' ideal answers "
                "(tests/eval/room_signals_golden.json, `simulated`). Replaced by real answers when "
                "the live record mode runs (tests/test_room_signals_eval.py)."
            ),
            "entries": dict(sorted(entries.items())),
        }
    return out


def write_simulated() -> list[Path]:
    written = []
    for ref, body in simulated_recordings().items():
        path = RECORDINGS_DIR / f"{ref}.json"
        if path.exists() and json.loads(path.read_text("utf-8")).get("source") == "recorded":
            continue  # a live recording wins
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in write_simulated():
        print(p)
