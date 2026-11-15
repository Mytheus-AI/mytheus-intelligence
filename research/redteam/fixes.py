"""The post-fix conditions: everything on (FIX), leave-one-out ablations, and the extractive
option. Flags are the production ones (`services/...`), set only while a condition runs."""

from __future__ import annotations

from dataclasses import replace

from research.redteam.harness import CLASSES, Condition, template

ON: dict[str, str] = {
    "A2A_RESPONSE_MAC_BODY": "1",
    "A2A_MEETING_CONTEXT_EXPLICIT": "1",
    "A2A_GRANT_CACHE_HMAC": "1",
    "A2A_GRANT_CACHE_SECRET": "redteam-grant-cache-secret-0123456789abcdef",
    "CONTEXT_GAP_USER_GAPS_PER_HOUR": "20",
    "CONTEXT_GAP_MEETING_A2A_CAP": "3",
    "CONTEXT_GAP_EXTRACTIVE": "0",
}
# The tenant side of "meeting_context needs an explicit grant": the platform template covers
# decision_lineage and action_status org-wide, and meeting_context is granted explicitly, here
# at same_team (a backend template change, recommended in the README).
FIX_GRANTS = frozenset(
    template("org", ("decision_lineage", "action_status"))
    | template("same_team", ("meeting_context",))
)
OLD_GRANTS = frozenset(template("org", CLASSES))


# Every control off: what origin/main does, for runs of a baseline from this branch.
OFF: dict[str, str] = {
    "A2A_RESPONSE_MAC_BODY": "0",
    "A2A_MEETING_CONTEXT_EXPLICIT": "0",
    "A2A_GRANT_CACHE_HMAC": "0",
    "CONTEXT_GAP_USER_GAPS_PER_HOUR": "0",
    "CONTEXT_GAP_MEETING_A2A_CAP": "0",
    "CONTEXT_GAP_EXTRACTIVE": "0",
}


def as_origin(cond: Condition) -> Condition:
    """A baseline condition with every post-fix flag off and the v2 prompt pinned."""
    return replace(cond, env=tuple(OFF.items()), pins=(("context_gap_agent", "v2"),))


def env(**over: str) -> tuple[tuple[str, str], ...]:
    return tuple({**ON, **over}.items())


FIX_CONDITIONS: dict[str, Condition] = {
    "FIX": Condition("FIX", grants=FIX_GRANTS, env=env(), note="every control on"),
    "FIX-mac": Condition(
        "FIX-mac", grants=FIX_GRANTS, env=env(A2A_RESPONSE_MAC_BODY="0"), note="LOO: body MAC"
    ),
    "FIX-mc": Condition(
        "FIX-mc",
        grants=OLD_GRANTS,
        env=env(A2A_MEETING_CONTEXT_EXPLICIT="0"),
        note="LOO: meeting_context default class and template grant",
    ),
    "FIX-limits": Condition(
        "FIX-limits",
        grants=FIX_GRANTS,
        env=env(CONTEXT_GAP_USER_GAPS_PER_HOUR="0", CONTEXT_GAP_MEETING_A2A_CAP="0"),
        note="LOO: rate limit and extraction cap",
    ),
    "FIX-spot": Condition(
        "FIX-spot",
        grants=FIX_GRANTS,
        env=env(),
        pins=(("context_gap_agent", "v2"),),
        note="LOO: spotlighting (prompt v2)",
    ),
    "FIX-hmac": Condition(
        "FIX-hmac", grants=FIX_GRANTS, env=env(A2A_GRANT_CACHE_HMAC="0"), note="LOO: sealed cache"
    ),
    "FIX+extract": Condition(
        "FIX+extract",
        grants=FIX_GRANTS,
        env=env(CONTEXT_GAP_EXTRACTIVE="1"),
        note="every control plus the extractive-only fill",
    ),
    "B1+spot": Condition(
        "B1+spot",
        grants=OLD_GRANTS,
        env=env(
            A2A_MEETING_CONTEXT_EXPLICIT="0",
            CONTEXT_GAP_USER_GAPS_PER_HOUR="0",
            CONTEXT_GAP_MEETING_A2A_CAP="0",
        ),
        note="origin/main policy with spotlighting only",
    ),
}
