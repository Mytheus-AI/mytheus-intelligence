"""Budget units: model tokens weighed by what they cost (the dollar-weighted unit).

Per-tenant budgets used to count raw tokens, so a Haiku gate token and a Sonnet output token
cost the tenant the same, although one is 100 times the price of the other, and thinking and
cache economics were invisible. The budget API is unchanged (`usage-events` `agent_token`
`quantity`, `agent_budgets.tokens_consumed` / `token_quota` / `hard_cap`, the Redis mirror and
`BudgetGate.check(org_id, est_tokens)` all keep their names and integer semantics); what one
"token" in them means is now defined here:

    1 budget unit = the list price of ONE uncached Claude Sonnet 5.5 input token
                  = $2.00 / 1,000,000 = $0.000002

    units = ceil( (uncached_in * in_price
                   + cache_write_in * in_price * 1.25
                   + cache_read_in * in_price * 0.10
                   + out * out_price) / $2.00 )        (prices per million tokens)

Prices (Anthropic first-party list, per million tokens, cached 2026-10-06):

    tier         model               input    output   unit weight in / out
    triage       Claude Haiku 5.5    $0.10    $0.50    0.05 / 0.25
    extract      Claude Sonnet 5.5   $2.00    $10.00   1    / 5
    high_stakes  Claude Opus 5.5     $4.00    $20.00   2    / 10

Cache writes (5-minute TTL) are 1.25x the input price and cache reads 0.1x. Thinking tokens are
output tokens. So a plan's `included_agent_tokens` reads as "this many Sonnet-input-token
equivalents", i.e. `quota * $0.000002` of model list price: 10,000,000 units = $20.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

UNIT_USD_PER_MTOK = 2.00  # one budget unit per million = one Sonnet 5.5 input token
CACHE_WRITE = 1.25
CACHE_READ = 0.10


@dataclass(frozen=True)
class Price:
    input: float  # USD per million tokens
    output: float


PRICES: dict[str, Price] = {
    "triage": Price(0.10, 0.50),  # Claude Haiku 5.5
    "extract": Price(2.00, 10.00),  # Claude Sonnet 5.5
    "high_stakes": Price(4.00, 20.00),  # Claude Opus 5.5
}


def usd(
    tier: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """List price of one call in USD. `input_tokens` is the UNCACHED input only."""
    p = PRICES[tier]
    millionths = (
        input_tokens * p.input
        + cache_write_tokens * p.input * CACHE_WRITE
        + cache_read_tokens * p.input * CACHE_READ
        + output_tokens * p.output
    )
    return millionths / 1_000_000


def budget_units(
    tier: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> int:
    """The call's cost in budget units, rounded up (any call that used tokens costs >= 1)."""
    dollars = usd(
        tier,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    return math.ceil(round(dollars * 1_000_000 / UNIT_USD_PER_MTOK, 6))
