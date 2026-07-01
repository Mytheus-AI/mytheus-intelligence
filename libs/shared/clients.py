"""Bridge to the published `mytheus-events` Avro contracts.

The AI tier CONSUMES the frozen contracts from `mytheus-events`; it never hand-rolls Avro
serializers (per this repo's CLAUDE.md). Until that wheel is installed in the dev env, this module
exposes a lazy loader so importing it never hard-fails the scaffold. The eventing band wires the
real Schema-Registry-backed producers/consumers on top of the returned client.
"""

from __future__ import annotations

from typing import Any


def load_events_client() -> Any:
    """Import and return the generated `mytheus_events` package (the Avro contract client).

    Raises `ModuleNotFoundError` with an actionable hint if the contract package is not yet
    installed in this environment.
    """
    try:
        import mytheus_events
    except ModuleNotFoundError as exc:  # pragma: no cover - wired in the eventing band
        raise ModuleNotFoundError(
            "mytheus-events is not installed. Install the published Avro client "
            "(`pip install mytheus-events`) or add it as a path dependency before "
            "wiring producers/consumers."
        ) from exc
    return mytheus_events
