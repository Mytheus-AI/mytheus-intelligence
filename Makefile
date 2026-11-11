.PHONY: install lint typecheck test ci eval eval-live eval-record eval-fixtures

install:
	uv sync

lint:
	uv run ruff check . && uv run black --check .

typecheck:
	uv run mypy .

test:
	uv run pytest

ci: lint typecheck test

# Evals that can fail (tests/evalkit/harness.py). Offline replays recordings: no spend.
eval:
	uv run pytest -m eval tests/test_eval_harness.py

# Real model calls, both prompt variants and the note judge (about 800 calls, 3-5 USD).
eval-live:
	EVAL_LIVE=1 uv run pytest -m eval tests/test_eval_harness.py -k live -s

# The same, and the answers replace the recordings the offline gate replays.
eval-record:
	EVAL_LIVE=1 EVAL_RECORD=1 uv run pytest -m eval tests/test_eval_harness.py -k live -s

# Regenerate the corpus and the simulated recordings (seeded, deterministic).
eval-fixtures:
	uv run python -m tests.evalkit.generate
