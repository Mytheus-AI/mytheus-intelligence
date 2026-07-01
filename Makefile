.PHONY: install lint typecheck test ci

install:
	uv sync

lint:
	uv run ruff check . && uv run black --check .

typecheck:
	uv run mypy .

test:
	uv run pytest

ci: lint typecheck test
