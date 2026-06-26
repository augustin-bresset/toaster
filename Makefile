.PHONY: env install test lint format check

# Create the virtual environment and install the package with dev extras.
env:
	uv venv
	uv pip install -e ".[dev]"

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

# What CI should run: lint + tests.
check: lint test
