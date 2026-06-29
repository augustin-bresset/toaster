# Contributing to Toaster

Thanks for helping! This is a small, focused project — the bar is simply:
**green checks, and the architecture stays clean.**

## Setup

```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
pre-commit install                         # optional: ruff on every commit
```

## Before you push

```bash
make check     # ruff check + ruff format --check + pytest
```

CI runs exactly this on Python 3.11 and 3.12 for every push and PR. Please keep
it green and add a test for any behaviour change (the suite is headless and
fast — no GUI needed).

## Architecture rules (please keep these)

- `toaster.core` is **numpy-only and headless** — it must never import a GUI,
  a renderer, or a web framework. `tests/test_architecture.py` enforces this.
- Dependencies flow one way: `core` ← `io / segment / persistence` ←
  `interaction` ← `api / web`. The browser receives only numpy arrays and a flat
  snapshot, never colour buffers.
- The two public extension seams are `Segmenter` (`@register_segmenter`) and
  `Loader` (`register_loader`). New built-ins go through them; see the README for
  worked examples.

## Style

- `ruff` for lint **and** formatting (config in `pyproject.toml`).
- Type hints and `from __future__ import annotations` everywhere; docstrings with
  Args/Returns on public API.
- Code, comments and docs in English.
