# Repository Guidelines

## Project Structure & Module Organization
- Core engine lives in `teadata/engine.py`, `entities.py`, `query.py`, and `geometry.py`; they define the `DataEngine`, entity models, query operators, and spatial helpers.
- Enrichment logic sits in `teadata/enrichment/`; database bridges are under `teadata/persistence/`; utility runners live in `teadata/scripts/`.
- Config and sample data references are in `teadata/teadata_sources.yaml` and `teadata/data/`. Snapshots are written to `.cache/` (ignored by Git); keep large `.pkl`/spatial files out of commits.
- Tests reside in `tests/` (e.g., `tests/test_snapshot_loading.py`), and example notebooks/scripts are in `examples/`. Packaging artifacts land in `build/` and `teadata.egg-info/`.

## Build, Test, and Development Commands
- Install for iterative work: `uv sync --all-extras` (this handles dev, database, and notebook extras automatically).
- Run the suite: `uv run pytest` or target a test (`uv run pytest tests/test_entities.py::test_campus_to_dict_includes_percent_enrollment_change`).
- Optional: build a fresh snapshot from the configured spatial files with `uv run python -m teadata.load_data` (uses `teadata_sources.yaml` and writes to `.cache/`).
- Packaging sanity check: `uv build` after a clean `git status` if you need a wheel/sdist.
- Linting and type checking: `uv run ruff check .` and `uv run mypy .`.

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8 defaults, 4-space indents, and type hints preferred (modules use `from __future__ import annotations`).
- Modules and functions use `snake_case`; classes use `PascalCase`; keep public attributes stable for pickled snapshots.
- Keep data loaders/enrichers deterministic: avoid implicit network calls and prefer explicit file paths resolved via config helpers.
- Use `ruff check .` and `mypy .` when available; keep formatting minimal and readable.

## Testing Guidelines
- Add pytest cases under `tests/` with `test_*.py` naming; exercise both happy paths and failure branches (e.g., gzip/no-extension snapshot handling).
- Use `tmp_path`/fixtures to avoid touching real data; prefer lightweight synthetic inputs over large datasets.
- For coverage when changing core logic: `pytest --cov=teadata --cov-report=term-missing`.
- Keep assertions specific (types, counts, and key fields) to guard query/enrichment regressions.

## Dependency & Tooling Updates
- When adding a new library or tool, update `pyproject.toml` (dependencies/extras), refresh the lockfile, and adjust test/tooling configs (`pyproject.toml` tool sections, `pytest` coverage settings) so CI stays aligned.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative subject lines (e.g., “Create .gitattributes”, “Adding gzip support of snapshot repo cache”).
- PRs should state what changed, why, data/config files touched, and the tests/commands run; link related issues.
- Exclude generated snapshots, local configs (`*.local.*`), and large data files that are covered by `.gitignore`.
- If behavior changes are user-visible, include a brief reproduction snippet or before/after note in the PR description.

## Security & Configuration Tips
- Keep credentials and machine-local paths out of tracked files; rely on `teadata/teadata_sources.yaml` and untracked `*.local.*` overrides.
- Treat `.cache/` artifacts and enrichment outputs as ephemeral; do not publish them unless explicitly sanitized.
- When using optional extras (`[database]`, notebooks), avoid embedding connection strings in code or examples—use env vars or local config instead.

## Downstream Dependencies
- `teadata-app` consumes the public `DataEngine`/`Query` APIs and snapshot loading behavior directly; avoid breaking signatures, return types, or query semantics, and coordinate any behavioral changes with that app before merging.
