# Repository Guidelines

## Main Authority and Task Closeout

- `origin/main` is the sole integration authority. Start by inspecting live remote main, local branches/worktrees, dirty changes and existing work ownership. Prefer parallel work in one checkout with disjoint file ownership. Create a temporary branch/worktree only when isolation is necessary; record its owner, base and purpose before use. Do not build chains of dependent review branches or create another checkout to recover existing work.
- Land each coherent change promptly after independent adversarial review and applicable checks. Merge or fast-forward to `main`, push, and verify the live remote commit. Source integration and production deployment are separate gates; inspect automatic deployment before pushing application changes.
- Before ending a normal task turn, leave one checkout on `main`, with local `HEAD` and `main` equal to live `origin/main`, a clean working tree, and no remaining task branches locally or remotely. Close completed PRs and remove superseded worktrees/branches as part of the work, not as a later optional cleanup.
- Inventory unique commits, dirty/untracked changes, retained source bytes, provenance, qualification receipts and deliverables before removal. Move necessary evidence to its canonical retained location and verify custody. Never discard unfinished work, another owner's changes, or published tags/releases/assets to make the closeout check pass.
- Run the canonical read-only workspace closeout auditor, `scripts/check_workspace_closeout.py`, against every repository in the task, including independent clones in the workspace. Its failures are unresolved work. It does not merge, push, delete or certify tests. If authorization, authentication or a conflicting owner prevents safe closure, report the exact retained state and blocker; do not claim completion.
- Keep this policy synchronized across TEA Data, TEA Data app, ReproQ Django and ReproQ Worker. Reuse the canonical auditor rather than introducing competing cleanup scripts or automatic destructive hooks.

## Project Structure & Module Organization
- Core engine lives in `teadata/engine.py`, `entities.py`, `query.py`, and `geometry.py`; they define the `DataEngine`, entity models, query operators, and spatial helpers.
- Enrichment logic sits in `teadata/enrichment/`; database bridges are under `teadata/persistence/`; utility runners live in `teadata/scripts/`.
- Config and sample data references are in `teadata/teadata_sources.yaml` and `teadata/data/`. Snapshots are written to `.cache/` and are committed; keep `.cache` artifacts in version control.
- Tests reside in `tests/` (e.g., `tests/test_snapshot_loading.py`), and example notebooks/scripts are in `examples/`. Packaging artifacts land in `build/` and `teadata.egg-info/`.

## Build, Test, and Development Commands
- Code-only clones and worktrees must set `GIT_LFS_SKIP_SMUDGE=1` during checkout. Hydrate only explicitly selected retained-release artifacts when an artifact check requires them.
- Install for iterative work: `uv sync --all-extras` (this handles dev, database, and notebook extras automatically).
- Always run Python commands through `uv run` and manage packages with `uv`; avoid bare `python` or `pip` to keep environments consistent.
- `.python-version` is the default development and `scripts/check` interpreter authority; `TEADATA_CHECK_PYTHON` selects a compatibility-check override.
- Run the suite: `uv run pytest` or target a test (`uv run pytest tests/test_entities.py::test_campus_to_dict_includes_percent_enrollment_change`).
- Optional: build a fresh snapshot from the configured spatial files with `uv run python -m teadata.load_data` (uses `teadata_sources.yaml` and writes to `.cache/`).
- When code changes affect anything serialized into the snapshot (pickled data), run `uv run python -m teadata.load_data` as the final task to refresh snapshot artifacts.
- Packaging sanity check: `uv build` after a clean `git status` if you need a wheel/sdist.
- Linting and type checking: `uv run ruff check .` and `uv run ty check .`.

## Coding Style & Naming Conventions

Write “public school,” “charter school,” and “private school” as two words in authored prose and catalog notes, including when they modify another noun.

Python 3.14 and 3.15 are required compatibility targets alongside supported earlier versions. Dependency and compatibility changes require isolated checks and real retained-bundle loading on those interpreters; classifiers alone do not establish support.
- Python 3.11+, PEP 8 defaults, 4-space indents, and type hints preferred (modules use `from __future__ import annotations`).
- Modules and functions use `snake_case`; classes use `PascalCase`; keep public attributes stable for pickled snapshots.
- Keep data loaders/enrichers deterministic: avoid implicit network calls and prefer explicit file paths resolved via config helpers.
- Use `uv run ruff check .` and `uv run ty check .` when available; keep formatting minimal and readable.

## Testing Guidelines
- Add pytest cases under `tests/` with `test_*.py` naming; exercise both happy paths and failure branches (e.g., gzip/no-extension snapshot handling).
- Use `tmp_path`/fixtures to avoid touching real data; prefer lightweight synthetic inputs over large datasets.
- For coverage when changing core logic: `pytest --cov=teadata --cov-report=term-missing`.
- Keep assertions specific (types, counts, and key fields) to guard query/enrichment regressions.

## Dependency & Tooling Updates
- When adding a new library or tool, update `pyproject.toml` (dependencies/extras), refresh the lockfile, and adjust test/tooling configs (`pyproject.toml` tool sections, `pytest` coverage settings) so CI stays aligned.

## Versioning Policy
- **Release tags:** Always use the thousandths place (e.g., `v0.0.101`, `v0.0.102`). If no tags exist, start at `v0.0.101`.
- **Standard SemVer (X.Y.Z):** Used for Major, Minor, and Patch updates to the codebase.
- **Data Refreshes (X.Y.Zy):** Append an additional digit for data-only updates (e.g., `0.0.71` implies a refresh of version `0.0.7`).
- **Release Process:** Increment the version in `pyproject.toml` prior to executing a production `load_data.py` run or distributing new snapshots.
- **Published history:** Preserve published tags, releases, and assets for reproducibility. Retire obsolete code and active cache artifacts instead. Never move a released tag or replace published assets.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative subject lines (e.g., “Create .gitattributes”, “Adding gzip support of snapshot repo cache”).
- PRs should state what changed, why, data/config files touched, and the tests/commands run; link related issues.
- Exclude local configs (`*.local.*`) and large data files that are covered by `.gitignore`; `.cache` artifacts are tracked and should remain in commits.
- If behavior changes are user-visible, include a brief reproduction snippet or before/after note in the PR description.

## Security & Configuration Tips
- Keep credentials and machine-local paths out of tracked files; rely on `teadata/teadata_sources.yaml` and untracked `*.local.*` overrides.
- Treat `.cache/` artifacts as required repo assets; avoid publishing them outside the repo unless explicitly sanitized.
- When using optional extras (`[database]`, notebooks), avoid embedding connection strings in code or examples—use env vars or local config instead.

## Downstream Dependencies
- `teadata-app` consumes the public `DataEngine`/`Query` APIs and snapshot loading behavior directly; avoid breaking signatures, return types, or query semantics, and coordinate any behavioral changes with that app before merging.

## Completing Source Integration

- Registration, acquisition, a standalone candidate and a separate source release are intermediate stages. Report them separately from qualified fields available through the selected normal entity release and DSL. Use the dated [integration inventory](docs/source-integration-status.md) as an audit record; recompute its comparison when configuration or the selected release changes.
- Feed qualified source contracts through the normal ingestion/enrichment and release-composition path. Reuse the catalog, field bindings and lazy metadata mapping for direct fields, dataset namespaces, discovery, queries and provenance. Avoid per-dataset copies of entity properties, query logic, storage or export mappings when the existing contract can drive them.
- Preserve source-specific parsing and semantic qualification where needed. Registration must not automatically qualify or publish unreviewed values. Keep multi-row source dimensions explicit; do not flatten them into conflicting scalars, infer suppressed values, or substitute another population/year.
- Close a source tranche with representative publisher-grounded checks through its released normal access and applicable exports, including missing/suppressed/zero states. State the exact remaining integration gate if blocked. Do not count an unconnected bundle as completion, repeat unchanged downloads, or accumulate more candidates instead of finishing a ready source family.

## Landing Review Gate

Use the [Elephant & Goldfish Model](https://github.com/google/elephant-goldfish-model) for every planned change before implementation. Scale the design to the change. Give one fresh Goldfish the draft, applicable standards, and the context needed to challenge it. Record the decisions and use the acceptance criteria to choose checks. Run reviewers one at a time.

Require independent adversarial review of the exact landing diff. After each repair that affects correctness, data integrity, security, or operations, review the repair and adjacent paths that could repeat the failure. Repeat until the exact diff has no open actionable findings. Reuse checks whose inputs still match. A separate broader audit may look for independent failure modes beyond the changed paths.

## Publisher Authority and Source Correctness

- TEA's original data and year-specific documents are the authority. Retain publisher bytes and their acquisition records; check headings, definitions, periods, populations, units and masking rules for each new dataset or edition. Download success, matching hashes and structural validation alone do not establish value correctness.
- Compare retained raw values through preparation, stores and exports, and keep independent publisher-report goldens for semantic bindings. Preserve raw tokens alongside normalized values and statuses. Documented transformations must follow the exact source contract; do not infer masking, units or comparability from field names or another year's rules.
- Every unexplained discrepancy blocks qualification. An intentional deviation for a publisher anomaly requires a narrowly scoped, evidence-backed record of the identified problem, understood repair and the user's explicit approval. General repair authorization is not a discrepancy waiver. Never auto-record expected values, add broad tolerances or silently accept known differences.
- Report checked coverage and unresolved lineage explicitly. Unsupported readers, unreviewed definitions and missing source evidence remain uncovered/HOLD; they cannot be counted as passing source qualification. Keep integrity, raw-value parity, semantic goldens and researcher reproducibility as distinct claims.

## Local Verification

`scripts/check` is the primary verification entry point; it runs locked isolated local Ruff, ty, deptry, and pytest checks and records receipts/logs in ignored `.local/checks/`. Use `scripts/check --verify-latest` to verify evidence for the current exact revision. After serialized-code or data changes, rebuild the bundle and run `scripts/check --release teadata/.cache/release.json`. Code-only receipts do not certify raw data or artifacts, and all receipts remain separate from adversarial review. `TEADATA_CHECK_PYTHON` selects another supported Python version for compatibility checks. Hosted workflows are manual only; do not restore automatic hosted triggers without a user request. Optional hook installation must preserve Git LFS and refuse custom hooks/configuration.
