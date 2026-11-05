# Versioning Strategy

`teadata` uses a versioning strategy designed to track both code changes and data refreshes.

## Semantic Versioning (X.Y.Z)

We follow [Semantic Versioning 2.0.0](https://semver.org/) for codebase changes:

*   **MAJOR** version for incompatible API changes.
*   **MINOR** version for adding functionality in a backwards compatible manner.
*   **PATCH** version for backwards compatible bug fixes.

## Data Refresh Suffix (X.Y.Zy)

Because `teadata` is a data-heavy library where binary snapshots are critical artifacts, we use a four-digit extension for data-only refreshes.

If the library code remains at version `0.0.7` but the underlying TEA data is refreshed (e.g., a new 2025 release of the accountability ratings), the version is incremented to `0.0.71`.

*   `0.0.7` -> Original release
*   `0.0.71` -> First data refresh of 0.0.7
*   `0.0.72` -> Second data refresh of 0.0.7

## When to Increment

1.  **Code Change:** Increment the Major, Minor, or Patch version as appropriate in `pyproject.toml`.
2.  **Data Refresh:** Increment the fourth digit (or start at `1` if none exists) in `pyproject.toml` before running `teadata/load_data.py` for a production release.

## Software distribution builds

Use the maintainer command from a clean, committed checkout:

```bash
uv run --locked python scripts/build_release.py --out-dir .local/release-build/dist
```

Add `--commit <full-sha>` to require a particular commit, or
`--build-constraints <requirements-file>` to pin the build backend with
self-contained named registry requirements (no includes, local paths or URLs).
The constraints are captured before use and their hash is recorded. The command
builds through `uv` from a temporary Git archive with LFS smudging disabled.
Ignored build directories and local inputs stay outside that archive, and the
temporary source is removed on success or failure. Existing distributions are
preserved; choose a fresh output directory when a destination already exists.
The versioned artifacts JSON records the source commit, archive hash, build
tool and constraints, and both distribution hashes. Packaging does not run
tests, qualify source data or publish anything.

The canonical `scripts/check` allows one hour per library check and 15 minutes
per app check. Use `--timeout <seconds>` for an explicit finite positive limit.
Timeouts remain failed receipts; increasing a budget does not turn an incomplete
run into a pass.

Software versions, data builds and dataset editions are separate identities.
Record the selected data manifest hash and each measure's actual period;
neither a software release number nor its publication date is a school year.
Publishing an additional native supplement does not activate it in the default
engine or replace a saved export's source identity.
