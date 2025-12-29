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
