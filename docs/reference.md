# Reference

## Performance Notes

- Entity lookups use the selected snapshot and spatial sidecars. See [spatial tools](spatial.md) for distance and containment behavior.
- Native source tables project fields and filter rows through Arrow. Select only the fields you need and stream batches for large exports. See [streaming projections](querying.md#streaming-projections-and-memory).
- Fetching and verifying cold artifacts can dominate first-use latency. Subsequent calls can reuse verified files. See [release runtime behavior](data-releases.md#runtime-and-rollout).

## Repo Layout

- `teadata/entities.py`, `teadata/engine.py`, `teadata/query.py`, and `teadata/geometry.py` define entities, loading, query operators, and spatial operations.
- `teadata/load_data.py` — build and publish a catalog-bound release from configured raw sources; see [dataset contracts and data releases](data-releases.md).
- `teadata/teadata_config.py` — config reader/validator (YAML/TOML), year-resolution, schema validation.
- `teadata/enrichment/` — modular enrichment functions (districts, campuses, charter networks, etc.).
- `teadata/default_release.json` pins the normal data release. `teadata/.cache/` contains historical artifacts that can be selected explicitly.

## FAQ

**Q: Which snapshot does the engine load?**

An active `using_release()` context takes precedence, followed by
`TEADATA_RELEASE`, `TEADATA_RELEASE_URL`, and the packaged default pin.
The manifest identifies the snapshot and other artifacts. See [release selection and caching](data-releases.md#runtime-and-rollout).

**Q: Does `district_number` need an apostrophe?**
A: Normalizers accept either digits-only or apostrophe-prefixed strings and standardize internally.

**Q: Can I chain spatial + attribute filters?**
A: Use `engine >> ("charters_within", district) >> (lambda c: c.rating in {"A", "B"})`.
See [Query operator discovery](querying.md#discover-query-operators) for the syntax after an engine command.

## License

This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file in the repository root for the full terms.
