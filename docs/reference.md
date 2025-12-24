# Reference

## Performance Notes

- **Shapely 2.x + STRtree** powers nearest and containment queries at scale.
- **One-time precomputation** during repo build: district ➜ campuses index, ID maps, clean normalizations.
- **Cache everything**: Snapshots persist the derived indices; reloads are **milliseconds**, not minutes.
- **Slow-path sanity check** on spatial containment can be toggled, and only runs when the fast path returns 0 where >0 is expected.

## Repo Layout

- `teadata/classes.py` — models (`District`, `Campus`, `DataEngine`), query object, spatial index.
- `teadata/load_data.py` / `teadata/load_data2.py` — examples/scripts to build and snapshot repos from raw sources.
- `teadata/teadata_config.py` — config reader/validator (YAML/TOML), year-resolution, schema validation.
- `teadata/enrichment/` — modular enrichment functions (districts, campuses, charter networks, etc.).
- `.cache/` — binary snapshots (`repo_*.pkl`) for instant loads.

## FAQ

**Q: Where do snapshots live?**
A: Prefer the package-internal `.cache/` for the “best known” snapshot. The loader also searches your repo root `.cache/`.

**Q: Does `district_number` need an apostrophe?**
A: Normalizers accept either digits-only or apostrophe-prefixed strings and standardize internally.

**Q: Can I chain spatial + attribute filters?**
A: Yes — use `>>` to compose: `engine >> ("within", d, "campuses") >> (lambda c: c.rating in {"A","B"})`.

## License

This project is licensed under the **Business Source License 1.1**. It allows for personal, educational, and non-commercial research use. Production or commercial use requires a separate license.

See the `LICENSE` file in the repository root for the full terms.
