# TEA Data Engine

**A high-performance, spatially-aware Python framework for the analysis and modeling of Texas public education data.**

`teadata` provides a robust, object-oriented interface for large-scale geographic and demographic datasets. By utilizing a high-speed, snapshot-based architecture, the engine enables rapid data integration and sophisticated spatial querying through an intuitive, Pythonic DSL.

---

## 📚 Documentation

**[Read the full documentation here](docs/index.md)** (or run `mkdocs serve` to browse locally).

---

## Quick Start

### Installation

**Using `uv` (Recommended for development)**

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
uv sync
```

**Using `pip`**

```bash
pip install teadata
```

When adding a new library or tool, update `pyproject.toml` (dependencies or
extras), refresh the lockfile, and adjust test/tooling configuration so CI and
local environments stay in sync.

**Installing from GitHub (for Render or other CI/CD)**

If you are using this in a Django application deployed on **Render**, add it to your `requirements.txt`:

```text
teadata @ git+https://github.com/adpena/teadata.git
```

Or install it directly via CLI:

```bash
pip install git+https://github.com/adpena/teadata.git
```

### Usage

```python
from teadata import DataEngine

# Fast-path: load the latest discovered snapshot
engine = DataEngine.from_snapshot(search=True)

# Retrieve district by TEA campus number
aldine = engine.get_district("101902")
print(aldine.name)  # -> "Aldine ISD"

# Iterate campuses inside the district
for c in aldine.campuses:
    print(c.name)
```

## Release Policy

- Tags always use the thousandths place (e.g., `v0.0.101`, `v0.0.102`). If no tags exist, start at `v0.0.101`.
- Keep only the three most recent tags/releases; delete older tags and their GitHub release assets.

## Features

- **Rich domain objects**: `District` and `Campus` with geometry.
- **Fluent query language**: Chain filters like `engine >> ("district", "101902") >> ("campuses_in",)`.
- **Spatial acceleration**: Nearest neighbors, containment checks.
- **Enrichment**: Attach external datasets (finance, accountability) easily.

## License

This project is licensed under the **Business Source License 1.1**.

- **Personal & Educational Use**: You may use, copy, modify, and distribute this software for personal, educational, or non-commercial research purposes.
- **Commercial & Production Use**: Prohibited without obtaining a separate commercial license from the Licensor.

See the [LICENSE](LICENSE) file for the full text.
