"""Build explicit, offline PEIMS Single File Financial Actuals candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from teadata.manifest import canonical_json
from teadata.peims_actuals import materialize_peims_actuals
from teadata.provenance import ProvenanceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Retained original archives and provenance records")
    parser.add_argument("--config", type=Path, required=True, help="Reviewed source configuration")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--year", type=int, action="append", required=True, help="Exact fiscal end year")
    parser.add_argument("--output", type=Path, required=True, help="New unpublished candidate directory")
    args = parser.parse_args(argv)
    result = materialize_peims_actuals(ProvenanceStore(args.store), config=args.config,
        config_sha256=args.config_sha256, years=args.year, output=args.output)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
