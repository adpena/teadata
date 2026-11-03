"""Materialize explicitly reviewed original financial workbooks offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Explicit reviewed Config; never edits default sources")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("DATASET", "CATALOG_YEAR"))
    parser.add_argument("--output", type=Path, required=True, help="New exclusive unpublished native candidate directory")
    args = parser.parse_args(argv)
    try:
        selections = [(dataset, int(year)) for dataset, year in args.source]
    except ValueError:
        parser.error("--source CATALOG_YEAR must be an explicit integer")
    from teadata.financial_leaves import materialize_financial_workbooks
    from teadata.provenance import ProvenanceStore

    result = materialize_financial_workbooks(ProvenanceStore(args.store), config=args.config,
        config_sha256=args.config_sha256, selections=selections, output=args.output)
    print(json.dumps({key: result[key] for key in ("status", "build_id", "field_correctness_qualified", "release_activation")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
