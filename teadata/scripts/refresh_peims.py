"""Materialize explicitly selected PEIMS student source editions offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from teadata.peims_leaves import materialize_peims_students
from teadata.provenance import ProvenanceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Retained provenance records and original bytes")
    parser.add_argument("--config", type=Path, required=True, help="Reviewed source Config")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("DATASET", "YEAR"))
    parser.add_argument("--output", type=Path, required=True, help="New exclusive unpublished candidate directory")
    args = parser.parse_args(argv)
    try:
        selections = [(dataset, int(year)) for dataset, year in args.source]
    except ValueError:
        parser.error("--source YEAR requires an integer annual edition")
    report = materialize_peims_students(ProvenanceStore(args.store), config=args.config,
        config_sha256=args.config_sha256, selections=selections, output=args.output)
    print(json.dumps({name: report[name] for name in ("status", "build_id", "release_activation")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
