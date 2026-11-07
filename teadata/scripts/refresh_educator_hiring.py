"""Build reviewed district and Texas teacher-hiring measurement years offline."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from teadata.educator_hiring import SCHOOL_YEARS, YEAR, materialize_educator_hiring
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--school-year", type=int, action="append", help="School-year ending year; repeat to select several")
    selection.add_argument("--all-years", action="store_true", help="Select all eleven reviewed paired source years")
    args = parser.parse_args(argv)
    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained source store is required")
        years = SCHOOL_YEARS if args.all_years else args.school_year or (YEAR,)
        result = materialize_educator_hiring(ProvenanceStore(args.store), args.output, school_years=years)
    except (ContractError, OSError, ValueError) as error:
        parser.exit(2, f"Educator hiring build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
