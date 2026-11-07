"""Build reviewed statewide educator retention cohorts from retained TEA bytes."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from teadata.educator_retention import COHORT_YEARS, materialize_educator_retention
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-year", type=int, action="append", help="First-teaching school-year ending year; repeat to select")
    args = parser.parse_args(argv)
    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained source store is required")
        result = materialize_educator_retention(ProvenanceStore(args.store), args.output,
                                               cohort_years=args.cohort_year or COHORT_YEARS)
    except (ContractError, OSError, ValueError) as error:
        parser.exit(2, f"Educator retention build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
