"""Build receipt-pinned statewide, region and district PEIMS College Credit leaves offline."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Existing retained provenance store")
    parser.add_argument("--config", type=Path, required=True, help="Exact reviewed source configuration")
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--edition", nargs=2, action="append", required=True, metavar=("YEAR", "SCOPE"))
    parser.add_argument("--output", type=Path, required=True, help="New unpublished candidate directory")
    args = parser.parse_args(argv)
    from teadata.manifest import ContractError, canonical_json
    from teadata.peims_college_credit import materialize_peims_college_credit
    from teadata.provenance import ProvenanceStore

    try:
        editions = [(int(year), scope) for year, scope in args.edition]
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained provenance store is required")
        result = materialize_peims_college_credit(ProvenanceStore(args.store), config=args.config,
            config_sha256=args.config_sha256, editions=editions, output=args.output)
    except (ContractError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"PEIMS College Credit build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
