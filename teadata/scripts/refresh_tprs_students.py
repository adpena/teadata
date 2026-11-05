"""Build a receipt-selected 2026 TPRS Membership/Enrollment projection offline."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--intake-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore
    from teadata.tprs_students import materialize_students

    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained provenance store is required")
        result = materialize_students(ProvenanceStore(args.store), config=args.config,
            config_sha256=args.config_sha256, intake=args.intake, intake_sha256=args.intake_sha256,
            output=args.output)
    except (ContractError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Student report build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
