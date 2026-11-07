"""Build the scoped, independently reconciled educator certification release."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from teadata.educator_certification import materialize_educator_certification
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--archive-receipt", required=True, help="Retained exact archive readback observation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained source store is required")
        result = materialize_educator_certification(ProvenanceStore(args.store), args.output,
                                                    archive_receipt=args.archive_receipt)
    except (ContractError, OSError, ValueError) as error:
        parser.exit(2, f"Educator certification build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
