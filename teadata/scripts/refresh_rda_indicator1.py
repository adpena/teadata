"""Build a local 2025 special-education RDA Indicator 1 candidate offline."""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import SubprocessError


def main(argv: list[str] | None = None) -> None:
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore
    from teadata.rda_native import materialize_rda_special_education_indicator1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained provenance store is required")
        result = materialize_rda_special_education_indicator1(ProvenanceStore(args.store), args.output)
    except (ContractError, OSError, ValueError, SubprocessError) as error:
        parser.exit(2, f"RDA Indicator 1 build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
