"""Build the complete retained district facility family offline, without activation."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from teadata.facility_usage import materialize_facility_usage
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not all((args.store / name).is_dir() for name in ("records", "objects")):
            raise ContractError("An existing private retained source store is required")
        result = materialize_facility_usage(ProvenanceStore(args.store), args.output)
    except (ContractError, OSError, ValueError):
        parser.exit(2, "Facility build failed; source or contract validation did not pass. No source values emitted.\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
