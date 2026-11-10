"""Materialize a private TPRS class-size candidate."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from teadata.manifest import ContractError, canonical_json
    from teadata.provenance import ProvenanceStore
    from teadata.release import DataRelease
    from teadata.tprs_class_size_native import materialize_class_sizes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-attachment", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-campus", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained source store is required")
        release = DataRelease(args.student_attachment, offline=True)
        result = materialize_class_sizes(release, ProvenanceStore(args.store), args.output,
                                         include_campus=args.include_campus)
    except (ContractError, OSError, ValueError) as error:
        parser.exit(2, f"TPRS class-size build failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
