"""Prepare reviewed retained attendance inputs for normal native-source composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("sof", "chronic"))
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--dictionary-origin", type=Path)
    parser.add_argument("--intake", type=Path)
    parser.add_argument("--intake-sha256")
    args = parser.parse_args(argv)
    from teadata.provenance import ProvenanceStore

    store = ProvenanceStore(args.store)
    if args.family == "sof":
        if args.dictionary is None or args.dictionary_origin is None or args.intake is not None or args.intake_sha256 is not None:
            parser.error("SOF requires --dictionary, --dictionary-origin and a separate output")
        from teadata.sof_attendance import materialize_sof_attendance

        result = materialize_sof_attendance(store, dictionary=args.dictionary, dictionary_origin=args.dictionary_origin,
                                           output=args.output)
    else:
        if args.intake is None or args.intake_sha256 is None or args.dictionary is not None or args.dictionary_origin is not None:
            parser.error("Chronic absence requires --intake and --intake-sha256")
        from teadata.attendance_chronic import materialize_attendance_chronic

        result = materialize_attendance_chronic(store, intake=args.intake, intake_sha256=args.intake_sha256, output=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
