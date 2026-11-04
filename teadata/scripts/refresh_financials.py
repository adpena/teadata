"""Materialize explicitly reviewed original financial workbooks offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Explicit reviewed Config; never edits default sources")
    source.add_argument("--campus-intake", type=Path, help="Exact FY2025 campus actual acquisition manifest")
    source.add_argument("--qualification-case", type=Path, help="Reviewed FY2025 campus and district witness selection")
    parser.add_argument("--config-sha256")
    parser.add_argument("--campus-intake-sha256")
    parser.add_argument("--qualification-case-sha256")
    parser.add_argument("--source", nargs=2, action="append", metavar=("DATASET", "CATALOG_YEAR"))
    parser.add_argument("--output", type=Path, required=True, help="New exclusive unpublished native candidate directory")
    args = parser.parse_args(argv)
    from teadata.provenance import ProvenanceStore

    if args.qualification_case is not None:
        if not args.qualification_case_sha256 or args.campus_intake_sha256 is not None or args.config_sha256 is not None or args.source is not None:
            parser.error("Financial qualification requires --qualification-case-sha256 and runs separately from other modes")
        from teadata.finance_qualification import qualify_financial_actuals

        result = qualify_financial_actuals(ProvenanceStore(args.store), case_path=args.qualification_case,
            case_sha256=args.qualification_case_sha256, output=args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.qualification_case_sha256 is not None:
        parser.error("--qualification-case-sha256 requires --qualification-case")
    if args.campus_intake is not None:
        if not args.campus_intake_sha256 or args.config_sha256 is not None or args.source is not None:
            parser.error("Campus intake requires --campus-intake-sha256 and runs separately from Config workbooks")
        from teadata.campus_financial_leaves import materialize_campus_financial_workbooks

        result = materialize_campus_financial_workbooks(ProvenanceStore(args.store), intake=args.campus_intake,
            intake_sha256=args.campus_intake_sha256, output=args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.config_sha256 or not args.source or args.campus_intake_sha256 is not None:
        parser.error("Config workbooks require --config-sha256 and explicit --source selections")
    try:
        selections = [(dataset, int(year)) for dataset, year in args.source]
    except ValueError:
        parser.error("--source CATALOG_YEAR must be an explicit integer")
    from teadata.financial_leaves import materialize_financial_workbooks

    result = materialize_financial_workbooks(ProvenanceStore(args.store), config=args.config,
        config_sha256=args.config_sha256, selections=selections, output=args.output)
    print(json.dumps({key: result[key] for key in ("status", "build_id", "field_correctness_qualified", "release_activation")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
