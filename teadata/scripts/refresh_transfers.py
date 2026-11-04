"""Explicit transfer intake and unpublished native candidates; no source activation or engine work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from teadata.provenance import ProvenanceStore
from teadata.transfer_sources import acquire_regional_transfers, prepare_transfer_sources, read_regional_form


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire", help="Acquire original district CSVs from a pinned regional form")
    acquire.add_argument("--store", type=Path, required=True)
    acquire.add_argument("--form-record", required=True)
    acquire.add_argument("--form-sha256", required=True)
    acquire.add_argument("--year", type=int, required=True, help="Exact offered school-year control")
    selection = acquire.add_mutually_exclusive_group(required=True)
    selection.add_argument("--regions", nargs="+", help="Exact offered IDs, e.g. 01 13; order is preserved")
    selection.add_argument("--all-offered-regions", action="store_true")
    acquire.add_argument("--resume", nargs=2, action="append", default=[], metavar=("REGION", "RECORD"))
    acquire.add_argument("--timeout", type=float, default=20)
    acquire.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    acquire.add_argument("--output", type=Path, required=True, help="New exclusive batch-index directory")
    prepare = sub.add_parser("prepare", help="Prepare pinned originals offline; never activate Config or an engine")
    prepare.add_argument("--store", type=Path, required=True)
    prepare.add_argument("--grain", choices=["campus", "district"], required=True)
    prepare.add_argument("--year", type=int, required=True)
    sources = prepare.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", nargs=2, action="append", default=[], metavar=("RECORD", "SHA256"))
    sources.add_argument("--batch-index", type=Path)
    prepare.add_argument("--batch-index-sha256")
    prepare.add_argument("--encoding", help="Explicit codec required when original HTTP MIME has no charset")
    prepare.add_argument("--policy-config", type=Path, required=True)
    prepare.add_argument("--policy-config-sha256", required=True)
    prepare.add_argument("--policy", required=True, help="Existing Config ValuePolicy ID")
    prepare.add_argument("--compare-legacy-input", nargs=2, metavar=("RECORD", "SHA256"))
    prepare.add_argument("--compare-encoding", help="Explicit codec for the separately retained legacy CSV")
    prepare.add_argument("--max-examples", type=int, default=10)
    prepare.add_argument("--output", type=Path, required=True)
    materialize = sub.add_parser("materialize", help="Build unpublished native district leaves from exact accepted preparations")
    materialize.add_argument("--store", type=Path, required=True)
    materialize.add_argument("--config", type=Path, required=True, help="Explicit reviewed source Config; never changes default YAML")
    materialize.add_argument("--config-sha256", required=True)
    materialize.add_argument("--prepared-input", nargs=2, action="append", required=True, metavar=("YEAR", "PROCESSING_RECORD"))
    materialize.add_argument("--output", type=Path, required=True, help="New exclusive unpublished candidate directory")
    compare = sub.add_parser("compare-campus-district", help="Check wholly visible cross-district transfer rollups offline")
    compare.add_argument("--store", type=Path, required=True, help="Retained campus preparation and its original provenance")
    compare.add_argument("--year", type=int, required=True)
    compare.add_argument("--campus-preparation", required=True)
    compare.add_argument("--district-release", type=Path, required=True)
    compare.add_argument("--district-pin", required=True)
    args = parser.parse_args(argv)
    store = ProvenanceStore(args.store)
    if args.command == "compare-campus-district":
        from teadata.transfer_rollups import PASS_OBSERVED_TRANSFER_ROLLUP, compare_transfer_rollup

        result = compare_transfer_rollup(store, year=args.year, campus_preparation=args.campus_preparation,
                                         district_release=args.district_release, district_pin=args.district_pin)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == PASS_OBSERVED_TRANSFER_ROLLUP else 2
    if args.command == "materialize":
        from teadata.transfer_leaves import materialize_district_transfers
        try:
            preparations = [(int(year), identifier) for year, identifier in args.prepared_input]
        except ValueError:
            parser.error("--prepared-input YEAR requires an integer annual edition")
        result = materialize_district_transfers(store, config=args.config, config_sha256=args.config_sha256,
            preparations=preparations, output=args.output)
        print(json.dumps({name: result[name] for name in ("status", "build_id", "release_activation")}, sort_keys=True))
        return 0
    if args.command == "prepare":
        if (args.batch_index is None) != (args.batch_index_sha256 is None):
            parser.error("--batch-index and --batch-index-sha256 must be supplied together")
        result = prepare_transfer_sources(store, grain=args.grain, year=args.year, output=args.output,
            policy_config=args.policy_config, policy_config_sha256=args.policy_config_sha256, value_policy=args.policy,
            inputs=[(identifier, digest) for identifier, digest in args.input],
            batch_index=(args.batch_index, args.batch_index_sha256) if args.batch_index is not None else None,
            encoding=args.encoding,
            compare_input=(args.compare_legacy_input[0], args.compare_legacy_input[1]) if args.compare_legacy_input is not None else None,
            compare_encoding=args.compare_encoding, max_examples=args.max_examples)
        print(json.dumps({name: result.get(name) for name in ["status", "rows", "processing_record_id"]}, sort_keys=True))
        return 0 if result["status"] == "prepared" else 2
    form = read_regional_form(store, args.form_record, args.form_sha256)
    resume = dict(args.resume)
    if len(resume) != len(args.resume):
        parser.error("Each resume region may be supplied only once")
    regions = list(form.regions) if args.all_offered_regions else args.regions
    report = acquire_regional_transfers(store, form, year=args.year, regions=regions,
        output=args.output, resume=resume, timeout=args.timeout, max_bytes=args.max_bytes)
    print(f"{report['status']}: {len(report['items'])} regional originals; {args.output / 'BATCH_INDEX.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
