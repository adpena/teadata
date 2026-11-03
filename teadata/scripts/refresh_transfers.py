"""Explicit regional transfer acquisition. No source activation or engine work."""

from __future__ import annotations

import argparse
from pathlib import Path

from teadata.provenance import ProvenanceStore
from teadata.transfer_sources import acquire_regional_transfers, read_regional_form


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
    args = parser.parse_args(argv)
    store = ProvenanceStore(args.store)
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
