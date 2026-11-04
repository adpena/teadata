"""Materialize exact public SPP Indicator 11 workbook editions offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from teadata.provenance import ProvenanceStore
from teadata.spp_leaves import materialize_spp_ind11


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="Retained TEA public provenance store")
    parser.add_argument("--config", required=True, type=Path, help="Reviewed source Config")
    parser.add_argument("--config-sha256", required=True, help="Exact Config SHA-256")
    parser.add_argument("--year", required=True, type=int, action="append", help="Exact school-year ending year")
    parser.add_argument("--output", required=True, type=Path, help="New exclusive candidate directory")
    args = parser.parse_args(argv)
    result = materialize_spp_ind11(ProvenanceStore(args.store), config=args.config, config_sha256=args.config_sha256,
                                  years=args.year, output=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
