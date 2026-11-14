"""Report configuration registration separately from one pinned release catalog."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from teadata.release import DataRelease
from teadata.manifest import ContractError
from teadata.teadata_config import load_config_bytes


def _report(args: argparse.Namespace) -> dict[str, Any]:
    config_raw = args.config.read_bytes()
    config = load_config_bytes(config_raw, path=args.config)
    selected = DataRelease(args.release, expected_sha256=args.sha256, offline=not args.online, cache_dir=args.cache_dir)
    inventory = selected.catalog_inventory(include_variables=args.include_variables)
    attached = {row["dataset"] for row in inventory["editions"]}
    base = {row["dataset"] for row in inventory["base_inputs"]}
    registered: list[dict[str, Any]] = []
    source_only = set()
    for section in ("data_sources", "spatial"):
        for name, editions in sorted(getattr(config, section).items()):
            years = editions.available_years()
            preparation = [year for year in years if config.is_source_only(name, year, section=section)]
            if preparation:
                source_only.add(name)
            registered.append({"dataset": name, "section": section, "registered_editions": list(years),
                               "source_only_editions": preparation, "base_input_present": name in base,
                               "native_attachment_present": name in attached,
                               "attached_report_editions": sorted({row["edition"] for row in inventory["editions"]
                                                                    if row["dataset"] == name}),
                               "attached_scopes": sorted({row["scope"] for row in inventory["editions"]
                                                          if row["dataset"] == name})})
    # Compact report: retain precise field periods, but avoid duplicating thousands
    # of physical bindings in a coverage document. The library exposes those on demand.
    if args.include_variables:
        for row in inventory["editions"]:
            metadata = row.pop("metadata")
            variables = row.pop("variables")
            periods = {}
            for item in variables:
                key = json.dumps([item["period"], item["unit"]], sort_keys=True)
                periods.setdefault(key, {"period": item["period"], "unit": item["unit"], "fields": []})["fields"].append(item["field"])
            row.update(title=metadata.get("title"), publisher=metadata.get("publisher"),
                       source_url=metadata.get("source_url"), declared_period=metadata.get("period"),
                       source_qualifiers=metadata.get("source_qualifiers"), variable_count=len(variables),
                       field_periods=[periods[key] for key in sorted(periods)])
    code_root = Path(__file__).resolve().parents[1]
    report = {"schema_version": 1, "observed_at": datetime.now(timezone.utc).isoformat(),
              "generator": {"software_version": version("teadata"), "working_source_sha256": {
                  name: sha256((code_root/name).read_bytes()).hexdigest() for name in
                  ("scripts/source_inventory.py", "teadata/release_inventory.py", "teadata/release.py", "teadata/teadata_config.py")}},
              "config_sha256": sha256(config_raw).hexdigest(),
              "scope": "Configuration registration and selected catalog metadata only. No observation, source acquisition or qualification replay.",
              "summary": {"registered_ids": len({row["dataset"] for row in registered}),
                          "registered_editions": sum(len(row["registered_editions"]) for row in registered),
                          "source_only_ids": len(source_only), "source_only_ids_not_attached": len(source_only-attached),
                          "base_inputs": len(base), "native_attachments": len(inventory["attachments"]),
                          "native_dataset_ids": len(attached), "native_edition_scope_keys": len(inventory["editions"])},
              "registration": registered, "selected_catalog": inventory,
              "unestablished_by_this_report": ["Publisher availability beyond retained registration", "Acquisition and parsing status of unattached sources",
                                                "Semantic qualification beyond separately retained release evidence", "Value queries and downstream deployment"]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="Exact local or HTTP(S) release manifest.")
    parser.add_argument("--sha256", required=True, help="Trusted release manifest SHA-256.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--include-variables", action="store_true", help="Explicitly inspect verified metadata, never data leaves.")
    parser.add_argument("--online", action="store_true", help="Allow missing manifest/metadata downloads. Offline by default.")
    args = parser.parse_args()
    try:
        result = _report(args)
    except (ContractError, OSError, ValueError) as error:
        parser.error(f"Cannot inspect this configuration/release: {error}. Offline is the default; supply retained metadata or explicitly permit downloads with --online.")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
