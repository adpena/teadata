"""Compare published campus and district transfer details without filling masks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .dataset_metadata import DataPeriod
from .provenance import ProvenanceStore
from .release import DataRelease
from .transfer_sources import _captured_input

_KEYS = ["transfer_direction", "transfer_reporting_district_number", "transfer_partner_district_number"]
_CAMPUS_IDS = ["transfer_reporting_number", "transfer_partner_number"]
_STATUS = "transfer_count_status"
_CAMPUS_VALUE = "transfer_count_value"
_DISTRICT_VALUE = "student_transfer_count"
PASS_OBSERVED_TRANSFER_ROLLUP = "PASS_OBSERVED_TRANSFER_ROLLUP"
pc: Any = arrow_compute


def _validate_details(table: pa.Table, value: str) -> None:
    if not table.num_rows:
        return
    if (any(table[name].null_count for name in [*_KEYS, _STATUS])
            or not pc.all(pc.is_in(table[_KEYS[0]], value_set=pa.array(["in", "out"]))).as_py()
            or not pc.all(pc.is_in(table[_STATUS], value_set=pa.array(["observed", "suppressed"]))).as_py()
            or not pc.all(pc.equal(pc.equal(table[_STATUS], "observed"), pc.is_valid(table[value]))).as_py()
            or pc.any(pc.less(table[value], 0)).as_py()):
        raise ValueError("Transfer detail keys, directions or count statuses differ from their reviewed types")


def _compare_details(campus: pa.Table, district: pa.Table) -> dict:
    _validate_details(campus, _CAMPUS_VALUE)
    _validate_details(district, _DISTRICT_VALUE)
    if any(campus[name].null_count for name in _CAMPUS_IDS):
        raise ValueError("Campus transfer detail requires both campus IDs")
    campus_pairs = campus.group_by([_KEYS[0], *_CAMPUS_IDS]).aggregate([(_STATUS, "count")])
    if campus_pairs.num_rows != campus.num_rows:
        raise ValueError("Duplicate directional campus transfer detail")
    within = pc.equal(campus[_KEYS[1]], campus[_KEYS[2]])
    if pc.any(pc.equal(district[_KEYS[1]], district[_KEYS[2]])).as_py():
        raise ValueError("District transfer detail contains a within-district pair")
    cross = campus.filter(pc.invert(within))
    grouped = cross.group_by(_KEYS).aggregate([(_STATUS, "count"), (_CAMPUS_VALUE, "count")])
    campus_groups = {
        tuple(row[key] for key in _KEYS): row[f"{_STATUS}_count"] == row[f"{_CAMPUS_VALUE}_count"]
        for row in grouped.to_pylist()
    }
    eligible = []
    district_keys = set()
    coverage: Counter[str] = Counter()
    for row in district.to_pylist():
        key = tuple(row[column] for column in _KEYS)
        if key in district_keys:
            raise ValueError("Duplicate directional district transfer detail")
        district_keys.add(key)
        if key not in campus_groups:
            coverage[f"district_only_{row[_STATUS]}"] += 1
        elif row[_STATUS] == "suppressed":
            coverage["district_suppressed"] += 1
        elif not campus_groups[key]:
            coverage["campus_suppressed"] += 1
        else:
            eligible.append({name: row[name] for name in [*_KEYS, _DISTRICT_VALUE]})
    coverage["campus_only"] = len(campus_groups.keys() - district_keys)
    mismatches = []
    if eligible:
        exact = pa.Table.from_pylist(eligible, schema=district.select([*_KEYS, _DISTRICT_VALUE]).schema)
        # Sum only groups with both a public district count and wholly visible campus counts.
        totals = cross.join(exact.select(_KEYS), keys=_KEYS, join_type="left semi")
        wide = pa.decimal128(38, 0)
        totals = totals.set_column(totals.schema.get_field_index(_CAMPUS_VALUE), _CAMPUS_VALUE,
                                   pc.cast(totals[_CAMPUS_VALUE], wide))
        totals = totals.group_by(_KEYS).aggregate([(_CAMPUS_VALUE, "sum")])
        joined = totals.join(exact, keys=_KEYS, join_type="inner")
        if joined.num_rows != len(eligible):
            raise ValueError("Comparable transfer rollup lost a directional district pair")
        total = f"{_CAMPUS_VALUE}_sum"
        differences = joined.filter(pc.not_equal(joined[total], pc.cast(joined[_DISTRICT_VALUE], wide)))
        coverage["different"] = differences.num_rows
        coverage["exact"] = joined.num_rows - differences.num_rows
        mismatches = differences.sort_by([(key, "ascending") for key in _KEYS]).slice(0, 10).to_pylist()
        for example in mismatches:
            example[total] = int(example[total])
    return {
        "status": PASS_OBSERVED_TRANSFER_ROLLUP if eligible and not (
            coverage["different"] or coverage["campus_only"] or coverage["district_only_observed"]
        ) else "HOLD",
        "coverage": {key: coverage[key] for key in (
            "exact", "different", "campus_suppressed", "district_suppressed",
            "campus_only", "district_only_observed", "district_only_suppressed",
        )},
        "campus_cross_district_detail_rows": cross.num_rows,
        "campus_within_district_detail_rows_excluded": campus.num_rows - cross.num_rows,
        "district_detail_rows": district.num_rows,
        "visible_mismatch_examples": mismatches,
    }


def compare_transfer_rollup(store: ProvenanceStore, *, year: int, campus_preparation: str,
                            district_release: Path, district_pin: str) -> dict:
    graph = store.verify_record_closure([campus_preparation])
    record = graph[campus_preparation]
    validation = record.get("validation", {})
    if (record["kind"] != "processing" or record["dataset"] != "campus_transfer_reports"
            or record.get("processor") != "teadata.scripts.refresh_transfers"
            or validation.get("raw_string_readback_equal") is not True
            or not (validation.get("status") == "PASS" or (
                validation.get("status") == "HOLD" and validation.get("prepared_source_qualified") is True
            ))):
        raise ValueError("Expected a retained campus transfer preparation with accepted source rows")
    outputs = record["outputs"]
    report_digest = outputs["validation_report"]["sha256"]
    report = json.loads(_captured_input(store.verify_artifact(report_digest), report_digest))
    if report.get("year") != year or report.get("grain") != "campus" or report.get("rows") != validation.get("rows"):
        raise ValueError("Campus transfer preparation differs from the selected school year")
    campus_digest = outputs["prepared_source"]["sha256"]
    campus_path = store.verify_artifact(campus_digest)
    with pq.ParquetFile(campus_path) as campus_file:
        if campus_file.metadata.num_rows != report["rows"]:
            raise ValueError("Campus transfer rows differ from the retained preparation")
    release = DataRelease(district_release, expected_sha256=district_pin, offline=True)
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    selected = [ref for ref in refs if (ref.edition.dataset, ref.edition.edition, ref.edition.scope, ref.edition.table)
                == ("district_transfer_reports", str(year), "district", "report_rows")]
    if len(selected) != 1:
        raise ValueError("Expected one district transfer leaf for the selected school year")
    ref = selected[0]
    if release.catalog.metadata(ref.edition.dataset, edition=ref.edition).period != DataPeriod("school_year", year - 1, year):
        raise ValueError("District transfer period differs from the selected school year")
    filters = [("transfer_row_kind", "=", "detail")]
    result = _compare_details(
        pq.read_table(campus_path, columns=[*_KEYS, *_CAMPUS_IDS, _STATUS, _CAMPUS_VALUE], filters=filters),
        pq.read_table(release.artifact(ref.identity), columns=[*_KEYS, _STATUS, _DISTRICT_VALUE], filters=filters),
    )
    return {**result, "year": year, "campus_preparation": campus_preparation,
            "campus_prepared_sha256": campus_digest, "campus_preparation_status": validation["status"],
            "district_manifest_sha256": release.manifest_sha256, "district_leaf_sha256": ref.sha256,
            "canonical_edges_or_release_qualified": False}
