"""Compare published transfer details without filling masks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, TypedDict

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .dataset_metadata import ArtifactRef, DataPeriod, EditionKey, _parse_school_year
from .district_types import DISTRICT_TYPE_REFERENCE, project_district_types, reviewed_district_types
from .provenance import ProvenanceStore
from .release import DataRelease
from .transfer_sources import _captured_input

_KEYS = ["transfer_direction", "transfer_reporting_district_number", "transfer_partner_district_number"]
_CAMPUS_IDS = ["transfer_reporting_number", "transfer_partner_number"]
_STATUS = "transfer_count_status"
_CAMPUS_VALUE = "transfer_count_value"
_DISTRICT_VALUE = "student_transfer_count"
_DISTRICT_DATASET = "district_transfer_reports"
_REPORT_CHARTER = "reporting_charter_status"
_CHARTER = "Open Enrollment Charter"
_PAIR = ["origin", "destination"]
PASS_OBSERVED_TRANSFER_ROLLUP = "PASS_OBSERVED_TRANSFER_ROLLUP"
PASS_RECIPROCAL_DISTRICT_DIRECTIONS = "PASS_RECIPROCAL_DISTRICT_DIRECTIONS"
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


def _district_editions(release: DataRelease) -> tuple[EditionKey, ...]:
    return tuple(
        EditionKey.from_dict({name: value for name, value in row.items() if name != "edition_id"})
        for row in release.catalog.editions(dataset=_DISTRICT_DATASET, scope="district", include_metadata=False)
        if row["table"] == "report_rows"
    )


def _district_leaf(release: DataRelease, year: int) -> ArtifactRef:
    keys = _district_editions(release)
    selected = [key for key in keys if key.edition == str(year)]
    if not selected:
        years = ", ".join(sorted({key.edition for key in keys})) or "none"
        raise ValueError(f"No district transfer edition for {year}. Available school-year ends: {years}.")
    if len(selected) != 1:
        raise ValueError(f"District transfer edition for {year} is ambiguous in this release.")
    key, = selected
    period = DataPeriod("school_year", year - 1, year)
    if release.catalog.metadata(key.dataset, edition=key).period != period:
        raise ValueError("District transfer period differs from the selected school year")
    plans = release.catalog.resolve_artifacts([(key, _DISTRICT_VALUE)], period=period)
    if len(plans) != 1 or plans[0]["artifact"].edition != key or plans[0]["artifact"].output != "leaf":
        raise ValueError("Expected one exact district transfer leaf for the selected school year")
    ref = plans[0]["artifact"]
    # Schema 3 embeds edition descriptors. Schema 4 indexes them by identity.
    edition = key.to_dict()
    registered = [identifier for identifier, entry in release.catalog.to_dict()["artifacts"].items()
                  if entry == key.identity or (isinstance(entry, dict) and entry.get("edition") == edition)]
    if registered != [ref.identity]:
        raise ValueError("District transfer edition requires exactly one registered leaf")
    return ref


class DirectionCoverage(TypedDict):
    matched: int
    out_only: int
    in_only: int
    matching_observed: int
    matching_suppressed: int
    status_differences: int
    observed_count_differences: int
    raw_token_only_differences: int


def _count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    return pc.sum(mask).as_py() or 0


def _direction_coverage(joined: pa.Table) -> DirectionCoverage:
    out, incoming = joined["out_status"], joined["in_status"]
    has_out, has_in = pc.is_valid(out), pc.is_valid(incoming)
    matched = pc.and_(has_out, has_in)
    status_differs = pc.not_equal(out, incoming)
    both_observed = pc.and_(pc.equal(out, "observed"), pc.equal(incoming, "observed"))
    value_differs = pc.not_equal(joined["out_value"], joined["in_value"])
    equal_observed = pc.and_(both_observed, pc.invert(value_differs))
    equal_suppressed = pc.and_(pc.equal(out, "suppressed"), pc.equal(incoming, "suppressed"))
    raw_differs = pc.not_equal(joined["out_raw"], joined["in_raw"])
    return {
        "matched": _count_true(matched),
        "out_only": _count_true(pc.and_(has_out, pc.invert(has_in))),
        "in_only": _count_true(pc.and_(has_in, pc.invert(has_out))),
        "matching_observed": _count_true(equal_observed),
        "matching_suppressed": _count_true(equal_suppressed),
        "status_differences": _count_true(status_differs),
        "observed_count_differences": _count_true(pc.and_(both_observed, value_differs)),
        "raw_token_only_differences": _count_true(pc.and_(pc.or_kleene(equal_observed, equal_suppressed), raw_differs)),
    }


def _district_pairs(details: pa.Table, raw: str) -> pa.Table:
    if not pa.types.is_integer(details[_DISTRICT_VALUE].type):
        raise ValueError("District transfer detail requires integral counts")
    _validate_details(details, _DISTRICT_VALUE)
    direction, reporting, partner = _KEYS
    for name in (reporting, partner):
        ids = details[name]
        valid = pc.and_(pc.match_substring_regex(ids, r"^'[0-9]{6}$"), pc.not_equal(ids, "'000000"))
        if details.num_rows and not pc.all(valid).as_py():
            raise ValueError("District transfer detail requires canonical six-digit endpoint IDs")
    if details[raw].null_count or pc.any(pc.equal(details[reporting], details[partner])).as_py():
        raise ValueError("District transfer detail has a missing raw count or a within-district pair")

    sides = []
    for view, endpoints in (("out", (reporting, partner)), ("in", (partner, reporting))):
        selected = details.filter(pc.equal(details[direction], view)).select([*endpoints, _STATUS, _DISTRICT_VALUE, raw])
        selected = selected.rename_columns([*_PAIR, f"{view}_status", f"{view}_value", f"{view}_raw"])
        if selected.group_by(_PAIR, use_threads=False).aggregate([]).num_rows != selected.num_rows:
            raise ValueError(f"Duplicate {view} district transfer pair")
        sides.append(selected)
    return sides[0].join(sides[1], keys=_PAIR, join_type="full outer", coalesce_keys=True, use_threads=False)


def _historical_roster(store: ProvenanceStore, year: int) -> tuple[dict[str, str], dict]:
    from .teadata_config import load_config

    # Reuse the already reviewed annual TEA reference. No PEIMS rows are opened.
    metadata = load_config(Path(__file__).with_name("teadata_sources.yaml")).resolve_dataset(
        "peims_special_education_by_primary_disability", year, reader_profile="source", exact=True,
    ).metadata
    ref, = (ref for ref in metadata.references if ref.reference_id == DISTRICT_TYPE_REFERENCE)
    records = store.verify_record_closure(ref.provenance_ids)
    roster = reviewed_district_types(metadata, store, records)
    if roster is None:
        raise ValueError("Historical district transfer audit lacks the reviewed same-year district types")
    receipt, = ref.provenance_ids
    return roster, {"basis": "same_year_tea_district_types", "receipt": receipt,
                    "sha256": ref.reviewed_sha256, "url": ref.url}


def _district_types(table: pa.Table, pairs: pa.Table, *, year: int, store: ProvenanceStore | None) -> dict:
    if _REPORT_CHARTER in table.column_names:
        reporting = _KEYS[1]
        flags = table.select([reporting, _REPORT_CHARTER]).group_by(
            [reporting, _REPORT_CHARTER], use_threads=False,
        ).aggregate([])
        flags = flags.filter(pc.is_valid(flags[reporting]))
        allowed = pa.array(["Traditional ISD/CSD", _CHARTER])
        if (len(flags[reporting].unique()) != flags.num_rows
                or (flags.num_rows and not pc.all(pc.is_in(flags[_REPORT_CHARTER], value_set=allowed)).as_py())):
            raise ValueError("District report contains conflicting or unsupported reporter charter flags")
        source = {"basis": "same_edition_publisher_flags"}
        origin, destination = (
            pc.equal(pc.take(flags[_REPORT_CHARTER], pc.index_in(pairs[name], value_set=flags[reporting])), _CHARTER)
            for name in _PAIR
        )
    elif store is not None and 2013 <= year <= 2016:
        roster, source = _historical_roster(store, year)
        origin, destination = (
            project_district_types(pc.utf8_slice_codeunits(pairs[name], start=1), roster)[1] for name in _PAIR
        )
    else:
        source = {"basis": "unavailable"}
        origin = destination = pa.nulls(pairs.num_rows, type=pa.bool_())
    return {
        **source,
        "unknown_origin_pairs": origin.null_count,
        "unknown_destination_pairs": destination.null_count,
        "charter_origin_pairs": _count_true(origin),
        "charter_destinations": (
            _direction_coverage(pairs.filter(destination)) if source["basis"] != "unavailable" else None
        ),
    }


def compare_district_directions(release: DataRelease, *, school_year: int | str,
                                roster_store: ProvenanceStore | None = None) -> dict:
    """Audit independent publisher views in one exact offline district edition."""
    year = _parse_school_year(school_year)
    if year is None or not release.offline:
        raise ValueError("District direction audit requires an explicit school year and offline release")
    ref = _district_leaf(release, year)
    support = ref.column_support
    count_support = support.projection(_DISTRICT_VALUE) if support is not None else {}
    raw = count_support.get("raw")
    if not isinstance(raw, str) or count_support.get("status") != _STATUS:
        raise ValueError("District direction audit requires bound raw counts and statuses")
    kind = "transfer_row_kind"
    columns = [*_KEYS, _DISTRICT_VALUE, _STATUS, raw, kind]
    leaf = release.analytical_dataset(ref.identity)
    if _REPORT_CHARTER in leaf.schema.names:
        columns.append(_REPORT_CHARTER)
    table = leaf.to_table(columns=columns, use_threads=False)
    kinds = pa.array(["detail", "publisher_total", "separator"])
    if table.num_rows and not pc.all(pc.is_in(table[kind], value_set=kinds)).as_py():
        raise ValueError("District transfer leaf contains an unrecognized row kind")
    details = table.filter(pc.equal(table[kind], "detail"))
    pairs = _district_pairs(details, raw)
    coverage = _direction_coverage(pairs)
    classification = _district_types(table, pairs, year=year, store=roster_store)
    passed = coverage["matched"] > 0 and not any(coverage[name] for name in (
        "out_only", "in_only", "status_differences", "observed_count_differences",
    )) and classification["charter_origin_pairs"] == 0
    return {
        "status": PASS_RECIPROCAL_DISTRICT_DIRECTIONS if passed else "HOLD_RECIPROCAL_DISTRICT_DIRECTIONS",
        "edition": ref.edition.to_dict(),
        "manifest_sha256": release.manifest_sha256,
        "leaf_sha256": ref.sha256,
        "directional_detail_rows": details.num_rows,
        "coverage": coverage,
        "classification": classification,
    }


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
    ref = _district_leaf(release, year)
    filters = [("transfer_row_kind", "=", "detail")]
    result = _compare_details(
        pq.read_table(campus_path, columns=[*_KEYS, *_CAMPUS_IDS, _STATUS, _CAMPUS_VALUE], filters=filters),
        pq.read_table(release.artifact(ref.identity), columns=[*_KEYS, _STATUS, _DISTRICT_VALUE], filters=filters),
    )
    return {**result, "year": year, "campus_preparation": campus_preparation,
            "campus_prepared_sha256": campus_digest, "campus_preparation_status": validation["status"],
            "district_manifest_sha256": release.manifest_sha256, "district_leaf_sha256": ref.sha256,
            "canonical_edges_or_release_qualified": False}
