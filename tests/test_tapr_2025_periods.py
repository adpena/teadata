"""Regression for report headings, not periods inferred from annual field names.

The retained 2025 Student Information reports identify Mobility, Attrition and
Disciplinary Placements as 2023-24. District HTML b3a93839... uses lines 2824,
3050 and 3174; campus HTML 7dbe9ffb... uses lines 2700, 2926 and 3050.
Synthetic values below exercise transport, not independently verified TEA facts.
"""

from dataclasses import replace
import csv
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from teadata.analytical import measurement_from_row
from teadata.analytical_build import build_analytical_artifacts
from teadata.dataset_metadata import ResolvedDataset
from teadata.manifest import DatasetCatalog, write_sqlite_catalog
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import load_config
from teadata.values import ValueStatus


ROOT = Path(__file__).resolve().parents[1]
LAGGED = {"kind": "school_year", "start_year": 2023, "end_year": 2024}
REPORT = {"kind": "school_year", "start_year": 2024, "end_year": 2025}
GROUPS = (
    ("ALL", "all_students"), ("BLA", "african_american"),
    ("HIS", "hispanic"), ("WHI", "white"), ("IND", "american_indian"),
    ("ASI", "asian"), ("PCI", "pacific_islander"),
    ("TWO", "two_or_more_races"), ("ECO", "econ_disadv"),
    ("LEP", "eb_el"), ("SPE", "special_ed"),
)


def reviewed_fields(scope):
    """The complete observed report families, including count companions."""
    prefix = "C" if scope == "campus" else "D"
    result = []
    for code, group in GROUPS:
        for suffix, unit in (("C", "count"), ("P", "percent")):
            result.append((
                f"{prefix}PEM{code}{suffix}",
                f"student_membership_mobility_{group}_{unit}",
                f"{scope}_2025_student_membership_2024_mobility_{group}_{unit}", unit,
            ))
    for suffix, unit in (("C", "count"), ("P", "percent")):
        result.extend((
            (f"{prefix}PETATT{suffix}", f"student_membership_attrition_all_students_{unit}",
             f"{scope}_2025_student_membership_2023_24_attrition_all_students_{unit}", unit),
            (f"{prefix}PETDIS{suffix}", f"daep_{unit}", f"{scope}_2024_daep_{unit}", unit),
        ))
    return result


@pytest.mark.parametrize("scope", ["campus", "district"])
def test_complete_lagged_families_override_report_and_unknown_source_periods(scope):
    cfg = load_config(ROOT / "teadata/teadata_sources.yaml")
    for family in ("student_staff_profile", "student_profile"):
        dataset = f"{scope}_tapr_{family}"
        source = cfg.resolve_dataset(dataset, 2025)
        assert source.metadata.period.to_dict() == (
            REPORT if family == "student_staff_profile" else {"kind": "unknown"}
        )
        records = cfg.header_records(dataset, edition="2025", reader_profile="source")
        by_code = {row["code"]: row for row in records}
        for code, selector, field, unit in reviewed_fields(scope):
            definition = source.metadata.metrics[selector]
            assert definition.field == by_code[code]["field"] == field
            assert definition.unit == unit
            assert definition.period.to_dict() == by_code[code]["period"] == LAGGED
        # A reviewed family must not turn a whole annual table into that period.
        prefix = "C" if scope == "campus" else "D"
        assert by_code[prefix + "PERRAKR"]["period"] == {"kind": "unknown"}
        if family == "student_profile":
            assert source.metadata.value_policy is None
            assert source.metadata.field_year is None
            assert cfg.is_source_only(dataset, 2025)
    staff = cfg.header_records(f"{scope}_tapr_staff_profile", edition="2025", reader_profile="source")
    if scope == "district":
        turnover = [row for row in staff if row["code"] in {"DPSTURNN", "DPSTURND", "DPSTURNR"}]
        assert len(turnover) == 3
        assert {row["code"]: row["period"] for row in turnover} == {
            "DPSTURNN": {"kind": "unknown"}, "DPSTURND": {"kind": "unknown"}, "DPSTURNR": REPORT,
        }


def synthetic_catalog(tmp_path, family):
    """Use actual period definitions with explicit synthetic acquisition/policy.

    This does not activate a source-only contract or borrow production receipts.
    """
    cfg = load_config(ROOT / "teadata/teadata_sources.yaml")
    store = ProvenanceStore(tmp_path / "provenance")
    datasets, receipts = {}, {}
    for scope in ("campus", "district"):
        dataset = f"{scope}_tapr_{family}"
        configured = cfg.resolve_dataset(dataset, 2025)
        specs = reviewed_fields(scope)
        metadata = replace(
            configured.metadata, provenance_ids=(), references=(), value_policy="period_fixture",
            metrics={selector: configured.metadata.metrics[selector] for _, selector, _, _ in specs},
        )
        path = tmp_path / f"{scope}.csv"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(([field for _, _, field, _ in specs], ["0"] * len(specs)))
        source = ResolvedDataset(dataset, "data_sources", 2025, 2025, str(path), metadata)
        identifier = store.record_acquisition(path, source)
        record = store.read_record(identifier)
        receipts[identifier] = record
        datasets[dataset] = {
            "section": "data_sources", "status": "loaded", "resolved_year": 2025,
            "artifact": record["artifact"],
            "metadata": replace(metadata, provenance_ids=(identifier,)).to_dict(),
            "reads": [{"rows": 1, "columns": [field for _, _, field, _ in specs]}],
        }
    return DatasetCatalog.create(
        datasets=datasets, records=receipts, processor="2" * 64,
        value_policies={"period_fixture": {"policy_id": "period_fixture", "tokens": {}}},
        metric_bindings={"reviewed.*": {"dataset": "{scope}_tapr_" + family,
            "field": "{metric}", "field_mode": "metric", "minimum": 0}},
    )


def synthetic_entities(path, catalog):
    """Two small, fictional entities for the existing analytical export API."""
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE districts (id TEXT PRIMARY KEY, name TEXT, district_number TEXT,
                district_number_canon TEXT, enrollment, rating, aea, aea_raw, meta TEXT);
            CREATE TABLE campuses (id TEXT PRIMARY KEY, district_id TEXT, campus_number TEXT,
                campus_number_canon TEXT, name TEXT, is_charter, is_private, enrollment,
                rating, aea, aea_raw, grade_range TEXT, lon, lat, meta TEXT, lookup_key TEXT,
                source_namespace TEXT, source_id TEXT);
        """)
        values = {
            scope: json.dumps({field: "0" for _, _, field, _ in reviewed_fields(scope)})
            for scope in ("campus", "district")
        }
        connection.execute("INSERT INTO districts VALUES (?,?,?,?,?,?,?,?,?)", (
            "d", "Period Fixture ISD", "012345", "'012345", 0, None, None, None, values["district"],
        ))
        connection.execute("INSERT INTO campuses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "c", "d", "012345678", "'012345678", "Period Fixture Campus", False, False,
            0, None, None, None, "KG-12", None, None, values["campus"], "012345678", None, None,
        ))
        write_sqlite_catalog(connection, catalog)
    return path


@pytest.mark.parametrize("family", ["student_staff_profile", "student_profile"])
def test_catalog_measurements_and_analytical_exports_keep_reviewed_field_periods(tmp_path, family):
    catalog = synthetic_catalog(tmp_path, family)
    # Serialized catalogs, not mutable Config, govern the exported interpretation.
    catalog = DatasetCatalog(catalog.to_dict())
    for scope in ("campus", "district"):
        for _, selector, field, unit in reviewed_fields(scope):
            assert catalog.metric_metadata("reviewed." + selector, scope=scope)["period"] == LAGGED
            assert catalog.field_metadata(field, scope=scope)["period"] == LAGGED
            entity = SimpleNamespace(meta={field: "0"}, source_build_id=catalog.build_id)
            value = catalog.measurement(entity, "reviewed." + selector, scope=scope)
            assert value.numeric == 0 and value.unit == unit
            assert value.source["period"] == LAGGED
            missing = catalog.measure_field("", field, scope=scope)
            assert missing.status == ValueStatus.MISSING
            assert missing.source["period"] == LAGGED
    source = synthetic_entities(tmp_path / "entities.sqlite", catalog)
    paths, _ = build_analytical_artifacts(source, catalog, tmp_path / "analytical", metrics=[])
    sources = pq.read_table(paths["analytical_sources"]).to_pylist()
    assert len(sources) == 52
    for scope in ("campus", "district"):
        rows = pq.read_table(paths[f"{scope}_analytics"]).to_pylist()
        assert {row["field"] for row in rows} == {field for _, _, field, _ in reviewed_fields(scope)}
        assert len(rows) == 26
        for row in rows + [entry for entry in sources if entry["scope"] == scope]:
            assert row["period"]["kind"] == "school_year"
            assert (row["period"]["start_year"], row["period"]["end_year"]) == (2023, 2024)
            assert row["dataset_period"]["kind"] == (
                "school_year" if family == "student_staff_profile" else "unknown"
            )
        assert all(measurement_from_row(row).source["period"] == LAGGED for row in rows)
