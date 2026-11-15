"""Publisher goldens through an explicitly pinned normal parent attachment.

These checks do not build a release, acquire sources or qualify unreviewed
earlier years. The full retained source gate is in test_native_college_credit.
"""

import csv
from decimal import Decimal
from io import BytesIO, StringIO
import json
import os
import sqlite3
from uuid import UUID

from openpyxl import load_workbook
import pyarrow as pa
import pytest
from sqlalchemy import Column, Integer, JSON, MetaData, String, Table, Text, create_engine, select

from teadata.engine import DataEngine
from teadata.entities import District
from teadata.manifest import ContractError
from teadata.native_college_credit import reviewed_college_credit
from teadata.release import DataRelease
from teadata.verification import _native_review, _open_reviewed_native_evidence


_DATASET = "peims_college_credit"
_PIN = "19fffe1f8d481d3c08bb01fe7cb4f73ae9ca8393d523a0d973411500cc33e3b8"
_COUNT = "students_earning_college_credit_count"
_HOURS = "college_credit_hours"
_AVERAGE = "average_credit_hours_per_student"


def test_normal_lazy_college_credit_publisher_workflow(tmp_path, monkeypatch):
    manifest = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_MANIFEST")
    pin = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_SHA256")
    if manifest is None and pin is None:
        pytest.skip("Provide the retained normal-source manifest and its trusted SHA-256")
    assert manifest and pin, "Both TEADATA_NORMAL_SOURCE_GOLDEN_MANIFEST and _SHA256 are required"
    monkeypatch.setattr("teadata.release._download", lambda *a, **k: pytest.fail("Implicit source download"))
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("Network access during offline normal-source workflow"))
    monkeypatch.setattr("socket.socket.connect_ex", lambda *a, **k: pytest.fail("Network access during offline normal-source workflow"))
    release = DataRelease(manifest, expected_sha256=pin, offline=True)
    child, = (ref for ref in release.catalog.native_sources if ref.manifest_sha256 == _PIN)
    assert {(key.edition, key.scope) for key in child.editions} == {
        (str(year), scope) for year in range(2022, 2027) for scope in ("state", "region", "district")}
    review = reviewed_college_credit(_PIN)
    assert review is not None
    selected_child, adjacent = _open_reviewed_native_evidence(release, child, "College Credit")
    evidence = _native_review(selected_child, adjacent).evidence()
    assert isinstance(evidence, dict)
    assert {witness["receipt"] for witness in review["publisher_witnesses"]} <= set(evidence["publisher"])
    # Expectations are frozen from publisher originals, not current candidate values.
    for case in review["goldens"]:
        view = release.observations(_DATASET, *case["facts"], scope=case["scope"], school_year=case["edition"])
        params = {"service": case["service_key"]}
        expression = "college_credit_service_key == @service"
        if case["scope"] != "state":
            expression += f" and {case['scope']}_number == @id"
            params["id"] = case["entity"]
        row, = view.query(expression, params=params).to_arrow().to_pylist()
        assert row["peims_source_record"] == case["source_record"]
        for field, expected in case["facts"].items():
            value = expected["value"]
            if value is not None and expected["type"].startswith("decimal"):
                value = Decimal(value)
            assert (row[field], row[f"peims_raw_{field}"], row[f"peims_status_{field}"]) == (
                value, expected["raw"], expected["status"]), (case["case_id"], field)

    engine = DataEngine()
    engine.catalog = release.catalog
    entities = []
    for identifier, name in ((1902, "CAYUGA ISD"), (57829, "A+ ACADEMY")):
        entity = District(id=UUID(int=identifier), name=name,
                          district_number=identifier)  # ty: ignore[invalid-argument-type]
        engine.add_district(entity)
        entities.append(entity)
    engine.bind_release(release)
    entity = entities[0]
    with monkeypatch.context() as discovery:
        discovery.setattr(DataRelease, "analytical_dataset", lambda *a, **k: pytest.fail("Discovery opened a leaf"))
        assert _HOURS in entity.peims_college_credit.variables()
        assert entity.peims_college_credit.years(_HOURS) == (2022, 2023, 2024, 2025, 2026)
    total = "college_credit_service_key == @service"
    params = {"service": "Total"}
    direct = entity.peims_college_credit.select(_COUNT, _HOURS).query(total, params=params)
    explicit = entity.dataset(_DATASET).at(26).select(_COUNT, _HOURS).query(total, params=params)
    assert direct.to_arrow().equals(explicit.to_arrow(), check_metadata=False)
    latest, = direct.to_arrow().to_pylist()
    assert (latest[_COUNT], latest[_HOURS]) == (136, 676)
    with pytest.raises(ContractError, match="Multiple published rows|select|query"):
        _ = entity.peims_college_credit.college_credit_hours
    earlier, = entity.peims_college_credit.at(22).select(_HOURS).query(total, params=params).to_arrow().to_pylist()
    assert earlier[_HOURS] == 424
    cohort = (engine >> ("district", "*")).observations(_DATASET, _HOURS, school_year=26)
    assert {row["district_number"] for row in cohort.query(total, params=params).to_arrow().to_pylist()} == {"001902", "057829"}
    for identifier in (1902, "001902", "'001902"):
        selected = release.series(_DATASET, _HOURS, scope="district", years=[26], ids=[identifier])
        rows = [row for row in selected.to_arrow().to_pylist() if row["college_credit_service_key"] == "Total"]
        assert len(rows) == 1 and rows[0][_HOURS] == 676
    historical = release.observations(_DATASET, _HOURS, scope="district", school_year=22).query(
        "district_number == @id and college_credit_service_key == @service", params={"id": "220819", **params})
    past, = historical.to_arrow().to_pylist()
    assert past[_HOURS] == 381
    current_missing = release.observations(_DATASET, _HOURS, scope="district", school_year=26).query(
        "district_number == @id", params={"id": "220819"})
    assert current_missing.to_arrow().num_rows == 0  # No replacement with an earlier source row.
    regions = release.observations(_DATASET, _HOURS, scope="region", school_year=26).query(total, params=params)
    assert set(regions.to_arrow()["region_number"].to_pylist()) == {f"{n:02d}" for n in range(1, 21)}
    for scope, year in (("campus", 2026), ("district", 2021)):
        with pytest.raises(ContractError, match="available|Available|unavailable|Unavailable"):
            release.observations(_DATASET, _HOURS, scope=scope, school_year=year)
    with pytest.raises(ContractError, match="2023|2022"):
        release.series(_DATASET, "cte_excluding_participants_college_credit_hours", scope="state", years=[22, 23])

    # Same selected source, literal publisher export and linked provenance.
    projected = entity.peims_college_credit.at(26).select(_HOURS).query(total, params=params, record=True)
    frame = projected.to_df(restore_original_headers=True)
    assert frame.iloc[0].tolist() == ["001902", "", "Total", "676"]
    provenance = frame.attrs["teadata_provenance"]
    assert provenance["release"]["manifest_sha256"] == _PIN
    assert provenance["edition_acquisitions"][0]["downloaded_at"]
    csv_stream = BytesIO()
    projected.to_csv(csv_stream, restore_original_headers=True)
    assert list(csv.reader(StringIO(csv_stream.getvalue().decode("utf-8")))) == [frame.columns.tolist(), frame.iloc[0].tolist()]
    sidecar = tmp_path / "college-credit.provenance.json"
    sidecar.write_text(json.dumps(provenance))
    assert json.loads(sidecar.read_text()) == provenance
    workbook = BytesIO()
    projected.to_excel(workbook, restore_original_headers=True)
    book = load_workbook(workbook, read_only=True)
    try:
        assert book.sheetnames == ["Data", "Provenance"]
        # Excel stores the literal empty Service token as a blank cell; its
        # reader returns None. Numeric0/masks/text are not blank-normalized.
        assert list(book["Data"].values) == [tuple(frame.columns), tuple(None if value == "" else value for value in frame.iloc[0])]
    finally:
        book.close()

    series = release.series(_DATASET, _HOURS, _AVERAGE, scope="district", years=[22, 23, 24, 25, 26], ids=[1902, 57829])
    values = series.to_arrow()
    totals = {(row["measurement_school_year"], row["district_number"]): row for row in values.to_pylist()
              if row["college_credit_service_key"] == "Total"}
    for case in review["goldens"]:
        if case["scope"] == "district" and case["entity"] in {"001902", "057829"} and case["service_key"] == "Total":
            row = totals[case["edition"], case["entity"]]
            assert row[_HOURS] == case["facts"][_HOURS]["value"]
            assert row[_AVERAGE] == Decimal(case["facts"][_AVERAGE]["value"])
    descriptor = series.provenance()
    assert len(descriptor["sources"]) == 5
    assert descriptor["cross_year_comparability"] == "unassessed"
    metadata = MetaData()
    receipts = Table("imports", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("facts", metadata, Column("import_id", String(64), nullable=False))
    for field in series.schema:
        facts.append_column(Column(field.name, Integer if pa.types.is_integer(field.type) else Text))
    legacy = getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)
    sql = create_engine("sqlite://", connect_args={"autocommit": legacy} if legacy is not None else {})
    metadata.create_all(sql)
    try:
        with sql.begin() as connection:
            result = series.to_sql(connection, facts, receipts=receipts)
        with sql.connect() as connection:
            restored = pa.Table.from_batches(series.iter_sql_batches(connection, facts, receipts=receipts, import_id=result.import_id))
            receipt = connection.execute(select(receipts.c.receipt)).scalar_one()
        assert receipt["selection"] == descriptor
        def sorted_rows(table):
            return sorted(table.to_pylist(), key=lambda row: (row["measurement_school_year"], row["district_number"], row["college_credit_service_key"]))
        assert sorted_rows(restored) == sorted_rows(values)
    finally:
        sql.dispose()
