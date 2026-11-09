from copy import deepcopy
import csv
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
import json

from openpyxl import load_workbook
import pyarrow as pa
import pytest

from teadata import peims_courses as courses
from teadata.analytical import write_leaf_artifact
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ReferenceDocument, ResolvedDataset
from teadata.export_provenance import ExportProvenance
from teadata.manifest import BuildRecorder, ContractError, canonical_json
from teadata.peims_course_sources import DATASET, course_csv_preamble
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease, publish_release
from teadata.teadata_config import load_config


@pytest.mark.parametrize(("year", "state_rows", "district_rows", "campus_rows"), [
    (2023, 1386, 200743, 504825), (2024, 1397, 201999, 511268),
    (2025, 1422, 204303, 519697), (2026, 1453, 207751, 533161),
])
def test_packaged_course_sources_are_exact_and_opt_in(year, state_rows, district_rows, campus_rows):
    config = load_config()
    assert config.is_source_only(DATASET, year)
    assert config.column_schema(DATASET, year) is None
    state = courses._table(config, year, "state")
    assert state["layout"]["last_data_record"] - 9 == state_rows
    district = courses._table(config, year, "district")
    assert district["layout"]["last_data_record"] - 9 == district_rows
    assert district["input_sha256"] != state["input_sha256"]
    campus = courses._table(config, year, "campus")
    assert campus["layout"]["last_data_record"] - 9 == campus_rows
    assert campus["input_sha256"] not in (state["input_sha256"], district["input_sha256"])
    assert [(row["code"], row["field"]) for row in campus["fields"][3:7]] == [
        ("DISTRICT", "source_district_number"), ("DISTRICT NAME", "source_district_name"),
        ("CAMPUS", "campus_number"), ("CAMPUS NAME", "campus_name"),
    ]
    metadata = config.resolve_dataset(DATASET, year, reader_profile="source", exact=True).metadata
    assert "represented by DISTRICT are unconfirmed" in (metadata.notes or "")


def _raw_source(tmp_path, scope, records):
    config = load_config()
    table = deepcopy(courses._table(config, 2026, scope))
    original = config.resolve_dataset(DATASET, 2026, reader_profile="source", exact=True)
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows(course_csv_preamble(2026, scope))
    start = stream.getvalue().encode("cp1252")
    writer.writerows(records)
    raw = stream.getvalue().encode("cp1252")
    footer = b"\r\nReviewed footer\r\n"
    path = tmp_path / "source.csv"
    path.write_bytes(raw + footer)
    first = 10 + len(records)
    table["layout"]["last_data_record"] = first - 1
    table["layout"]["annotations"] = [
        {"first_record": 1, "last_record": 8, "kind": "preamble",
         "sha256": sha256(b"".join(start.splitlines(keepends=True)[:8])).hexdigest()},
        {"first_record": first, "last_record": first + 1, "kind": "footer", "sha256": sha256(footer).hexdigest()},
    ]
    return path, courses._CourseSource(2026, scope, original, "", "", table, ())


@pytest.mark.parametrize("scope", ["state", "district"])
def test_native_course_batches_preserve_codes_precision_and_masks(tmp_path, scope):
    state = scope == "state"
    geo = ["STATE", "STATE", "2025-2026"] if state else [
        "STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
    ]
    path, selection = _raw_source(tmp_path, scope, [
        [*geo, "ART", "00123456", "ART, GRADE 2", "GRADE 2", "1", "0.30", "0", "0", "-999"],
        [*geo, "ART", "00123457", "ART, GRADE 3", "GRADE 3", "0", "0", "1", "0.20", "0"],
    ])
    table = pa.Table.from_batches(list(courses._batches(path, selection)))
    assert table.schema.field("teacher_fte_count").type == pa.decimal128(16, 2)
    assert table["teacher_fte_count"].to_pylist() == [Decimal("0.30"), Decimal("0.00")]
    assert table["course_code"].to_pylist() == ["00123456", "00123457"]
    if not state:
        assert table["district_number"].to_pylist() == ["001902", "001902"]
        assert table["region_number"].to_pylist() == ["07", "07"]
    assert table["peims_source_record"].to_pylist() == [10, 11]
    assert table["peims_raw_teacher_fte_count"].to_pylist() == ["0.30", "0"]
    assert table["course_enrollment_count"].to_pylist() == [None, 0]
    assert table["peims_raw_course_enrollment_count"].to_pylist() == ["-999", "0"]
    assert table["peims_status_course_enrollment_count"].to_pylist() == ["suppressed", "observed"]


@pytest.mark.parametrize(("damage", "match"), [
    ("duplicate", "Duplicate PEIMS course code"), ("identity", "Conflicting PEIMS course district identity"),
    ("district", "Malformed district ID"), ("scope", "selected scope or year"), ("year", "selected scope or year"),
    ("precision", "numeric token"), ("headcount-mask", "numeric token"), ("footer", "raw preamble or footer"),
    ("reopened", "district group reappeared"),
])
def test_native_course_stream_rejects_mismatched_scope_identity_and_values(tmp_path, monkeypatch, damage, match):
    monkeypatch.setattr(courses, "_CSV_BLOCK_BYTES", 192)
    rows = [
        ["STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
         "ART", "00123456", "ART, GRADE 2", "GRADE 2", "1", "0.30", "0", "0", "-999"],
        ["STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
         "ART", "00123457", "ART, GRADE 3", "GRADE 3", "1", "0.30", "0", "0", "0"],
    ]
    replacements = {"duplicate": (8, "00123456"), "identity": (4, "OTHER ISD"), "district": (3, "1902"),
                    "scope": (1, "STATE"), "year": (2, "2024-2025"), "precision": (12, "0.301"),
                    "headcount-mask": (11, "-999")}
    if damage in replacements:
        index, value = replacements[damage]
        rows[1][index] = value
    if damage == "reopened":
        rows.insert(1, [*rows[0][:3], "001903", *rows[0][4:]])
    path, selection = _raw_source(tmp_path, "district", rows)
    if damage == "footer":
        path.write_bytes(path.read_bytes().replace(b"Reviewed footer", b"Changed footer"))
    with pytest.raises(ContractError, match=match):
        list(courses._batches(path, selection))


@pytest.mark.parametrize(("damage", "match"), [
    (None, ""), ("same-group", "Duplicate PEIMS course code"), ("campus-id", "Malformed campus ID"),
    ("campus-suffix", "Zero campus ID suffix"), ("campus-name", "Conflicting PEIMS course campus identity"),
])
def test_campus_source_district_and_campus_are_separate_key_parts(tmp_path, monkeypatch, damage, match):
    monkeypatch.setattr(courses, "_CSV_BLOCK_BYTES", 192)
    prefix = ["STATE", "COUNTYREGIONDISTRICTCAMPUS", "2025-2026"]
    campus = ["019901102", "DEKALB EL", "TRADITIONAL ISD/CSD", "08"]
    values = ["SPECIAL EDUCATION", "SE000001", "SPECIAL EDUCATION", "ALL GRADE LEVELS", "1", "0.30", "0", "0", "-999"]
    first = [*prefix, "019901", "DEKALB ISD", *campus, *values]
    second = [*prefix, "019905", "NEW BOSTON ISD", *campus, *values]
    if damage in {"same-group", "campus-name"}:
        second[3:5] = first[3:5]
    if damage == "campus-id":
        second[5] = "19901102"
    if damage == "campus-suffix":
        second[5] = "019901000"
    if damage == "campus-name":
        second[6] = "WRONG NAME"
    path, selection = _raw_source(tmp_path, "campus", [first, second])
    if damage:
        with pytest.raises(ContractError, match=match):
            list(courses._batches(path, selection))
        return
    table = pa.Table.from_batches(list(courses._batches(path, selection)))
    assert selection.keys == ["school_year", "source_district_number", "campus_number", "course_code"]
    assert "district_number" not in table.schema.names
    assert table["source_district_number"].to_pylist() == ["019901", "019905"]
    assert table["campus_number"].to_pylist() == ["019901102", "019901102"]
    assert table["course_code"].to_pylist() == ["SE000001", "SE000001"]
    assert table["peims_raw_course_enrollment_count"].to_pylist() == ["-999", "-999"]


@pytest.mark.parametrize("damage", [None, "swapped-raw", "extra-label", "wrong-scope", "wrong-disposition", "missing-identity"])
def test_course_publisher_exports_reuse_exact_literals_and_keys(tmp_path, monkeypatch, damage):
    identity = ["STATE", "COUNTYREGIONDISTRICTCAMPUS", "2025-2026", "019905", "NEW BOSTON ISD",
                "019901102", "DEKALB EL", "TRADITIONAL ISD/CSD", "08", "ART"]
    raw, selected = _raw_source(tmp_path, "campus", [
        [*identity, "00123456", "ART, GRADE 2", "GRADE 2", "1", "0.30", "0", "0", "-999"],
        [*identity, "00123457", "ART, GRADE 3", "GRADE 3", "2", "0", "1", "0.20", "0"],
        [*identity, "00123458", "ART, GRADE 4", "GRADE 4", "0", "0", "1", "0.20", "3"],
    ])
    cfg = load_config()
    crosswalk = cfg.column_contract(DATASET, 2026, reader_profile="source")
    document = json.loads(crosswalk["text"])
    dataset, = [entry for entry in document["datasets"] if entry["edition"] == "2026"]
    selected.table["input_sha256"] = digest = sha256(raw.read_bytes()).hexdigest()
    if damage == "extra-label":
        selected.table["layout"]["label_record"] = 15
    if damage == "wrong-disposition":
        selected.table["fields"][-1]["disposition"] = "identity"
    if damage == "missing-identity":
        dataset.pop("source_only")
        dataset["prepared_input_sha256"] = digest
        for field in ("disposition", "dictionary_status", "publisher_type", "publisher_description"):
            selected.table["fields"][0].pop(field)
    dataset["tables"] = [selected.table]
    document["datasets"] = [dataset]
    text = canonical_json(document)
    schema = tmp_path / "course-crosswalk.json"
    schema.write_bytes(text.encode())
    schema_sha = sha256(text.encode()).hexdigest()
    store = ProvenanceStore(tmp_path / "provenance")
    schema_receipt = store.record_acquisition(schema, ResolvedDataset(
        "reference:course_fixture", "reference", None, None, str(schema), DatasetMetadata()))
    reference = ReferenceDocument("columns", "schema", path=schema.name, reviewed_sha256=schema_sha,
                                  provenance_ids=(schema_receipt,))
    source = replace(selected.source, metadata=replace(selected.source.metadata, references=(reference,), value_policy=None))
    monkeypatch.setattr(cfg, "resolve_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(cfg, "column_contract", lambda *args, **kwargs: {"sha256": schema_sha, "text": text})
    receipt = store.record_acquisition(raw, source)
    recorder = BuildRecorder(cfg, store)
    scope = "region" if damage == "wrong-scope" else "campus"
    key = EditionKey(DATASET, "2026", EditionKey.source_revision((digest,)), scope, "campus", "one row per course")
    support = selected.support
    if damage == "swapped-raw":
        first, second = (support["fields"][name] for name in ("teacher_fte_count", "contract_teacher_fte_count"))
        first["raw"], second["raw"] = second["raw"], first["raw"]
    recipe = {"operation": "fixture"}
    contract = recorder.leaf_contract(key, source, native_schema=selected.schema.serialize().to_pybytes(),
        columns=selected.schema.names, key_columns=selected.keys, rows=selected.rows, inputs=[digest],
        recipe=recipe, toolchain={"test": "test"}, column_support=support)
    path = tmp_path / "courses.parquet"
    physical = write_leaf_artifact(path, courses._batches(raw, selected), contract=contract)
    artifact_id = sha256(canonical_json(contract).encode()).hexdigest()
    processing = store.record_processing(dataset=DATASET, inputs=[receipt], outputs={"leaf": path},
        processor="test", code_version=recorder.processor_sha256,
        parameters={"leaf_contract_sha256": artifact_id, "recipe": recipe, "toolchain": {"test": "test"}})
    artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], path.name, canonical_json(contract), processing, "leaf")
    recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[receipt])
    release = DataRelease(publish_release(tmp_path, recorder.catalog(schema_version=4), schema_version=4), offline=True)
    table = release.table(key).select("course_enrollment_count", "teacher_fte_count").query(
        "teacher_head_count >= @minimum", params={"minimum": 1}, record=True)
    if damage:
        with pytest.raises(ContractError, match="reviewed positional and raw column contract"):
            table.provenance(restore_original_headers=True)
        return
    expected = [
        ["YEAR", "DISTRICT", "CAMPUS", "COURSE", "TEACHER FTE COUNT", "COURSE ENROLLMENT"],
        ["2025-2026", "019905", "019901102", "00123456", "0.30", "-999"],
        ["2025-2026", "019905", "019901102", "00123457", "0", "0"],
    ]
    frame = table.to_df(restore_original_headers=True)
    assert [list(frame.columns), *frame.to_numpy().tolist()] == expected
    csv_buffer, xlsx_buffer = BytesIO(), BytesIO()
    table.to_csv(csv_buffer, restore_original_headers=True)
    assert list(csv.reader(StringIO(csv_buffer.getvalue().decode()))) == expected
    table.to_excel(xlsx_buffer, restore_original_headers=True)
    workbook = load_workbook(xlsx_buffer, read_only=True)
    try:
        assert [list(row) for row in workbook["Data"].values] == expected
        assert "Provenance" in workbook.sheetnames
    finally:
        workbook.close()
    saved = ExportProvenance.from_dict(frame.attrs["teadata_provenance"])
    column = saved.to_dict()["columns"][-1]
    assert column["column"] == "peims_raw_course_enrollment_count"
    assert column["fields"][0]["field"] == "course_enrollment_count"
    assert column["support_for"] == [{"field": "course_enrollment_count", "role": "raw"}]
    assert saved.replay_table(release).to_df(restore_original_headers=True).equals(frame)
