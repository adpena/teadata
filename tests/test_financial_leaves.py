"""Real tiny workbooks exercise source binding, not TEA scientific correctness."""

from hashlib import sha256
import json

from openpyxl import Workbook
from openpyxl.styles import PatternFill
import pytest

from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DatasetMetadata, EditionKey, ResolvedDataset
from teadata.financial_leaves import materialize_financial_workbooks
from teadata.manifest import canonical_json
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease
from teadata.scripts.refresh_financials import main
from teadata.values import measure


HEADERS = ["DISTRICT ID", "DISTRICT NAME", "SOF Run ID", "ENROLLMENT", "AMOUNT"]
FIELDS = ["district_number", "district_name", "sof_run", "student_count", "amount_currency"]


def financial_fixture(tmp_path, *, years=(2024,), damage=None, tables=1, duplicate_headers=False, fractional_fte=False,
                      uninterpreted_values=None):
    store = ProvenanceStore(tmp_path / "store")
    notes = tmp_path / "publisher-notes.txt"
    notes.write_text("Synthetic fixture: -999 means suppressed numeric cell. Publication run P3 may include older row runs. Amounts are actual dollars.")
    notes_id = store.record_acquisition(notes, ResolvedDataset("reference:fixture_notes", "reference", None, None, "",
        DatasetMetadata(title="Explicit synthetic source policy; not TEA methodology")))
    notes_sha = sha256(notes.read_bytes()).hexdigest()
    entries = {}
    policy = {"reference_id": "notes", "tokens": {"-999": {"status": "suppressed", "reason": "synthetic_source_rule"}}}
    sheet_names = {2024: "2024 AF Act", 2025: "24_25 Actual", 2026: "Human SOF Header"}
    headers = HEADERS + ["AMOUNT"] if duplicate_headers else list(HEADERS)
    fields = FIELDS[:4] + ["amount_currency_at_e", "amount_currency_at_f"] if duplicate_headers else list(FIELDS)
    if fractional_fte:
        headers[3], fields[3] = "STAFF FTE", "staff_full_time_equivalent"
    if uninterpreted_values is not None:
        headers.append("UNREVIEWED TAX DETAIL")
        fields.append("unreviewed_tax_detail")
    for year in years:
        workbook = Workbook()
        workbook.remove(workbook.active)
        selections, maps = {}, []
        for number in range(tables):
            name = sheet_names[year] + (" secondary" if number else "")
            sheet = workbook.create_sheet(name)
            sheet.append(headers)
            if uninterpreted_values is None:
                sheet.append(["001001", "First district", "P1", 1.5 if fractional_fte else -999, -12.5] + ([33.25] if duplicate_headers else []))
                sheet.append(["001002", "Second district", "P3", 2.25 if fractional_fte else 0, 0] + ([44.5] if duplicate_headers else []))
            else:
                for index, token in enumerate(uninterpreted_values):
                    sheet.append([f"{1001 + index:06}", "Synthetic district", "P1", -999 if index == 0 else 0,
                        -12.5 if index == 0 else 0] + ([33.25] if duplicate_headers else []) + [token])
            sheet.cell(1, len(headers) + 1).fill = PatternFill(fill_type="solid", fgColor="FFFF00")
            if damage == "extra_header":
                sheet.cell(1, 6, "UNREVIEWED")
            elif damage == "populated_unnamed":
                sheet.cell(2, 6, "unreviewed data")
            elif damage == "footer":
                sheet.append([None, "Unreviewed footer", None, None, None])
            elif damage == "formula":
                sheet.cell(2, 5, "=1+1")
            elif damage == "duplicate_id":
                sheet.cell(3, 1, "001001")
            elif damage == "bad_id":
                sheet.cell(2, 1, "state total")
            elif damage == "fractional_count":
                sheet.cell(2, 4, 1.5)
            elif damage == "bad_currency":
                sheet.cell(2, 5, "unknown money")
            elif damage == "merge":
                sheet.merge_cells("A2:B2")
            logical = "actual" + str(number)
            selections[logical] = {"sheet": name, "header_row": 1, "scope": "district",
                "grain": "one publisher district row per worksheet", "key_columns": ["district_number"],
                "identifier_field": "district_number"}
            if uninterpreted_values is not None:
                selections[logical]["uninterpreted_fields"] = ["unreviewed_tax_detail"]
            maps.append({"table": logical, "source_schema_sha256": source_schema_fingerprint(headers, headers),
                "metric_period": {"kind": "unknown"},
                "fields": [{"position": position, "code": code, "label": code, "field": field,
                    "disposition": "identity" if position <= 3 else "fact", "publisher_type": None,
                    "publisher_description": None, "dictionary_status": "not_applicable" if position <= 3 else "not_documented"}
                    for position, (code, field) in enumerate(zip(headers, fields, strict=True), 1)]})
            if duplicate_headers:
                maps[-1]["header_identity"] = "position"
        workbook.create_sheet("Literal Notes").append(["Synthetic source notes retained in original workbook"])
        path = tmp_path / f"original-{year}.xlsx"
        workbook.save(path)
        source_sha = sha256(path.read_bytes()).hexdigest()
        for table in maps:
            table["input_sha256"] = "0" * 64 if damage == "wrong_source" else source_sha
        resource = tmp_path / f"headers-{year}.json"
        resource.write_text(canonical_json({"schema_version": 2, "mapping_version": 1,
            "datasets": [{"dataset": "fixture_finance", "edition": str(year), "source_only": True, "tables": maps}]}) + "\n")
        schema_id = store.record_acquisition(resource, ResolvedDataset("reference:fixture_columns", "reference", None, None, "",
            DatasetMetadata(title="Synthetic reviewed literal workbook headers")))
        school = {"kind": "school_year", "start_year": year - 1, "end_year": year}
        fiscal = {"kind": "fiscal_year", "start_year": year - 1, "end_year": year}
        metadata = {"version": str(year), "period": {"kind": "unknown"}, "identifiers": ["district_number"],
            "value_policy": "financial_fixture", "metrics": {
                "students": {"field": fields[3], "unit": "fte" if fractional_fte else "count", "period": school},
                "dollars": {"field": fields[4], "unit": "currency", "period": fiscal}},
            "references": [{"reference_id": "columns", "role": "schema", "path": resource.name, "status": "available",
                "reviewed_sha256": sha256(resource.read_bytes()).hexdigest(), "provenance_ids": [schema_id]},
                {"reference_id": "notes", "role": "source_notes", "path": notes.name, "status": "available",
                 "reviewed_sha256": notes_sha, "provenance_ids": [notes_id]}],
            "source_qualifiers": ({"family": "sof", "reference_id": "notes", "publication_run_id": "P3",
                "row_run_column": "sof_run", "publisher_release_status": "Synthetic Final"} if year == 2026
                else {"family": "peims_financial", "reference_id": "notes", "financial_basis": "actual"})}
        if duplicate_headers:
            metadata["metrics"]["second_dollars"] = {"field": fields[5], "unit": "currency", "period": fiscal}
        if fractional_fte:
            metadata["metrics"]["staff_fte"] = metadata["metrics"].pop("students")
        if uninterpreted_values is not None:
            metadata["metrics"]["unreviewed_tax"] = {"field": "unreviewed_tax_detail", "unit": None,
                "period": {"kind": "unknown"}}
        if damage == "undefined_raw_metric":
            metadata["metrics"].pop("unreviewed_tax")
        elif damage == "known_raw_unit":
            metadata["metrics"]["unreviewed_tax"]["unit"] = "currency"
        if damage == "missing_units":
            metadata["metrics"]["dollars"].pop("unit")
        raw_id = store.record_acquisition(path, ResolvedDataset("fixture_finance", "data_sources", year, year, str(path),
            DatasetMetadata.from_dict(metadata)))
        metadata["provenance_ids"] = [raw_id]
        entries[year] = {"path": str(path), "metadata": metadata,
            "reader": {"profiles": {"source": {"column_schema": resource.name,
                "workbook": {"tables": selections, "notes_sheets": ["Literal Notes"]}}}}}
    config = tmp_path / "sources.json"
    config.write_text(canonical_json({"data_sources": {"fixture_finance": entries},
        "value_policies": {"financial_fixture": policy}}))
    return store, config, sha256(config.read_bytes()).hexdigest()


def test_one_shared_native_producer_preserves_exact_sheets_periods_raw_status_and_mixed_runs(tmp_path):
    store, config, digest = financial_fixture(tmp_path, years=(2024, 2025, 2026), tables=2)
    result = materialize_financial_workbooks(store, config=config, config_sha256=digest,
        selections=[("fixture_finance", year) for year in (2024, 2025, 2026)], output=tmp_path / "native")
    assert len(result["editions"]) == 6
    assert not result["field_correctness_qualified"] and not result["release_activation"]
    release = DataRelease(tmp_path / "native/release.json", offline=True)
    for entry in result["editions"]:
        key = EditionKey.from_dict(entry["key"])
        variables = {row["field"]: row for row in release.catalog.variables(edition=key)}
        assert set(variables) == set(FIELDS)
        assert variables["student_count"]["period"]["kind"] == "school_year"
        assert variables["amount_currency"]["period"]["kind"] == "fiscal_year"
        plan, = release.catalog.resolve_artifacts([(key, "student_count")])
        selected = release.analytical_dataset(entry["artifact_id"]).to_table(columns=list(plan["columns"]))
        assert selected["student_count"].to_pylist() == [None, 0]
        assert selected["financial_raw_student_count"].to_pylist() == ["-999", "0"]
        assert selected["financial_status_student_count"].to_pylist() == ["suppressed", "observed"]
        assert selected["financial_source_row"].to_pylist() == [2, 3]
        assert "amount_currency" not in selected.column_names
        if key.edition == "2026":
            assert selected["sof_run"].to_pylist() == ["P1", "P3"]
            assert "sof_run" not in release.catalog.to_dict()["artifacts"][entry["artifact_id"]]["contract"]["key_columns"]
        full = release.analytical_dataset(entry["artifact_id"]).to_table()
        assert full["district_number"].to_pylist() == ["'001001", "'001002"]
        assert full["financial_raw_district_number"].to_pylist() == ["001001", "001002"]
        assert full["amount_currency"].to_pylist() == [-12.5, 0]
        assert full["financial_source_sheet"].to_pylist()[0].startswith({"2024": "2024 AF", "2025": "24_25", "2026": "Human SOF"}[key.edition])
        assert measure(full["financial_raw_amount_currency"][0].as_py(), release.catalog.policy(key.dataset, edition=key), unit="currency").numeric == -12.5
        acquisitions = release.catalog.acquisitions(key.dataset, edition=key)
        assert len(acquisitions) == 1 and acquisitions[0]["method"] == "local_observation"
        descriptor = release.catalog.artifact_ref(entry["artifact_id"])
        processing = release.catalog.to_dict()["provenance"][descriptor.processing_id]
        assert "uninterpreted_fields" not in processing["parameters"]["recipe"]["worksheet"]


def test_explicit_uninterpreted_fact_stays_discoverable_without_numeric_or_status_inference(tmp_path):
    import pyarrow as pa

    raw = ["<2,790", ".", "-999", "-12.5", "0", None]
    store, config, digest = financial_fixture(tmp_path, uninterpreted_values=raw)
    result = materialize_financial_workbooks(store, config=config, config_sha256=digest,
        selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    entry, = result["editions"]
    assert not result["field_correctness_qualified"] and not result["release_activation"]
    key = EditionKey.from_dict(entry["key"])
    release = DataRelease(tmp_path / "native/release.json", offline=True)
    variable, = [row for row in release.catalog.variables(edition=key) if row["field"] == "unreviewed_tax_detail"]
    assert variable["unit"] is None and variable["period"]["kind"] == "unknown"
    assert variable["support"] == {"context": ["financial_source_sheet", "financial_source_row"]}
    header, = [row for row in release.catalog.header_records(key.dataset, edition_key=key)
               if row["field"] == "unreviewed_tax_detail"]
    assert header["disposition"] == "fact" and header["dictionary_status"] == "not_documented"
    plan, = release.catalog.resolve_artifacts([(key, "unreviewed_tax_detail")])
    selected = release.analytical_dataset(entry["artifact_id"]).to_table(columns=list(plan["columns"]))
    assert selected["unreviewed_tax_detail"].to_pylist() == raw
    field_type = selected.schema.field("unreviewed_tax_detail").type
    assert pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
    assert selected["financial_source_row"].to_pylist() == [2, 3, 4, 5, 6, 7]
    assert selected["financial_source_sheet"].to_pylist() == ["2024 AF Act"] * 6
    assert not any(name.startswith(("financial_raw_unreviewed_tax", "financial_status_unreviewed_tax")) for name in selected.column_names)
    assert "amount_currency" not in selected.column_names and "student_count" not in selected.column_names
    full = release.analytical_dataset(entry["artifact_id"]).to_table()
    assert full["amount_currency"].to_pylist() == [-12.5, 0, 0, 0, 0, 0]
    assert full["financial_status_student_count"].to_pylist() == ["suppressed", "observed", "observed", "observed", "observed", "observed"]
    descriptor = release.catalog.artifact_ref(entry["artifact_id"])
    processing = release.catalog.to_dict()["provenance"][descriptor.processing_id]
    recipe = processing["parameters"]["recipe"]
    assert recipe["worksheet"]["uninterpreted_fields"] == ["unreviewed_tax_detail"]
    assert sha256(canonical_json(recipe).encode()).hexdigest() == descriptor.contract["recipe_sha256"]


@pytest.mark.parametrize("declaration", [None, "unreviewed_tax_detail", ["unreviewed_tax_detail"] * 2,
    ["unavailable_field"], ["district_name"], ["district_number"], ["student_count"]])
def test_uninterpreted_opt_in_rejects_malformed_nonfact_identifier_and_known_unit_fields(tmp_path, declaration):
    store, config, _ = financial_fixture(tmp_path, uninterpreted_values=["<2,790", "."])
    document = json.loads(config.read_text())
    table = document["data_sources"]["fixture_finance"]["2024"]["reader"]["profiles"]["source"]["workbook"]["tables"]["actual0"]
    table["uninterpreted_fields"] = declaration
    config.write_text(canonical_json(document))
    with pytest.raises(ValueError, match="Financial uninterpreted"):
        materialize_financial_workbooks(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize("damage", ["undefined_raw_metric", "known_raw_unit"])
def test_uninterpreted_fact_requires_an_explicit_unknown_unit_definition(tmp_path, damage):
    store, config, digest = financial_fixture(tmp_path, damage=damage, uninterpreted_values=["<2,790", "."])
    with pytest.raises(ValueError, match="explicit unknown-unit facts"):
        materialize_financial_workbooks(store, config=config, config_sha256=digest,
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    assert not (tmp_path / "native").exists()


def test_unknown_unit_fact_does_not_implicitly_enable_uninterpreted_projection(tmp_path):
    store, config, _ = financial_fixture(tmp_path, uninterpreted_values=["<2,790", "."])
    document = json.loads(config.read_text())
    table = document["data_sources"]["fixture_finance"]["2024"]["reader"]["profiles"]["source"]["workbook"]["tables"]["actual0"]
    del table["uninterpreted_fields"]
    config.write_text(canonical_json(document))
    with pytest.raises(ValueError, match="explicitly reviewed canonical units"):
        materialize_financial_workbooks(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize("damage,message", [
    ("extra_header", "Financial source contains unclaimed physical column payload"), ("populated_unnamed", "unclaimed physical"),
    ("footer", "identifier/footer"), ("formula", "formulas"), ("merge", "merged"),
    ("duplicate_id", "duplicate"), ("bad_id", "identifier"),
    ("fractional_count", "invalid value"), ("bad_currency", "invalid value"),
    ("wrong_source", "different original"), ("missing_units", "units"),
])
def test_source_surprises_fail_without_publishing_candidate(tmp_path, damage, message):
    store, config, digest = financial_fixture(tmp_path, damage=damage)
    with pytest.raises((ValueError, RuntimeError), match=message):
        materialize_financial_workbooks(store, config=config, config_sha256=digest,
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    assert not (tmp_path / "native").exists()


def test_exact_config_pin_and_cli_do_not_select_a_fallback_edition(tmp_path):
    store, config, digest = financial_fixture(tmp_path)
    with pytest.raises(ValueError, match="exact pin"):
        materialize_financial_workbooks(store, config=config, config_sha256="0" * 64,
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    with pytest.raises(KeyError):
        main(["--store", str(store.root), "--config", str(config), "--config-sha256", digest,
            "--source", "fixture_finance", "2025", "--output", str(tmp_path / "native")])
    assert not (tmp_path / "native").exists()


def test_projected_support_expansion_has_its_own_native_byte_budget(tmp_path, monkeypatch):
    import fastexcel

    from teadata import financial_leaves
    from teadata.manifest import BuildRecorder
    from teadata.teadata_config import load_config

    store, config, digest = financial_fixture(tmp_path)
    cfg = load_config(config)
    selected = financial_leaves._selection(cfg, "fixture_finance", 2024, BuildRecorder(cfg, store))
    source = selected.source
    declaration, = selected.tables
    reader = fastexcel.read_excel(source.path)
    batch = reader.load_sheet_eager(declaration.sheet, header_row=declaration.header_row - 1, dtypes="string", dtype_coercion="strict")
    projected, _ = financial_leaves._project(batch, declaration, source, cfg.value_policies["financial_fixture"])
    assert batch.nbytes + 1 < projected.nbytes
    ceiling = (batch.nbytes + projected.nbytes) // 2
    assert batch.nbytes <= ceiling < projected.nbytes
    monkeypatch.setattr(financial_leaves, "MAX_BATCH_BYTES", ceiling)
    with pytest.raises(ValueError, match="projected table"):
        materialize_financial_workbooks(store, config=config, config_sha256=digest,
            selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    assert not (tmp_path / "native").exists()


def test_real_duplicate_headers_retain_both_positional_values_and_unambiguous_fields(tmp_path):
    store, config, digest = financial_fixture(tmp_path, duplicate_headers=True)
    result = materialize_financial_workbooks(store, config=config, config_sha256=digest,
        selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    entry, = result["editions"]
    key = EditionKey.from_dict(entry["key"])
    release = DataRelease(tmp_path / "native/release.json", offline=True)
    headers = release.catalog.header_records(key.dataset, edition_key=key)
    assert [(row["position"], row["code"], row["label"], row["header_identity"]) for row in headers[-2:]] == [
        (5, "AMOUNT", "AMOUNT", "position"), (6, "AMOUNT", "AMOUNT", "position")]
    with pytest.raises(RuntimeError, match="Ambiguous header"):
        release.catalog.header_mapping(key.dataset, table=key.table, edition_key=key)
    for field, expected, raw in [("amount_currency_at_e", [-12.5, 0], ["-12.5", "0"]),
                                  ("amount_currency_at_f", [33.25, 44.5], ["33.25", "44.5"])]:
        plan, = release.catalog.resolve_artifacts([(key, field)])
        selected = release.analytical_dataset(entry["artifact_id"]).to_table(columns=list(plan["columns"]))
        assert selected.num_rows == 2  # The literal publisher header was skipped once.
        assert selected[field].to_pylist() == expected
        assert selected["financial_raw_" + field].to_pylist() == raw
        assert selected["financial_status_" + field].to_pylist() == ["observed", "observed"]
        assert selected["financial_source_row"].to_pylist() == [2, 3]
        assert selected["financial_source_sheet"].to_pylist() == ["2024 AF Act", "2024 AF Act"]


def test_declared_fractional_fte_uses_the_canonical_unit_without_count_coercion(tmp_path):
    store, config, digest = financial_fixture(tmp_path, fractional_fte=True)
    result = materialize_financial_workbooks(store, config=config, config_sha256=digest,
        selections=[("fixture_finance", 2024)], output=tmp_path / "native")
    entry, = result["editions"]
    key = EditionKey.from_dict(entry["key"])
    release = DataRelease(tmp_path / "native/release.json", offline=True)
    variable, = [row for row in release.catalog.variables(edition=key) if row["field"] == "staff_full_time_equivalent"]
    assert variable["unit"] == "fte" and variable["code"] == "STAFF FTE"
    plan, = release.catalog.resolve_artifacts([(key, variable["field"])])
    selected = release.analytical_dataset(entry["artifact_id"]).to_table(columns=list(plan["columns"]))
    assert selected["staff_full_time_equivalent"].to_pylist() == [1.5, 2.25]
    assert selected["financial_raw_staff_full_time_equivalent"].to_pylist() == ["1.5", "2.25"]
    assert selected["financial_status_staff_full_time_equivalent"].to_pylist() == ["observed", "observed"]
