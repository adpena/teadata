"""Original-source decimals and explicit DPE semantic boundaries."""

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from zipfile import ZipFile

from openpyxl import Workbook
import pytest

from teadata import sof_dpe
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease


def test_frozen_admissions_preserve_held_sources_and_correct_units():
    grid = sof_dpe.definition()
    fields = {field["field"]: field for table in grid["tables"].values() for field in table["facts"]}
    assert len(fields) == 593
    assert grid["admission_counts"] == {"selected": 486, "held": 104, "already_live": 3}
    assert fields["inputs_dpe_nifa_ada"]["unit"] == "attendance_equivalent"
    assert fields["outputs_dpe_tier_ii_ent_lv3"]["unit"] == "currency"
    assert fields["outputs_dpe_local_rev_shortfall23"]["period"]["end_year"] == 2023
    for name in ("inputs_dpe_mentor_allot", "inputs_dpe_ft_staff_cnt", "inputs_dpe_pt_staff_cnt",
                 "inputs_dpe_eda_state_share_adj", "outputs_dpe_sm_mid_sped", "outputs_dpe_adj_allot"):
        assert fields[name]["admission"] == "held" and fields[name]["unit"] is None
    assert fields["inputs_dpe_ada_tot_refined"]["admission"] == "already_live"
    assert fields["outputs_dpe_ch41_flag"]["unit"] == "text"


def test_exact_original_precision_negative_zero_and_absent_cells():
    field = dict(field="money", unit="currency", precision=23, scale=19)
    raw = ["-12.1234567890123456789", "0", None, "1e-19"]
    values = sof_dpe._numbers(raw, field)
    assert values.to_pylist() == [Decimal("-12.1234567890123456789"), Decimal(0), None, Decimal("1e-19")]
    assert values.type.scale == 19
    wide = sof_dpe._numbers(["123456789012345678901234567890123456789.00001"],
                            dict(field="wide", unit="currency", precision=44, scale=5))
    assert wide[0].as_py() == Decimal("123456789012345678901234567890123456789.00001")


@pytest.mark.parametrize("token", ["1.5", "-1", "9223372036854775808", "*", "", "NaN", "(12)", "$12"])
def test_no_fractional_headcounts_overflow_or_unapproved_numeric_tokens(token):
    with pytest.raises(ContractError):
        sof_dpe._numbers([token], dict(field="students", unit="count", precision=25, scale=3))


@pytest.mark.parametrize("field", [dict(precision=True, scale=0), dict(precision=77, scale=1),
                                   dict(precision=2, scale=3), dict(precision=3, scale=-1)])
def test_explicit_precision_contract_is_required(field):
    with pytest.raises(ContractError):
        sof_dpe._numbers(["0"], dict(field="amount", unit="currency", **field))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Small synthetic source, distinct from the real publisher qualification."""
    original = sof_dpe.definition()
    grid = {key: original[key] for key in ("schema_version", "dataset", "edition", "publisher_release_status", "funding_period")}
    store = ProvenanceStore(tmp_path / "sources")
    metadata = DatasetMetadata(title="Synthetic source", version="47422")
    descriptor = ResolvedDataset("synthetic", "data_sources", 2026, 2026, "", metadata)
    dictionary = Workbook()
    listing = dictionary.active
    listing.title = "Variable Listing"
    listing.append(["Variable", "Worksheet", "Type", "Definition"])
    listing.append(["AMOUNT", "all", "Num", "Synthetic USD amount"])
    listing.append(["COUNT", "all", "Num", "Synthetic integral count"])
    listing.append(["FLAG", "all", "Char", "Synthetic literal code"])
    dictionary_path = tmp_path / "dictionary.xlsx"
    dictionary.save(dictionary_path)
    dictionary.close()
    workbook = Workbook()
    workbook.remove(workbook.active)
    tables = {}
    for name in ("inputs_dpe", "is_asahe_dpe", "mo_asahe_dpe", "outputs_dpe"):
        sheet = workbook.create_sheet(name)
        headers = ["DISTRICT_ID", "SOF_RUN_ID", "SCHOOLYEAR", "AMOUNT", "COUNT", "FLAG", "HELD"]
        sheet.append(headers)
        sheet.append(["001902", "47223", "2026", "-12.1234567890123456789", "2", "N", "not a numeric measure"])
        sheet.append(["001903", "47422", "2026", None, "0", "EMPTY_SENTINEL", "do not coerce"])
        workbook.create_sheet(name[:-4] + "_lpe").append(["Retained but not projected"])
        facts = [dict(field=name + "_" + code.lower(), position=position, publisher_header=code,
            dictionary_code=code, dictionary_row=position - 2, publisher_type="Char" if code == "FLAG" else "Num",
            publisher_description=description, admission="selected", unit=unit, period=grid["funding_period"],
            precision=23, scale=19) for position, code, description, unit in
            ((4, "AMOUNT", "Synthetic USD amount", "currency"), (5, "COUNT", "Synthetic integral count", "count"),
             (6, "FLAG", "Synthetic literal code", "text"))]
        tables[name] = dict(headers=headers, identities=[dict(position=i, field=field, publisher_header=headers[i-1])
            for i, field in enumerate(("district_number", "sof_run_id", "publisher_schoolyear"), 1)],
            facts=facts, rows=2, row_runs={"47223": 1, "47422": 1}, literal_goldens=[dict(district_number="001902",
                source_row=2, cells={name + "_amount": "-12.1234567890123456789", name + "_count": "2", name + "_flag": "N"})])
    source_path = tmp_path / "publisher.xlsx"
    workbook.save(source_path)
    workbook.close()
    # Write a fully framed stored empty string, rather than openpyxl's
    # unsupported empty inlineStr with no XML string container.
    rewritten = tmp_path / "framed.xlsx"
    with ZipFile(source_path) as original_book, ZipFile(rewritten, "w") as framed:
        for member in original_book.namelist():
            raw = original_book.read(member)
            if member.startswith("xl/worksheets/"):
                raw = raw.replace(b'<t>EMPTY_SENTINEL</t>', b'<t></t>')
            framed.writestr(member, raw)
    rewritten.replace(source_path)
    index = tmp_path / "index.html"
    index.write_text("Synthetic Near Final publicationrun47422")
    grid.update(tables=tables, workbook_sha256=sha256(source_path.read_bytes()).hexdigest(),
                dictionary_sha256=sha256(dictionary_path.read_bytes()).hexdigest(),
                admission_counts=dict(selected=12, held=4, already_live=0))
    grid["source_receipts"] = {key: store.record_acquisition(path, descriptor,
        requested_url="https://example.org/" + path.name, resolved_url="https://example.org/" + path.name,
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for key, path in (("sof_bulk_2026", source_path), ("sof_dictionary_2026", dictionary_path),
                          ("state_funding_index", index), ("sof_timeline_2026", index))}
    audit = tmp_path / "audit.json"
    audit.write_text('{"scope":"Synthetic preliminary inventory"}\n')
    grid["audit_sha256"] = store.retain_artifact(audit)["sha256"]
    report = tmp_path / "report.html"
    report.write_text("""'content':'<div id="ReportTitle1">Synthetic SOF</div><div id="FIELD1">-$12.123456789</div>'""")
    report_id = store.record_acquisition(report, descriptor, requested_url="https://example.org/report.html",
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), parameters={"cdn": "001902", "run": "47223"})
    grid["source_receipts"]["sof_detail_001902"] = report_id
    grid["report_goldens"] = [dict(district="001902", row_run="47223", receipt=report_id,
        sha256=sha256(report.read_bytes()).hexdigest(), headers={"ReportTitle1": "Synthetic SOF"},
        checks=[dict(table="inputs_dpe", field="inputs_dpe_amount", raw="-12.1234567890123456789",
                     item="FIELD1", display="-$12.123456789")])]
    resource = tmp_path / "resource.json"
    resource.write_text(json.dumps(grid) + "\n")
    monkeypatch.setattr(sof_dpe, "RESOURCE", resource.name)
    monkeypatch.setattr(sof_dpe, "files", lambda name: tmp_path)
    monkeypatch.setattr(sof_dpe, "CONTRACT_SHA256", sha256(resource.read_bytes()).hexdigest())
    return grid, source_path, store


def test_source_projection_and_native_roundtrip_keep_runs_codes_missing_and_provenance(fixture, tmp_path):
    grid, path, store = fixture
    tables = dict(sof_dpe.iter_sof_dpe_tables(path))
    for name, table in tables.items():
        assert table[name + "_amount"].to_pylist() == [Decimal("-12.1234567890123456789"), None]
        assert table["financial_status_" + name + "_amount"].to_pylist() == ["observed", "missing"]
        assert table["financial_raw_" + name + "_flag"].to_pylist() == ["N", ""]
        assert table[name + "_flag"].to_pylist() == ["N", None]
        assert table["sof_run_id"].to_pylist() == ["47223", "47422"]
        assert "HELD" not in table.column_names
    output = tmp_path / "candidate"
    result = sof_dpe.materialize_sof_dpe(store, output=output)
    assert result["release_activation"] is result["field_correctness_qualified"] is False
    release = DataRelease(output / "release.json", expected_sha256=result["manifest_sha256"], offline=True)
    receipt = sof_dpe.qualify_sof_dpe(release, ProvenanceStore(output / "provenance"))
    assert receipt["original_cells"] == 24 and receipt["publisher_formula_qualification"] is False
    assert receipt["independent_report_literal_goldens"] == 1
    assert receipt["original_download_citations"] == 4
    for artifact in release.catalog.to_dict()["artifacts"]:
        key = release.catalog.artifact_ref(artifact).edition
        metadata = release.catalog.metadata(sof_dpe.DATASET, edition=key)
        assert metadata.source_qualifiers is not None
        assert metadata.source_qualifiers.publication_run_id == "47422"
        assert metadata.source_qualifiers.row_run_column == "sof_run_id"
        original_id = grid["source_receipts"]["sof_bulk_2026"]
        original = store.read_record(original_id)
        assert metadata.source_url == "https://example.org/index.html"
        assert metadata.download_url == original["requested_url"]
        acquired, = release.catalog.acquisitions(sof_dpe.DATASET, edition=key)
        for field in ("method", "downloaded_at", "requested_url", "resolved_url", "response_headers"):
            assert acquired[field] == original[field]
        assert acquired["parameters"]["original_publisher_receipt"] == original_id
        assert acquired["recorded_at"] != original["recorded_at"]
        assert ProvenanceStore(output / "provenance").read_record(original_id) == original
        citation = release.table(key).select(key.table + "_amount").provenance()
        assert citation.to_dict()["source"]["url"] == metadata.source_url
        assert "Downloaded 2026-01-01 UTC." in citation.publisher_data_citation
        assert "unrecorded" not in citation.publisher_data_citation
    with pytest.raises(FileExistsError):
        sof_dpe.materialize_sof_dpe(store, output=output)


@pytest.mark.parametrize("changes", [
    {"method": "local_observation"}, {"downloaded_at": None},
    {"downloaded_at": "2026-01-01T00:00:00"}, {"resolved_url": None},
    {"artifact": {"sha256": "0" * 64}},
])
def test_missing_or_incorrect_download_cannot_be_relabelled(fixture, tmp_path, monkeypatch, changes):
    grid, _, store = fixture
    read = store.read_record
    def altered(identifier):
        record = read(identifier)
        if identifier == grid["source_receipts"]["sof_bulk_2026"]:
            record.update(changes)
        return record
    monkeypatch.setattr(store, "read_record", altered)
    with pytest.raises(ContractError, match="original dated publisher download"):
        sof_dpe._configuration(store, tmp_path)


def test_incorrect_source_and_interpretation_fail_without_fallback(fixture, monkeypatch):
    grid, path, _ = fixture
    monkeypatch.setattr(sof_dpe, "CONTRACT_SHA256", "0" * 64)
    with pytest.raises(ContractError, match="interpretation"):
        list(sof_dpe.iter_sof_dpe_tables(path))
    changed = deepcopy(grid)
    changed["workbook_sha256"] = "0" * 64
    monkeypatch.setattr(sof_dpe, "definition", lambda: changed)
    with pytest.raises(ContractError, match="exact retained"):
        list(sof_dpe.iter_sof_dpe_tables(path))


def test_report_reader_preserves_conflicting_repeated_ids_and_refuses_absent_content():
    source = b''' 'content':'<div id="value">1</div><div id="value">2</div>' '''
    assert sof_dpe._report_values(source, {"value"}) == {"value": {"1", "2"}}
    with pytest.raises(ContractError, match="Crystal page"):
        sof_dpe._report_values(b"No report available", {"value"})
