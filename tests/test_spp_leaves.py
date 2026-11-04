from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa
import pytest

from teadata import spp_leaves as spp
from teadata.analytical import write_leaf_artifact
from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from teadata.financial_leaves import NS, RID
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.release import DataRelease
from teadata.scripts.refresh_spp import main
from teadata.teadata_config import load_config
from teadata.values import ValuePolicy, ValueStatus


ROOT = Path(__file__).resolve().parents[1]
POLICY = {"reference_id": "masking", "tokens": {
    "-999": {"status": "suppressed", "reason": "ferpa_small_cell"},
    "NR": {"status": "missing", "reason": "publisher_no_report_reason_unspecified"},
}}
FIELDS = [*spp._IDENTITIES, *spp._METRICS]
HEADERS = ["Region", "CDN", "District Name", "Parent consent", "Timely evaluations", "Published percentage", "Met target"]


def _workbook(path, region, facts, *, district=None, damage=None):
    workbook = ET.Element(NS + "workbook")
    ET.SubElement(ET.SubElement(workbook, NS + "sheets"), NS + "sheet", {"name": "Ind11", RID: "rId1"})
    relationships = ET.Element("Relationships")
    ET.SubElement(relationships, "Relationship", Id="rId1", Target="worksheets/sheet1.xml")
    sheet = ET.Element(NS + "worksheet")
    data = ET.SubElement(sheet, NS + "sheetData")
    shared = ET.Element(NS + "sst")

    def add_row(row, values):
        row_element = ET.SubElement(data, NS + "row", r=str(row))
        for position, token in enumerate(values):
            if token is None:
                continue
            coordinate = chr(65 + position) + str(row)
            cell = ET.SubElement(row_element, NS + "c", r=coordinate)
            if type(token) is int:
                value = str(token)
            elif isinstance(token, Decimal):
                value = str(token)
            else:
                cell.set("t", "s")
                value = str(len(shared))
                ET.SubElement(ET.SubElement(shared, NS + "si"), NS + "t").text = token
            if damage == "formula" and coordinate == "E2":
                ET.SubElement(cell, NS + "f").text = "D2"
            ET.SubElement(cell, NS + "v").text = value

    add_row(1, HEADERS if damage != "header" else [*HEADERS[:-1], "Changed target"])
    add_row(2, [region, district or f"{region}1901", "Fixture district", *facts])
    with ZipFile(path, "w") as zip_file:
        for name, root in {
            "xl/workbook.xml": workbook, "xl/_rels/workbook.xml.rels": relationships,
            "xl/sharedStrings.xml": shared, "xl/worksheets/sheet1.xml": sheet,
        }.items():
            zip_file.writestr(name, ET.tostring(root))


def _candidate_inputs(tmp_path, *, damage=None):
    store = ProvenanceStore(tmp_path / "store")
    period = DataPeriod("school_year", 2023, 2024)
    base = DatasetMetadata(title="Synthetic SPP workbook", period=period, version="2025", source_url=spp.SELECTOR)
    identifiers = {}
    hashes = []
    for region in spp.REGIONS:
        path = tmp_path / f"{region}.xlsx"
        facts = {
            "01": (1672, 1529, "91.5", "NO"),
            "02": (2434, 2433, "100.0", "NO"),
            "03": (-999, 0, "0.0", "NO"),
            "04": (None, None, "NR", "NR"),
            "05": (3705, 3704, Decimal("99.97"), "NO"),
        }.get(region, (8, 8, "100.0", "YES"))
        _workbook(path, region, facts, district="011901" if damage == "duplicate" and region == "02" else None)
        url = spp.workbook_url(2024, region)
        resolved = ResolvedDataset("reference:tea_spp_district_detail", "reference", 2024, 2024, "", base)
        identifiers[region] = store.record_acquisition(path, resolved, downloaded_at=datetime.now(timezone.utc),
            requested_url=url, resolved_url=url, response_headers={"content-type": spp.MIME},
            parameters={"esc_region": region, "report_edition": 2025, "http_method": "GET", "http_status": 200})
        hashes.append(sha256(path.read_bytes()).hexdigest())
    if damage == "wrong_region_receipt":
        identifiers["01"], identifiers["02"] = identifiers["02"], identifiers["01"]
    mappings = [{"position": i, "code": label, "label": label, "field": name,
                 "disposition": "identity" if i < 4 else "fact",
                 "dictionary_status": "not_applicable" if i < 4 else "not_documented",
                 "publisher_type": None, "publisher_description": None}
                for i, (label, name) in enumerate(zip(HEADERS, FIELDS, strict=True), 1)]
    schema = tmp_path / "crosswalk.json"
    schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [{
        "dataset": spp.DATASET, "edition": "2025", "source_only": True, "tables": [{
            "table": "district", "header_identity": "position", "input_sha256s": hashes,
            "source_schema_sha256": source_schema_fingerprint(HEADERS, HEADERS),
            "metric_period": period.to_dict(), "fields": mappings,
        }],
    }]}))
    masking = tmp_path / "masking.txt"
    masking.write_text("Fixture public report masking methodology")
    references = []
    for identifier, role, path in (("masking", "masking_methodology", masking), ("schema", "schema", schema)):
        url = spp.SELECTOR if identifier == "masking" else None
        receipt = store.record_acquisition(path, ResolvedDataset(f"reference:{identifier}", "reference", None, None, "",
            DatasetMetadata(title=identifier, source_url=url)))
        references.append({"reference_id": identifier, "role": role, "status": "available", "path": path.name, "url": url,
                           "reviewed_sha256": sha256(path.read_bytes()).hexdigest(), "provenance_ids": [receipt]})
    config = tmp_path / "config.json"
    config.write_text(canonical_json({"data_sources": {spp.DATASET: {2024: {
        "path": "unused.xlsx", "metadata": {**base.to_dict(), "references": references, "value_policy": "fixture",
            "metrics": {name: {"field": name, "unit": unit} for name, unit in spp._METRICS.items()}},
        "reader": {"profiles": {"source": {"column_schema": schema.name, "acquisition_ids": identifiers}}},
    }}}, "value_policies": {"fixture": POLICY}}))
    return store, config, sha256(config.read_bytes()).hexdigest()


def test_spp_native_leaf_retains_exact_values_raw_tokens_and_source_cells(tmp_path, capsys):
    store, config, pin = _candidate_inputs(tmp_path)
    output = tmp_path / "candidate"
    assert main(["--store", str(store.root), "--config", str(config), "--config-sha256", pin,
                 "--year", "2024", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["release_activation"] and not result["field_correctness_qualified"]
    release = DataRelease(output / "release.json", expected_sha256=result["manifest"]["sha256"], offline=True)
    release.verify()
    edition, = result["editions"]
    assert edition["rows"] == 20
    key = EditionKey.from_dict(edition["key"])
    assert key.edition == "2025" and key.scope == "district"
    variables = {row["field"]: row for row in release.catalog.variables(edition=key)}
    assert variables[spp._PERCENT]["period"] == DataPeriod("school_year", 2023, 2024).to_dict()
    assert len(release.catalog.acquisitions(spp.DATASET, edition=key)) == 20
    table = release.analytical_dataset(edition["artifact_id"]).to_table().replace_schema_metadata(None)
    actual = table.to_pylist()
    exceptional, rounded_100, masked, missing, numeric_percent = actual[:5]
    assert exceptional[spp._PERCENT] == Decimal("91.50")
    assert exceptional[f"spp_raw_{spp._PERCENT}"] == "91.5"
    assert exceptional["spp_display_consistency"] == "consistent_with_two_stage_display_rounding"
    assert rounded_100[spp._PERCENT] == Decimal("100.00") and rounded_100[spp._TARGET] is False
    assert masked[spp._CONSENT] is None and masked[f"spp_raw_{spp._CONSENT}"] == "-999" and masked[spp._TIMELY] == 0
    assert masked[f"spp_status_{spp._CONSENT}"] == "suppressed"
    assert missing[f"spp_cell_type_{spp._CONSENT}"] == "absent" and missing[spp._TARGET] is None
    assert missing[f"spp_raw_{spp._TARGET}"] == "NR" and missing[f"spp_status_{spp._TARGET}"] == "missing"
    assert numeric_percent[spp._PERCENT] == Decimal("99.97") and numeric_percent[f"spp_cell_type_{spp._PERCENT}"] == "numeric"
    assert (numeric_percent["spp_source_sheet"], numeric_percent["spp_source_record"], numeric_percent["spp_source_input"]) == ("Ind11", 2, 4)
    contract = release.catalog.artifact_ref(edition["artifact_id"]).contract
    for name in (spp._PERCENT, spp._TARGET):
        status_name = f"spp_status_{name}"
        invalid = table.set_column(table.schema.get_field_index(status_name), status_name,
                                   pa.array(["missing", *table[status_name].to_pylist()[1:]]))
        with pytest.raises(ContractError, match="disagrees with normalized value presence"):
            write_leaf_artifact(tmp_path / "invalid.parquet", invalid.to_batches(), contract=contract)


@pytest.mark.parametrize(("damage", "message"), [
    ("duplicate", "duplicate district IDs"),
    ("wrong_region_receipt", "acquisition differs"),
])
def test_spp_refuses_duplicate_district_or_wrong_region_receipt(tmp_path, damage, message):
    store, config, pin = _candidate_inputs(tmp_path, damage=damage)
    output = tmp_path / "candidate"
    with pytest.raises(ContractError, match=message):
        spp.materialize_spp_ind11(store, config=config, config_sha256=pin, years=[2024], output=output)
    assert not output.exists()


@pytest.mark.parametrize("changes", [
    {"tokens": {**POLICY["tokens"], "8": {"status": "suppressed", "reason": "unreviewed"}}},
    {"tokens": {**POLICY["tokens"], "-999": {"status": "missing", "reason": "unreviewed"}}},
    {"tokens": {**POLICY["tokens"], "-999": {**POLICY["tokens"]["-999"], "upper": 5}}},
    {"reference_id": "schema"},
])
def test_spp_refuses_semantically_changed_policy_even_with_matching_config_pin(tmp_path, changes):
    store, config, _ = _candidate_inputs(tmp_path)
    raw = json.loads(config.read_text())
    raw["value_policies"]["fixture"].update(changes)
    config.write_text(canonical_json(raw))
    pin = sha256(config.read_bytes()).hexdigest()
    output = tmp_path / "candidate"
    with pytest.raises(ContractError, match="reviewed publisher masking policy and exact token rules"):
        spp.materialize_spp_ind11(store, config=config, config_sha256=pin, years=[2024], output=output)
    assert not output.exists()


@pytest.mark.parametrize(("facts", "damage", "message"), [
    ((8, 8, "100.0", "YES"), "header", "headings differ"),
    ((8, 8, "100.0", "YES"), "formula", "formula"),
    ((8, 7, "99.0", "NO"), None, "established display patterns"),
    ((8, 7, "87.5", "YES"), None, "target disagrees"),
    ((None, None, "0.0", "NO"), None, "lacks expected counts"),
    ((8, -1, "0.0", "NO"), None, "unsupported count token"),
])
def test_spp_refuses_unreviewed_workbook_tokens_and_arithmetic(tmp_path, facts, damage, message):
    path = tmp_path / "test.xlsx"
    _workbook(path, "01", facts, damage=damage)
    policy = ValuePolicy.from_dict({"policy_id": "fixture", **POLICY})
    with pytest.raises(ContractError, match=message):
        spp._sheet(path, HEADERS, "01", 0, policy)


def test_spp_decodes_published_zero_no_report_and_decimal_tokens():
    policy = ValuePolicy.from_dict({"policy_id": "fixture", **POLICY})
    assert spp._counts(spp._Cell("0", "numeric"), policy) == (0, ValueStatus.OBSERVED)
    assert spp._target(spp._Cell("NR", "shared_string"), policy) == (None, ValueStatus.MISSING)
    assert spp._percent(spp._Cell("99.97", "numeric"), policy) == (Decimal("99.97"), ValueStatus.OBSERVED)


@pytest.mark.parametrize(("year", "report"), [(2024, "2025"), (2025, "2026")])
def test_canonical_spp_discovery_binds_all_regional_receipts_offline(year, report):
    cfg = load_config()
    assert cfg.is_source_only(spp.DATASET, year)
    source = cfg.resolve_dataset(spp.DATASET, year, exact=True, reader_profile="source")
    assert source.metadata.version == report and source.metadata.period == DataPeriod("school_year", year - 1, year)
    acquisitions = source.reader["profiles"]["source"]["acquisition_ids"]
    assert set(acquisitions) == set(spp.REGIONS)
    refs = [identifier for ref in source.metadata.references for identifier in ref.provenance_ids]
    store = ProvenanceStore(ROOT / "provenance")
    records = validate_record_graph(store.read_record, [*acquisitions.values(), *refs])
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    for region, identifier in acquisitions.items():
        assert records[identifier]["requested_url"] == spp.workbook_url(year, region)
