"""Synthetic publisher-shaped leaves; no TEA field-correctness qualification."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
import pytest

from test_transfer_preparation import DISTRICT_HEADER, POLICY, original
from test_transfer_sources import Response, install_transport, retained_form
from teadata.column_schema import crosswalk_dataset, parse_column_schema, source_schema_fingerprint, table_input_sha256s
from teadata.dataset_metadata import DatasetMetadata, EditionKey, ResolvedDataset
from teadata.manifest import canonical_json
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.release import DataRelease
from teadata.scripts.refresh_transfers import main
from teadata.teadata_config import load_config
from teadata.transfer_leaves import materialize_district_transfers
from teadata.transfer_rollups import _compare_details
from teadata.transfer_sources import acquire_regional_transfers, prepare_transfer_sources
from teadata.values import measure


FIELDS = ["school_year", "reporting_region", "reporting_region_name", "reporting_district_number",
    "reporting_charter_status", "report_number", "transfer_direction", "line_group_number",
    "partner_district_number", "student_transfer_count"]


def test_campus_district_rollup_checks_only_fully_observed_cross_district_groups():
    def detail(rows, value):
        fields = {
            "transfer_direction": [row[0] for row in rows],
            "transfer_reporting_district_number": [row[1] for row in rows],
            "transfer_partner_district_number": [row[2] for row in rows],
            "transfer_count_status": ["observed" if row[3] is not None else "suppressed" for row in rows],
            value: pa.array([row[3] for row in rows], type=pa.int64()),
        }
        if value == "transfer_count_value":
            fields["transfer_reporting_number"] = [f"{row[1]}{index:03}" for index, row in enumerate(rows)]
            fields["transfer_partner_number"] = [f"{row[2]}001" for row in rows]
        return pa.table(fields)

    campus = detail([("in", "001", "002", 11), ("in", "001", "002", 15), ("out", "002", "001", 26),
                     ("in", "003", "003", 100), ("in", "003", "004", 12), ("in", "003", "004", None),
                     ("in", "005", "006", 32)], "transfer_count_value")
    district = detail([("in", "001", "002", 26), ("out", "002", "001", 26), ("in", "003", "004", 22),
                       ("in", "005", "006", None), ("out", "006", "005", None)], "student_transfer_count")
    result = _compare_details(campus, district)
    assert result["status"] == "PASS_OBSERVED_TRANSFER_ROLLUP"
    assert result["coverage"] == {"exact": 2, "different": 0, "campus_suppressed": 1, "district_suppressed": 1,
                                  "campus_only": 0, "district_only_observed": 0, "district_only_suppressed": 1}
    assert result["campus_within_district_detail_rows_excluded"] == 1
    assert result["visible_mismatch_examples"] == []
    changed = district.set_column(district.num_columns - 1, "student_transfer_count", pa.array([27, 26, 22, None, None]))
    mismatch = _compare_details(campus, changed)
    assert mismatch["status"] == "HOLD"
    assert mismatch["coverage"]["different"] == 1
    assert [row["student_transfer_count"] for row in mismatch["visible_mismatch_examples"]] == [27]
    assert _compare_details(campus.slice(0, 1), district.slice(0, 2))["coverage"]["district_only_observed"] == 1
    assert _compare_details(campus, district.slice(0, 1))["coverage"]["campus_only"] == 3
    with pytest.raises(ValueError, match="Duplicate directional campus"):
        _compare_details(pa.concat_tables([campus.slice(0, 1), campus]), district)
    very_large = detail([("in", "001", "002", 2**63 - 1), ("in", "001", "002", 2**63 - 1),
                         ("in", "001", "002", 3)], "transfer_count_value")
    overflow = _compare_details(very_large, detail([("in", "001", "002", 1)], "student_transfer_count"))
    assert overflow["status"] == "HOLD"
    assert overflow["visible_mismatch_examples"][0]["transfer_count_value_sum"] == 2**64 + 1
    assert _compare_details(campus.slice(0, 0), district.slice(0, 0))["status"] == "HOLD"


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_packaged_district_transfer_discovery_binds_all_regional_originals(year):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "teadata/teadata_sources.yaml")
    dataset = "district_transfer_reports"
    source = config.resolve_dataset(dataset, year, reader_profile="source", exact=True)
    assert config.is_source_only(dataset, year)
    assert config.column_schema(dataset, year) is None
    assert config.column_schema(dataset, year, reader_profile="source") == FIELDS
    read_record = ProvenanceStore(root / "provenance").read_record
    roots = [*source.metadata.provenance_ids,
             *(identifier for ref in source.metadata.references for identifier in ref.provenance_ids)]
    graph = validate_record_graph(read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    originals = [graph[identifier] for identifier in source.metadata.provenance_ids]
    assert [dict(record["parameters"]["form_controls"])["selected_id"] for record in originals] == [
        f"{region:02}" for region in range(1, 21)
    ]
    assert all(record["dataset"] == dataset and record["method"] == "download"
               and record["requested_year"] == record["resolved_year"] == year
               for record in originals)
    contract = config.column_contract(dataset, year, reader_profile="source")
    assert contract is not None
    table, = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, str(year))["tables"]
    table_input_sha256s(table, acquisition_sha256s=[record["artifact"]["sha256"] for record in originals])


def prepared_fixture(tmp_path, monkeypatch, years=(2024, 2025), *, complete=True, damage=None):
    store, form = retained_form(tmp_path)
    policy = deepcopy({key: value for key, value in POLICY.items() if key != "policy_id"})
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(canonical_json({"value_policies": {"campus_transfers": policy}}))
    policy_sha = sha256(policy_path.read_bytes()).hexdigest()
    entries, preparations, reports = {}, [], []
    for year in years:
        bodies = {}
        for region, district, partner in (("01", "000101", "000202"), ("13", "000202", "000101")):
            charter = region == "13"
            rows = []
            for report_number, direction, token in [("100", "Transfers In From", "5" if charter else "0")]:
                prefix = [f"{year - 1}-{year}", region, "Synthetic region", f"Fixture  -  {district}",
                          "Charter" if charter else "Traditional ISD/CSD", report_number, direction]
                rows.extend([prefix + ["100", f"Partner  -  {partner}", token],
                             prefix + ["200", "Total Transfers In", token], prefix + ["300", "", "."]])
            if not charter:
                prefix = [f"{year - 1}-{year}", region, "Synthetic region", f"Fixture  -  {district}",
                          "Traditional ISD/CSD", "200", "Transfers Out To"]
                rows.extend([prefix + ["100", f"Partner  -  {partner}", "-999"],
                             prefix + ["200", "Total Transfers Out", "-999"], prefix + ["300", "", "."]])
            raw, _ = original(tmp_path, rows, grain="district", report_scope="Region", filename=f"{year}-{region}.csv")
            bodies[region] = raw.read_bytes()

        class CsvResponse(Response):
            def __init__(self, request):
                super().__init__(request)
                self.seek(0)
                self.truncate()
                self.write(bodies[parse_qs(urlparse(request.full_url).query)["selected_id"][0]])
                self.seek(0)

        install_transport(monkeypatch, CsvResponse)
        batch = tmp_path / f"batch-{year}"
        acquired = acquire_regional_transfers(store, form, year=year, regions=list(form.regions) if complete else ["01"], output=batch)
        index = batch / "BATCH_INDEX.json"
        # A synthetic past identity exercises admission without current-code equality.
        with monkeypatch.context() as past:
            past.setattr("teadata.manifest.processor_identity", lambda: "a" * 64)
            prepared = prepare_transfer_sources(store, grain="district", year=year, output=tmp_path / f"prepared-{year}",
                batch_index=(index, sha256(index.read_bytes()).hexdigest()), encoding="utf-8",
                policy_config=policy_path, policy_config_sha256=policy_sha, value_policy="campus_transfers")
        assert prepared["status"] == "prepared"
        preparations.append((year, prepared["processing_record_id"]))
        reports.append(prepared)
        raw_hashes = [item["artifact"]["sha256"] for item in acquired["items"]]
        table = {"table": "report_rows", "input_sha256s": raw_hashes,
            "source_schema_sha256": source_schema_fingerprint(DISTRICT_HEADER, DISTRICT_HEADER),
            "metric_period": {"kind": "school_year", "start_year": year - 1, "end_year": year},
            "fields": [{"position": number, "code": code, "label": code, "field": field,
                        "disposition": "fact" if code == "TRANSFERS_IN_OUT" else "identity",
                        "publisher_type": None, "publisher_description": None,
                        "dictionary_status": "not_documented" if code == "TRANSFERS_IN_OUT" else "not_applicable"}
                       for number, (code, field) in enumerate(zip(DISTRICT_HEADER, FIELDS, strict=True), 1)]}
        if damage == "reordered_inputs":
            table["input_sha256s"].reverse()
        elif damage == "count_identity":
            table["fields"][-1].update(disposition="identity", dictionary_status="not_applicable")
        resource = tmp_path / f"district-{year}-headers.json"
        resource.write_text(canonical_json({"schema_version": 2, "mapping_version": 1,
            "datasets": [{"dataset": "district_transfer_reports", "edition": str(year), "source_only": True, "tables": [table]}]}) + "\n")
        schema_id = store.record_acquisition(resource, ResolvedDataset("reference:district_headers", "reference", None, None, "",
            DatasetMetadata(title="Synthetic reviewed literal district headers")))
        schema_sha = sha256(resource.read_bytes()).hexdigest()
        first = acquired["items"][0]
        first_record = store.read_record(first["record_id"])
        entries[year] = {"path": str(tmp_path / f"prepared-{year}/prepared-source.parquet"),
            "metadata": {"version": str(year), "period": {"kind": "school_year", "start_year": year - 1, "end_year": year},
                "provenance_ids": [item["record_id"] for item in acquired["items"]], "value_policy": "campus_transfers",
                "metrics": {"transfers": {"field": "student_transfer_count", "unit": "count"}},
                "references": [{"reference_id": "columns", "role": "schema", "path": resource.name,
                    "reviewed_sha256": schema_sha, "status": "available", "provenance_ids": [schema_id]},
                    {"reference_id": "masking", "role": "masking_methodology", "url": first_record["requested_url"],
                     "reviewed_sha256": first["artifact"]["sha256"], "status": "available", "provenance_ids": [first["record_id"]]}]},
            "reader": {"profiles": {"source": {"column_schema": resource.name}}}}
        if damage == "count_unknown_period":
            entries[year]["metadata"]["metrics"]["transfers"]["period"] = {"kind": "unknown"}
    if damage == "policy":
        policy["tokens"]["-999"]["lower"] = 1
        policy["tokens"]["-999"]["upper"] = 9
    config = tmp_path / "sources.json"
    config.write_text(canonical_json({"data_sources": {"district_transfer_reports": entries}, "value_policies": {"campus_transfers": policy}}))
    return store, config, sha256(config.read_bytes()).hexdigest(), preparations, reports


def test_one_statewide_district_leaf_per_year_preserves_history_raw_support_totals_and_keys(tmp_path, monkeypatch):
    store, config, config_sha, preparations, reports = prepared_fixture(tmp_path, monkeypatch)
    before = {identifier: store.read_record(identifier) for _, identifier in preparations}
    result = materialize_district_transfers(store, config=config, config_sha256=config_sha,
        preparations=preparations, output=tmp_path / "native")
    assert result["status"] == "unpublished_native_transfer_candidate"
    assert not result["release_activation"] and not result["field_correctness_qualified"]
    release = DataRelease(tmp_path / "native/release.json", offline=True)
    assert len(list(release.catalog.editions())) == len(release.catalog.to_dict()["artifacts"]) == 2
    for edition, (_, prepared_id), report in zip(result["editions"], preparations, reports, strict=True):
        key = EditionKey.from_dict(edition["key"])
        variables = {row["field"]: row for row in release.catalog.variables(edition=key)}
        assert set(variables) == set(FIELDS) and variables["student_transfer_count"]["code"] == "TRANSFERS_IN_OUT"
        plan, = release.catalog.resolve_artifacts([(key, "student_transfer_count")])
        assert "REPORT_DISTNAME_NUMBER" not in plan["columns"]
        selected = release.analytical_dataset(edition["artifact_id"]).to_table(columns=list(plan["columns"]))
        assert selected.num_rows == report["rows"]
        assert selected["TRANSFERS_IN_OUT"].to_pylist() == ["0", "0", ".", "-999", "-999", ".", "5", "5", "."]
        assert selected["student_transfer_count"].to_pylist() == [0, 0, None, None, None, None, 5, 5, None]
        assert selected["transfer_row_kind"].to_pylist().count("publisher_total") == 3
        assert len(set(zip(selected["transfer_source_input"].to_pylist(), selected["transfer_source_record"].to_pylist(), strict=True))) == report["rows"]
        support = variables["student_transfer_count"]["support"]
        decoded = [measure(raw, release.catalog.policy(key.dataset, edition=key), unit="count", minimum=0)
                   for raw in selected[support["raw"]].to_pylist()]
        assert [value.status.value for value in decoded] == selected[support["status"]].to_pylist()
        assert len(release.catalog.acquisitions(key.dataset, edition=key)) == 2  # every offered synthetic region
        full = release.analytical_dataset(edition["artifact_id"]).to_table()
        assert full.num_columns == 23 and "REPORT_DISTNAME_NUMBER" in full.column_names
        assert full["reporting_district_number"][0].as_py() == "'000101"
        assert full["REPORT_DISTNAME_NUMBER"][0].as_py() == "Fixture  -  000101"
        assert store.read_record(prepared_id) == before[prepared_id] and before[prepared_id]["code_version"] == "a" * 64
        graph = release.catalog.to_dict()["provenance"]
        assert graph[edition["processing_record_id"]]["inputs"][0] == prepared_id
        assert graph[edition["processing_record_id"]]["validation"]["field_correctness_qualified"] is False


@pytest.mark.parametrize("damage,complete,message", [("reordered_inputs", True, "bytes/order"),
    ("policy", True, "value policy"), (None, False, "every offered region"),
    ("count_unknown_period", True, "exact school-year period"), ("count_identity", True, "exact school-year period")])
def test_unaccepted_scope_policy_or_reviewed_inputs_fail_before_candidate_creation(tmp_path, monkeypatch, damage, complete, message):
    store, config, digest, preparations, _ = prepared_fixture(tmp_path, monkeypatch, years=(2025,), complete=complete, damage=damage)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=message):
        materialize_district_transfers(store, config=config, config_sha256=digest, preparations=preparations, output=output)
    assert not output.exists()


def test_materialize_cli_uses_the_existing_command_and_explicit_pins(tmp_path, monkeypatch, capsys):
    store, config, digest, preparations, _ = prepared_fixture(tmp_path, monkeypatch, years=(2025,))
    year, identifier = preparations[0]
    assert main(["materialize", "--store", str(store.root), "--config", str(config), "--config-sha256", digest,
        "--prepared-input", str(year), identifier, "--output", str(tmp_path / "cli-native")]) == 0
    assert json.loads(capsys.readouterr().out)["release_activation"] is False


def test_diagnostic_hold_processing_cannot_become_a_native_leaf_input(tmp_path, monkeypatch):
    store, config, digest, _, reports = prepared_fixture(tmp_path, monkeypatch, years=(2025,))
    original_input = reports[0]["inputs"][0]
    policy = tmp_path / "policy.json"
    # Explicit manual district scope must reject this physically regional CSV.
    failed = prepare_transfer_sources(store, grain="district", year=2025, output=tmp_path / "hold",
        inputs=[(original_input["record_id"], original_input["artifact"]["sha256"])], encoding="utf-8",
        policy_config=policy, policy_config_sha256=sha256(policy.read_bytes()).hexdigest(), value_policy="campus_transfers")
    assert failed["status"] == "HOLD"
    output = tmp_path / "not-a-leaf"
    with pytest.raises(ValueError, match="accepted complete source preparation"):
        materialize_district_transfers(store, config=config, config_sha256=digest,
            preparations=[(2025, failed["processing_record_id"])], output=output)
    assert not output.exists()


def recorded_toolchain_variant(tmp_path, store, identifier, variant):
    """Canonical adversarial fixture event; never rewrite the retained parent."""
    parent = store.read_record(identifier)
    parameters = deepcopy(parent["parameters"])
    report = json.loads(store.verify_artifact(parent["outputs"]["validation_report"]["sha256"]).read_bytes())
    for document in (parameters, report):
        if variant == "missing":
            document.pop("toolchain")
        elif variant == "null":
            document["toolchain"] = None
        elif variant == "empty":
            document["toolchain"] = {}
        elif variant.startswith("blank_"):
            document["toolchain"][variant.removeprefix("blank_")] = " "
        elif variant == "nonstring":
            document["toolchain"]["architecture"] = False
        else:
            assert variant == "unknown_extra"
            document["toolchain"].update(wheel_sha256=None, compiler=None)
    folder = tmp_path / "restated-toolchain"
    folder.mkdir()
    changed_report = folder / "VALIDATION.json"
    changed_report.write_text(canonical_json(report) + "\n")
    prepared_folder = tmp_path / "prepared-2025"
    changed = store.record_processing(dataset=parent["dataset"], inputs=parent["inputs"],
        outputs={"prepared_source": prepared_folder / "prepared-source.parquet",
                 "annotations": prepared_folder / "ANNOTATIONS.json", "validation_report": changed_report},
        processor=parent["processor"], code_version=parent["code_version"], parameters=parameters, validation=parent["validation"])
    return changed, parent


@pytest.mark.parametrize("variant", ["missing", "null", "empty", "blank_python", "blank_pyarrow", "blank_platform",
                                     "blank_architecture", "nonstring"])
def test_incomplete_matching_report_and_record_toolchains_are_not_admitted(tmp_path, monkeypatch, variant):
    store, config, digest, preparations, _ = prepared_fixture(tmp_path, monkeypatch, years=(2025,))
    _, original_id = preparations[0]
    changed, parent = recorded_toolchain_variant(tmp_path, store, original_id, variant)
    output = tmp_path / "not-admitted"
    with pytest.raises(ValueError, match="recorded toolchain"):
        materialize_district_transfers(store, config=config, config_sha256=digest, preparations=[(2025, changed)], output=output)
    assert not output.exists() and store.read_record(original_id) == parent


def test_unknown_extra_wheel_and_compiler_lineage_is_preserved_without_current_version_equality(tmp_path, monkeypatch):
    store, config, digest, preparations, _ = prepared_fixture(tmp_path, monkeypatch, years=(2025,))
    _, original_id = preparations[0]
    changed, parent = recorded_toolchain_variant(tmp_path, store, original_id, "unknown_extra")
    result = materialize_district_transfers(store, config=config, config_sha256=digest,
        preparations=[(2025, changed)], output=tmp_path / "unknown-lineage")
    release = DataRelease(tmp_path / "unknown-lineage/release.json", offline=True)
    retained = release.catalog.to_dict()["provenance"][changed]
    assert retained["parameters"]["toolchain"]["wheel_sha256"] is None
    assert retained["parameters"]["toolchain"]["compiler"] is None
    assert retained["code_version"] == "a" * 64 and result["field_correctness_qualified"] is False
    assert store.read_record(original_id) == parent
