from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow.dataset as ds
import pytest

from teadata.column_schema import parse_column_schema
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.manifest import canonical_json
from teadata.peims_actuals import BUDGET_DATASET, DATASET, materialize_peims_actuals, materialize_peims_budgets
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease
from teadata.teadata_config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "teadata/schemas"
ACTUALS_SCHEMA = SCHEMAS / "peims--2023-2025--single-file-financial-actuals-crosswalk.json"
BUDGET_SCHEMA = SCHEMAS / "peims--2026--single-file-financial-budgets-crosswalk.json"
YEAR = 2025
NOW = datetime(2026, 11, 4, tzinfo=UTC)
LISTING = "https://example.test/financial"
ROW = "001902,199,2026,11,6100,101,23,-125,18FEB2026\n"
ZERO = "001902,199,2026,11,6100,101,33,0,18FEB2026\n"


def _case(tmp_path, damage=None, *, budget=False, year_override=None, publisher_order=(),
          no_cost_center=None):
    store = ProvenanceStore(tmp_path / "provenance")
    schema = BUDGET_SCHEMA if budget else ACTUALS_SCHEMA
    basis = "budget" if budget else "actual"
    edition = parse_column_schema(schema.read_bytes())["datasets"][-1]
    dataset, year = edition["dataset"], int(edition["edition"])
    if year_override is not None:
        year = year_override
        edition["edition"] = str(year)
        for table in edition["tables"]:
            table["metric_period"] = DataPeriod("fiscal_year", year - 1, year).to_dict()
    metadata = DatasetMetadata(version=str(year), period=DataPeriod("fiscal_year", year - 1, year),
                               source_url=LISTING, publisher="Texas Education Agency")
    original = ResolvedDataset(dataset, "data_sources", year, year, "", metadata)
    archives = {}
    for table in edition["tables"]:
        kind = table["table"]
        url = f"{LISTING}/{kind}.zip"
        member = f"{kind}.csv" if year_override is None else f"{kind.title()}.csv"
        path = tmp_path / f"{kind}.zip"
        fields = ",".join(field["code"] for field in table["fields"])
        keeps_order = kind in publisher_order
        body = ZERO + ROW if keeps_order else ROW + ZERO
        if keeps_order and kind == "district" and damage == "duplicate":
            body += ZERO.replace(",0,18FEB2026", ",123,19FEB2026")
        if kind == no_cost_center and damage != "missing-unit":
            unit_row = ROW.replace(",101,", ",NUL,")
            if damage == "foreign-unit":
                unit_row = unit_row.replace("001902", "001903")
            body = body.replace(ROW, unit_row)
        if kind == "district":
            if damage == "duplicate" and not keeps_order:
                body = ROW + ROW
            elif damage == "header":
                fields = fields.replace("PROGRAM_INTENT", "PROGRAM_CODE")
            elif damage == "actual-amount":
                fields = fields.replace("BUDGAMT", "ACTAMT")
            elif damage == "swapped-aliases":
                function, program = table["fields"][3], table["fields"][6]
                function["field"], program["field"] = program["field"], function["field"]
            elif damage == "token":
                body = body.replace("-125", ".")
        with ZipFile(path, "w") as package:
            package.writestr(member, fields + "\n" + body)
            if kind == "district" and damage == "member":
                package.writestr("unexpected.csv", body)
        parameters = {"http_method": "GET", "http_status": 200}
        if year_override == 2014:
            archive_source = ResolvedDataset(f"reference:peims_financial_archive:{path.name}", "reference",
                year, year, "", DatasetMetadata(version=str(year), period=metadata.period,
                                               source_url=url, publisher=metadata.publisher))
        else:
            archive_source = original
            parameters["population"] = kind
        receipt = store.record_acquisition(path, archive_source, downloaded_at=NOW, recorded_at=NOW, requested_url=url,
            resolved_url=url, parameters=parameters)
        table["input_sha256s"] = [sha256(path.read_bytes()).hexdigest()]
        archives[kind] = {"receipt_id": receipt, "member": member,
                          "rows": 3 if damage == "rows" else body.count("\n"),
                          "requested_url": url, "resolved_url": url}
        if keeps_order:
            archives[kind]["key_order"] = "publisher"
        if kind == no_cost_center:
            archives[kind]["no_cost_center"] = {"reference_id": "publisher_units",
                                               "district_counts": {"001902": 1}}
    if damage == "custody":
        archives["charter"]["receipt_id"] = archives["district"]["receipt_id"]
    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [edition]}))
    listing = tmp_path / "listing.html"
    listing.write_text("Fixture publisher listings for district and charter actuals")
    references = []
    documents = [("listing", "source_notes", listing), ("crosswalk", "schema", crosswalk)]
    if no_cost_center:
        lookup = tmp_path / "publisher_units.txt"
        lookup.write_text("001902,NUL,00,NO COST CENTER")
        documents.append(("publisher_units", "data_dictionary", lookup))
    for identifier, role, path in documents:
        receipt = store.record_acquisition(path, ResolvedDataset(f"reference:{identifier}", "reference", None, None,
            "", DatasetMetadata(title=identifier)))
        references.append({"reference_id": identifier, "role": role, "title": identifier, "status": "available",
                           "path": path.name, "reviewed_sha256": sha256(path.read_bytes()).hexdigest(),
                           "provenance_ids": [receipt]})
    config = tmp_path / "sources.json"
    amount = f"{basis}_amount"
    config.write_text(canonical_json({"data_sources": {dataset: {str(year): {
        "path": "unused.zip", "metadata": {**metadata.to_dict(), "references": references,
            "metrics": {amount: {"field": amount, "unit": "currency"}},
            "source_qualifiers": {"family": "peims_financial", "reference_id": "listing",
                                  "financial_basis": "actual" if damage == "actual-basis" else basis}},
        "reader": {"profiles": {"source": {"column_schema": crosswalk.name, "archives": archives}}},
    }}}}))
    return store, config


@pytest.mark.parametrize("budget", [False, True], ids=["actuals", "budgets"])
def test_native_financial_accounts_keep_distinct_scopes_exact_codes_and_signed_amounts(tmp_path, budget):
    store, config = _case(tmp_path, budget=budget)
    basis = "budget" if budget else "actual"
    amount = f"{basis}_amount"
    raw, status = f"raw_{amount}", f"{amount}_status"
    output = tmp_path / "candidate"
    materialize = materialize_peims_budgets if budget else materialize_peims_actuals
    result = materialize(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                         years=[2026 if budget else YEAR], output=output)
    assert not result["cross_report_reconciliation_qualified"] and not result["release_activation"]
    path = output / "release.json"
    release = DataRelease(path, expected_sha256=sha256(path.read_bytes()).hexdigest(), offline=True)
    assert len(release.verify()) == 2
    editions = list(release.catalog.editions())
    assert {row["table"] for row in editions} == {"district", "charter"}
    assert {row["dataset"] for row in editions} == {BUDGET_DATASET if budget else DATASET}
    for entry in result["editions"]:
        leaf = release.analytical_dataset(entry["artifact_id"])
        fields = leaf.schema.names
        assert ("fund_year" in fields) == (entry["key"]["table"] == "district")
        assert ("fiscal_year_code" in fields) == (entry["key"]["table"] == "charter")
        assert leaf.to_table(filter=ds.field("district_number") == "001902",
            columns=[amount, raw, status, "source_record"]).to_pylist() == [
            {amount: -125, raw: "-125", status: "observed", "source_record": 2},
            {amount: 0, raw: "0", status: "observed", "source_record": 3},
        ]


@pytest.mark.parametrize(("damage", "message"), [
    ("actual-amount", "CSV header changed"), ("actual-basis", "require explicit budget basis"),
])
def test_budget_candidate_refuses_actuals_amount_or_basis(tmp_path, damage, message):
    store, config = _case(tmp_path, damage, budget=True)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match=message):
        materialize_peims_budgets(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                                  years=[2026], output=output)
    assert not output.exists()


@pytest.mark.parametrize("budget", [False, True], ids=["actuals", "budgets"])
def test_financial_accounts_refuse_swapped_same_width_field_aliases(tmp_path, budget):
    store, config = _case(tmp_path, "swapped-aliases", budget=budget)
    output = tmp_path / "candidate"
    materialize = materialize_peims_budgets if budget else materialize_peims_actuals
    with pytest.raises(ValueError, match="columns differ from the packaged source crosswalk"):
        materialize(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                     years=[2026 if budget else YEAR], output=output)
    assert not output.exists()


@pytest.mark.parametrize(("damage", "message"), [
    ("duplicate", "repeated or out of order"), ("header", "header changed"), ("token", "unknown amount token"),
    ("member", "single reviewed CSV member"), ("rows", "reviewed record count"), ("custody", "differs from its reviewed contract"),
])
def test_financial_archive_drift_and_cross_batch_duplicate_never_publish(tmp_path, monkeypatch, damage, message):
    store, config = _case(tmp_path, damage=damage)
    monkeypatch.setattr("teadata.peims_actuals._BLOCK_BYTES", len(ROW.encode()))
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match=message):
        materialize_peims_actuals(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                                years=[YEAR], output=output)
    assert not output.exists()


@pytest.mark.parametrize("damage", [None, "duplicate", "omit", "null", "custody"], ids=[
    "preserve-order", "duplicate-across-batches", "omit-order", "null-order", "substitute-receipt",
])
def test_historical_actuals_bind_publisher_order_and_check_full_keys(tmp_path, monkeypatch, damage):
    year = 2019
    store, config = _case(tmp_path, damage if damage == "duplicate" else None,
                          year_override=year, publisher_order=("district",))
    canonical = load_config(config)
    monkeypatch.setattr("teadata.peims_actuals.load_config", lambda: canonical)
    monkeypatch.setattr("teadata.peims_actuals._BLOCK_BYTES", len(ROW.encode()) + 1)
    candidate = tmp_path / "candidate[exact]"
    if damage in {"omit", "null", "custody"}:
        data = json.loads(config.read_bytes())
        sources = data["data_sources"][DATASET][str(year)]["reader"]["profiles"]["source"]["archives"]
        if damage == "omit":
            del sources["district"]["key_order"]
        elif damage == "null":
            sources["district"]["key_order"] = None
        else:
            sources["district"]["receipt_id"] = sources["charter"]["receipt_id"]
        config.write_text(canonical_json(data))
    parameters = dict(config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                      years=[year], output=candidate)
    if damage is not None:
        match = {"duplicate": "repeated account keys", "omit": "differs from the packaged contract",
                 "null": "requires a reviewed historical", "custody": "differs from the packaged contract"}[damage]
        with pytest.raises(ValueError, match=match):
            materialize_peims_actuals(store, **parameters)
        assert not candidate.exists()
        assert not list(tmp_path.glob(".peims-actuals-*"))
        return

    result = materialize_peims_actuals(store, **parameters)
    path = candidate / "release.json"
    release = DataRelease(path, expected_sha256=sha256(path.read_bytes()).hexdigest(), offline=True)
    assert len(release.verify()) == 2
    for edition in result["editions"]:
        ref = release.catalog.artifact_ref(edition["artifact_id"])
        leaf = release.analytical_dataset(edition["artifact_id"])
        data = leaf.to_table(columns=["actual_amount", "raw_actual_amount", "source_record"])
        processing = ProvenanceStore(candidate / "provenance").read_record(ref.processing_id)["parameters"]
        if edition["key"]["table"] == "district":
            expected = (0, -125)
            assert processing["recipe"]["account_key_order"] == "publisher"
            assert processing["toolchain"]["duckdb"]
        else:
            expected = (-125, 0)
            assert "account_key_order" not in processing["recipe"]
            assert "duckdb" not in processing["toolchain"]
        assert data.to_pylist() == [
            {"actual_amount": amount, "raw_actual_amount": str(amount), "source_record": index}
            for index, amount in enumerate(expected, 2)
        ]


@pytest.mark.parametrize("budget", [False, True], ids=["actuals", "budgets"])
def test_modern_financial_sources_cannot_opt_into_publisher_order(tmp_path, budget):
    store, config = _case(tmp_path, budget=budget, publisher_order=("district",))
    materialize = materialize_peims_budgets if budget else materialize_peims_actuals
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="requires a reviewed historical"):
        materialize(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                    years=[2026 if budget else YEAR], output=output)
    assert not output.exists()


@pytest.mark.parametrize(("damage", "year", "sector", "message"), [
    (None, 2014, "district", None),
    ("foreign-unit", 2014, "district", "malformed account code"),
    ("missing-unit", 2014, "district", "counts differ"),
    ("missing-lookup", 2014, "district", "metadata differs from the packaged contract"),
    ("foreign-listing", 2014, "district", "metadata differs from the packaged contract"),
    (None, 2014, "charter", "historical district FIN_UNIT contract"),
    (None, 2025, "district", "historical district FIN_UNIT contract"),
])
def test_no_cost_center_requires_exact_district_scope_and_lookup(tmp_path, monkeypatch, damage, year, sector, message):
    historical = year < 2023
    store, config = _case(tmp_path, damage, year_override=year if historical else None,
                          publisher_order=(sector,) if historical else (), no_cost_center=sector)
    canonical = load_config(config)
    monkeypatch.setattr("teadata.peims_actuals.load_config", lambda: canonical)
    monkeypatch.setattr("teadata.peims_actuals._BLOCK_BYTES", len(ROW.encode()) + 1)
    if damage in {"missing-lookup", "foreign-listing"}:
        document = json.loads(config.read_bytes())
        metadata = document["data_sources"][DATASET][str(year)]["metadata"]
        if damage == "missing-lookup":
            metadata["references"].pop()
        else:
            metadata["source_url"] = "https://example.test/other-listing"
        config.write_text(canonical_json(document))
    candidate = tmp_path / "candidate"
    options = dict(config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                   years=[year], output=candidate)
    if message:
        with pytest.raises(ValueError, match=message):
            materialize_peims_actuals(store, **options)
        assert not candidate.exists()
        return
    result = materialize_peims_actuals(store, **options)
    manifest = candidate / "release.json"
    release = DataRelease(manifest, expected_sha256=sha256(manifest.read_bytes()).hexdigest(), offline=True)
    district, = [item for item in result["editions"] if item["key"]["table"] == "district"]
    table = release.analytical_dataset(district["artifact_id"]).to_table(columns=["financial_unit_code"])
    assert table["financial_unit_code"].to_pylist() == ["101", "NUL"]
    processing = ProvenanceStore(candidate / "provenance").read_record(district["processing_record_id"])
    assert processing["parameters"]["recipe"]["no_cost_center"]["district_counts"] == {"001902": 1}
    edition = release.catalog.artifact_ref(district["artifact_id"]).edition
    acquisition, = release.catalog.acquisitions(DATASET, edition=edition)
    assert acquisition["record_id"] == district["acquisition_id"]
    assert acquisition["dataset"] == "reference:peims_financial_archive:district.zip"
    lookup, = [ref for ref in release.catalog.metadata(DATASET, edition=edition).references
               if ref.reference_id == "publisher_units"]
    assert set(lookup.provenance_ids) <= set(processing["inputs"])


@pytest.mark.parametrize(("dataset", "years", "basis"), [
    (DATASET, (2023, 2024, 2025), "actual"), (BUDGET_DATASET, (2026,), "budget"),
])
def test_annual_financial_config_has_reviewed_separate_publisher_bindings(dataset, years, basis):
    cfg = load_config()
    assert set(cfg.data_sources[dataset].entries) == set(years) | (set(range(2013, 2023)) if dataset == DATASET else set())
    store = ProvenanceStore(ROOT / "provenance")
    # Admission reads archived bytes; this static check uses only checked-in receipt headers.
    for year in years:
        source = cfg.resolve_dataset(dataset, year, reader_profile="source", exact=True)
        assert cfg.is_source_only(dataset, year)
        assert source.metadata.period == DataPeriod("fiscal_year", year - 1, year)
        assert source.metadata.source_qualifiers is not None
        assert source.metadata.source_qualifiers.financial_basis == basis
        archives = source.reader["profiles"]["source"]["archives"]
        assert set(archives) == {"district", "charter"}
        for population, archive in archives.items():
            record = store.read_record(archive["receipt_id"])
            assert record["dataset"] == dataset
            assert record["parameters"]["population"] == population
            assert record["resolved_url"] == archive["resolved_url"]
