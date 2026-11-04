from datetime import UTC, datetime
from hashlib import sha256
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


def _case(tmp_path, damage=None, *, budget=False):
    store = ProvenanceStore(tmp_path / "provenance")
    schema = BUDGET_SCHEMA if budget else ACTUALS_SCHEMA
    basis = "budget" if budget else "actual"
    edition = parse_column_schema(schema.read_bytes())["datasets"][-1]
    dataset, year = edition["dataset"], int(edition["edition"])
    metadata = DatasetMetadata(version=str(year), period=DataPeriod("fiscal_year", year - 1, year),
                               source_url=LISTING)
    original = ResolvedDataset(dataset, "data_sources", year, year, "", metadata)
    archives = {}
    for table in edition["tables"]:
        kind = table["table"]
        url = f"{LISTING}/{kind}.zip"
        member = f"{kind}.csv"
        path = tmp_path / f"{kind}.zip"
        fields = ",".join(field["code"] for field in table["fields"])
        body = ROW + ZERO
        if kind == "district":
            if damage == "duplicate":
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
        receipt = store.record_acquisition(path, original, downloaded_at=NOW, recorded_at=NOW, requested_url=url,
            resolved_url=url, parameters={"http_method": "GET", "http_status": 200, "population": kind})
        table["input_sha256s"] = [sha256(path.read_bytes()).hexdigest()]
        archives[kind] = {"receipt_id": receipt, "member": member, "rows": 3 if damage == "rows" else 2,
                          "requested_url": url, "resolved_url": url}
    if damage == "custody":
        archives["charter"]["receipt_id"] = archives["district"]["receipt_id"]
    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [edition]}))
    listing = tmp_path / "listing.html"
    listing.write_text("Fixture publisher listings for district and charter actuals")
    references = []
    for identifier, role, path in (("listing", "source_notes", listing), ("crosswalk", "schema", crosswalk)):
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


@pytest.mark.parametrize(("dataset", "years", "basis"), [
    (DATASET, (2023, 2024, 2025), "actual"), (BUDGET_DATASET, (2026,), "budget"),
])
def test_annual_financial_config_has_reviewed_separate_publisher_bindings(dataset, years, basis):
    cfg = load_config()
    assert set(cfg.data_sources[dataset].entries) == set(years)
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
