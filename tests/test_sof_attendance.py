"""SOF attendance uses the original numeric decimal tokens and explicit units."""

from decimal import Decimal
from datetime import datetime, timezone
from hashlib import sha256
import json

import pyarrow as pa
import pytest

from teadata.dataset_metadata import MetricDefinition
from teadata.manifest import ContractError
from teadata.sof_attendance import exact_decimal


@pytest.mark.parametrize("raw", ["543.68900000000008", "-0.000000000000001", "0", "1.2300", "1e-15"])
def test_original_sof_decimal_roundtrips_without_float(raw):
    value = exact_decimal(raw, precision=21, scale=15)
    assert value == Decimal(raw)
    projected = pa.array([value], type=pa.decimal128(21, 15))
    assert projected[0].as_py() == value
    assert -value.as_tuple().exponent == -Decimal(raw).as_tuple().exponent


@pytest.mark.parametrize("raw", ["0.0000000000000001", "1000000", "NaN", "inf", "-", "", "1,000"])
def test_sof_unreviewed_numeric_token_or_overflow_fails(raw):
    with pytest.raises(ContractError):
        exact_decimal(raw, precision=21, scale=15)


def test_sof_missing_is_distinct_from_observed_zero_and_headcounts():
    assert exact_decimal(None, precision=23, scale=17) is None
    assert exact_decimal("0", precision=23, scale=17) == Decimal(0)
    metadata = MetricDefinition.from_dict({"field": "inputs_lpe_ada_tot_refined", "unit": "attendance_equivalent"})
    assert metadata.unit == "attendance_equivalent"


def test_sof_generated_source_contract_has_closed_dictionary_and_reference_bindings(tmp_path, monkeypatch):
    from openpyxl import Workbook

    from teadata import sof_attendance as source_module
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
    from teadata.manifest import BuildRecorder
    from teadata.provenance import ProvenanceStore
    from teadata.teadata_config import load_config_bytes

    store = ProvenanceStore(tmp_path / "provenance")
    grid = source_module.definition()
    raw = tmp_path / "source.xlsx"
    raw.write_bytes(b"synthetic workbook source; source semantic qualification is separate")
    grid["workbook_sha256"] = sha256(raw.read_bytes()).hexdigest()
    descriptor = ResolvedDataset("fixture", "reference", 2026, 2026, "", DatasetMetadata(version="47422"))
    receipt = store.record_acquisition(raw, descriptor, downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                       requested_url="https://example.org/source.xlsx")
    grid["workbook_receipt"] = receipt
    index = tmp_path / "index.html"
    index.write_text("synthetic source publication index")
    grid["index_receipt"] = store.record_acquisition(index, descriptor,
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), requested_url="https://example.org/index.html")
    dictionary = tmp_path / "dictionary.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Variable Listing"
    for code, description in grid["definitions"].items():
        sheet.append((code, "inputs", "Num", description))
    workbook.save(dictionary)
    workbook.close()
    grid["dictionary_sha256"] = sha256(dictionary.read_bytes()).hexdigest()
    origin = tmp_path / "origin.json"
    origin.write_text(json.dumps({"url": "https://example.org/dictionary.xlsx",
        "sha256": grid["dictionary_sha256"], "size_bytes": dictionary.stat().st_size}) + "\n")
    grid["dictionary_origin_sha256"] = sha256(origin.read_bytes()).hexdigest()
    monkeypatch.setattr(source_module, "definition", lambda: grid)
    stage = tmp_path / "stage"
    stage.mkdir()
    raw_config, acquisition = source_module._configuration(store, stage, dictionary, origin)
    path = stage / "source-config.input"
    path.write_bytes(raw_config)
    config = load_config_bytes(raw_config, path=path)
    selected = config.resolve_dataset(source_module.DATASET, 2026, exact=True, reader_profile="source")
    contract = config.column_contract(source_module.DATASET, 2026, reader_profile="source")
    assert contract is not None and BuildRecorder(config, store).processor_sha256
    closure = store.verify_record_closure([acquisition])
    references = {item.reference_id: item for item in selected.metadata.references}
    dictionary_record = closure[references["dictionary"].provenance_ids[0]]
    assert dictionary_record["method"] == "local_observation" and dictionary_record["downloaded_at"] is None
    assert dictionary_record["requested_url"] is None and dictionary_record["resolved_url"] is None
    assert dictionary_record["source_metadata"]["source_url"] == "https://example.org/dictionary.xlsx"
    assert dictionary_record["parameters"]["origin_observation_receipt"] in closure
    assert references["dictionary_origin"].reviewed_sha256 == grid["dictionary_origin_sha256"]
