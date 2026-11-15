"""Fail-closed College Credit qualification, independent of source preparation."""

from copy import deepcopy
import csv
from decimal import Decimal
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from urllib.parse import urlencode

import pyarrow as pa
import pytest

from teadata import native_college_credit as gate, verification
from teadata.dataset_metadata import EditionKey
from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease


_PIN = "19fffe1f8d481d3c08bb01fe7cb4f73ae9ca8393d523a0d973411500cc33e3b8"


def plan():
    result = gate.reviewed_college_credit(_PIN)
    assert result is not None
    return result


def identity_release(review):
    catalog = {"schema_version": 4, "editions": {}, "artifacts": {}}
    refs, acquisitions = {}, {}
    for identifier, case in review["editions"].items():
        catalog["editions"][identifier] = {"key": case["key"], "metadata": case["metadata"]}
        catalog["artifacts"][case["artifact_id"]] = identifier
        refs[case["artifact_id"]] = SimpleNamespace(
            edition=EditionKey.from_dict(case["key"]), identity=case["artifact_id"],
            sha256=case["leaf_sha256"], size_bytes=case["leaf_size_bytes"], processing_id=case["processing_id"],
            contract={"rows": case["rows"]})
        acquisitions[identifier] = [{"record_id": case["source"]["original_acquisition_id"],
                                     "artifact": case["source"]["artifact"]}]
    return SimpleNamespace(manifest_sha256=_PIN, build_id=review["build_id"], path=Path("child/release.json"),
        catalog=SimpleNamespace(to_dict=lambda: catalog, artifact_ref=refs.__getitem__,
                                metadata=lambda dataset, edition: SimpleNamespace(period=DataPeriod("school_year", int(edition.edition) - 1, int(edition.edition))),
                                acquisitions=lambda dataset, edition: acquisitions[edition.identity])), catalog, refs


def test_review_is_complete_and_preserves_difficult_publisher_goldens():
    review = plan()
    assert gate.reviewed_college_credit("0" * 64) is None
    assert len(review["editions"]) == 15 and len(review["publisher_witnesses"]) == 5
    assert {(row["key"]["edition"], row["key"]["scope"]) for row in review["editions"].values()} == {
        (str(year), scope) for year in range(2022, 2027) for scope in ("state", "region", "district")}
    assert len(review["goldens"]) == 23 and sum(len(row["facts"]) for row in review["goldens"]) == 138
    cases = {case["case_id"]: case for case in review["goldens"]}
    zero = cases["2026-district-101903-03100500"]["facts"]
    assert zero["cte_excluding_explorers_average_credit_hours_per_student"] == {
        "raw": "0", "status": "observed", "value": "0.0", "type": "decimal128(16,1)"}
    assert zero["students_earning_college_credit_count"]["status"] == "suppressed"
    assert zero["average_credit_hours_per_student"]["value"] == "3.0"
    assert cases["2026-district-057829-03020000"]["facts"]["college_credit_hours"]["value"] == 24
    assert cases["2022-district-220819-Total"]["facts"]["college_credit_hours"]["value"] == 381


@pytest.mark.parametrize("damage", ["manifest", "build", "schema", "edition", "key", "metadata", "leaf", "size", "processing", "rows", "acquisition"])
def test_changed_identity_cannot_reuse_review(damage):
    review = plan()
    selected, catalog, refs = identity_release(review)
    gate.validate_college_credit_identity(selected, review)
    identifier, case = next(iter(review["editions"].items()))
    ref = refs[case["artifact_id"]]
    # Keep fixture and simulated actual catalog independent.
    catalog["editions"] = deepcopy(catalog["editions"])
    if damage == "manifest":
        selected.manifest_sha256 = "0" * 64
    elif damage == "build":
        selected.build_id = "0" * 64
    elif damage == "schema":
        catalog["schema_version"] = 3
    elif damage == "edition":
        catalog["editions"].pop(identifier)
    elif damage == "key":
        catalog["editions"][identifier]["key"]["scope"] = "campus"
    elif damage == "metadata":
        catalog["editions"][identifier]["metadata"]["sha256"] = "0" * 64
    elif damage == "leaf":
        ref.sha256 = "0" * 64
    elif damage == "size":
        ref.size_bytes += 1
    elif damage == "processing":
        ref.processing_id = "0" * 64
    elif damage == "rows":
        ref.contract["rows"] += 1
    else:
        selected.catalog.acquisitions = lambda *a, **k: [{"record_id": "0" * 64, "artifact": case["source"]["artifact"]}]
    with pytest.raises(ContractError, match="College Credit.*differs"):
        gate.validate_college_credit_identity(selected, review)


def synthetic_report(tmp_path, *, format="csv"):
    review = plan()
    case = deepcopy(next(case for case in review["editions"].values()
                         if case["key"]["edition"] == "2026" and case["key"]["scope"] == "state"))
    source = case["source"]
    row = ["Total", "", "0", "0", "0", "", "-999", "-999", "3.0"]
    text = StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows([["TEXAS EDUCATION AGENCY"], [], ["College Credit Reports - Statewide Totals"], [],
                     ["PEIMS data for 2025-2026"], [], source["header"], row, ["FERPA"]])
    csv_data = text.getvalue().encode("cp1252")
    source.update(header_record=7, first_data_record=8, last_data_record=8, source_rows=1,
                  preamble_sha256=sha256(b"".join(csv_data.splitlines(keepends=True)[:6])).hexdigest(),
                  footer_sha256=sha256(b"FERPA\n").hexdigest())
    html = ("<p>College Credit Reports - Statewide Totals 2025-2026</p><table><tr>" +
            "".join(f"<th>{label}</th>" for label in source["header"]) + "</tr><tr>" +
            "".join(f"<td>{'N/A' if value == '-999' else value}</td>" for value in row) +
            "</tr></table><p>FERPA</p>").encode("cp1252")
    data = csv_data if format == "csv" else html
    artifact = {"filename": f"report.{format}", "sha256": sha256(data).hexdigest(), "size_bytes": len(data)}
    source["artifact"] = artifact
    source["retrieved"] = source["downloaded_at"]
    controls = [["_service", "marykay"], ["_program", "adhoc.download_college_credit_report.sas"],
                ["RptClass", "CollegeCred"], ["_debug", "0"], ["school_year", "2026"], ["report", "State"], ["format", format]]
    url = f"{gate._BROKER}?{urlencode([(pair[0], pair[1]) for pair in controls])}"
    record = {"kind": "acquisition", "method": "download", "section": "reference", "dataset": f"reference:peims_college_credit:state:{format}",
              "requested_year": 2026, "resolved_year": 2026, "requested_url": url, "resolved_url": url,
              "downloaded_at": source["downloaded_at"], "artifact": artifact,
              "source_metadata": {"publisher": "Texas Education Agency", "version": "2026", "period": {"kind": "unknown"}},
              "parameters": {"http_method": "GET", "http_status": 200, "form_url": gate._BROKER,
                             "form_controls": controls, "form_content_type": None},
              "response_headers": {"content-type": "text/csv" if format == "csv" else "text/html"}}
    path = tmp_path / f"report.{format}"
    path.write_bytes(data)
    store = Mock(spec=ProvenanceStore)
    store.read_record.return_value = record
    store.verify_artifact.return_value = path
    return case, source, record, store, data, csv_data


@pytest.mark.parametrize("damage", ["year", "dataset", "url", "method", "status", "type", "controls", "date", "period", "raw-missing", "raw-tampered"])
def test_request_and_retained_raw_fail_closed(tmp_path, damage):
    _, source, record, store, data, _ = synthetic_report(tmp_path)
    assert gate._report_bytes(store, source, year=2026, scope="state", format="csv", receipt="receipt") == data
    if damage == "year":
        record["requested_year"] = 2025
    elif damage == "dataset":
        record["dataset"] = "reference:peims_college_credit:district:csv"
    elif damage == "url":
        record["resolved_url"] += "&wrong=1"
    elif damage == "method":
        record["parameters"]["http_method"] = "POST"
    elif damage == "status":
        record["parameters"]["http_status"] = 404
    elif damage == "type":
        record["response_headers"]["content-type"] = "application/pdf"
    elif damage == "controls":
        record["parameters"]["form_controls"].reverse()
    elif damage == "date":
        record["downloaded_at"] = None
    elif damage == "period":
        record["source_metadata"]["period"] = {"kind": "school_year", "start_year": 2025, "end_year": 2026}
    elif damage == "raw-missing":
        store.verify_artifact.side_effect = FileNotFoundError("retained raw unavailable")
    else:
        store.verify_artifact.return_value.write_bytes(data + b"tamper")
    with pytest.raises((ContractError, FileNotFoundError), match="College Credit|retained raw"):
        gate._report_bytes(store, source, year=2026, scope="state", format="csv", receipt="receipt")


@pytest.mark.parametrize("damage", [None, "linebreaks", "year", "value", "mask", "duplicate", "missing", "label"])
def test_independent_html_year_services_values_and_masks(tmp_path, monkeypatch, damage):
    _, source, _, store, html, csv_data = synthetic_report(tmp_path, format="html")
    witness = {"edition": 2026, "receipt": "html", "artifact": source["artifact"]}
    # CSV source framing must remain separate from the independent HTML object.
    if damage == "linebreaks":
        html = html.replace(b"Total Students", b"Total<br/>Students")
    elif damage == "year":
        html = html.replace(b"2025-2026", b"2024-2025")
    elif damage == "value":
        html = html.replace(b">3.0<", b">3.1<")
    elif damage == "mask":
        html = html.replace(b">N/A<", b">0<")
    elif damage in {"duplicate", "missing"}:
        start, end = html.index(b"<tr><td>"), html.index(b"</tr></table>") + len(b"</tr>")
        html = html[:start] + (html[start:end] * 2 if damage == "duplicate" else b"") + html[end:]
    elif damage == "label":
        html = html.replace(b">Total</td>", b">Wrong</td>")
    monkeypatch.setattr(gate, "_report_bytes", lambda *a, **k: html)
    if damage in (None, "linebreaks"):
        assert gate._html_parity(store, witness, source, csv_data)["matched_fact_cells"] == 6
    else:
        with pytest.raises((ContractError, ValueError), match="HTML"):
            gate._html_parity(store, witness, source, csv_data)


def test_html_parser_preserves_internal_literal_label_spaces():
    # These doubles occur in TEA's language course labels in every reviewed
    # year. Header line wrapping must not erase literal publisher body spacing.
    rows = gate._PublisherRows("<table><tr><td>03110300</td><td>LANG O/T ENGLISH III  - ARABIC</td></tr></table>").rows
    assert rows == [["03110300", "LANG O/T ENGLISH III  - ARABIC"]]


@pytest.mark.parametrize("damage", [None, "value", "status", "raw", "record", "service", "processing", "framing"])
def test_independent_csv_preserves_zero_suppression_decimal_and_positions(tmp_path, damage):
    case, source, record, store, data, _ = synthetic_report(tmp_path)
    fields = gate._fields(2026)
    tokens = ["0", "0", "0", "-999", "-999", "3.0"]
    columns = {"peims_source_record": [8], "college_credit_service_key": ["Total"]}
    for field, token in zip(fields, tokens, strict=True):
        masked = token == "-999"
        columns[field] = [None if masked else Decimal(token) if "average" in field else int(token)]
        columns[f"peims_raw_{field}"] = [token]
        columns[f"peims_status_{field}"] = ["suppressed" if masked else "observed"]
    if damage == "value":
        columns[fields[0]] = [1]
    elif damage == "status":
        columns[f"peims_status_{fields[0]}"] = ["suppressed"]
    elif damage == "raw":
        columns[f"peims_raw_{fields[0]}"] = ["-999"]
    elif damage == "record":
        columns["peims_source_record"] = [9]
    elif damage == "service":
        columns["college_credit_service_key"] = ["00000000"]
    elif damage == "framing":
        source["preamble_sha256"] = "0" * 64
    table = Mock()
    table.to_arrow.return_value = pa.table(columns)
    def literal_export(stream, **kwargs):
        literal = b"".join(data.splitlines(keepends=True)[6:8])
        stream.write(literal)
    table.to_csv.side_effect = literal_export
    processing = {"processor": "wrong" if damage == "processing" else "teadata.peims_college_credit", "dataset": "peims_college_credit",
                  "inputs": [source["original_acquisition_id"]], "parameters": {"leaf_contract_sha256": case["artifact_id"],
                  "recipe": {"operation": "reviewed_peims_college_credit_csv_projection", "encoding": "cp1252"}}}
    store.verify_record_closure.return_value = {case["processing_id"]: processing, source["original_acquisition_id"]: record}
    release = cast(DataRelease, SimpleNamespace(catalog=SimpleNamespace(artifact_ref=lambda _: SimpleNamespace(identity=case["artifact_id"]),
        acquisitions=lambda *a, **k: [{"record_id": source["original_acquisition_id"], **record}]), table=lambda _: table))
    if damage is None:
        assert gate._raw_parity(release, store, case)["masked_cells"] == 2
    else:
        with pytest.raises(ContractError, match="College Credit"):
            gate._raw_parity(release, store, case)


def test_normal_dispatch_relocation_verifies_both_evidence_stores(monkeypatch):
    review = plan()
    child, _, _ = identity_release(review)
    processing, publisher = Mock(spec=ProvenanceStore), Mock(spec=ProvenanceStore)
    processing.verify_record_closure.return_value = {"processing-proof": {}}
    publisher.verify_record_closure.return_value = {"publisher-proof": {}}
    monkeypatch.setattr(verification, "_retained_store", lambda path: processing)
    selected = verification._native_review(child, publisher)
    assert selected.label == "College Credit"
    assert selected.evidence() == {"processing": ["processing-proof"], "publisher": ["publisher-proof"]}
    processing.verify_record_closure.assert_called_once_with([case["processing_id"] for case in review["editions"].values()])
    publisher.verify_record_closure.assert_called_once_with([w["receipt"] for w in review["publisher_witnesses"]])
    publisher.verify_record_closure.side_effect = FileNotFoundError("fifth HTML witness missing")
    with pytest.raises(FileNotFoundError, match="fifth HTML"):
        selected.evidence()


def test_full_retained_college_credit_qualification(monkeypatch):
    manifest = os.environ.get("TEADATA_COLLEGE_CREDIT_MANIFEST")
    witness_root = os.environ.get("TEADATA_COLLEGE_CREDIT_WITNESS_STORE")
    if manifest is None and witness_root is None:
        pytest.skip("Provide retained College Credit manifest and independent witness store for the full source replay")
    assert manifest and witness_root, "Both retained College Credit evidence paths are required"
    monkeypatch.setattr("teadata.release._download", lambda *a, **k: pytest.fail("Implicit source download"))
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("Network access during offline source qualification"))
    monkeypatch.setattr("socket.socket.connect_ex", lambda *a, **k: pytest.fail("Network access during offline source qualification"))
    release = DataRelease(manifest, expected_sha256=_PIN, offline=True)
    parent_path = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_MANIFEST")
    if parent_path:
        parent_pin = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_SHA256")
        assert parent_pin, "The normal parent route requires its trusted manifest pin"
        parent = DataRelease(parent_path, expected_sha256=parent_pin, offline=True)
        ref, = (ref for ref in parent.catalog.native_sources if ref.manifest_sha256 == _PIN)
        # This is the same opener/store route used by normal release validation
        # and receipt reuse. It must work without a separate witness override.
        child, adjacent = verification._open_reviewed_native_evidence(parent, ref, "College Credit")
        assert child.manifest_sha256 == release.manifest_sha256
        result = verification._native_review(child, adjacent).qualify()
    else:
        result = verification._native_review(release, ProvenanceStore(witness_root)).qualify()
    assert result["status"] == "PASS_REVIEWED_NATIVE"
    assert result["parity"]["coverage"] == {
        "rows": 125435, "source_cells": 1675932, "typed_facts": 752610, "masked_cells": 235378,
        "skipped_separator_records": 5345, "requested_editions": 15, "checked_editions": 15,
        "uncovered_editions": 0, "failed_editions": 0}
    assert sum(item["matched_fact_cells"] for item in result["publisher_witnesses"]["reports"]) == 20442
    destination = os.environ.get("TEADATA_COLLEGE_CREDIT_QUALIFICATION_OUTPUT")
    if destination:
        Path(destination).write_text(json.dumps(result, indent=2) + "\n")
