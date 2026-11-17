"""The canonical gate must use independent source observations and fail closed."""

from copy import deepcopy
from importlib.resources import files
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pyarrow as pa
import pytest

from teadata import verification as gate
from teadata.dataset_metadata import DataPeriod, EditionKey
from teadata.release import DataRelease


def plans():
    return json.loads(
        files("teadata")
        .joinpath("reference_cases/native_educator_aggregates.json")
        .read_text()
    )


def small_case(monkeypatch, kind="retention"):
    plan = deepcopy(plans()["children"][kind])
    identifier, entry = next(iter(plan["catalog"]["editions"].items()))
    key = EditionKey.from_dict(entry["key"])
    year = int(key.table.removeprefix("state_"))
    label = f"{year - 1}-{year % 100:02d}"
    plan["catalog"]["editions"] = {identifier: entry}
    plan["catalog"]["artifacts"] = {
        a: e for a, e in plan["catalog"]["artifacts"].items() if e == identifier
    }
    plan["rows"] = 1
    if kind == "retention":
        values = [label, "Male", 10, *[v for _ in range(5) for v in (5, 0.5)]]
        item = {"dataset": key.dataset, "row": 2, "values": values}
    else:
        values = [
            label,
            "All",
            key.dataset.rsplit("_", 1)[-1].title(),
            "All",
            "All",
            12,
        ]
        item = {"row": 2, "values": values}
    proof = {kind: {"rows": [item]}}
    raw = gate._educator_aggregate_rows(proof, kind, plan["fields"])[
        key.dataset, key.table
    ]
    catalog = deepcopy(plan["catalog"])
    period = DataPeriod("school_year", year - 1, year).to_dict()
    selected = SimpleNamespace(
        manifest_sha256=plan["manifest_sha256"],
        build_id=catalog["build_id"],
        path=Path("relocated/release.json"),
        verify=Mock(),
        table=Mock(
            side_effect=lambda _: SimpleNamespace(
                to_arrow=lambda: pa.Table.from_pylist(raw)
            )
        ),
        catalog=SimpleNamespace(
            to_dict=lambda: catalog,
            field_metadata=Mock(return_value={"period": period}),
        ),
    )
    evidence = Mock(
        return_value={"processing": ["processing"], "publisher": ["publisher"]}
    )
    witness = Mock(return_value=proof)
    monkeypatch.setattr(gate, "_native_record_evidence", evidence)
    monkeypatch.setattr(gate, "_educator_aggregate_witness", witness)
    return selected, plan, proof, raw, catalog, evidence, witness


def test_packaged_plan_has_only_two_immutable_children_and_explicit_source_roots():
    document = plans()
    assert document["schema_version"] == 1
    assert set(document["children"]) == {"retention", "certification"}
    for kind, editions, rows in (("retention", 35, 151), ("certification", 33, 33)):
        plan = document["children"][kind]
        assert gate._EDUCATOR_AGGREGATE_PINS[plan["manifest_sha256"]] == kind
        assert len(plan["catalog"]["editions"]) == editions and plan["rows"] == rows
        assert (
            len(set(plan["publisher_roots"])) == 8
            and gate._EDUCATOR_WITNESS in plan["publisher_roots"]
        )
        assert "qualification" not in json.dumps(plan).lower()
        keys = [v["key"] for v in plan["catalog"]["editions"].values()]
        assert all(k["scope"] == "state" for k in keys)
        assert all(
            not (
                k["dataset"].endswith("_preparation_route")
                and k["table"] == "state_2018"
            )
            for k in keys
        )


def test_unknown_pin_does_not_open_new_plan(monkeypatch):
    resource = Mock(side_effect=AssertionError("Unselected plan I/O"))
    monkeypatch.setattr("importlib.resources.files", resource)
    assert (
        gate._reviewed_educator_aggregate(
            cast(DataRelease, SimpleNamespace(manifest_sha256="0" * 64)), Mock()
        )
        is None
    )
    resource.assert_not_called()


@pytest.mark.parametrize("kind", ["retention", "certification"])
def test_exact_small_family_uses_independent_witness_and_all_physical_fields(
    monkeypatch, kind
):
    release, plan, _, _, _, evidence, witness = small_case(monkeypatch, kind)
    processing, publisher = Mock(), Mock()
    report = gate._qualify_educator_aggregate(
        release, processing, publisher, kind, plan
    )
    assert (
        report["status"] == "PASS_REVIEWED_NATIVE"
        and report["rows"] == report["editions"] == 1
    )
    assert report["certification_detail_qualified"] is False
    evidence.assert_called_once_with(
        release, processing, publisher, plan["publisher_roots"]
    )
    witness.assert_called_once_with(publisher, plan["publisher_roots"])


@pytest.mark.parametrize(
    "damage", ["manifest", "build", "edition", "routing", "metadata", "extra_held"]
)
def test_changed_identity_rejected_before_any_observations(monkeypatch, damage):
    release, plan, _, _, catalog, evidence, witness = small_case(monkeypatch)
    if damage == "manifest":
        release.manifest_sha256 = "0" * 64
    elif damage == "build":
        catalog["build_id"] = "0" * 64
    elif damage == "edition":
        catalog["editions"].clear()
    elif damage == "routing":
        catalog["artifacts"].clear()
    elif damage == "metadata":
        next(iter(catalog["editions"].values()))["metadata"]["sha256"] = "0" * 64
    else:
        catalog["editions"]["held"] = {
            "key": {
                "dataset": "educator_state_retention_by_preparation_route",
                "table": "state_2018",
            }
        }
    with pytest.raises(gate.ReceiptError, match="reviewed identity"):
        gate._qualify_educator_aggregate(release, Mock(), Mock(), "retention", plan)
    release.verify.assert_not_called()
    release.table.assert_not_called()
    evidence.assert_not_called()
    witness.assert_not_called()


@pytest.mark.parametrize(
    "damage",
    [
        "count",
        "float_count",
        "bool_count",
        "ratio",
        "raw",
        "status",
        "identity",
        "coordinate",
        "missing",
        "extra",
        "duplicate",
        "absent",
    ],
)
def test_native_values_support_identity_and_exact_row_coverage_fail_closed(
    monkeypatch, damage
):
    release, plan, _, raw, _, _, _ = small_case(monkeypatch)
    row = raw[0]
    if damage == "count":
        row["first_year_teacher_count"] += 1
    elif damage == "float_count":
        row["first_year_teacher_count"] = 10.0
    elif damage == "bool_count":
        row["first_year_teacher_count"] = True
    elif damage == "ratio":
        row["retained_year_1_ratio"] = math.nextafter(0.5, 1)
    elif damage == "raw":
        row["raw_retained_year_1_ratio"] = "0.50"
    elif damage == "status":
        row["status_retained_year_1_ratio"] = "suppressed"
    elif damage == "identity":
        row["publisher_group"] = "All Groups"
    elif damage == "coordinate":
        row["publisher_worksheet_row"] = 3
    elif damage == "missing":
        del row["publisher_worksheet_row"]
    elif damage == "extra":
        row["detail"] = "unqualified"
    elif damage == "duplicate":
        raw.append(dict(row))
    else:
        raw.clear()
    with pytest.raises(gate.ReceiptError, match="parity differs"):
        gate._qualify_educator_aggregate(release, Mock(), Mock(), "retention", plan)


def test_missing_followup_remains_missing_and_ratio_signed_zero_is_not_equal(
    monkeypatch,
):
    release, plan, proof, raw, _, _, _ = small_case(monkeypatch)
    proof["retention"]["rows"][0]["values"][-2:] = [None, None]
    raw[:] = gate._educator_aggregate_rows(proof, "retention", plan["fields"])[
        next(iter(plan["catalog"]["editions"].values()))["key"]["dataset"],
        next(iter(plan["catalog"]["editions"].values()))["key"]["table"],
    ]
    assert (
        raw[0]["raw_retained_year_5_ratio"] == ""
        and raw[0]["status_retained_year_5_ratio"] == "missing"
    )
    gate._qualify_educator_aggregate(release, Mock(), Mock(), "retention", plan)
    proof["retention"]["rows"][0]["values"][4] = 0.0
    raw[0]["retained_year_1_ratio"] = -0.0
    raw[0]["raw_retained_year_1_ratio"] = "0.0"
    with pytest.raises(gate.ReceiptError, match="parity differs"):
        gate._qualify_educator_aggregate(release, Mock(), Mock(), "retention", plan)


@pytest.mark.parametrize("damage", ["held", "detail", "duplicate", "missing", "period"])
def test_witness_roster_population_and_period_cannot_expand(monkeypatch, damage):
    kind = "certification" if damage == "detail" else "retention"
    release, plan, proof, _, _, _, _ = small_case(monkeypatch, kind)
    item = proof[kind]["rows"][0]
    if damage == "held":
        item["dataset"] = "educator_state_retention_by_preparation_route"
        item["values"][0] = "2017-18"
    elif damage == "detail":
        item["values"][1] = "Alternative"
    elif damage == "duplicate":
        proof[kind]["rows"].append(deepcopy(item))
    elif damage == "missing":
        proof[kind]["rows"].clear()
    else:
        release.catalog.field_metadata.return_value = {"period": {"kind": "unknown"}}
    with pytest.raises(gate.ReceiptError):
        gate._qualify_educator_aggregate(release, Mock(), Mock(), kind, plan)


@pytest.mark.parametrize(
    "damage", [None, "record", "artifact", "roots", "scope", "source_bytes"]
)
def test_independent_witness_requires_complete_verified_publisher_closure(damage):
    roots = [gate._EDUCATOR_WITNESS, "1" * 64]
    record = {
        "kind": "acquisition",
        "method": "local_observation",
        "artifact": {"sha256": gate._EDUCATOR_WITNESS_SHA},
        "parameters": {"source_receipts": ["1" * 64]},
    }
    proof = {
        "status": "FROZEN_SOURCE_WITNESSES_NOT_MATERIALIZATION",
        "schema_version": 1,
        "source_release_activation": False,
    }
    store = Mock()
    store.read_record.return_value = record
    store.verify_artifact.return_value.read_bytes.side_effect = lambda: json.dumps(
        proof
    ).encode()
    if damage == "record":
        record["method"] = "download"
    elif damage == "artifact":
        record["artifact"]["sha256"] = "0" * 64
    elif damage == "roots":
        record["parameters"]["source_receipts"] = []
    elif damage == "scope":
        proof["source_release_activation"] = True
    elif damage == "source_bytes":
        store.verify_record_closure.side_effect = ValueError("Publisher bytes changed")
    if damage is None:
        assert gate._educator_aggregate_witness(store, roots) == proof
    else:
        with pytest.raises((gate.ReceiptError, ValueError)):
            gate._educator_aggregate_witness(store, roots)
    store.verify_record_closure.assert_called_once_with(roots)


def test_exact_dispatch_captures_relocated_processing_and_all_publisher_evidence(
    monkeypatch,
):
    plan = plans()["children"]["retention"]
    release = SimpleNamespace(
        manifest_sha256=plan["manifest_sha256"],
        build_id=plan["catalog"]["build_id"],
        path=Path("relocated/child/release.json"),
        catalog=SimpleNamespace(to_dict=lambda: plan["catalog"]),
    )
    processing, publisher = Mock(), Mock()
    retained = Mock(return_value=processing)
    qualifier, evidence = (
        Mock(return_value={"status": "PASS_REVIEWED_NATIVE"}),
        Mock(return_value={"publisher": []}),
    )
    monkeypatch.setattr(gate, "_retained_store", retained)
    monkeypatch.setattr(gate, "_qualify_educator_aggregate", qualifier)
    monkeypatch.setattr(gate, "_native_record_evidence", evidence)
    review = gate._native_review(cast(DataRelease, release), publisher)
    review.qualify()
    review.evidence()
    retained.assert_called_once_with(Path("relocated/child/provenance"))
    qualifier.assert_called_once_with(release, processing, publisher, "retention", plan)
    evidence.assert_called_once_with(
        release, processing, publisher, plan["publisher_roots"]
    )
