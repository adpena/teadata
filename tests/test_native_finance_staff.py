from html import escape
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from teadata.manifest import ContractError
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.native_finance_staff import REVIEWED_MANIFEST, _staff_cells, qualify_finance_staff, retained_finance_staff_evidence
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease


def _report(entity, scope, *, edition="2024-25", duplicate=False):
    district = scope == "district"
    labels = ["Count", "Percent", "Count", "Percent"] if district else ["Count/Average", "Percent", "District", "State"]
    teachers = ["Teachers", "19.1", "", "", ""]
    rows = [
        ["", "Texas Education Agency", ""], [f"{edition} Staff Information"], [f"Example ({entity})"],
        ["2024-25"], ["", scope.title(), ""], ["Staff Information", *labels],
        ["Professional Staff", "23.1", "", "", ""], teachers,
        *([teachers] if duplicate else []), ["Professional Support", "2.0", "", "", ""],
        ["Number of Students per Teacher", "21.2"], ["Average Years Experience of Campus Staff"],
        ["Teachers", "7.1"], ["Average Teacher Salary by Years of Experience (regular duties only)"],
        ["Average Actual Salaries (regular duties only)"], ["Teachers", "$67,496"],
    ]
    if district:
        rows.append(["Turnover Rate for Teachers", "30.3%"])
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"


@pytest.mark.parametrize(("scope", "entity"), [("campus", "084908042"), ("district", "084908")])
def test_staff_witness_uses_teacher_row_in_each_report_section(scope, entity):
    result = _staff_cells(_report(entity, scope), scope, entity)
    expected = {
        "teacher_total_full_time_equiv_count": ("fte", "19.1"),
        "teacher_experience_average": ("years", "7.1"),
        "teacher_total_base_salary_average": ("currency", "$67,496"),
        "teacher_student_ratio": ("ratio", "21.2"),
    }
    if scope == "district":
        expected["teacher_turnover_ratio"] = ("percent", "30.3%")
    assert result == expected
    with pytest.raises(ContractError, match="period or geography"):
        _staff_cells(_report(entity, scope, edition="2023-24"), scope, entity)
    with pytest.raises(ValueError):
        _staff_cells(_report(entity, scope, duplicate=True), scope, entity)


@pytest.mark.parametrize(("manifest", "offline"), [("a" * 64, True), (REVIEWED_MANIFEST, False)])
def test_finance_staff_gate_rejects_unreviewed_or_remote_release(manifest, offline):
    release = Mock(spec=DataRelease, manifest_sha256=manifest, offline=offline)
    with pytest.raises(ContractError, match="exact reviewed offline release"):
        qualify_finance_staff(release, Mock(spec=ProvenanceStore))


def test_composed_receipt_rehashes_source_bytes(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("value\n5\n")
    store = ProvenanceStore(tmp_path / "provenance")
    receipt = store.record_acquisition(source, ResolvedDataset(
        "sample", "data_sources", 2025, 2025, str(source), DatasetMetadata(title="Sample"),
    ))
    processing = store.record_processing(
        dataset="sample", inputs=[receipt], outputs={"leaf": source}, processor="test",
        code_version="a" * 64,
    )
    monkeypatch.setattr("teadata.native_finance_staff._reviewed_leaves",
                        lambda _: ({"state": SimpleNamespace(processing_id=processing)}, {}))
    release = Mock(spec=DataRelease)
    assert retained_finance_staff_evidence(release, store) == sorted((receipt, processing))
    path = store.verify_artifact(store.read_record(receipt)["artifact"]["sha256"])
    path.write_text("value\n6\n")
    with pytest.raises(ValueError, match="Archived artifact checksum mismatch"):
        retained_finance_staff_evidence(release, store)
