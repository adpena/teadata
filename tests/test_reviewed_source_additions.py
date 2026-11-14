"""Exact source admission is distinct from successful materialization."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from teadata import attendance_chronic, native_witnesses, sof_attendance, verification
from teadata.manifest import ContractError
from teadata.release import DataRelease


@pytest.mark.parametrize("entry", [
    None, {}, {"family": "other", "references": {}},
    {"family": "sof_attendance", "references": {"unexpected": {}}},
    {"family": "chronic_absence", "references": []},
    {"family": "bound_transfers", "references": {}},
    {"family": "sof_attendance", "references": {}, "approved": True},
])
def test_admission_registry_requires_exact_closed_review(tmp_path, monkeypatch, entry):
    resource = tmp_path / "reference_cases/native_source_additions.json"
    resource.parent.mkdir()
    resource.write_text(json.dumps({"schema_version": 1, "releases": {"a" * 64: entry}}))
    monkeypatch.setattr(native_witnesses, "files", lambda _: tmp_path)
    if entry is None:
        assert native_witnesses.reviewed_source_additions("a" * 64) is None
    else:
        with pytest.raises(ContractError, match="Invalid exact source-addition review"):
            native_witnesses.reviewed_source_additions("a" * 64)
    assert native_witnesses.reviewed_source_additions("b" * 64) is None


@pytest.mark.parametrize("family,module,qualifier,evidence", [
    ("sof_attendance", sof_attendance, "qualify_sof_attendance", "retained_sof_evidence"),
    ("chronic_absence", attendance_chronic, "qualify_attendance_chronic", "retained_attendance_evidence"),
])
def test_source_review_and_reuse_bind_adjacent_retained_store(monkeypatch, family, module, qualifier, evidence):
    selected = SimpleNamespace(manifest_sha256="a" * 64, path=Path("frozen/child/release.json"))
    store = Mock()
    retained = Mock(return_value=store)
    qualify = Mock(return_value={"status": "PASS_REVIEWED_NATIVE"})
    records = Mock(return_value=["receipt"])
    monkeypatch.setattr(native_witnesses, "reviewed_source_additions", lambda pin: (family, {}))
    monkeypatch.setattr(verification, "_retained_store", retained)
    monkeypatch.setattr(module, qualifier, qualify)
    monkeypatch.setattr(module, evidence, records)
    review = verification._native_review(cast(DataRelease, selected), Mock())
    assert review.label == family
    assert review.qualify() == {"status": "PASS_REVIEWED_NATIVE"}
    assert review.evidence() == ["receipt"]
    retained.assert_called_once_with(Path("frozen/child/provenance"))
    qualify.assert_called_once_with(selected, store)
    records.assert_called_once_with(selected, store)


def test_new_transfer_review_preserves_original_independent_reports():
    selected = native_witnesses.reviewed_source_additions(
        "17e8582abdda6ad9cd1ab60a597da73d2d9ef0049868de25f1a7c1441d9a1afa")
    assert selected is not None and selected[0] == "bound_transfers"
    original = native_witnesses.reviewed_native_references()
    assert selected[1].keys() == original.keys()
    unchanged = 0
    for key, new in selected[1].items():
        assert new["reports"] == original[key]["reports"]
        if new == original[key]:
            unchanged += 1
        else:
            # The new role binding changes Arrow schema metadata and therefore
            # leaf bytes, while the independent source cases stay unchanged.
            assert new["artifact"] != original[key]["artifact"]
            assert new["processing_id"] != original[key]["processing_id"]
    assert unchanged == 6
