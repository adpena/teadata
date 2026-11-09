from types import SimpleNamespace

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from teadata.dataset_metadata import DataPeriod, EditionKey
from teadata.provenance import ProvenanceStore
from teadata.scripts.refresh_transfers import main
from teadata import transfer_rollups
from teadata.transfer_rollups import compare_district_directions


def _report(rows, *, publisher_flags=True):
    columns = {
        "transfer_direction": pa.array([row[0] for row in rows], type=pa.string()),
        "transfer_reporting_district_number": pa.array([row[1] for row in rows], type=pa.string()),
        "transfer_partner_district_number": pa.array([row[2] for row in rows], type=pa.string()),
        "student_transfer_count": pa.array([row[3] for row in rows], type=pa.int64()),
        "transfer_count_status": pa.array([row[4] for row in rows], type=pa.string()),
        "TRANSFERS_IN_OR_OUT": pa.array([row[5] for row in rows], type=pa.string()),
        "transfer_row_kind": pa.array([row[6] for row in rows], type=pa.string()),
    }
    if publisher_flags:
        columns["reporting_charter_status"] = pa.array([row[7] for row in rows], type=pa.string())
    return pa.table(columns)


_ISD = "Traditional ISD/CSD"
_CHARTER = "Open Enrollment Charter"
_OUT = ("out", "'000001", "'000002", 0, "observed", "0", "detail", _ISD)
_IN = ("in", "'000002", "'000001", 0, "observed", "00", "detail", _CHARTER)


def _release(table, *, year=2026, revisions=("a",), measurement_year=None, extra_artifact=False):
    keys = [EditionKey("district_transfer_reports", str(year), revision * 64, "district", "report_rows", "publisher detail")
            for revision in revisions]
    support = SimpleNamespace(projection=lambda field: {"raw": "TRANSFERS_IN_OR_OUT", "status": "transfer_count_status"})
    ref = SimpleNamespace(edition=keys[0], output="leaf", identity="leaf-id", column_support=support, sha256="b" * 64)
    end = measurement_year or year

    def read(artifact_id):
        assert artifact_id == ref.identity and table is not None
        return ds.dataset(table)

    def resolve(fields, *, period):
        assert fields == [(ref.edition, "student_transfer_count")]
        assert period == DataPeriod("school_year", year - 1, year)
        return [{"artifact": ref}]

    artifacts = {ref.identity: {"edition": ref.edition.to_dict()}}
    if extra_artifact:
        artifacts["another-leaf"] = {"edition": ref.edition.to_dict()}
    catalog = SimpleNamespace(editions=lambda **kwargs: [{"edition_id": key.identity, **key.to_dict()} for key in keys],
                              metadata=lambda *args, **kwargs: SimpleNamespace(period=DataPeriod("school_year", end - 1, end)),
                              resolve_artifacts=resolve, to_dict=lambda: {"artifacts": artifacts})
    return SimpleNamespace(offline=True, catalog=catalog, analytical_dataset=read, manifest_sha256="c" * 64)


def test_direction_audit_orients_pairs_and_separates_masking_from_values():
    table = _report([
        _OUT, _IN,
        ("out", "'000001", "'000003", None, "suppressed", "-999", "detail", _ISD),
        ("in", "'000003", "'000001", 15, "observed", "15", "detail", _CHARTER),
        ("out", "'000001", "'000004", 12, "observed", "12", "detail", _ISD),
        ("out", "'000001", "'000005", 10, "observed", "10", "detail", _ISD),
        ("in", "'000005", "'000001", 11, "observed", "11", "detail", _ISD),
        ("in", "'000004", None, 12, "observed", "12", "publisher_total", _CHARTER),
        ("out", "'000001", None, None, "missing", ".", "separator", _ISD),
    ])
    result = compare_district_directions(_release(table), school_year="2025–26")

    assert result["status"] == "HOLD_RECIPROCAL_DISTRICT_DIRECTIONS"
    assert result["directional_detail_rows"] == 7
    assert result["coverage"] == {
        "matched": 3, "out_only": 1, "in_only": 0, "matching_observed": 1, "matching_suppressed": 0,
        "status_differences": 1, "observed_count_differences": 1, "raw_token_only_differences": 1,
    }
    classification = result["classification"]
    assert classification["unknown_destination_pairs"] == 0
    assert classification["charter_destinations"]["status_differences"] == 1
    assert classification["charter_destinations"]["out_only"] == 1


def test_direction_audit_passes_equivalent_raw_tokens_and_keeps_unknown_classification():
    table = _report([_OUT, _IN], publisher_flags=False)
    result = compare_district_directions(_release(table, year=2013), school_year=13)

    assert result["status"] == "PASS_RECIPROCAL_DISTRICT_DIRECTIONS"
    assert result["coverage"]["raw_token_only_differences"] == 1
    assert result["classification"] == {
        "basis": "unavailable", "unknown_origin_pairs": 1, "unknown_destination_pairs": 1,
        "charter_origin_pairs": 0, "charter_destinations": None,
    }


def test_direction_audit_binds_early_roster_to_selected_year(monkeypatch, tmp_path):
    store = ProvenanceStore(tmp_path)

    def roster(selected, year):
        assert selected is store and year == 2013
        return {"000002": "I"}, {"basis": "same_year_tea_district_types", "receipt": "e" * 64}

    monkeypatch.setattr(transfer_rollups, "_historical_roster", roster)
    table = _report([_OUT, _IN], publisher_flags=False)
    result = compare_district_directions(_release(table, year=2013), school_year="2012-13", roster_store=store)
    classification = result["classification"]
    assert classification["receipt"] == "e" * 64
    assert classification["unknown_origin_pairs"] == 1
    assert classification["unknown_destination_pairs"] == 0
    assert classification["charter_destinations"]["matching_observed"] == 1


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([_OUT, _OUT, _IN], "Duplicate out"),
        ([_OUT, (*_IN[:-1], "Charter")], "unsupported reporter charter flags"),
        ([_OUT, _IN, (*_IN[:-2], "publisher_total", _ISD)], "conflicting"),
        ([_OUT, _IN, (*_IN[:-2], "unknown", _CHARTER)], "unrecognized row kind"),
        ([_OUT, _IN, (*_IN[:-2], None, _CHARTER)], "unrecognized row kind"),
        ([("out", "000001", *_OUT[2:]), _IN], "canonical six-digit"),
        ([_OUT, (*_IN[:2], "'000002", *_IN[3:])], "within-district"),
    ],
)
def test_direction_audit_rejects_malformed_detail_or_reporter_type(rows, error):
    with pytest.raises(ValueError, match=error):
        compare_district_directions(_release(_report(rows)), school_year=2026)


def test_direction_audit_validates_the_edition_before_opening_the_leaf():
    with pytest.raises(ValueError, match="Available school-year ends: 2026"):
        compare_district_directions(_release(None), school_year=2024)
    with pytest.raises(ValueError, match="ambiguous"):
        compare_district_directions(_release(None, revisions=("a", "d")), school_year=2026)
    with pytest.raises(ValueError, match="period differs"):
        compare_district_directions(_release(None, measurement_year=2025), school_year=2026)
    with pytest.raises(ValueError, match="exactly one registered leaf"):
        compare_district_directions(_release(None, extra_artifact=True), school_year=2026)


def test_direction_audit_reports_a_charter_origin_and_empty_detail_as_hold():
    charter_origin = _report([(*_OUT[:-1], _CHARTER), (*_IN[:-1], _ISD)])
    result = compare_district_directions(_release(charter_origin), school_year=26)
    assert result["status"] == "HOLD_RECIPROCAL_DISTRICT_DIRECTIONS"
    assert result["classification"]["charter_origin_pairs"] == 1
    empty = compare_district_directions(_release(_report([])), school_year=26)
    assert empty["status"] == "HOLD_RECIPROCAL_DISTRICT_DIRECTIONS"
    assert empty["coverage"]["matched"] == 0


def test_direction_audit_cli_rejects_wrong_manifest_pin(tmp_path, capsys):
    path = tmp_path / "release.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["audit-district-directions", "--district-release", str(path), "--district-pin", "a" * 64,
              "--year", "2025-26"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "checksum differs from its trusted pin" in output.err
