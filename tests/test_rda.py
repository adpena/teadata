from decimal import Decimal
from hashlib import sha256
import json

import pytest

from teadata import rda
from teadata.manifest import ContractError
from teadata.rda_native import _district_projection, _state_projection


def test_indicator1_registry_preserves_direct_state_and_absent_pooled_field():
    prior = rda.rda_indicator1_metadata(2024, "01")
    current = rda.rda_indicator1_metadata(2025, "01")

    assert prior["test_administration"] == "Spring 2024"
    assert current["test_administration"] == "Spring 2025"
    assert prior["fields"][0]["multi_year_rate_code"] is None
    assert current["fields"][0]["multi_year_rate_code"] == "BB_SM_PA25"
    assert current["fields"][0]["direct_texas"] == {
        "numerator": 118_871, "denominator": 172_496, "rate_percent": "68.9", "report_page": 1,
    }
    current["fields"].clear()
    assert len(rda.rda_indicator1_metadata(2025, "01")["fields"]) == 4
    with pytest.raises(ContractError, match="is not reviewed"):
        rda.rda_indicator1_metadata(2026, "01")


def test_rda_projection_preserves_raw_and_disclosure_states(tmp_path, monkeypatch):
    raw = (
        b"DISTRICT,REGION,P,N,D,PA\n"
        b"000001,01,.,12,26,54\n"
        b"000002,01,-1.0,-1,-1,.\n"
        b"000003,02,0.0,0,17,.\n"
        b"000004,20,59.6,53,89,.\n"
        b"000005,20,62.0,-1,-1,.\n"
    )
    metadata = _fixture_metadata(raw, 5)
    monkeypatch.setattr(rda, "rda_indicator1_metadata", lambda *_: metadata)
    path = tmp_path / "district.csv"
    path.write_bytes(raw)
    table = rda.read_rda_indicator1(path, edition=2025, area="03")

    assert table["district_number"].to_pylist() == [f"{code:06d}" for code in range(1, 6)]
    assert table["single_year_rate"].to_pylist() == [None, None, Decimal("0.0"), Decimal("59.6"), Decimal("62.0")]
    assert table["single_year_rate_raw"].to_pylist() == [".", "-1.0", "0.0", "59.6", "62.0"]
    assert table["single_year_rate_status"].to_pylist() == ["missing", "suppressed", "observed", "observed", "observed"]
    assert table["counts_status"].to_pylist() == ["observed", "suppressed", "observed", "observed", "suppressed"]
    assert table["numerator"].to_pylist() == [12, None, 0, 53, None]
    assert table["denominator_raw"].to_pylist() == ["26", "-1", "17", "89", "-1"]
    assert table["multi_year_rate"].to_pylist() == [Decimal("54.0"), None, None, None, None]
    assert table["multi_year_rate_raw"].to_pylist() == ["54", ".", ".", ".", "."]
    assert json.loads(table.schema.metadata[b"teadata.rda_indicator1"]) == metadata

    path.write_bytes(raw.replace(b"000001", b"000006"))
    with pytest.raises(ContractError, match="publisher bytes"):
        rda.read_rda_indicator1(path, edition=2025, area="03")


@pytest.mark.parametrize(("row", "message"), [
    (b"000001,01,66.6,2,3,.\n", "differs from the published counts"),
    (b"000001,01,.,2,3,.\n", "separately published multi-year rate"),
    (b"000000,01,66.7,2,3,.\n", "district row count or identifiers"),
])
def test_rda_rejects_projection_inconsistency(row, message):
    raw = b"DISTRICT,REGION,P,N,D,PA\n" + row
    with pytest.raises(ContractError, match=message):
        rda._project(raw, _fixture_metadata(raw, 1))


def _fixture_metadata(raw, rows):
    return {
        "district_csv": {"size_bytes": len(raw), "sha256": sha256(raw).hexdigest()},
        "district_rows": rows,
        "fields": [{
            "subject": "Mathematics", "single_year_rate_code": "P",
            "numerator_code": "N", "denominator_code": "D", "multi_year_rate_code": "PA",
        }],
    }


_STATE_PAGE = """
2025 Results Driven Accountability
State Report
Special Education
Indicator       Rate       Numerator       Denominator
1. SPED STAAR 3-8 Passing Rate
   (i) Mathematics  70.0 - 100  44.7  192,262  430,232  2  44.7  44.3  49.3  -4.6
   (ii) Reading Language Arts  70.0 - 100  48.8  212,721  435,743  2  48.8  47.5  49.3  -0.5
   (iii) Science  65.0 - 100  41.3  55,572  134,448  1  41.3  36.6  43.8  -2.5
   (iv) Social Studies  65.0 - 100  31.0  17,056  55,078  3  31.0  31.6  33.8  -2.8
2. SPED Year-After-Exit (YAE) STAAR 3-8 Passing Rate
"""


def test_rda_state_native_keeps_literal_counts_and_direct_rate():
    table = _state_projection(_STATE_PAGE, rda.rda_indicator1_metadata(2025, "04"))
    assert table.num_rows == 4
    assert table.to_pylist()[0] == {
        "subject": "Mathematics", "state_rate": "44.7", "state_numerator": "192,262", "state_denominator": "430,232",
    }


@pytest.mark.parametrize(("before", "after", "message"), [
    ("State Report", "District Report", "reviewed edition, area or table headings"),
    ("(i) Mathematics", "(i) Science", "subject or report layout"),
    ("44.7  192,262", "44.6  192,262", "reviewed direct Texas report"),
    ("192,262", "192262", "reviewed direct Texas report"),
])
def test_rda_state_native_refuses_mismatched_page_or_cells(before, after, message):
    text = _STATE_PAGE.replace(before, after, 1)
    with pytest.raises(ContractError, match=message):
        _state_projection(text, rda.rda_indicator1_metadata(2025, "04"))


def test_rda_native_binds_csv_order_and_statistic_names(tmp_path, monkeypatch):
    raw = b"DISTRICT,REGION,P,PA,N,D\n000001,01,.,54,12,26\n"
    metadata = _fixture_metadata(raw, 1)
    monkeypatch.setattr(rda, "rda_indicator1_metadata", lambda *_: metadata)
    path = tmp_path / "source.csv"
    path.write_bytes(raw)
    names = ("district_number", "education_service_center", "mathematics_single_year_rate",
             "mathematics_multi_year_rate", "mathematics_numerator", "mathematics_denominator")
    mapping = {"fields": [{"code": code, "field": name} for code, name in
                          zip(("DISTRICT", "REGION", "P", "PA", "N", "D"), names, strict=True)]}
    table = _district_projection(path, mapping, metadata)
    assert table.column_names == list(names)
    assert table["mathematics_single_year_rate"].to_pylist() == ["."]
    assert table["mathematics_multi_year_rate"].to_pylist() == ["54"]
    fields = mapping["fields"]
    fields[2]["field"], fields[3]["field"] = fields[3]["field"], fields[2]["field"]
    with pytest.raises(ContractError, match="subject or statistic binding"):
        _district_projection(path, mapping, metadata)
