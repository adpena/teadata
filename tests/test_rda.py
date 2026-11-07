from decimal import Decimal
from hashlib import sha256
import json

import pytest

from teadata import rda
from teadata.manifest import ContractError


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
