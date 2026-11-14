"""Opt-in publisher goldens through an explicitly pinned normal entity release.

The portable suite skips this test without both environment inputs. The source
release gate runs it against its complete retained normal bundle. Expectations
come from the reviewed SPP workbooks and independent HTML witnesses in
``reference_cases/native_staff_spp.json``; this test does not reacquire TEA data
or replace that source's full raw replay and semantic qualification.
"""

from decimal import Decimal
import json
import os
from uuid import UUID

import pytest

from teadata.entities import District
from teadata.engine import DataEngine
from teadata.manifest import ContractError
from teadata.release import DataRelease


_SPP = "spp_timely_initial_evaluation"
_PIN = "c7a02280b6228040552cb895da02a6fb9f9773887fc96e7121a257dc40adf95b"
_RATE = "timely_initial_evaluation_reported_percent"
_CONSENT = "initial_evaluation_consent_count"
_TIMELY = "timely_initial_evaluation_count"
_TARGET = "timely_initial_evaluation_met_target"


def test_publisher_spp_goldens_through_the_normal_lazy_release(tmp_path, monkeypatch):
    manifest = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_MANIFEST")
    pin = os.environ.get("TEADATA_NORMAL_SOURCE_GOLDEN_SHA256")
    if manifest is None and pin is None:
        pytest.skip("Set both TEADATA_NORMAL_SOURCE_GOLDEN_MANIFEST and _SHA256 for retained-source proof")
    assert manifest and pin, "A retained normal-source golden requires an explicit manifest and trusted SHA-256"

    monkeypatch.setattr("teadata.release._download", lambda *a, **kw: pytest.fail("Implicit source download"))
    release = DataRelease(manifest, expected_sha256=pin, offline=True)
    child, = (ref for ref in release.catalog.native_sources if ref.manifest_sha256 == _PIN)
    assert {(key.edition, key.scope) for key in child.editions} == {("2025", "district"), ("2026", "district")}

    # These values were selected before the new parent was built. In particular,
    # observed0, publisher NR and suppressed-999 are different source states.
    expected = {
        "028903": (Decimal("0.00"), "0.0", "observed", False, "NO"),
        # A rounded display of100.0 does not replace the publisher's NO flag.
        "101912": (Decimal("100.00"), "100.0", "observed", False, "NO"),
        "108804": (Decimal("100.00"), "100.0", "observed", True, "YES"),
        "214902": (None, "NR", "missing", None, "NR"),
        "246913": (Decimal("91.50"), "91.5", "observed", False, "NO"),
    }
    fields = (_RATE, _CONSENT, _TIMELY, _TARGET)
    historical = release.observations(_SPP, *fields, scope="district", school_year="2023-24")
    table = historical.query("district_number in @ids", params={"ids": list(expected)}).to_arrow()
    actual = {row["district_number"]: row for row in table.to_pylist()}
    assert set(actual) == set(expected), "Publisher rows must not disappear through a current-entity roster join"
    for identifier, (value, raw, status, target, target_raw) in expected.items():
        row = actual[identifier]
        assert (row[_RATE], row[f"spp_raw_{_RATE}"], row[f"spp_status_{_RATE}"]) == (value, raw, status)
        assert (row[_TARGET], row[f"spp_raw_{_TARGET}"]) == (target, target_raw)
    assert actual["028903"][_TIMELY] == 0
    assert actual["028903"][f"spp_status_{_TIMELY}"] == "observed"
    for field in (_CONSENT, _TIMELY):
        assert actual["108804"][field] is None
        assert actual["108804"][f"spp_raw_{field}"] == "-999"
        assert actual["108804"][f"spp_status_{field}"] == "suppressed"

    # The workbook rounds this rate to99.6; its independent HTML displays99.62.
    # Keep the workbook observation and the qualified display distinction;
    # neither a quotient nor the HTML display silently replaces the source.
    latest = release.observations(_SPP, _RATE, scope="district")
    austin = latest.query("district_number == @id", params={"id": "227901"})
    austin_row, = austin.to_arrow().to_pylist()
    assert (austin_row[_RATE], austin_row[f"spp_raw_{_RATE}"]) == (Decimal("99.60"), "99.6")
    source = latest.provenance().to_dict()
    assert source["edition"]["edition"] == "2026"
    assert source["selector"]["measurement_period"] == {
        "kind": "school_year", "start_year": 2024, "end_year": 2025,
    }

    # Real parent attachment, integer-ID normalization and namespace access.
    entity = District(
        id=UUID(int=28903), name="Source golden identity",
        district_number=28903,  # ty: ignore[invalid-argument-type]
    )
    engine = DataEngine()
    engine.catalog = release.catalog
    engine.add_district(entity)
    engine.bind_release(release)
    observed = entity.dataset(_SPP).at("2023-24").select(_RATE).to_arrow().to_pylist()
    assert len(observed) == 1 and observed[0][_RATE] == Decimal("0.00")
    direct = release.for_entity(entity, edition="2025").dataset(_SPP)
    assert direct.timely_initial_evaluation_reported_percent == Decimal("0.00")

    series = release.series(_SPP, _RATE, scope="district", years=[24, 25], ids=[101912])
    assert [(row["measurement_school_year"], row[_RATE]) for row in series.to_arrow().to_pylist()] == [
        (2024, Decimal("100.00")), (2025, Decimal("99.97")),
    ]
    descriptor = series.provenance()
    assert len(descriptor["sources"]) == 2
    assert all(item["provenance"]["release"]["manifest_sha256"] == _PIN for item in descriptor["sources"])
    export = tmp_path / "spp-series.csv"
    series.to_csv(export)
    assert "100.00" in export.read_text() and "99.97" in export.read_text()
    # CSV's documented contract keeps provenance as a separate JSON descriptor.
    # A CSV file alone cannot carry its own worksheet or schema metadata.
    sidecar = tmp_path / "spp-series.provenance.json"
    sidecar.write_text(json.dumps(descriptor))
    assert json.loads(sidecar.read_text()) == descriptor

    with pytest.raises(ContractError, match="Available years|available years"):
        release.observations(_SPP, _RATE, scope="district", school_year=2026)
    for scope in ("campus", "state"):
        with pytest.raises(ContractError, match="unavailable|Unavailable"):
            release.observations(_SPP, _RATE, scope=scope, school_year=2025)
