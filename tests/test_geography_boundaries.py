from gzip import decompress
from hashlib import sha256
import json
from unittest.mock import Mock

import pytest

from teadata import geography_boundaries as boundaries
from teadata.legislative_rosters import LegislativePlan, RosterSource


def _source(tmp_path, district_ids):
    ring = [[-100, 30], [-100, 31], [-99, 31], [-99, 30], [-100, 30]]
    features = [{"type": "Feature", "properties": {"ESC_REGION": number},
                 "geometry": {"type": "Polygon", "coordinates": [ring]}} for number in district_ids]
    raw = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    digest = sha256(raw).hexdigest()
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / digest).write_bytes(raw)
    store = Mock(root=tmp_path)
    receipt = {"kind": "acquisition", "method": "download", "section": "spatial",
               "dataset": "reference:tea_esc_boundaries", "artifact": {"sha256": digest, "size_bytes": len(raw)},
               "source_metadata": {"version": "ESC2024", "publisher": "Texas Education Agency",
                                    "source_url": boundaries._ESC_URL},
               "requested_url": boundaries._ESC_URL + "/query", "downloaded_at": "2026-11-10T17:32:55Z"}
    store.verify_record_closure.return_value = {"test-receipt": receipt}
    store.read_record.return_value = receipt
    return store, RosterSource("test-receipt", digest)


def test_boundary_reader_keeps_provenance_and_negotiated_bytes_per_outline(tmp_path, monkeypatch):
    from shapely.geometry import shape

    store, source = _source(tmp_path, range(1, 21))
    artifact = tmp_path / "boundaries.sqlite3"
    pin = boundaries.build_geography_boundaries(store, sources={"ESC2024": source}, output=artifact)
    with boundaries.open_geography_boundaries(artifact, expected_sha256=pin) as reader:
        outline = reader.get("ESC2024", 1)
        compressed = reader.get("ESC2024", 1, encoding="gzip")
        headers = reader.get("ESC2024", 1, encoding="gzip", include_body=False)
        assert outline.body is not None and compressed.body is not None
        assert decompress(compressed.body) == outline.body
        assert outline.sha256 == sha256(outline.body).hexdigest()
        assert headers.sha256 == compressed.sha256 != outline.sha256
        assert headers.size_bytes == len(compressed.body) and headers.body is None
        assert outline.acquisition["receipt"] == source.receipt and outline.source_crs == "EPSG:4326"
        assert outline.acquisition["sha256"] == source.sha256 and outline.tolerance_m == 0
        document = json.loads(outline.body)
        assert document["properties"] == {"layer": "ESC2024", "number": 1}
        assert shape(document["geometry"]).exterior.is_ccw
        assert outline.relative_area_change == 0
        with pytest.raises(KeyError, match="unavailable"):
            reader.get(LegislativePlan.TEXAS_HOUSE, 1)
    with pytest.raises(ValueError, match="SHA-256"):
        with boundaries.open_geography_boundaries(artifact, expected_sha256="0" * 64):
            pytest.fail("An unpinned boundary file was accepted")

    held_artifact = tmp_path / "held.sqlite3"
    monkeypatch.setattr(boundaries, "_SIZE_LIMITS", (100, 100))
    held_pin = boundaries.build_geography_boundaries(store, sources={"ESC2024": source}, output=held_artifact)
    with boundaries.open_geography_boundaries(held_artifact, expected_sha256=held_pin) as reader:
        held = reader.get("ESC2024", 20)
        assert held.status == "held" and held.body is None and held.sha256 is None
        assert held.acquisition == outline.acquisition


@pytest.mark.parametrize("ids", [(*range(1, 20), 19), (1.5, *range(2, 21))])
def test_bad_source_geography_ids_block_the_artifact(tmp_path, ids):
    store, source = _source(tmp_path, ids)
    artifact = tmp_path / "boundaries.sqlite3"
    with pytest.raises(ValueError, match=r"publisher boundary ID"):
        boundaries.build_geography_boundaries(store, sources={"ESC2024": source}, output=artifact)
    assert not artifact.exists()
