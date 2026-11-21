"""Build bounded, independently pinned display outlines for Texas geography profiles."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from gzip import GzipFile
from hashlib import file_digest, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal
from zipfile import ZipFile

from ._sqlite import verified_database
from .geography_rosters import GeographyAcquisition, _acquisition
from .legislative_rosters import LegislativePlan, RosterSource
from .manifest import canonical_json
from .provenance import ProvenanceStore

if TYPE_CHECKING:
    from pyproj import Transformer
    from shapely.geometry.base import BaseGeometry


BoundaryLayer = LegislativePlan | Literal["ESC2024"]
BoundaryEncoding = Literal["identity", "gzip"]
_ESC = "ESC2024"
_ESC_URL = "https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/ESC2024/FeatureServer/0"
_TLC_URL = "https://data.capitol.texas.gov/dataset/"
_PLANS = dict(zip(LegislativePlan, (150, 31, 38, 38, 15), strict=True))
_SIZE_LIMITS = (128 * 1024, 48 * 1024)
_APPLICATION_ID = 0x54474244
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeographyBoundary:
    layer: BoundaryLayer
    geography_number: int
    acquisition: GeographyAcquisition
    source_crs: str
    status: Literal["available", "held"]
    encoding: BoundaryEncoding
    tolerance_m: int | None
    coordinate_decimals: int | None
    relative_area_change: float | None
    size_bytes: int | None
    sha256: str | None
    body: bytes | None


@dataclass(frozen=True, slots=True)
class _Display:
    identity: bytes
    gzip: bytes
    tolerance_m: int
    coordinate_decimals: int | None
    relative_area_change: float


def _shape(geometry: BaseGeometry) -> bool:
    return geometry.geom_type in {"Polygon", "MultiPolygon"} and not geometry.is_empty and geometry.is_valid


def _representation(geometry: BaseGeometry, layer: BoundaryLayer, number: int) -> tuple[bytes, bytes] | None:
    from shapely.geometry import mapping
    from shapely.ops import orient

    data = json.dumps({"type": "Feature", "geometry": mapping(orient(geometry, sign=1.0)),
                       "properties": {"layer": layer, "number": number}},
                      separators=(",", ":"), allow_nan=False).encode()
    if len(data) > _SIZE_LIMITS[0]:
        return None
    output = BytesIO()
    with GzipFile(fileobj=output, filename="", mode="wb", mtime=0, compresslevel=6) as stream:
        stream.write(data)
    compressed = output.getvalue()
    return (data, compressed) if len(compressed) <= _SIZE_LIMITS[1] else None


@lru_cache(maxsize=8)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    from pyproj import Transformer

    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _reproject(geometry: BaseGeometry, source_crs: str, target_crs: str) -> BaseGeometry:
    import numpy as np
    from shapely import transform

    transformer = _transformer(source_crs, target_crs)
    return transform(geometry, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])))


def _display(original: BaseGeometry, source_crs: str, layer: BoundaryLayer, number: int) -> _Display | None:
    import numpy as np
    from shapely import transform

    wgs84 = _reproject(original, source_crs, "EPSG:4326")
    west, south, east, north = wgs84.bounds
    if not _shape(wgs84) or not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError(f"Invalid WGS84 reprojection for {layer} {number}")

    projected = _reproject(original, source_crs, "EPSG:5070")
    area = projected.area
    if not _shape(projected) or area <= 0:
        raise ValueError(f"Invalid equal-area reprojection for {layer} {number}")

    def measure(candidate: BaseGeometry, tolerance: int, precision: int | None) -> _Display | None:
        if not _shape(candidate) or not (payload := _representation(candidate, layer, number)):
            return None
        measured = _reproject(candidate, "EPSG:4326", "EPSG:5070")
        if not _shape(measured):
            return None
        difference = projected.symmetric_difference(measured).area / area
        return _Display(*payload, tolerance, precision, difference) if difference <= 0.001 else None

    if display := (measure(wgs84, 0, None) or measure(transform(wgs84, lambda xy: np.round(xy, 5)), 0, 5)):
        return display
    for tolerance in (5, 25, 50, 100):
        simplified = projected.simplify(tolerance, preserve_topology=True)
        if not _shape(simplified):
            continue
        candidate = _reproject(simplified, "EPSG:5070", "EPSG:4326")
        if display := (measure(candidate, tolerance, None)
                       or measure(transform(candidate, lambda xy: np.round(xy, 5)), tolerance, 5)):
            return display
    return None


def _geometry_source(path: Path, esc: bool) -> str:
    if esc:
        return f"GeoJSON:{path}"
    with ZipFile(path) as archive:
        members = [item.filename for item in archive.infolist() if item.filename.lower().endswith(".shp")]
    if len(members) != 1:
        raise ValueError("Expected one shapefile in the retained legislative boundary archive")
    member = PurePosixPath(members[0])
    if member.is_absolute() or ".." in member.parts or "\\" in members[0]:
        raise ValueError("Unsafe retained legislative boundary member")
    return f"/vsizip/{{{path}}}/{member}"


def _layer(store: ProvenanceStore, layer: BoundaryLayer, pin: RosterSource) -> tuple[
    GeographyAcquisition, str, dict[int, BaseGeometry],
]:
    from pyogrio.raw import read
    from pyproj import CRS
    from shapely import from_wkb

    esc = layer == _ESC
    label = "ESC_REGION" if esc else "District"
    source_crs = "EPSG:4326" if esc else "EPSG:3081"
    expected = 20 if esc else _PLANS[LegislativePlan(layer)]
    dataset = "reference:tea_esc_boundaries" if esc else "reference:tlc_legislative_boundaries"
    url = _ESC_URL if esc else _TLC_URL + layer.lower()
    publisher = "Texas Education Agency" if esc else "Texas Legislative Council"
    acquisition = store.verify_record_closure([pin.receipt])[pin.receipt]
    metadata = acquisition.get("source_metadata", {})
    artifact = acquisition.get("artifact", {})
    if (acquisition["kind"] != "acquisition" or acquisition.get("method") != "download"
            or acquisition.get("dataset") != dataset or acquisition.get("section") != "spatial"
            or metadata.get("version") != layer or metadata.get("source_url") != url
            or metadata.get("publisher") != publisher or artifact.get("sha256") != pin.sha256
            or not 0 < artifact["size_bytes"] <= 128 * 1024**2):
        raise ValueError(f"Retained boundary receipt does not bind {layer} and its expected source bytes")

    path = store.root / "objects" / pin.sha256
    description, _, wkb, fields = read(_geometry_source(path.resolve(), esc), columns=[label], max_features=expected + 1)
    if not description.get("crs") or not CRS.from_user_input(description["crs"]).equals(CRS(source_crs)):
        raise ValueError(f"Unexpected publisher CRS for {layer}")
    if list(description["fields"]) != [label] or len(fields) != 1 or wkb is None:
        raise ValueError(f"Expected {label} IDs and publisher geometry in {layer}")
    identifiers = fields[0].tolist()
    if any(type(value) is not int and not (isinstance(value, str) and value.isascii() and value.isdecimal())
           for value in identifiers):
        raise ValueError(f"Invalid publisher boundary ID for {layer}")
    numbers = [int(number) for number in identifiers]
    if len(numbers) != expected or set(numbers) != set(range(1, expected + 1)):
        raise ValueError(f"Missing, repeated or unexpected publisher boundary IDs for {layer}")
    geometries = from_wkb(wkb)
    if any(geometry is None or not _shape(geometry) for geometry in geometries):
        raise ValueError(f"Invalid publisher geometry in {layer}")
    return _acquisition(store, pin.receipt), source_crs, dict(zip(numbers, geometries, strict=True))


def build_geography_boundaries(store: ProvenanceStore, *, sources: Mapping[BoundaryLayer, RosterSource],
                               output: Path) -> str:
    """Build one independently pinned display artifact from exact publisher layers."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not sources or any(layer != _ESC and not isinstance(layer, LegislativePlan) for layer in sources):
        raise ValueError("Choose ESC2024 or one or more exact legislative plans")
    with TemporaryDirectory(prefix=".geography-boundaries-", dir=output.parent) as temp:
        path = Path(temp) / "boundaries.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.executescript(f"""
                PRAGMA application_id = {_APPLICATION_ID};
                PRAGMA user_version = {_SCHEMA_VERSION};
                PRAGMA foreign_keys = ON;
                CREATE TABLE source(layer TEXT PRIMARY KEY, acquisition TEXT NOT NULL, crs TEXT NOT NULL) WITHOUT ROWID;
                CREATE TABLE boundary(layer TEXT REFERENCES source NOT NULL, number INTEGER NOT NULL,
                    tolerance_m INTEGER, coordinate_decimals INTEGER, relative_area_change REAL,
                    identity_bytes INTEGER, identity_sha TEXT, identity_body BLOB,
                    gzip_bytes INTEGER, gzip_sha TEXT, gzip_body BLOB,
                    PRIMARY KEY(layer, number)) WITHOUT ROWID;
            """)
            for layer, pin in sorted(sources.items()):
                source, crs, geometries = _layer(store, layer, pin)
                connection.execute("INSERT INTO source VALUES (?, ?, ?)", (layer, canonical_json(source), crs))
                for number, geometry in sorted(geometries.items()):
                    display = _display(geometry, crs, layer, number)
                    if display is None:
                        connection.execute("INSERT INTO boundary(layer, number) VALUES (?, ?)", (layer, number))
                        continue
                    data, zipped = display.identity, display.gzip
                    connection.execute("INSERT INTO boundary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (layer, number, display.tolerance_m, display.coordinate_decimals, display.relative_area_change,
                         len(data), sha256(data).hexdigest(), data, len(zipped), sha256(zipped).hexdigest(), zipped))
        with path.open("rb") as stream:
            published_sha = file_digest(stream, "sha256").hexdigest()
        os.link(path, output)
    return published_sha


class GeographyBoundaries:
    """Read one boundary payload or its headers without hydrating the other outlines."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, layer: BoundaryLayer, number: int, *, encoding: BoundaryEncoding = "identity",
            include_body: bool = True) -> GeographyBoundary:
        if (layer != _ESC and not isinstance(layer, LegislativePlan)) or type(number) is not int or number < 1:
            raise ValueError("Choose an exact boundary layer and positive geography number")
        if encoding not in ("identity", "gzip"):
            raise ValueError("Choose identity or gzip encoding")
        column = "identity" if encoding == "identity" else "gzip"
        body = f"{column}_body" if include_body else "NULL"
        row = self._connection.execute(f"""SELECT acquisition, crs, tolerance_m, coordinate_decimals, relative_area_change,
                {column}_bytes, {column}_sha, {body} FROM boundary JOIN source USING (layer)
                WHERE layer = ? AND number = ?""", (layer, number)).fetchone()
        if row is None:
            raise KeyError(f"Boundary {number} is unavailable for {layer}")
        acquisition, crs, tolerance, decimals, change, size, digest, payload = row
        return GeographyBoundary(layer, number, json.loads(acquisition), crs,
                                 "held" if size is None else "available", encoding, tolerance, decimals, change,
                                 size, digest, payload)


@contextmanager
def open_geography_boundaries(path: Path, *, expected_sha256: str) -> Generator[GeographyBoundaries]:
    """Verify the artifact once, then allow indexed metadata and body reads."""
    with verified_database(path, expected_sha256=expected_sha256, application_id=_APPLICATION_ID,
                           schema_version=_SCHEMA_VERSION, label="Geography boundary") as connection:
        yield GeographyBoundaries(connection)
