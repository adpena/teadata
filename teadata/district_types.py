"""Project historical TEA district classifications with exact annual provenance."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow as pa

    from .dataset_metadata import DatasetMetadata
    from .provenance import ProvenanceStore


DISTRICT_TYPE_REFERENCE = "district_type_roster"
DISTRICT_TYPE_COLUMNS = ("tea_district_type_code", "tea_is_charter_lea")


def reviewed_district_types(metadata: DatasetMetadata, store: ProvenanceStore,
                            records: Mapping[str, Any]) -> dict[str, str] | None:
    """Load the exact same-year publisher roster when the edition binds one."""
    from .native_witnesses import _Rows

    references = [ref for ref in metadata.references if ref.reference_id == DISTRICT_TYPE_REFERENCE]
    if not references:
        return None
    ref, = references
    period = metadata.period
    receipt, = ref.provenance_ids
    record = records[receipt]
    artifact = record["artifact"]
    year = period.end_year if period is not None else None
    if (ref.status != "available" or ref.url != record["resolved_url"]
            or type(year) is not int or record["requested_year"] != record["resolved_year"] or record["resolved_year"] != year
            or record["dataset"] != "reference:tea_historical_district_type"
            or record["kind"] != "acquisition" or record["method"] != "download"
            or ref.reviewed_sha256 != artifact["sha256"] or not 0 < artifact["size_bytes"] <= 4 * 1024**2):
        raise ValueError("Historical district type receipt differs from the exact annual reference")
    with store.verify_artifact(artifact["sha256"]).open("rb") as stream:
        raw = stream.read(artifact["size_bytes"] + 1)
    if len(raw) != artifact["size_bytes"] or sha256(raw).hexdigest() != artifact["sha256"]:
        raise ValueError("Historical district type roster changed during the read")
    html = raw.decode("utf-8-sig")
    if f"Texas Public School Districts Categorized by Type, {year - 1}-{year % 100:02}" not in html:
        raise ValueError("Historical district type roster has a different school year")
    rows = _Rows(html).rows
    header = ["NAME", "DISTRICT", "TYPE", "DESCRIPTION"]
    if rows.count(header) != 1:
        raise ValueError("Historical district type roster has a different header")
    roster = {}
    for row in rows[rows.index(header) + 1:]:
        if len(row) != 4 or not any(row[1:]):
            continue
        _, identifier, code, description = row
        if (not re.fullmatch(r"[0-9]{6}", identifier) or identifier == "000000" or identifier in roster
                or not re.fullmatch(r"[A-I]", code) or (code == "I") != (description == "CHARTER SCHOOL DISTRICTS")):
            raise ValueError("Historical district type roster has an invalid or duplicate classification")
        roster[identifier] = code
    if not roster:
        raise ValueError("Historical district type roster is empty")
    return roster


def project_district_types(identifiers: pa.ChunkedArray, roster: Mapping[str, str]) -> tuple[pa.ChunkedArray, pa.ChunkedArray]:
    """A one-to-one lookup leaves unknown classifications null."""
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    codes = pc.take(pa.array(tuple(roster.values()), type=pa.string()),
                    pc.index_in(identifiers, value_set=pa.array(tuple(roster), type=pa.string())))
    return codes, pc.equal(codes, "I")
