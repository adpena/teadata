"""Read reviewed Indicator 1 Results Driven Accountability observations."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.csv as csv

from .manifest import ContractError, canonical_json
from .values import ValueStatus

_RESOURCE = "methodologies/rda-indicator-1.json"
_RATE = pa.decimal128(4, 1)
pc: Any = compute


@cache
def _editions() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath(_RESOURCE).read_bytes())["editions"]


def rda_indicator1_metadata(edition: int, area: str) -> dict[str, Any]:
    """Describe one reviewed area, including the direct Texas report values.

    Area codes are TEA's two-character selectors: 01 for bilingual education,
    03 for other special populations, and 04 for special education.
    """
    if type(edition) is not int or not isinstance(area, str):
        raise ContractError("Select an exact RDA monitoring year and two-character area code")
    try:
        entry = _editions()[str(edition)]
        selection = entry["areas"][area]
    except KeyError as error:
        raise ContractError(f"RDA Indicator 1 edition {edition}, area {area!r} is not reviewed") from error
    return deepcopy({
        "edition": edition, "area": area,
        "test_administration": entry["test_administration"],
        "references": entry["references"],
        "district_report_witnesses": entry["district_report_witnesses"],
        **selection,
    })


def _verified_bytes(path: Path, source: dict[str, Any]) -> bytes:
    size = source["size_bytes"]
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or sha256(raw).hexdigest() != source["sha256"]:
        raise ContractError("RDA district CSV does not match the reviewed publisher bytes")
    return raw


def _numeric(raw: pa.ChunkedArray, pattern: str, data_type: pa.DataType,
             tokens: tuple[str, ...]) -> pa.ChunkedArray:
    match = pc.match_substring_regex(raw, pattern)
    if pc.any(pc.invert(pc.or_(match, pc.is_in(raw, value_set=pa.array(tokens))))).as_py():
        raise ContractError("Unreviewed RDA Indicator 1 value token")
    value = pc.cast(pc.if_else(match, raw, None), data_type)
    if data_type == _RATE and pc.any(pc.greater(value, 100)).as_py():
        raise ContractError("RDA Indicator 1 rate exceeds 100 percent")
    return value


def _status(value: pa.ChunkedArray, other: str | pa.ChunkedArray) -> pa.ChunkedArray:
    return pc.if_else(pc.is_valid(value), ValueStatus.OBSERVED.value, other)


def _check_reported_rate(rate: pa.ChunkedArray, numerator: pa.ChunkedArray, denominator: pa.ChunkedArray) -> None:
    complete = pc.and_(pc.is_valid(rate), pc.is_valid(numerator))
    n = pc.filter(numerator, complete)
    d = pc.filter(denominator, complete)
    p = pc.cast(pc.multiply(pc.filter(rate, complete), 10), pa.int64())
    if pc.any(pc.or_(pc.less_equal(d, 0), pc.greater(n, d))).as_py():
        raise ContractError("Invalid RDA Indicator 1 numerator or denominator")
    rounded = pc.divide_checked(pc.add_checked(pc.multiply_checked(n, 2000), d),
                               pc.multiply_checked(d, 2))
    if pc.any(pc.not_equal(p, rounded)).as_py():
        raise ContractError("RDA single-year rate differs from the published counts")


def _subject(table: pa.Table, field: dict[str, Any]) -> pa.Table:
    raw_rate = table[field["single_year_rate_code"]]
    raw_n = table[field["numerator_code"]]
    raw_d = table[field["denominator_code"]]
    rate = _numeric(raw_rate, r"^[0-9]{1,3}\.[0-9]$", _RATE, (".", "-1.0"))
    n = _numeric(raw_n, r"^[0-9]+$", pa.int64(), ("-1",))
    d = _numeric(raw_d, r"^[0-9]+$", pa.int64(), ("-1",))
    if pc.any(pc.not_equal(pc.is_null(n), pc.is_null(d))).as_py():
        raise ContractError("RDA numerator and denominator disclosure states differ")
    _check_reported_rate(rate, n, d)

    multi_code = field["multi_year_rate_code"]
    raw_multi = table[multi_code] if multi_code else pa.chunked_array([pa.nulls(table.num_rows, type=pa.string())])
    multi = (_numeric(raw_multi, r"^[0-9]{1,3}(?:\.[0-9])?$", _RATE, (".",))
             if multi_code else pc.cast(raw_multi, _RATE))
    period = pc.equal(raw_rate, ".")
    if pc.any(pc.not_equal(period, pc.is_valid(multi))).as_py():
        raise ContractError("RDA single-year omission and separately published multi-year rate do not reconcile")
    rate_unobserved = pc.if_else(period, ValueStatus.MISSING.value, ValueStatus.SUPPRESSED.value)
    return pa.table({
        "district_number": table["DISTRICT"],
        "education_service_center": table["REGION"],
        "subject": pa.repeat(pa.scalar(field["subject"]), table.num_rows),
        "single_year_rate": rate,
        "single_year_rate_raw": raw_rate,
        "single_year_rate_status": _status(rate, rate_unobserved),
        "single_year_rate_reason": pc.if_else(period, "publisher_reports_multi_year_rate", None),
        "numerator": n,
        "numerator_raw": raw_n,
        "denominator": d,
        "denominator_raw": raw_d,
        "counts_status": _status(n, ValueStatus.SUPPRESSED.value),
        "multi_year_rate": multi,
        "multi_year_rate_raw": raw_multi,
        "multi_year_rate_status": _status(multi, ValueStatus.MISSING.value),
    })


def _project(raw: bytes, metadata: dict[str, Any]) -> pa.Table:
    columns = ["DISTRICT", "REGION"]
    for field in metadata["fields"]:
        columns.extend(field[key] for key in ("single_year_rate_code", "numerator_code", "denominator_code"))
        if field["multi_year_rate_code"] is not None:
            columns.append(field["multi_year_rate_code"])
    table = csv.read_csv(
        pa.BufferReader(raw),
        read_options=csv.ReadOptions(encoding="windows-1252"),
        convert_options=csv.ConvertOptions(
            include_columns=columns, column_types=dict.fromkeys(columns, pa.string()),
            strings_can_be_null=False, null_values=[],
        ),
    )
    ids = table["DISTRICT"]
    if (table.num_rows != metadata["district_rows"] or pc.count_distinct(ids).as_py() != table.num_rows
            or pc.any(pc.equal(ids, "000000")).as_py()
            or not pc.all(pc.match_substring_regex(ids, r"^[0-9]{6}$")).as_py()
            or not pc.all(pc.match_substring_regex(table["REGION"], r"^(?:0[1-9]|1[0-9]|20)$")).as_py()):
        raise ContractError("RDA district row count or identifiers differ from the reviewed scope")
    return pa.concat_tables([_subject(table, field) for field in metadata["fields"]]).replace_schema_metadata({
        "teadata.rda_indicator1": canonical_json(metadata),
    })


def read_rda_indicator1(path: str | Path, *, edition: int, area: str) -> pa.Table:
    """Read a verified original TEA district CSV, one row per district and subject.

    Single-year and pooled rates have separate fields. The single-year value
    remains absent where TEA only publishes a pooled rate. A null raw pooled
    value indicates the source schema omits that column.
    """
    metadata = rda_indicator1_metadata(edition, area)
    raw = _verified_bytes(Path(path), metadata["district_csv"])
    try:
        return _project(raw, metadata)
    except pa.ArrowInvalid as error:
        raise ContractError("RDA district CSV does not satisfy the reviewed schema") from error
