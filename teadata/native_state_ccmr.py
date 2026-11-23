"""Direct Texas annual-graduate CCMR totals from TEA accountability reports."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path
import re
from shutil import which
from subprocess import check_output
from typing import TYPE_CHECKING, Any

from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .native_witnesses import _Rows

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


DATASET = "state_accountability_ccmr"
FIELDS = ("annual_graduates", "ccmr_credit_displayed", "ccmr_percent_reported")
ROW_LABELS = ("Total graduates", "Total credit for CCMR criteria")
EDITIONS = (2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
_COUNT = r"(?:0|[1-9][0-9]{0,2})(?:,[0-9]{3})*"


@lru_cache(maxsize=1)
def state_ccmr_reference() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath("reference_cases/state_accountability_ccmr.json").read_bytes())


def _pdf_page(path: Path, page: int) -> str:
    tool = which("pdftotext")
    if tool is None:
        raise ContractError("State CCMR PDF qualification requires Poppler pdftotext")
    return check_output([tool, "-layout", "-enc", "UTF-8", "-f", str(page), "-l", str(page), str(path), "-"],
                        encoding="utf-8", timeout=15)


def _tokens(raw: tuple[str, str, str]) -> tuple[int, Decimal, int]:
    count, credit, percent = raw
    if (re.fullmatch(_COUNT, count) is None or re.fullmatch(_COUNT + r"(?:\.[05])?", credit) is None
            or re.fullmatch(r"(?:100|[1-9]?[0-9])%", percent) is None):
        raise ContractError("State CCMR report contains an unexpected display token")
    d, n, rate = int(count.replace(",", "")), Decimal(credit.replace(",", "")), int(percent[:-1])
    if not 0 <= n <= d or d == 0:
        raise ContractError("State CCMR report values are outside its annual graduate population")
    return d, n, rate


def state_ccmr_source_labels(edition: int) -> tuple[str, str, str]:
    if edition == 2018:
        return (*ROW_LABELS, "Percent")
    return (*ROW_LABELS, "Percentage" if edition <= 2022 else "Component Score")


def _html_total(path: Path, edition: int, graduate_year: int) -> tuple[str, str, str]:
    rows = _Rows(path.read_text(encoding="iso-8859-1")).rows
    if rows[1:3] != [[f"{edition} College, Career, and Military Readiness"], ["State"]]:
        raise ContractError("State CCMR report has an unexpected edition or geography")
    total = [i for i, row in enumerate(rows) if row == ["Total"]]
    if len(total) != 1 or not 2 <= total[0] < len(rows) - 2:
        raise ContractError("State CCMR report needs one Total section")
    i = total[0]
    graduate_label, credit_label, heading = state_ccmr_source_labels(edition)
    graduates, credit = rows[i + 1:i + 3]
    if (rows[i - 2:i] != [["", f"{graduate_year} Annual Graduates"], ["", "CountCredit", heading]]
            or len(graduates) != 3 or len(credit) != 3 or graduates[0] != graduate_label
            or graduates[-1] != "" or credit[0] != credit_label):
        raise ContractError("State CCMR report has unexpected graduate year, labels or columns")
    return graduates[1], credit[1], credit[2]


def _publisher_path(store: ProvenanceStore, receipt: str, digest: str, url: str, edition: int) -> Path:
    record = store.read_record(receipt)
    if (record["kind"] != "acquisition" or record["method"] != "download" or not record["downloaded_at"]
            or record["resolved_url"] != url or record["artifact"]["sha256"] != digest
            or record["source_metadata"]["version"] != str(edition)
            or record["source_metadata"]["publisher"] != "Texas Education Agency"):
        raise ContractError("State CCMR publisher acquisition differs from its exact pin")
    return store.verify_artifact(digest)


def read_state_ccmr(store: ProvenanceStore, row: dict[str, Any], method: dict[str, str]) -> tuple[str, str, str]:
    year, graduates = row["rating_edition"], row["annual_graduates_year_end"]
    if year not in EDITIONS or graduates != year - 1:
        raise ContractError("State CCMR source period is outside the reviewed editions")
    path = _publisher_path(store, row["source_receipt"], row["source_sha256"], row["direct_report"], year)
    if year == 2018:
        adopted = _pdf_page(_publisher_path(store, method["receipt"], method["sha256"], method["url"], year), 4)
        if "2018 Accountability Manual" not in adopted or "Number of 2017 Annual Graduates" not in adopted:
            raise ContractError("2018 accountability manual does not bind the 2017 graduate period")
        report = _pdf_page(path, 3)
        if re.search(r"2018 College, Career, and Military Readiness Data Table\s+State\s+Annual Graduates\s+"
                     r"Count/Credit\s+Percent\s+Total\b", report) is None:
            raise ContractError("2018 State CCMR report edition, scope or headers differ")
        matches = re.findall(r"(?m)^\s*Total graduates\s+(\S+)\s+100%\s*\n"
                             r"\s*Total credit for CCMR criteria\s+(\S+)\s+(\S+)", report)
        if len(matches) != 1:
            raise ContractError("2018 State CCMR report needs one Total pair")
        raw = matches[0]
    else:
        raw = _html_total(path, year, graduates)
    _tokens(raw)
    if raw != tuple(row[key] for key in ("total_graduates_display", "ccmr_credit_display", "rate_display")):
        raise ContractError(f"State CCMR {year} publisher display differs from the retained witness")
    return raw


def state_ccmr_table(raw: tuple[str, str, str]):
    import pyarrow as pa

    numeric = _tokens(raw)
    columns = {name: pa.array([value], type=dtype)
               for name, value, dtype in zip(FIELDS, numeric, (pa.int64(), pa.decimal128(12, 1), pa.int64()), strict=True)}
    fields = {}
    for name, token in zip(FIELDS, raw, strict=True):
        raw_name, status = name + "_raw", name + "_status"
        columns.update({raw_name: pa.array([token]), status: pa.array(["observed"])})
        fields[name] = {"raw": raw_name, "status": status}
    return pa.table(columns), {"fields": fields, "context": []}


def qualify_state_ccmr(release: DataRelease, processing_store: ProvenanceStore,
                       publisher_store: ProvenanceStore) -> dict[str, Any]:
    reference = state_ccmr_reference()
    if not release.offline or release.manifest_sha256 != reference.get("manifest_sha256"):
        raise ContractError("State CCMR qualification requires the pinned offline candidate")
    expected = {str(row["rating_edition"]): row for row in reference["reports"]}
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    if (len(refs) != len(expected) or {ref.edition.edition for ref in refs} != expected.keys()
            or any((ref.edition.dataset, ref.edition.scope, ref.edition.table) != (DATASET, "state", "state")
                   or ref.contract["rows"] != 1 or ref.contract["key_columns"] != [] for ref in refs)):
        raise ContractError("State CCMR manifest has a different family, edition or grain")
    processing_store.verify_record_closure([ref.processing_id for ref in refs])
    for ref in refs:
        row = expected[ref.edition.edition]
        raw = read_state_ccmr(publisher_store, row, reference["method_2018"])
        table, support = state_ccmr_table(raw)
        actual = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        graduate_year = row["annual_graduates_year_end"]
        period = DataPeriod("school_year", graduate_year - 1, graduate_year)
        metadata = release.catalog.metadata(DATASET, edition=ref.edition)
        if (list(ref.contract["inputs"]) != [row["source_sha256"]]
                or ref.contract.get("column_support") != support or metadata.period != period
                or tuple(metadata.metrics[name].period for name in FIELDS) != (period,) * len(FIELDS)
                or not actual.equals(table, check_metadata=True)):
            raise ContractError(f"State CCMR {ref.edition.edition} leaf differs from the published display")
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "report_editions": sorted(expected), "compared_values": 3 * len(refs),
            "limits": "Published display precision. Cross-edition comparability is not established."}
