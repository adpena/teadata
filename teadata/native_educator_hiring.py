"""Retained-source qualification of one exact educator hiring family.

This is a verification plan, not a second materializer or data access path.
Report editions, measurement years and the dated dictionary remain distinct.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Mapping
from contextlib import ExitStack
from importlib.resources import files
import json
import math
import struct
from typing import TYPE_CHECKING, Any

from .dataset_metadata import DataPeriod
from .manifest import ContractError

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


REVIEWED_MANIFEST = "8989e9860aadd5d43bcb30bcc40aab2af59f230429638fb0723a1496ca821626"
_DISTRICT = "educator_district_new_hires"
_EPP = "educator_district_new_hires_by_epp"
_STATE = "educator_state_new_hires"
_SOURCE_ROW = "publisher_worksheet_row"


def reviewed_educator_hiring(manifest_sha256: str) -> dict[str, Any] | None:
    """Select the pinned plan without loading it for unrelated source families."""
    if manifest_sha256 != REVIEWED_MANIFEST:
        return None
    plan = json.loads(files("teadata").joinpath("reference_cases/native_educator_hiring.json").read_text())
    if plan["manifest_sha256"] != REVIEWED_MANIFEST:
        raise ContractError("Packaged educator hiring review identity differs")
    return plan


def validate_educator_hiring_identity(release: DataRelease, plan: Mapping[str, Any]) -> None:
    """Check exact metadata and routing before any observation leaf is opened."""
    catalog = release.catalog.to_dict()
    cases = plan["editions"]
    if (release.manifest_sha256 != plan["manifest_sha256"] or release.build_id != plan["build_id"]
            or catalog["schema_version"] != plan["catalog_schema_version"]
            or catalog["editions"].keys() != cases.keys()
            or catalog["artifacts"] != {case["artifact_id"]: identifier for identifier, case in cases.items()}):
        raise ContractError("Educator hiring manifest, build or edition set differs from its reviewed identity")
    for identifier, case in cases.items():
        ref = release.catalog.artifact_ref(case["artifact_id"])
        source = plan["sources"][case["key"]["dataset"]]
        acquisitions = release.catalog.acquisitions(ref.edition.dataset, edition=ref.edition)
        if (catalog["editions"][identifier] != {"key": case["key"], "metadata": case["metadata"]}
                or ref.edition.to_dict() != case["key"] or ref.identity != case["artifact_id"]
                or ref.sha256 != case["leaf_sha256"] or ref.size_bytes != case["leaf_size_bytes"]
                or ref.processing_id != case["processing_id"] or ref.contract["rows"] != case["rows"]
                or len(acquisitions) != 1 or acquisitions[0]["record_id"] != source["receipt"]
                or acquisitions[0]["artifact"] != source["artifact"]):
            raise ContractError(f"Educator hiring {ref.edition.dataset}/{case['school_year_end']} identity differs")
        period = DataPeriod("school_year", case["school_year_end"] - 1, case["school_year_end"]).to_dict()
        for field in source["fields"]:
            if field.endswith(("_count", "_ratio")):
                metadata = release.catalog.field_metadata(field, scope=ref.edition.scope, edition=ref.edition)
                if metadata is None or metadata["period"] != period:
                    raise ContractError(f"Educator hiring {field} measurement period differs")


def _same(value: Any, expected: Any, *, ratio: bool = False) -> bool:
    if ratio:
        return (type(value) is float and type(expected) in (int, float) and math.isfinite(expected)
                and struct.pack(">d", value) == struct.pack(">d", float(expected)))
    return type(value) is type(expected) and value == expected


def _check_row(native: Mapping[str, Any], raw: tuple, ordinal: int, source: Mapping[str, Any]) -> None:
    fields = source["fields"]
    numeric = [name for name in fields if name.endswith(("_count", "_ratio"))]
    expected_columns = {*fields, _SOURCE_ROW, *(f"raw_{name}" for name in numeric),
                        *(f"status_{name}" for name in numeric)}
    if (len(raw) != len(fields) or native.keys() != expected_columns
            or type(native[_SOURCE_ROW]) is not int or native[_SOURCE_ROW] != ordinal):
        raise ContractError(f"Educator hiring publisher row {ordinal} width, fields or coordinate differs")
    for name, expected in zip(fields, raw, strict=True):
        ratio = name.endswith("_ratio")
        if name in numeric:
            allowed = (int, float) if ratio else (int,)
            if (type(expected) not in allowed or not math.isfinite(expected)
                    or not (0 <= expected <= 1 if ratio else expected >= 0)
                    or native[f"raw_{name}"] != str(expected) or native[f"status_{name}"] != "observed"):
                raise ContractError(f"Educator hiring publisher row {ordinal}/{name} raw value or status differs")
        if not _same(native[name], expected, ratio=ratio):
            raise ContractError(f"Educator hiring publisher row {ordinal}/{name} value or type differs")


def _native_rows(release: DataRelease, case: Mapping[str, Any]) -> Generator[dict[str, Any], None, None]:
    dataset = release.analytical_dataset(case["artifact_id"])
    for batch in dataset.to_batches(batch_size=512, use_threads=False):
        yield from batch.to_pylist()


def _cross_checks(district: Mapping[tuple[int, str, str], int], epp: Mapping[tuple[int, str, str], int],
                  state: Mapping[int, Mapping[str, Any]], categories: list) -> dict[str, Any]:
    if district.keys() != epp.keys() or any(value != epp[key] for key, value in district.items()):
        raise ContractError("Educator hiring district/EPP keys or counts differ, including zero keys")
    totals: Counter[tuple[int, str]] = Counter()
    for (year, _, category), count in district.items():
        totals[year, category] += count
    for year, row in state.items():
        total = row["total_new_hires_count"]
        if total <= 0 or sum(totals[year, category] for _, category, _ in categories) != total:
            raise ContractError(f"Educator hiring {year} district total differs from direct Texas")
        for _, category, name in categories:
            count = row[name + "_count"]
            if (totals[year, category] != count or not _same(row[name + "_ratio"], count / total, ratio=True)):
                raise ContractError(f"Educator hiring {year}/{category} count or ratio differs from direct Texas")
    return {"district_epp_years": sorted({key[0] for key in district}),
            "district_epp_keys": len(district), "direct_texas_years": sorted(state),
            "categories_per_direct_texas_year": len(categories), "zero_keys_preserved": True}


def qualify_educator_hiring(release: DataRelease, processing_store: ProvenanceStore,
                            publisher_store: ProvenanceStore, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Replay publisher cells once per workbook, with bounded native-row batches."""
    from openpyxl import load_workbook

    validate_educator_hiring_identity(release, plan)
    release.verify()
    processing = processing_store.verify_record_closure([case["processing_id"] for case in plan["editions"].values()])
    publisher = publisher_store.verify_record_closure(plan["publisher_roots"])
    district: dict[tuple[int, str, str], int] = {}
    epp: Counter[tuple[int, str, str]] = Counter()
    state: dict[int, Mapping[str, Any]] = {}
    examples = {(case["dataset"], case["worksheet_row"]): case for case in plan["goldens"]}
    observed_examples = set()
    reports = {}
    for dataset, source in plan["sources"].items():
        record = publisher_store.read_record(source["receipt"])
        if (record["artifact"] != source["artifact"] or record["artifact"]["sha256"] != source["sha256"]
                or record.get("requested_url") != source["url"] or record.get("resolved_url") != source["url"]
                or record.get("method") != source["method"] or record.get("downloaded_at") != source["downloaded_at"]):
            raise ContractError(f"Educator hiring {dataset} publisher acquisition differs")
        cases = {case["school_year_end"]: case for case in plan["editions"].values()
                 if case["key"]["dataset"] == dataset}
        counts: Counter[int] = Counter()
        labels = {f"{year - 1}-{year % 100:02d}": year for year in cases}
        skipped = 0
        with ExitStack() as resources:
            raw_file = resources.enter_context(publisher_store.verify_artifact(source["sha256"]).open("rb"))
            book = load_workbook(raw_file, read_only=True, data_only=False, keep_links=False)
            resources.callback(book.close)
            raw_rows = iter(book[source["worksheet"]].values)
            if next(raw_rows, None) != tuple(source["headers"]):
                raise ContractError(f"Educator hiring {dataset} literal worksheet header differs")
            native = {year: _native_rows(release, case) for year, case in cases.items()}
            for iterator in native.values():
                resources.callback(iterator.close)
            for ordinal, raw in enumerate(raw_rows, 2):
                year = labels.get(raw[0]) if dataset == _STATE else raw[1]
                if type(year) is not int or year not in cases:
                    skipped += 1
                    continue
                row = next(native[year], None)
                if row is None:
                    raise ContractError(f"Educator hiring {dataset}/{year} native rows end before publisher row {ordinal}")
                _check_row(row, raw, ordinal, source)
                counts[year] += 1
                if dataset in (_DISTRICT, _EPP):
                    key = year, row["district_number"], row["certification_at_hire_category"]
                    if dataset == _DISTRICT:
                        if key in district:
                            raise ContractError("Educator hiring duplicate district/year/category key")
                        district[key] = row["new_hires_count"]
                    else:
                        epp[key] += row["new_hires_count"]
                else:
                    if year in state:
                        raise ContractError("Educator hiring duplicate direct Texas year")
                    state[year] = row
                example = examples.get((dataset, ordinal))
                if example is not None:
                    raw_values = dict(zip(source["fields"], raw, strict=True))
                    if any(not _same(row[name], expected) or not _same(raw_values[name], expected)
                           for name, expected in example["facts"].items()):
                        raise ContractError(f"Educator hiring publisher example {dataset}/{ordinal} differs")
                    observed_examples.add((dataset, ordinal))
            for year, iterator in native.items():
                if next(iterator, None) is not None or counts[year] != cases[year]["rows"]:
                    raise ContractError(f"Educator hiring {dataset}/{year} native row coverage differs")
        reports[dataset] = {"selected_source_rows": sum(counts.values()), "rows_by_measurement_year": dict(counts),
                            "excluded_source_rows": skipped, "acquisition_id": source["receipt"],
                            "artifact_sha256": source["sha256"], "method": source["method"],
                            "downloaded_at": source["downloaded_at"]}
    if observed_examples != examples.keys():
        raise ContractError("Educator hiring publisher example coverage is incomplete")
    comparisons = _cross_checks(district, epp, state, plan["category_mapping"])
    return {"status": "PASS_REVIEWED_NATIVE", "build_id": release.build_id,
            "manifest_sha256": release.manifest_sha256, "source_parity": reports,
            "cross_product_comparisons": comparisons, "publisher_source_examples": len(observed_examples),
            "processing_records": len(processing), "publisher_records": len(publisher),
            "limits": plan["limits"], "qualification_scope": "pinned_native_family"}
