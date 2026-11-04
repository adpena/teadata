"""Replay the reviewed 2024–25 statewide TAPR readiness selections offline."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from teadata.native_witnesses import _Rows
from teadata.provenance import ProvenanceStore
from teadata.tapr_dictionary import extract_tapr_dictionary
from teadata.teadata_config import load_config

DATASET = "state_tapr_college_readiness_components"
GRAD1_DATASET = "state_tapr_college_career_military_readiness"
EDITION = 2025
GROUPS = {
    "A": "State", "B": "AfricanAmerican", "H": "Hispanic", "W": "White",
    "I": "AmericanIndian", "3": "Asian", "4": "PacificIslander", "2": "Two orMoreRaces",
    "S": "SpecialEd", "E": "EconDisadv", "L": "EB/EL",
}
MEASURES = {
    "0GY": "AP / IB Met Criteria in Any Subject (Annual Graduates)",
    "0GJ": "Associate Degree (Annual Graduates)",
    "0GUA": "Dual Course Credits in Any Subject (Annual Graduates)",
    "0GZ": "Onramps Course Credits (Annual Graduates)",
    "DGG": "Dual Course Credits or Onramps Course Credits (Annual Graduates)",
}


def verify_rates(values: dict[str, str], rows: list[list[str]], *, measures: dict[str, str] = MEASURES) -> int:
    heading = ["2024-25 College, Career, and Military Readiness (CCMR)"]
    if (rows.count(heading) != 1 or rows[rows.index(heading) + 1:rows.index(heading) + 2] != [["State"]]
            or rows.count(["AcademicYear", *GROUPS.values()]) != 1):
        raise ValueError("Unexpected statewide CCMR report edition, scope, or groups")
    for measure, label in measures.items():
        if rows.count([label]) != 1:
            raise ValueError(f"Expected one statewide CCMR heading for {measure}")
        observed = rows[rows.index([label]) + 1:rows.index([label]) + 2]
        expected = ["2023-24"]
        for group in GROUPS:
            stem = f"S{group}{measure}24"
            numerator = int(values[stem + "N"])
            denominator = int(values[f"S{group}0GP24D"])
            rate = Decimal(values[stem + "R"])
            if denominator <= 0 or not 0 <= numerator <= denominator:
                raise ValueError(f"Invalid statewide count and denominator for {stem}")
            rounded = (Decimal(numerator) * 100 / denominator).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            if rounded != rate:
                raise ValueError(f"Statewide count and rate disagree for {stem}")
            expected.append(f"{rate:.1f}%")
        if observed != [expected]:
            raise ValueError(f"Statewide report rates differ for {measure}")
    if "DGG" in measures:
        for group in GROUPS:
            dual, onramps, union = (int(values[f"S{group}{measure}24N"])
                                     for measure in ("0GUA", "0GZ", "DGG"))
            if not max(dual, onramps) <= union <= dual + onramps:
                raise ValueError(f"Invalid statewide dual-credit or OnRamps count for {group}")
    return len(measures) * len(GROUPS)


def _verify_source(store: ProvenanceStore, dataset: str, measures: dict[str, str]) -> dict[str, int]:
    cfg = load_config()
    metadata = cfg.resolve_dataset(dataset, EDITION, exact=True, reader_profile="source").metadata
    refs = {ref.reference_id: ref for ref in metadata.references if ref.provenance_ids}
    records = store.verify_record_closure([identifier for ref in refs.values() for identifier in ref.provenance_ids])

    def original(role: str) -> Path:
        ref = refs[role]
        identifier, = ref.provenance_ids
        record = records[identifier]
        digest = record["artifact"]["sha256"]
        if ref.reviewed_sha256 != digest or ref.url != record["requested_url"]:
            raise ValueError(f"Statewide TAPR reference does not match its acquisition: {role}")
        return store.verify_artifact(digest)

    with TemporaryDirectory(prefix="teadata-state-readiness-") as temporary:
        request, = cfg.tapr_source_requests(dataset, EDITION, input_dir=temporary)
        raw = original("csv_headers")
        copyfile(raw, request.path)
        source, = cfg.read_tapr_sources(dataset, EDITION, input_dir=temporary)
    with raw.open(encoding="cp1252", newline="") as stream:
        labels, codes, values = csv.reader(stream, strict=True)
    fields = request.mapping["fields"]
    if (labels != [field["label"] for field in fields] or codes != [field["code"] for field in fields]
            or source.table.to_pylist() != [dict(zip(source.table.column_names, values, strict=True))]):
        raise ValueError("Statewide TAPR source reader differs from the original CSV")
    occurrences = extract_tapr_dictionary(original("dictionary").read_bytes(), encoding="cp1252")
    if [item.column_record() for item in occurrences] != request.mapping["dictionary_records"]:
        raise ValueError("Statewide TAPR dictionary differs from the reviewed occurrence mapping")
    rows = _Rows(original("state_ccmr_report").read_text(encoding="cp1252")).rows
    return {
        "raw_cells": len(values),
        "dictionary_occurrences": len(occurrences),
        "displayed_rates": verify_rates(dict(zip(codes, values, strict=True)), rows, measures=measures),
    }


def verify_state_grad2(store: ProvenanceStore) -> dict[str, int]:
    return _verify_source(store, DATASET, MEASURES)


def verify_state_grad1(store: ProvenanceStore) -> dict[str, int]:
    return _verify_source(store, GRAD1_DATASET, {"0GP": "College, Career, or Military Ready (Annual Graduates)"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--family", choices=["GRAD1", "GRAD2"], default="GRAD2")
    args = parser.parse_args()
    verifier = verify_state_grad1 if args.family == "GRAD1" else verify_state_grad2
    print(json.dumps(verifier(ProvenanceStore(args.store)), sort_keys=True))


if __name__ == "__main__":
    main()
