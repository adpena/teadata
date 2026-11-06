"""Replay reviewed 2024–25 statewide TAPR selections offline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
COMP4_DATASET = "state_tapr_four_year_longitudinal_rate"
STUDENT_DATASET = "state_tapr_student_profile"
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


def _verify_source(store: ProvenanceStore, dataset: str,
                   verify: Callable[[dict[str, str], Callable[[str], Path], set[str]], dict[str, int]]) -> dict[str, int]:
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
    field_codes = {field["field"]: field["code"] for field in fields}
    bound_codes = {field_codes[metric.field] for metric in metadata.metrics.values()}
    return {
        "raw_cells": len(values),
        "dictionary_occurrences": len(occurrences),
        **verify(dict(zip(codes, values, strict=True)), original, bound_codes),
    }


def _verify_ccmr(store: ProvenanceStore, dataset: str, measures: dict[str, str]) -> dict[str, int]:
    def verify(values: dict[str, str], original: Callable[[str], Path], _bound_codes: set[str]) -> dict[str, int]:
        rows = _Rows(original("state_ccmr_report").read_text(encoding="cp1252")).rows
        return {"displayed_rates": verify_rates(values, rows, measures=measures)}

    return _verify_source(store, dataset, verify)


def verify_state_grad2(store: ProvenanceStore) -> dict[str, int]:
    return _verify_ccmr(store, DATASET, MEASURES)


def verify_state_grad1(store: ProvenanceStore) -> dict[str, int]:
    return _verify_ccmr(store, GRAD1_DATASET, {"0GP": "College, Career, or Military Ready (Annual Graduates)"})


def verify_state_comp4(store: ProvenanceStore) -> dict[str, int]:
    def verify(values: dict[str, str], original: Callable[[str], Path], _bound_codes: set[str]) -> dict[str, int]:
        original("cohort_methodology")
        cohort, graduates = int(values["SA0C424D"]), int(values["SAGC424N"])
        rate = Decimal(values["SAGC424R"])
        if cohort <= 0 or not 0 <= graduates <= cohort:
            raise ValueError("Invalid Class of 2024 cohort counts")
        if (Decimal(graduates) * 100 / cohort).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) != rate:
            raise ValueError("Statewide Class of 2024 cohort counts and rate disagree")
        longitudinal = _Rows(original("state_longitudinal").read_text(encoding="cp1252")).rows
        state_heading = ["Grade 9 Four-Year Longitudinal Graduation and Dropout Rates, Texas Public Schools, Class of 2024"]
        if longitudinal.count(state_heading) != 1:
            raise ValueError("Statewide longitudinal cohort and graduation values disagree")
        header, data = longitudinal[longitudinal.index(state_heading) + 4:longitudinal.index(state_heading) + 6]
        if header[:3] != ["Class", "Number", "Rate (%)"] or data[:3] != [f"{cohort:,}", f"{graduates:,}", f"{rate:.1f}"]:
            raise ValueError("Statewide longitudinal cohort and graduation values disagree")
        school = _Rows(original("federal_garland_school_report_card").read_text(encoding="cp1252")).rows
        federal = ["4-Year Federal Graduation Rate Without Exclusions (Gr 9-12)"]
        if (["2025 School Report Card"] not in school
                or ["GARLAND H S (057909002) - GARLAND ISD - DALLAS COUNTY"] not in school
                or school.count(federal) != 1
                or not any(row[:4] == ["", "State", "District", "Campus"] for row in school[:school.index(federal)])
                or school[school.index(federal) + 1][:5] != ["Graduation", "Class of 2024", f"{rate:.1f}%", "92.5%", "91.7%"]):
            raise ValueError("Garland school report card federal graduation values disagree")
        return {"displayed_rates": 1}

    return _verify_source(store, COMP4_DATASET, verify)


def verify_state_student_profile(store: ProvenanceStore) -> dict[str, int]:
    def verify(values: dict[str, str], original: Callable[[str], Path], bound_codes: set[str]) -> dict[str, int]:
        rows = _Rows(original("state_student_report").read_text(encoding="cp1252")).rows
        if (rows[1:3] != [["2024-25 Student Information"], ["State"]]
                or rows[4:6] != [["", "Membership", "Enrollment"],
                                 ["Student Information", "Count", "Percent", "Count", "Percent"]]):
            raise ValueError("Unexpected statewide student report edition, scope, or populations")

        def section(start: str, stop: str) -> dict[str, list[str]]:
            if rows.count([start]) != 1 or rows.count([stop]) != 1:
                raise ValueError(f"Unexpected student profile report section: {start}")
            entries = rows[rows.index([start]) + 1:rows.index([stop])]
            by_label = {entry[0]: entry[1:] for entry in entries if entry}
            if len(by_label) != len(entries):
                raise ValueError(f"Duplicate statewide student report labels: {start}")
            return by_label

        def control(stem: str, cells: list[str], suffixes: str = "CP") -> dict[str, str]:
            return {stem + suffix: cell for suffix, cell in zip(suffixes, cells, strict=True)}

        races = {"BLA": "African American", "HIS": "Hispanic", "WHI": "White", "IND": "American Indian",
                 "ASI": "Asian", "PCI": "Pacific Islander", "TWO": "Two or More Races"}
        ethnic = section("Ethnic Distribution", "Sex")
        programs = section("Students by Instructional Program", "Students with Disabilities by Type of Primary Disability")
        other = section("Other Student Cohorts", "Students by Instructional Program")
        mobility = section("Mobility (2023-24)", "Student Attrition (2023-24)")
        if rows[6][0] != "Total Students":
            raise ValueError("Unexpected statewide student total row")
        expected = {"SPETALLC": rows[6][1], "SPNTALLC": rows[6][3]}
        for code, label in races.items():
            expected.update(control("SPET" + code, ethnic[label][:2]))
            expected.update(control("SPNT" + code, ethnic[label][2:]))
            expected.update(control("SPEM" + code, mobility[label][:2]))
        for code, label in {"SPE": "Special Education", "GIF": "Gifted and Talented Education",
                            "DSL": "Students w/ Dyslexia", "LEP": "EB Students/EL", "504": "Section 504 Students"}.items():
            cells = (programs if label in programs else other)[label]
            expected.update(control("SPET" + code, cells[:2]))
            expected.update(control("SPNT" + code, cells[2:]))
        for code, label in {"ALL": "Total Mobile Students", "SPE": "Special Ed Students who are Mobile",
                            "LEP": "Count and Percent of EB Students/EL who are Mobile",
                            "ECO": "Count and Percent of Econ Dis Students who are Mobile"}.items():
            expected.update(control("SPEM" + code, mobility[label][:2]))
        expected.update(control("SPETDIS", other["Students w/ Disciplinary Placements (2023-24)"][:2]))
        attrition = rows[rows.index(["Student Attrition (2023-24)"]) + 1]
        if attrition[0] != "Total Student Attrition":
            raise ValueError("Unexpected statewide student attrition row")
        expected.update(control("SPETATT", attrition[1:3]))
        for code, displayed in expected.items():
            if Decimal(values[code]) != Decimal(displayed.removesuffix("%").replace(",", "")):
                raise ValueError(f"Statewide student report differs for {code}")
        held = {"SPETATTC", "SPETATTP"}
        if bound_codes != expected.keys() - held:
            raise ValueError("Statewide student bindings differ from the independently reviewed report fields")
        return {"displayed_comparable_cells": len(bound_codes), "displayed_held_cells": len(held)}

    return _verify_source(store, STUDENT_DATASET, verify)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    verifiers = {"GRAD1": verify_state_grad1, "GRAD2": verify_state_grad2, "COMP4": verify_state_comp4,
                 "STUD": verify_state_student_profile}
    parser.add_argument("--family", choices=verifiers, default="GRAD2")
    args = parser.parse_args()
    print(json.dumps(verifiers[args.family](ProvenanceStore(args.store)), sort_keys=True))


if __name__ == "__main__":
    main()
