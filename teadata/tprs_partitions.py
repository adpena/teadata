"""Translate retained Student campus partitions without copying their index."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from .dataset_metadata import EditionKey


STUDENT_DATASET = "tprs_student_membership_enrollment_reports"


@dataclass(frozen=True, slots=True)
class StudentDerivedRoute:
    topic: str
    memo_attribute: str


STUDENT_DERIVED_ROUTES = MappingProxyType({
    "tprs_class_size": StudentDerivedRoute("Class Size", "_campus_class_size_plan"),
    "tprs_student_continuity": StudentDerivedRoute("Student Continuity", "_campus_continuity_plan"),
})
TPRS_PARTITION_DATASETS = frozenset({STUDENT_DATASET, *STUDENT_DERIVED_ROUTES})


def derived_campus_edition(student: EditionKey, dataset: str) -> EditionKey:
    """Preserve a reviewed Student partition's revision and identity in a derived table."""
    if dataset not in STUDENT_DERIVED_ROUTES:
        raise ValueError("Expected a registered Student-derived campus dataset")
    if (student.dataset, student.edition, student.scope, student.table, student.grain) != (
        STUDENT_DATASET, "2026", "campus", "campus", "one row per campus_number",
    ):
        raise ValueError("Expected a 2025–26 TPRS Student campus edition key")
    return replace(student, dataset=dataset, table=f"campus-{student.identity}")
