"""The reviewed identity repair for the pinned 2021–22 district report."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .dataset_metadata import ResolvedDataset


REVIEW_YEAR = 2022
REVIEW_REFERENCE = "district_partner_identity_review"
CORRECTED = "transfer_partner_id_corrected"
REVIEW_PATH = Path(__file__).with_name("methodologies") / "district-transfer-2022-partner-ids.json"


@dataclass(frozen=True)
class PartnerIdentityRepair:
    labels: Mapping[str, str]
    expected_hits: Mapping[str, int]

    @staticmethod
    def count_hits(table: Any, partner_label: str) -> dict[str, int]:
        import pyarrow.compute as arrow_compute

        pc: Any = arrow_compute
        actual = pc.value_counts(table.filter(table[CORRECTED])[partner_label]).to_pylist()
        return {item["values"]: item["counts"] for item in actual}

    def validate_hits(self, actual: Mapping[str, int]) -> None:
        if actual != self.expected_hits:
            raise ValueError("District identity repair differs from the approved original label counts")


@dataclass(frozen=True)
class DistrictIdentityReview:
    sha256: str
    regions: Mapping[str, tuple[str, PartnerIdentityRepair]]

    def for_original(self, region: str | None, digest: str) -> PartnerIdentityRepair | None:
        selected = self.regions.get(region) if region is not None else None
        return selected[1] if selected is not None and selected[0] == digest else None


def load_identity_review() -> DistrictIdentityReview:
    raw = REVIEW_PATH.read_bytes()
    document = json.loads(raw)
    labels, regions = document["labels"], document["regions"]
    if (document["schema_version"] != 1 or document["school_year_end"] != REVIEW_YEAR
            or set(regions) != {f"{region:02d}" for region in range(1, 21)}
            or len(labels) != 34 or len(set(labels.values())) != 34
            or any(not re.fullmatch(r"[0-9]{6}", value) or value == "000000" for value in labels.values())
            or any(not set(region["label_hits"]) <= set(labels)
                   or not re.fullmatch(r"[0-9a-f]{64}", region["sha256"])
                   or any(type(count) is not int or count < 1 for count in region["label_hits"].values())
                   for region in regions.values())
            or sum(sum(region["label_hits"].values()) for region in regions.values()) != 1076):
        raise ValueError("Invalid approved district partner identity review")
    originals = {}
    for region, entry in regions.items():
        hits = entry["label_hits"]
        repair = PartnerIdentityRepair(MappingProxyType({label: labels[label] for label in hits}),
                                       MappingProxyType(hits))
        originals[region] = (entry["sha256"], repair)
    return DistrictIdentityReview(sha256(raw).hexdigest(), MappingProxyType(originals))


def reviewed_repair_receipt(source: ResolvedDataset, originals: Sequence[tuple[str, str]],
                            read_record: Callable[[str], Mapping[str, Any]]) -> tuple[str, DistrictIdentityReview]:
    """Require the reviewed document and all twenty original source hashes."""
    review = load_identity_review()
    reference, = (ref for ref in source.metadata.references if ref.reference_id == REVIEW_REFERENCE)
    receipt, = reference.provenance_ids
    record = read_record(receipt)
    if (source.resolved_year != REVIEW_YEAR or reference.role != "methodology" or reference.status != "available"
            or reference.reviewed_sha256 != review.sha256 or record["artifact"]["sha256"] != review.sha256
            or record["kind"] != "acquisition" or record["method"] != "local_observation"
            or record["dataset"] != f"reference:{REVIEW_REFERENCE}"
            or [sha for _, sha in originals] != [review.regions[region][0] for region in sorted(review.regions)]):
        raise ValueError("District identity repair is restricted to the twenty approved original CSVs")
    return receipt, review
