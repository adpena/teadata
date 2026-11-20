"""Qualify the reviewed statewide campus transfer observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .dataset_metadata import DataPeriod, EntityRowBinding
from .manifest import ContractError
from .source_parity import validate_source_parity
from .transfer_rollups import PASS_OBSERVED_TRANSFER_ROLLUP, _compare_detail_files

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


@dataclass(frozen=True, slots=True)
class CampusTransferReview:
    year: int
    district_processing: str
    comparable_groups: int
    district_only_suppressed: int
    canonical_edges_on_hold: bool = False


# These district processing records retain the separate regional publisher
# reports and their source-row lineage. They do not contain individual campus reports.
REVIEWED_CAMPUS_TRANSFERS = {
    "435affccc788e71bce14ef0fe86bbbbdb39da60d6a2da8965eb526361803c06b": CampusTransferReview(
        2021, "974d886e6dee838b9e3baa97425ab7d3eb27af527f8439fcf0333dc68e086467", 368, 0,
    ),
    "4f40beb6c56fb0d3e06c97f4318f8a37700446ab66e93b762a91b5a644d25074": CampusTransferReview(
        2022, "44e8b06228f8ef8a21d419f63ad0e8f494c3d95c6ef2807fc87ad5c5eae85485", 364, 0,
    ),
    "4771c65391a31c17bbf1e78147bf2d61660728d9eb4399d25c510cb41e332683": CampusTransferReview(
        2023, "83ff68b7c6b6e7887386e6bd7e60444125d048d0e009a90151ccdf56ffc5a3ea", 364, 0, canonical_edges_on_hold=True,
    ),
    "6f466968708b31ec1d0ec283d55fdbf5e40530873962428e1b451b3a0dcc81ef": CampusTransferReview(
        2024, "1ca5b30a7987f18c737ba51fbf37a861d853403269c3a7c5396f1c41abc9316c", 372, 6, canonical_edges_on_hold=True,
    ),
    "06c5988e150c80761d7231323c56ed1981e336e1c0933dbdd5b1415aaf78d09e": CampusTransferReview(
        2025, "0e7dea1c32deccc3a6b65f7f9936c0f938f5ff8447d11100ff0fd66dada6dfcb", 420, 0,
    ),
    "429355f35f834ad018f7b2b40e162d06062433e333711756a8ccf7353fd4e0cf": CampusTransferReview(
        2026, "7ea45b656a5ae28649e9fc58b7e65329b8b7a04c0706475c6b1c50db69d1fe96", 418, 12,
    ),
}


def qualify_campus_transfers(release: DataRelease, processing_store: ProvenanceStore,
                             district_store: ProvenanceStore) -> dict[str, Any]:
    review = REVIEWED_CAMPUS_TRANSFERS.get(release.manifest_sha256)
    if review is None or not release.offline:
        raise ContractError("Select the exact reviewed offline campus transfer candidate")
    year = review.year
    identifier, = release.catalog.to_dict()["artifacts"]
    ref = release.catalog.artifact_ref(identifier)
    key = ref.edition
    if ((key.dataset, key.edition, key.scope, key.table, ref.output) !=
            ("campus_transfer_reports", str(year), "campus", "report_rows", "leaf")
            or release.catalog.metadata(key.dataset, edition=key).period != DataPeriod("school_year", year - 1, year)
            or ref.entity_binding != EntityRowBinding("reporting_campus_number", "reporting campus")):
        raise ContractError("Campus transfer edition, period or reporting role differs from its review")
    parity = validate_source_parity(release, processing_store, edition_ids=[key.identity])
    if parity["status"] != "PASS_RAW_PARITY":
        raise ContractError(f"Campus transfer source parity failed: {parity['editions']}")

    # Typed parity has replayed the publisher rows, identities and mask tokens
    # against this leaf and checked the selected preparation's recorded directions.
    leaf = processing_store.read_record(ref.processing_id)
    if review.canonical_edges_on_hold and leaf["validation"].get("canonical_edges_qualified") is not False:
        raise ContractError("Reviewed campus transfer masking conflicts require the canonical-edge hold")
    preparation = processing_store.read_record(leaf["parameters"]["recipe"]["preparation"])
    campus = processing_store.verify_artifact(preparation["outputs"]["prepared_source"]["sha256"])
    district_graph = district_store.verify_record_closure([review.district_processing])
    district_leaf = district_graph[review.district_processing]["outputs"]["leaf"]
    district = district_store.verify_artifact(district_leaf["sha256"])
    rollup = _compare_detail_files(campus, district)
    coverage = rollup["coverage"]
    if (rollup["status"] != PASS_OBSERVED_TRANSFER_ROLLUP
            or coverage["exact"] != review.comparable_groups
            or coverage["district_only_suppressed"] != review.district_only_suppressed):
        raise ContractError(f"Campus transfer comparison differs from its reviewed coverage: {coverage}")
    return {"status": "PASS_REVIEWED_NATIVE", "build_id": release.build_id,
            "manifest_sha256": release.manifest_sha256, "parity": parity,
            "district_comparison": {**rollup, "processing_id": review.district_processing,
                                    "district_leaf_sha256": district_leaf["sha256"]},
            "qualification_scope": "statewide_campus_rows_and_complete_cross_district_groups",
            "canonical_edges_on_hold": review.canonical_edges_on_hold,
            "independent_individual_campus_report": False, "suppressed_values_inferred": False}
