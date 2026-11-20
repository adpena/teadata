"""Directional admission for exact statewide transfer preparations."""

from hashlib import sha256
import json
from typing import NamedTuple

from .values import ValuePolicy


class _ReviewedPreparation(NamedTuple):
    original: str
    acquisition: str
    processing: str
    prepared: str
    annotations: str
    report: str


# These publisher originals have exactly four mixed observed/suppressed pairs.
# See docs/transfer-sources.md. They cannot supply canonical edges.
_REVIEWED = {
    2023: _ReviewedPreparation(
        "bd736577ead791adc9c84d4e6f785d9dd2b06926aa895bafa8b1b8dfd8e946e5",
        "e3a1b7c0a5ae0d6ce080739c3847acd6f441cc2f12d4efad16a747b279652c8e",
        "04923e3c5d593a5f1e5a0ad6951a4aecdae12c40960adecac11991e9c2c9ba39",
        "ce13bc8f1740ab47a4a1483ca1b2b69271afa95fd11d5a17253a8335885cd238",
        "3940d7bb76cb817ade08332324df61cbdadef5cf37e8eaa958828d1a5f2d0bdd",
        "7ecd1dd275f3a3c4c90711651d7d8b10daaed978629312aed46d8d32255d9b90",
    ),
    2024: _ReviewedPreparation(
        "40df05470d7a2cf117ec793ab24a85c5e70b62338b6ca1785a6a8643c354ef34",
        "5349c45b292fc5d2e4d564412a9917ace24100b1883afdb1c34f3bd636bc690b",
        "e323415fb1b7d3f04e6c9cfc769e4a9af0480e637b8ef4eec5680da906b1e60d",
        "db5500a3e8826740ab685a94e303803eba429dedce8ff2bb7471952201c1bb25",
        "8bb0c37c1a4d35f4383a6aac140ece2d9604661a18aaacd9b936b686c3556662",
        "659197f33fe56177d02ab9584c608f99f0f768391ebc025064a8eb41a914d276",
    ),
}


def sort_direction_examples(directions: dict) -> dict:
    """Arrow joins need not emit discrepancy examples in physical row order."""
    return {**directions, "examples": sorted(directions["examples"], key=lambda row: row["source_record_out"])}


def preparation_witness(year: int, acquisition: str, original: str, processing: str, preparation: dict,
                        report: dict, policy: ValuePolicy) -> dict:
    """Bind the expected current comparison to the retained preparation and original."""
    if not isinstance(report, dict) or not isinstance(preparation, dict):
        raise ValueError("Campus preparation needs a structured directional witness")
    validation = preparation.get("validation", {})
    directions = report.get("source_validation", {})
    if (not isinstance(directions, dict) or not isinstance(validation, dict)
            or directions.get("input_sha256") != original
            or report.get("policy_sha256") != sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()
            or any(validation.get(key) is not True or report.get(key) is not True for key in (
                "prepared_source_qualified", "raw_string_readback_equal"))):
        raise ValueError("Campus preparation differs from its accepted directional witness")
    if (year not in _REVIEWED and validation.get("status") == "PASS" and report.get("status") == "prepared"
            and validation.get("canonical_edges_qualified") is True and report.get("canonical_edges_qualified") is True
            and directions.get("status") == "PASS" and directions.get("raw_count_token_differences") == 0):
        return directions

    reviewed = _REVIEWED.get(year)
    if reviewed is None:
        raise ValueError("Campus preparation differs from its accepted directional witness")
    outputs = preparation["outputs"]
    keys = ("prepared_source", "annotations", "validation_report")
    actual = _ReviewedPreparation(original, acquisition, processing, *(outputs[key]["sha256"] for key in keys))
    if (actual != reviewed or validation.get("status") != "HOLD" or report.get("status") != "HOLD"
            or any(item.get("canonical_edges_qualified") is not False for item in (report, validation))
            or directions.get("status") != "HOLD" or directions.get("status_mismatches") != 4
            or directions.get("raw_count_token_differences") != 4 or directions.get("value_mismatches") != 2
            or any(directions.get(key) != 0 for key in ("unknown_count_rows", "duplicate_out_keys",
                "duplicate_in_keys", "one_sided_pairs", "historical_district_mismatches"))
            or len(directions.get("examples", ())) != 4
            or any({row["count_status_out"], row["count_status_in"]} != {"observed", "suppressed"}
                   for row in directions["examples"])):
        raise ValueError("Campus preparation differs from its reviewed masking-only hold")
    # The old validator counted incomparable null/observed values. All other
    # counters and example records must match the current validator exactly.
    return sort_direction_examples({**directions, "value_mismatches": 0})
