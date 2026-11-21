"""Qualify the exact 2025–26 STAAR All Grades child for normal release admission."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from .manifest import ContractError
from .native_witnesses import _Rows
from .source_parity import validate_source_parity
from .staar_comparison import _aggregate_bindings, staar_rate_contract
from .values import ValueStatus

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease
    from .staar_comparison import StaarRateContract


REVIEWED_MANIFEST = "b399e271381f944ec3844d435f8e110c3c83049bdd75421e4eaa87e3ffd217c2"
_REPORT = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_TABLE = "staar_all_grades"
_RATE = re.compile(r"(?:100|[1-9]?[0-9])")
_SUBJECTS = {
    "All Grades All Subjects": "All Subjects",
    "All Grades ELA/Reading": "Reading/Language Arts",
    "All Grades Mathematics": "Mathematics",
    "All Grades Science": "Science",
    "All Grades Social Studies": "Social Studies",
}
_STANDARDS = {
    "At Approaches Grade Level or Above": "At Approaches GL Standard or Above",
    "At Meets Grade Level or Above": "At Meets GL Standard or Above",
    "At Masters Grade Level": "At Masters GL Standard",
}
_GROUPS = (
    "All Students", "African American", "Hispanic", "White", "American Indian", "Asian", "Pacific Islander",
    "Two or More Races", "Special Ed (Current)", "Special Ed (Former)", "Continuously Enrolled",
    "Non-Continuously Enrolled", "Econ Disadv", "EB (Current & Monitored)",
)


@dataclass(frozen=True, slots=True)
class _Witness:
    identifier: str
    column: str
    rows: int
    receipt: str


# Exactly one independent TEA HTML report per geographic scope.
_WITNESSES = {
    "campus": _Witness("227901002", "Campus", 9105, "25dec934d6ca63e3ce08032bbe69805f52ce2c96df5bc417527c1d211c841542"),
    "district": _Witness("227901", "District", 1202, "2d27735462ffb33a2bf567bc8023b602f4dfb9f4b9c412fbeaea8d3bd3fcdf98"),
    "region": _Witness("13", "Region 13", 20, "799bf79af389b9dd72d05995e9225b716c01123f89439ab7f2fed08a6bb1cd31"),
    "state": _Witness("S", "State", 1, "e978651ff7bd604f12755493c4370de3dfe56c45c65a72d046b6804260b4fcbe"),
}
PUBLISHER_ROOTS = tuple(witness.receipt for witness in _WITNESSES.values())


def _read_report(store: ProvenanceStore, record: dict, scope: str, witness: _Witness) -> str:
    url = urlsplit(record["resolved_url"])
    params = parse_qs(url.query)
    transport = record["parameters"]
    if (record["method"] != "download" or not record["downloaded_at"]
            or url._replace(query="").geturl() != _REPORT or record["requested_url"] != record["resolved_url"]
            or transport.get("http_method") != "GET" or transport.get("http_status") != 200
            or params.get("_program") != ["perfrept.perfmast.sas"]
            or params.get("prgopt") != ["reports/tapr/performance.sas"]
            or params.get("ccyy") != ["2026"] or params.get("lev") != [scope[0].upper()]
            or params.get("id") != [witness.identifier]):
        raise ContractError(f"STAAR All Grades publisher receipt differs for {scope}")
    return store.verify_artifact(record["artifact"]["sha256"]).read_text(encoding="latin-1")


def _group(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9()&]", "", text)


def _named_rates(content: str, scope: str, witness: _Witness) -> dict[str, str]:
    rows = _Rows(content).rows
    if len(rows) < 6 or rows[1] != ["2025-26 STAAR Performance"] or len(rows[2]) != 1:
        raise ContractError("STAAR All Grades publisher report heading differs")
    entity = rows[2][0]
    if scope == "state":
        same_entity = entity == "State"
    elif scope == "region":
        same_entity = entity.startswith(f"REGION {witness.identifier}: ")
    else:
        same_entity = f"({witness.identifier})" in entity
    header = rows[4]
    if not same_entity or header.count(witness.column) != 1:
        raise ContractError(f"STAAR All Grades publisher geography differs for {scope}")
    column = header.index(witness.column)
    if tuple(map(_group, header[column + 1:])) != tuple(map(_group, _GROUPS[1:])):
        raise ContractError(f"STAAR All Grades publisher groups differ for {scope}")

    rates: dict[str, str] = {}
    panels: set[tuple[str, str]] = set()
    heading = ""
    for row in rows[6:]:
        if len(row) == 1:
            heading = row[0]
        if len(row) < 3 or row[1] != "2026" or row[0] not in _STANDARDS or heading not in _SUBJECTS:
            continue
        subject, standard = _SUBJECTS[heading], _STANDARDS[row[0]]
        if (subject, standard) in panels or len(row[column:]) != len(_GROUPS):
            raise ContractError(f"STAAR All Grades publisher panel is duplicated or incomplete for {scope}")
        panels.add((subject, standard))
        for group, display in zip(_GROUPS, row[column:], strict=True):
            rates[f"STAAR Performance, {subject}, SY 2025-26, {group}, % {standard}"] = display
    if len(rates) != 210:
        raise ContractError(f"STAAR All Grades publisher report is missing rate cells for {scope}")
    return rates


def _rate_display(token: object, contract: StaarRateContract) -> tuple[str, str]:
    if token == "":
        return "-", "missing"
    if isinstance(token, str) and _RATE.fullmatch(token):
        return f"{token}%", "observed"
    if isinstance(token, str):
        rule = contract.value_policy.tokens.get(token)
        if rule is not None and rule.status is ValueStatus.SUPPRESSED:
            return "*", "suppressed"
    raise ContractError("STAAR All Grades contains an unsupported raw rate token")


def qualify_staar_all_grades(release: DataRelease, store: ProvenanceStore) -> dict:
    """Replay full raw exports and 210 independent published rates at each scope."""
    if not release.offline or release.manifest_sha256 != REVIEWED_MANIFEST:
        raise ContractError("STAAR All Grades qualification requires its exact reviewed offline manifest")
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    if (len(refs) != len(_WITNESSES) or {ref.edition.scope for ref in refs} != set(_WITNESSES)
            or any(ref.edition.table != _TABLE or ref.edition.edition != "2026"
                   or ref.contract["rows"] != _WITNESSES[ref.edition.scope].rows for ref in refs)):
        raise ContractError("STAAR All Grades qualification requires all four exact reviewed editions")
    from pyarrow import dataset as ds

    reports = store.verify_record_closure(PUBLISHER_ROOTS)
    bindings = _aggregate_bindings()["editions"]["2026"]["families"][_TABLE]["rates"]
    checked = []
    for ref in refs:
        scope = ref.edition.scope
        witness = _WITNESSES[scope]
        record = reports[witness.receipt]
        content = _read_report(store, record, scope, witness)
        rates = _named_rates(content, scope, witness)
        if rates.keys() != {binding["label"] for binding in bindings.values()}:
            raise ContractError(f"STAAR All Grades named and reviewed rate labels differ for {scope}")
        contract = staar_rate_contract(release, ref.edition)
        fields = {metric: definition.field for metric, definition in contract.metadata.metrics.items()}
        where = None if scope == "state" else ds.field(f"{scope}_number") == witness.identifier
        selected = release.analytical_dataset(ref.identity).to_table(columns=list(fields.values()), filter=where)
        if selected.num_rows != 1:
            raise ContractError(f"STAAR All Grades publisher witness is missing or duplicated for {scope}")
        row, = selected.to_pylist()
        statuses: Counter[str] = Counter()
        for metric, field in fields.items():
            display, status = _rate_display(row[field], contract)
            if display != rates[bindings[metric]["label"]]:
                raise ContractError(f"STAAR All Grades published rate differs for {scope} {witness.identifier} {metric}")
            statuses[status] += 1
        checked.append({"scope": scope, "entity": witness.identifier, "publisher_receipt": witness.receipt,
                        "source_downloaded_at": record["downloaded_at"], "compared_rates": sum(statuses.values()),
                        "statuses": dict(statuses)})
    # The 9,105-by-949 campus Arrow table exceeds the default 128 MiB
    # comparison budget's per-table slot. Keep a bounded source-specific cap.
    parity = validate_source_parity(release, store, edition_ids=[ref.edition.identity for ref in refs],
                                    working_bytes=256 * 1024**2, verify_export=True)
    return {"status": "PASS_REVIEWED_NATIVE" if parity["status"] == "PASS_RAW_PARITY" else "HOLD",
            "manifest_sha256": release.manifest_sha256, "build_id": release.build_id, "raw_parity": parity,
            "publisher_witnesses": checked,
            "limits": "2025–26 accountability-subset published rates only; other columns retain raw fidelity. "
                      "One retained independent named report per scope; no count inference or longitudinal equivalence."}
