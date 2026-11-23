"""Qualify reviewed STAAR native bundles using named TEA reports."""

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


ALL_GRADES_MANIFEST = "b399e271381f944ec3844d435f8e110c3c83049bdd75421e4eaa87e3ffd217c2"
ALL_GRADES_2024_MANIFEST = "85d89c21d2ad4e381bc78f2831638996146120d5df65ffaf00069cb42bbff032"
ALL_GRADES_2025_MANIFEST = "a2131525b24f30ebce71bba6d0969255ceabcb7c002e58f5d69efd4a31825fae"
END_OF_COURSE_MANIFEST = "2e62fb58db2c8afb9510c66feb6f385be8dd4439f6d5b71befa8c2bd4b317fc3"
_REPORT = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_RATE = re.compile(r"(?:100|[1-9]?[0-9])")
_SUBJECTS = {
    "staar_all_grades": {
        "All Grades All Subjects": "A001", "All Grades ELA/Reading": "AR01",
        "All Grades Mathematics": "AM01", "All Grades Science": "AC01", "All Grades Social Studies": "AS01",
    },
    "staar_end_of_course": {
        "End of Course English I": "AR11", "End of Course English II": "AR21",
        "End of Course Algebra I": "AA11", "End of Course Biology": "ABI1", "End of Course U.S. History": "AUS1",
    },
}
_STANDARDS = {
    "At Approaches Grade Level or Above": "S", "At Meets Grade Level or Above": "2", "At Masters Grade Level": "3",
}
_GROUPS = (
    "All Students", "African American", "Hispanic", "White", "American Indian", "Asian", "Pacific Islander",
    "Two or More Races", "Special Ed (Current)", "Special Ed (Former)", "Continuously Enrolled",
    "Non-Continuously Enrolled", "Econ Disadv", "EB (Current & Monitored)",
)
_GROUP_CODES = ("DA", "DB", "DH", "DW", "DI", "D3", "D4", "D2", "DS", "NS", "NC", "NM", "DE", "D0")


@dataclass(frozen=True, slots=True)
class _Witness:
    identifier: str
    column: str
    rows: int
    receipt: str


# Exactly one independent TEA HTML report per geographic scope.
_WITNESSES_2026 = {
    "campus": _Witness("227901002", "Campus", 9105, "25dec934d6ca63e3ce08032bbe69805f52ce2c96df5bc417527c1d211c841542"),
    "district": _Witness("227901", "District", 1202, "2d27735462ffb33a2bf567bc8023b602f4dfb9f4b9c412fbeaea8d3bd3fcdf98"),
    "region": _Witness("13", "Region 13", 20, "799bf79af389b9dd72d05995e9225b716c01123f89439ab7f2fed08a6bb1cd31"),
    "state": _Witness("S", "State", 1, "e978651ff7bd604f12755493c4370de3dfe56c45c65a72d046b6804260b4fcbe"),
}
_WITNESSES_BY_EDITION = {
    "2024": {
        "campus": _Witness("227901002", "Campus", 9082, "fb1944c749316706057a3e6aa3104cec9bb345a5f1fcdb1e15c8556a4fd924bd"),
        "district": _Witness("227901", "District", 1207, "502428d0d3cb0e2af018ed6a5560d47873d4d194e548bd66e7d09197d8945efe"),
        "region": _Witness("13", "Region 13", 20, "2b559a1306079e768a6186a38e9f45f909b04b884eaed0bfdc6e70c13d35ceed"),
        "state": _Witness("S", "State", 1, "9f9e446e999ff0a291c48abfa5bd1085242cfa851570854c6ea00ad0ccad36cc"),
    },
    "2025": {
        "campus": _Witness("227901002", "Campus", 9084, "e9785b815c796d90943291f9398d66d09b15808004765a36af12dde4e02685fa"),
        "district": _Witness("227901", "District", 1208, "0e9d7618ba99fe458101e0aa1044cb189b5fcb8bcd1d0a16f8e8ad15e8f727ff"),
        "region": _Witness("13", "Region 13", 20, "1573110287d37bdf900a555d735f8e4e8494802bedf8205927399292481123eb"),
        "state": _Witness("S", "State", 1, "05050d07fb233bf7908c8382ff76e95b11ecbfe81a109d994fd8ca6b1cfa5206"),
    },
    "2026": _WITNESSES_2026,
}


@dataclass(frozen=True, slots=True)
class StaarQualification:
    name: str
    table: str
    editions: tuple[str, ...]

    @property
    def publisher_roots(self) -> tuple[str, ...]:
        years = _aggregate_bindings()["editions"]
        named = (witness.receipt for year in self.editions for witness in _WITNESSES_BY_EDITION[year].values())
        methods = (reference["provenance_id"] for year in self.editions for reference in years[year]["methodology"])
        return tuple(dict.fromkeys((*named, *methods)))


QUALIFICATIONS = {
    ALL_GRADES_2024_MANIFEST: StaarQualification("2023–24 STAAR All Grades", "staar_all_grades", ("2024",)),
    ALL_GRADES_2025_MANIFEST: StaarQualification("2024–25 STAAR All Grades", "staar_all_grades", ("2025",)),
    ALL_GRADES_MANIFEST: StaarQualification("2025–26 STAAR All Grades", "staar_all_grades", ("2026",)),
    END_OF_COURSE_MANIFEST: StaarQualification("STAAR End-of-Course", "staar_end_of_course", ("2024", "2025", "2026")),
}


def _read_report(store: ProvenanceStore, record: dict, year: str, scope: str, witness: _Witness) -> str:
    url = urlsplit(record["resolved_url"])
    params = parse_qs(url.query)
    transport = record["parameters"]
    if (record["method"] != "download" or not record["downloaded_at"]
            or url._replace(query="").geturl() != _REPORT or record["requested_url"] != record["resolved_url"]
            or transport.get("http_method") != "GET" or transport.get("http_status") != 200
            or params.get("_program") != ["perfrept.perfmast.sas"]
            or params.get("prgopt") != ["reports/tapr/performance.sas"]
            or params.get("ccyy") != [year] or params.get("lev") != [scope[0].upper()]
            or params.get("id") != [witness.identifier]):
        raise ContractError(f"STAAR publisher receipt differs for {year} {scope}")
    return store.verify_artifact(record["artifact"]["sha256"]).read_text(encoding="latin-1")


def _group(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9()&]", "", text)


def _named_rates(content: str, year: str, table: str, scope: str, witness: _Witness) -> dict[str, str]:
    rows = _Rows(content).rows
    if len(rows) < 6 or rows[1] != [f"{int(year) - 1}-{year[-2:]} STAAR Performance"] or len(rows[2]) != 1:
        raise ContractError("STAAR publisher report heading differs")
    entity = rows[2][0]
    if scope == "state":
        same_entity = entity == "State"
    elif scope == "region":
        same_entity = entity.startswith(f"REGION {witness.identifier}: ")
    else:
        same_entity = f"({witness.identifier})" in entity
    header = rows[4]
    if not same_entity or header.count(witness.column) != 1:
        raise ContractError(f"STAAR publisher geography differs for {year} {scope}")
    column = header.index(witness.column)
    groups = _GROUPS[1:] if year == "2026" else (*_GROUPS[1:-1], "EB/EL (Current & Monitored)")
    if tuple(map(_group, header[column + 1:])) != tuple(map(_group, groups)):
        raise ContractError(f"STAAR publisher groups differ for {year} {scope}")

    rates: dict[str, str] = {}
    panels: set[tuple[str, str]] = set()
    subjects = _SUBJECTS[table]
    heading = ""
    for row in rows[6:]:
        if len(row) == 1:
            heading = row[0]
        if len(row) < 3 or row[1] != year or row[0] not in _STANDARDS or heading not in subjects:
            continue
        subject, standard = subjects[heading], _STANDARDS[row[0]]
        if (subject, standard) in panels or len(row[column:]) != len(_GROUP_CODES):
            raise ContractError(f"STAAR publisher panel is duplicated or incomplete for {year} {scope}")
        panels.add((subject, standard))
        for code, display in zip(_GROUP_CODES, row[column:], strict=True):
            rates[f"{code}00{subject}{standard}{year[-2:]}R"] = display
    if len(rates) != 210:
        raise ContractError(f"STAAR publisher report is missing rate cells for {year} {scope}")
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
    raise ContractError("STAAR contains an unsupported raw rate token")


def qualify_staar_rates(release: DataRelease, store: ProvenanceStore) -> dict:
    """Replay full raw exports and 210 independent published rates at each scope."""
    plan = QUALIFICATIONS.get(release.manifest_sha256)
    if not release.offline or plan is None:
        raise ContractError("STAAR qualification requires its exact reviewed offline manifest")
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    expected = {(year, scope): witness for year in plan.editions
                for scope, witness in _WITNESSES_BY_EDITION[year].items()}
    if (len(refs) != len(expected) or {(ref.edition.edition, ref.edition.scope) for ref in refs} != expected.keys()
            or any(ref.edition.table != plan.table
                   or ref.contract["rows"] != expected[ref.edition.edition, ref.edition.scope].rows for ref in refs)):
        raise ContractError("STAAR qualification requires every exact reviewed edition and scope")
    from pyarrow import dataset as ds

    reports = store.verify_record_closure(plan.publisher_roots)
    years = _aggregate_bindings()["editions"]
    for year in plan.editions:
        for reference in years[year]["methodology"]:
            record = reports[reference["provenance_id"]]
            if (record["artifact"]["sha256"] != reference["sha256"]
                    or record["resolved_url"] != reference["url"]):
                raise ContractError(f"STAAR publisher methodology differs for {year}")
    checked = []
    for ref in refs:
        year, scope = ref.edition.edition, ref.edition.scope
        witness = expected[year, scope]
        bindings = years[year]["families"][plan.table]["rates"]
        record = reports[witness.receipt]
        content = _read_report(store, record, year, scope, witness)
        rates = _named_rates(content, year, plan.table, scope, witness)
        if rates.keys() != {binding["code"] for binding in bindings.values()}:
            raise ContractError(f"STAAR named and reviewed rate codes differ for {year} {scope}")
        contract = staar_rate_contract(release, ref.edition)
        fields = {metric: definition.field for metric, definition in contract.metadata.metrics.items()}
        where = None if scope == "state" else ds.field(f"{scope}_number") == witness.identifier
        selected = release.analytical_dataset(ref.identity).to_table(columns=list(fields.values()), filter=where)
        if selected.num_rows != 1:
            raise ContractError(f"STAAR publisher witness is missing or duplicated for {year} {scope}")
        row, = selected.to_pylist()
        statuses: Counter[str] = Counter()
        for metric, field in fields.items():
            display, status = _rate_display(row[field], contract)
            if display != rates[bindings[metric]["code"]]:
                raise ContractError(f"STAAR published rate differs for {year} {scope} {witness.identifier} {metric}")
            statuses[status] += 1
        checked.append({"scope": scope, "entity": witness.identifier, "publisher_receipt": witness.receipt,
                        "source_downloaded_at": record["downloaded_at"], "compared_rates": sum(statuses.values()),
                        "statuses": dict(statuses), **({"edition": year} if len(plan.editions) > 1 else {})})
    # The wide campus Arrow tables exceed the default 128 MiB cap's per-table slot.
    parity = validate_source_parity(release, store, edition_ids=[ref.edition.identity for ref in refs],
                                    working_bytes=256 * 1024**2, verify_export=True)
    return {"status": "PASS_REVIEWED_NATIVE" if parity["status"] == "PASS_RAW_PARITY" else "HOLD",
            "manifest_sha256": release.manifest_sha256, "build_id": release.build_id, "raw_parity": parity,
            "publisher_witnesses": checked,
            "limits": "Accountability-subset published rates only; other columns retain raw fidelity. "
                      "One retained independent named report per scope; no count inference or longitudinal equivalence."}
