"""Admission of roster-bound 2026 TPRS student reports."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
import gzip

from hashlib import sha256
from io import BytesIO

import json
from stat import S_ISREG
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlsplit

from .column_schema import _unique_object
from .dataset_metadata import ArtifactRef, ColumnSchemaRef, EditionKey
from .manifest import ContractError, canonical_json

from .provenance import ProvenanceStore
from .tprs_students import (
    BROKER, DATASET, EDITION, _DIGEST, _STATEWIDE_ROW_GROUP_ROWS, _identifier, _need,
    _validate_report_acquisition, semantic_fields,
)
from .values import ValueStatus

if TYPE_CHECKING:
    from .release import DataRelease


REGIONS = tuple(f"{number:02}" for number in range(1, 21))
DISTRICT_COUNT = 1202
CAMPUS_INTAKE_VERSION = 3
CAMPUS_COHORT_VERSION = 4
CAMPUS_PARTITION_VERSION = 5
CAMPUS_PARTITION_REPORTS = 512
MAX_ROSTER_BYTES = 256 * 1024
MAX_CAMPUS_ROSTER_BYTES = 512 * 1024
MAX_CAMPUS_COHORT_REPORTS = 1500
MAX_STATEWIDE_HTML_BYTES = 160 * 1024**2
# Reviewed district/Texas projection and the complete campus composition.
# Each new revision needs its own source review.
REVIEWED_DISTRICT_STATE_MANIFEST = "373041dfb0eb994ccde4a88f1e7e630e7b395bf9f7a51199655d843788a2b5f6"
REVIEWED_CAMPUS_STUDENT_MANIFEST = "c96effc9bcbfd01c7cd969082cf701a2dc4e1900b6d6973b23c0ed2d21559f9e"
REVIEWED_STATEWIDE_TPRS_MANIFESTS = frozenset({REVIEWED_DISTRICT_STATE_MANIFEST, REVIEWED_CAMPUS_STUDENT_MANIFEST})


@dataclass(frozen=True, slots=True)
class TPRSStudentCampusIndex:
    """Exact campus routes derived from one immutable release's report ledgers."""

    manifest_sha256: str
    edition: str
    _routes: Mapping[str, EditionKey] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_routes", MappingProxyType(dict(self._routes)))

    def __len__(self) -> int:
        return len(self._routes)

    def edition_for(self, campus_number: str | int) -> EditionKey | None:
        """Return the exact key, or None if the selected release has no route."""
        from .identifiers import canonical_source_identifier

        if isinstance(campus_number, bool) or not isinstance(campus_number, (str, int)):
            raise ValueError("Campus number must be a string or integer")
        identifier = canonical_source_identifier(campus_number, width=9)
        if identifier is None:
            raise ValueError("Campus number must not be blank")
        return self._routes.get(identifier[1:])


@dataclass(frozen=True, slots=True)
class ReportSelection:
    scope: str
    identifier: str
    receipt: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RosterSelection:
    reports: tuple[ReportSelection, ...]
    roster: dict[str, dict[str, str]]
    partition: int | None = None


def _regional_roster(response: bytes, record: Mapping[str, Any], region: str, scope: str, limit: int) -> tuple[str, ...]:
    size = record["artifact"]["size_bytes"]
    _need(type(size) is int and 0 < len(response) == size <= limit
          and sha256(response).hexdigest() == record["artifact"]["sha256"], "Roster response differs from its receipt")
    url = urlsplit(record["requested_url"])
    controls = parse_qsl(url.query, keep_blank_values=True)
    expected = {"_service": "marykay", "_debug": "0", "_program": "perfrept.acct_json_response.sas",
                "prmv_level": {"campus": "C", "district": "D"}[scope],
                "ccyy": str(EDITION), "prmv_var": "region", "prmv_text": region}
    parameters = record["parameters"]
    _need(record["kind"] == "acquisition" and record["method"] == "download" and record["section"] == "reference"
          and record["requested_year"] == record["resolved_year"] == EDITION
          and record["source_metadata"].get("version") == str(EDITION)
          and record["resolved_url"] == record["requested_url"] and record["downloaded_at"]
          and url._replace(query="").geturl() == BROKER and not url.fragment and len(controls) == len(expected)
          and dict(controls) == expected and parameters.get("form_controls") == [list(pair) for pair in controls]
          and parameters.get("form_url") == BROKER and parameters.get("http_method") == "GET"
          and parameters.get("http_status") == 200
          and record["response_headers"].get("content-type", "").split(";")[0] == "application/json",
          "Roster receipt does not bind the exact public regional request")
    rows = json.loads(response.decode("utf-8"), object_pairs_hook=_unique_object)
    _need(isinstance(rows, list) and bool(rows), "Roster response requires entity records")
    members: set[str] = set()
    for row in rows:
        _need(isinstance(row, dict) and row.get("REGION") == region, "Roster row does not belong to its region")
        identifier = row.get(scope.upper())
        _identifier(scope, identifier)
        if scope == "campus":
            district = row.get("DISTRICT")
            _identifier("district", district)
            _need(identifier[:6] == district, "Roster campus district differs from its identifier")
        _need(identifier not in members, f"Duplicate {scope} in regional roster")
        members.add(identifier)
    return tuple(sorted(members))


def campus_roster(store: ProvenanceStore, region: str, receipt: str) -> tuple[str, ...]:
    """Read one exact 2026 campus region without inferring eligibility."""
    _need(region in REGIONS, "Invalid campus roster region")
    record = store.read_record(receipt)
    _need(record["kind"] == "acquisition", "Campus roster receipt must identify an acquisition")
    size = record["artifact"]["size_bytes"]
    limit = MAX_CAMPUS_ROSTER_BYTES
    _need(type(size) is int and 0 < size <= limit, "Campus roster exceeds its byte budget")
    # Bound the publisher JSON before asking the general provenance reader to hash it.
    digest = record["artifact"]["sha256"]
    _need(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None, "Invalid campus roster content pin")
    path = store.root / "objects" / digest
    _need(path.is_file() and path.stat().st_size == size, "Campus roster differs from its declared size")
    with path.open("rb") as stream:
        response = stream.read(size + 1)
    members = _regional_roster(response, record, region, "campus", limit)
    store.verify_record_closure([receipt])
    return members


def validate_campus_selection(store: ProvenanceStore, selection: Mapping[str, Any]) -> str:
    """Require the sealed report identifiers to match the retained regional roster."""
    region, receipt = selection["region"], selection["roster_receipt"]
    _need(isinstance(region, str) and isinstance(receipt, str) and _DIGEST.fullmatch(receipt) is not None,
          "Campus intake requires an exact regional roster receipt")
    expected = campus_roster(store, region, receipt)
    observed: list[str] = []
    for report in selection["reports"]:
        _need(isinstance(report, dict) and report.get("scope") == "campus", "Regional intake contains a non-campus report")
        identifier = report.get("identifier")
        _identifier("campus", identifier)
        observed.append(identifier)
    _need(tuple(observed) == expected, "Regional intake differs from its exact ordered campus roster")
    return receipt


@dataclass(frozen=True)
class _ReplayLeaf:
    artifact: ArtifactRef
    audit: ColumnSchemaRef
    reports: tuple[ReportSelection, ...]
    campus_rosters: tuple[tuple[str, str], ...] = ()


def _reviewed_leaves(release: DataRelease) -> tuple[_ReplayLeaf, ...]:
    from .release import _student_attachments

    _need(release.offline and release.manifest_sha256 in REVIEWED_STATEWIDE_TPRS_MANIFESTS,
          "Statewide TPRS replay requires the exact reviewed offline manifest")
    complete_campuses = release.manifest_sha256 == REVIEWED_CAMPUS_STUDENT_MANIFEST
    catalog = release.catalog.to_dict()
    _need(catalog["schema_version"] == 4 and len(catalog["artifacts"]) == len(catalog["editions"])
          and (len(catalog["artifacts"]) > 2 if complete_campuses else len(catalog["artifacts"]) == 2),
          "Reviewed statewide TPRS requires exact native leaves for its scopes")
    leaves = []
    cohorts = set()
    statewide_scopes = []
    for identifier in catalog["artifacts"]:
        artifact = release.catalog.artifact_ref(identifier)
        key = artifact.edition
        _need(key.dataset == DATASET and key.edition == str(EDITION), "Unexpected reviewed TPRS edition")
        refs = _student_attachments(release.catalog, artifact)
        _need(bool(refs), "Reviewed statewide TPRS lacks its report ledger")
        reports, cohort = release._student_report_ledger(artifact, refs)
        campus_rosters = ()
        if key.scope == "campus":
            _need(complete_campuses and cohort is None, "Reviewed statewide TPRS does not include this campus leaf")
            roster = json.loads(release._column_schema_bytes(refs["roster"]), object_pairs_hook=_unique_object)
            campus_rosters = tuple(sorted((region, value["receipt"]) for region, value in roster.items()))
        else:
            _need(cohort is not None, "Reviewed statewide TPRS requires a district and Texas cohort")
            cohorts.add(cohort)
            statewide_scopes.append(key.scope)
        leaves.append(_ReplayLeaf(artifact, refs["raw_audit"], reports, campus_rosters))
    _need(len(cohorts) == 1 and sorted(statewide_scopes) == ["district", "state"],
          "Reviewed TPRS leaves must share the district and Texas cohort")
    return tuple(leaves)


def _verify_campus_census(leaves: tuple[_ReplayLeaf, ...], store: ProvenanceStore) -> None:
    rosters = {}
    observed = set()
    for leaf in leaves:
        if leaf.artifact.edition.scope != "campus":
            continue
        for region, pin in leaf.campus_rosters:
            _need(rosters.setdefault(region, pin) == pin, "Campus leaves disagree on a regional roster receipt")
        identifiers = {item.identifier for item in leaf.reports}
        _need(len(identifiers) == len(leaf.reports) and observed.isdisjoint(identifiers),
              "Campus leaves repeat a publisher campus identifier")
        observed.update(identifiers)
    _need(set(rosters) == set(REGIONS), "Reviewed campus census requires all 20 regional rosters")
    expected = set()
    for region, pin in sorted(rosters.items()):
        identifiers = set(campus_roster(store, region, pin))
        _need(expected.isdisjoint(identifiers), "Publisher regional campus rosters overlap")
        expected.update(identifiers)
    _need(observed == expected, "Campus leaves do not cover the exact publisher roster")


def _retained_replay_records(leaves: tuple[_ReplayLeaf, ...], store: ProvenanceStore) -> list[str]:
    from .tprs_students import definition

    roots = {item.receipt for leaf in leaves for item in leaf.reports}
    roots.update(pin for leaf in leaves for _, pin in leaf.campus_rosters)
    roots.update(witness[role] for witness in definition()["witnesses"].values() for role in ("html_receipt", "xlsx_receipt"))
    # Each identifier commits the complete record and its artifact digest. The
    # store checks the actual artifact bytes again when verifying a check receipt.
    return sorted(store.verify_record_closure(sorted(roots)))


def retained_tprs_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    """Identify the exact retained inputs required by a composed release check."""
    return _retained_replay_records(_reviewed_leaves(release), store)


def qualify_statewide_tprs(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Qualify the reviewed TPRS projection from retained publisher evidence."""
    from .tprs_students import _verify_statewide_projection, verify_workbook_witnesses

    leaves = _reviewed_leaves(release)
    paths = release.verify()
    complete_campuses = release.manifest_sha256 == REVIEWED_CAMPUS_STUDENT_MANIFEST
    if complete_campuses:
        _verify_campus_census(leaves, store)
    evidence = _retained_replay_records(leaves, store)
    counts = {}
    for leaf in leaves:
        key = leaf.artifact.edition
        audit = release._resource_path(leaf.audit.location, leaf.audit.sha256, leaf.audit.size_bytes)
        _verify_statewide_projection(store, leaf.reports, release.catalog.policy(key.dataset, edition=key),
                                     paths[leaf.artifact.identity], audit)
        counts[key.scope] = counts.get(key.scope, 0) + len(leaf.reports)
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "structural_integrity": True,
            "report_replay_counts": counts, "retained_records": len(evidence),
            "publisher_witnesses": verify_workbook_witnesses(store),
            "scope": ("Reviewed 2025–26 campus, district and Texas student projection, with explicit raw field holds"
                      if complete_campuses else
                      "Reviewed 2025–26 district and Texas student projection, with explicit raw field holds")}


def _check_roster_regions(roster: Mapping[str, Any], scope: str) -> None:
    if scope == "campus":
        _need(bool(roster) and set(roster) <= set(REGIONS), "Campus cohort requires explicit valid regions")
    else:
        _need(set(roster) == set(REGIONS), "Statewide selection requires all 20 regional rosters")


def campus_partition_ids(identifiers: Iterable[str], partition: int) -> tuple[str, ...]:
    """Select one fixed slice of an already verified publisher campus roster."""
    _need(type(partition) is int and partition >= 0, "Campus partition requires a nonnegative integer ordinal")
    ordered = sorted(identifiers)
    start = partition * CAMPUS_PARTITION_REPORTS
    _need(start < len(ordered), "Campus partition is outside the pinned regional roster")
    return tuple(ordered[start:start + CAMPUS_PARTITION_REPORTS])


def roster_membership(roster: Mapping[str, Any], records: Mapping[str, Any], *,
                      scope: str = "district", partition: int | None = None) -> dict[str, str]:
    """Return publisher identifiers and exact regional roster receipts."""
    _check_roster_regions(roster, scope)
    if partition is not None:
        _need(scope == "campus" and len(roster) == 1, "Campus partition requires exactly one regional roster")
    limit = MAX_CAMPUS_ROSTER_BYTES if scope == "campus" else MAX_ROSTER_BYTES
    members: dict[str, str] = {}
    for region, entry in sorted(roster.items()):
        _need(isinstance(entry, dict) and set(entry) == {"receipt", "response"}
              and isinstance(entry["receipt"], str) and isinstance(entry["response"], str),
              "Invalid roster entry")
        receipt = entry["receipt"]
        record = records[receipt]
        response = entry["response"].encode("utf-8")
        for identifier in _regional_roster(response, record, region, scope, limit):
            _need(identifier not in members, f"Duplicate {scope} across regional rosters")
            members[identifier] = receipt
    if partition is not None:
        return {identifier: members[identifier] for identifier in campus_partition_ids(members, partition)}
    if scope == "campus":
        _need(len(members) <= MAX_CAMPUS_COHORT_REPORTS, "Campus cohort exceeds its report count budget")
    else:
        _need(len(members) == DISTRICT_COUNT, "Statewide roster size differs from the reviewed edition")
    return members


def campus_report_has_roster(record: Mapping[str, Any], receipt: str) -> bool:
    """A campus report must have an explicit edition and its sole roster parent."""
    metadata = record.get("source_metadata", {})
    return (isinstance(metadata, Mapping) and record.get("requested_year") == record.get("resolved_year") == EDITION
            and metadata.get("version") == str(EDITION) and metadata.get("provenance_ids") == [receipt])


def validate_report_ledger(reports: tuple[ReportSelection, ...], members: Mapping[str, str],
                           records: Mapping[str, Any], *, scope: str | None = None) -> None:
    if scope == "campus":
        _need(all(entry.scope == "campus" for entry in reports)
              and [entry.identifier for entry in reports] == sorted(members),
              "Campus report ledger differs from the exact ordered regional roster union")
    else:
        districts = [entry.identifier for entry in reports if entry.scope == "district"]
        state = [entry for entry in reports if entry.scope == "state"]
        _need(districts == ([] if scope == "state" else sorted(members))
              and [entry.identifier for entry in state] == ([] if scope == "district" else ["S"])
              and len(reports) == len(districts) + len(state), "Report ledger does not cover the exact roster and Texas singleton")
    _need(len({entry.receipt for entry in reports}) == len({entry.sha256 for entry in reports}) == len(reports),
          "Report ledger repeats a receipt or source body")
    total = 0
    for entry in reports:
        record = records[entry.receipt]
        _validate_report_acquisition(record, scope=entry.scope, identifier=entry.identifier, expected_sha256=entry.sha256)
        metadata = record["source_metadata"]
        if entry.scope == "campus":
            _need(campus_report_has_roster(record, members[entry.identifier]),
                  "Campus report receipt lacks its sole exact regional roster parent or explicit edition")
        else:
            _need(record["requested_year"] == record["resolved_year"] == EDITION
                  and metadata.get("version") == str(EDITION), "Statewide ledger requires original explicit-edition receipts")
            if entry.scope == "district":
                _need(members[entry.identifier] in metadata.get("provenance_ids", ()),
                      "District report receipt lacks its exact regional roster parent")
        total += record["artifact"]["size_bytes"]
    _need(total <= MAX_STATEWIDE_HTML_BYTES, "Roster report ledger exceeds the aggregate source byte budget")


def _admit_roster_selection(store: ProvenanceStore, document: dict[str, Any], *, scope: str) -> RosterSelection:
    campus = scope == "campus"
    partitioned = campus and document.get("schema_version") == CAMPUS_PARTITION_VERSION
    version = CAMPUS_COHORT_VERSION if campus else 2
    fields = {"schema_version", "dataset", "edition", "rosters", "reports"}
    if partitioned:
        version = CAMPUS_PARTITION_VERSION
        fields.add("partition")
    _need(set(document) == fields
          and type(document["schema_version"]) is int and document["schema_version"] == version
          and document["dataset"] == DATASET and document["edition"] == str(EDITION), "Invalid roster-bound student intake")
    rosters = document["rosters"]
    _need(isinstance(rosters, dict)
          and all(isinstance(value, str) and _DIGEST.fullmatch(value) for value in rosters.values()), "Invalid roster receipt pins")
    _check_roster_regions(rosters, scope)
    partition = document.get("partition")
    if partitioned:
        _need(type(partition) is int and partition >= 0, "Campus partition requires a nonnegative integer ordinal")
        _need(len(rosters) == 1, "Campus partition requires exactly one regional roster")
    roster = {}
    limit = MAX_CAMPUS_ROSTER_BYTES if campus else MAX_ROSTER_BYTES
    for region, receipt in sorted(rosters.items()):
        record = store.read_record(receipt)
        _need(record["kind"] == "acquisition", "Roster receipt must identify an acquisition")
        digest = record["artifact"]["sha256"]
        size = record["artifact"]["size_bytes"]
        _need(type(size) is int and 0 < size <= limit, "Roster response exceeds its byte budget")
        path = store.root / "objects" / digest
        _need(path.is_file() and path.stat().st_size == size, "Roster response differs from its declared size")
        with store.verify_artifact(digest).open("rb") as stream:
            response = stream.read(size + 1)
        roster[region] = {"receipt": receipt, "response": response.decode("utf-8")}
    records = {receipt: store.read_record(receipt) for receipt in rosters.values()}
    members = roster_membership(roster, records, scope=scope, partition=partition)
    rows = document["reports"]
    _need(isinstance(rows, list) and len(rows) == len(members) + (0 if campus else 1),
          "Incomplete campus cohort report selection" if campus else "Incomplete statewide report selection")
    reports = []
    for row in rows:
        _need(isinstance(row, dict) and set(row) == {"scope", "identifier", "receipt", "sha256"}
              and all(isinstance(value, str) for value in row.values())
              and _DIGEST.fullmatch(row["receipt"]) is not None and _DIGEST.fullmatch(row["sha256"]) is not None,
              "Invalid report receipt pins")
        reports.append(ReportSelection(**row))
    selected = tuple(reports if campus else sorted(reports, key=lambda item: (item.scope, item.identifier)))
    report_records = {entry.receipt: store.read_record(entry.receipt) for entry in selected}
    validate_report_ledger(selected, members, report_records, scope="campus" if campus else None)
    if campus:
        # The declared aggregate alone cannot bound a corrupt retained object.
        # Reject nonregular or expanded bodies before the provenance copier reads them.
        for entry in selected:
            state = (store.root / "objects" / entry.sha256).stat(follow_symlinks=False)
            _need(S_ISREG(state.st_mode) and state.st_size == report_records[entry.receipt]["artifact"]["size_bytes"],
                  "Retained campus report differs from its declared regular file size")
    return RosterSelection(selected, roster, partition)


def admit_statewide_selection(store: ProvenanceStore, document: dict[str, Any]) -> RosterSelection:
    """Admit the complete district and direct Texas selection."""
    return _admit_roster_selection(store, document, scope="district")


def admit_campus_cohort(store: ProvenanceStore, document: dict[str, Any]) -> RosterSelection:
    """Admit complete campus regions or one exact partition of a region."""
    return _admit_roster_selection(store, document, scope="campus")


STATEWIDE_OPERATION = "statewide_tprs_student_membership_enrollment"
CAMPUS_COHORT_OPERATION = "regional_cohort_tprs_student_membership_enrollment"
CAMPUS_COHORT_LEDGER_VERSION = 3
CAMPUS_PARTITION_OPERATION = "regional_partition_tprs_student_membership_enrollment"
CAMPUS_PARTITION_LEDGER_VERSION = 4
MAX_AUDIT_LINE_BYTES = 4096
MAX_AUDIT_BYTES = 128 * 1024**2
MAX_AUDIT_ROWS = 1_000_000


def student_report_resources(record: Mapping[str, Any]) -> dict[str, ColumnSchemaRef]:
    """Route reviewed attachments by their processing output, never by a path in the intake."""
    recipe = record.get("parameters", {}).get("recipe", {})
    operation = recipe.get("operation")
    if operation not in (STATEWIDE_OPERATION, CAMPUS_COHORT_OPERATION, CAMPUS_PARTITION_OPERATION):
        _need("report_ledger_version" not in recipe, "Unsupported statewide student ledger operation")
        return {}
    partitioned = operation == CAMPUS_PARTITION_OPERATION
    campus = operation != STATEWIDE_OPERATION
    version = CAMPUS_PARTITION_LEDGER_VERSION if partitioned else CAMPUS_COHORT_LEDGER_VERSION if campus else 2
    _need(type(recipe.get("report_ledger_version")) is int and recipe["report_ledger_version"] == version
          and record.get("processor") == "teadata.tprs_students"
          and set(record["outputs"]) == {"leaf", "roster", "raw_audit", "grid"}, "Invalid roster-bound student output contract")
    if campus:
        _need("cohort" not in recipe and type(recipe.get("parquet_row_group_max_rows")) is int
              and recipe["parquet_row_group_max_rows"] == _STATEWIDE_ROW_GROUP_ROWS, "Invalid campus student recipe")
        if partitioned:
            partition = recipe.get("campus_partition")
            _need(isinstance(partition, dict) and set(partition) == {"region", "ordinal"}
                  and partition["region"] in REGIONS and type(partition["ordinal"]) is int and partition["ordinal"] >= 0
                  and "campus_regions" not in recipe, "Invalid campus partition recipe")
        else:
            regions = recipe.get("campus_regions")
            _need(isinstance(regions, list) and bool(regions) and all(isinstance(region, str) and region in REGIONS for region in regions)
                  and sorted(set(regions)) == regions and "campus_partition" not in recipe, "Invalid campus cohort recipe")
    else:
        pair = recipe.get("cohort")
        _need(isinstance(pair, dict) and set(pair) == {"district", "state"}
              and all(isinstance(item, str) and _DIGEST.fullmatch(item) for item in pair.values())
              and pair["district"] != pair["state"], "Statewide student recipe requires exact district and Texas editions")
    refs = {}
    for role, suffix in (("roster", "json"), ("raw_audit", "jsonl.gz"), ("grid", "json")):
        try:
            artifact = record["outputs"][role]
            digest = artifact["sha256"]
            _need(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None, "Invalid student attachment identity")
            refs[role] = ColumnSchemaRef(digest, artifact["size_bytes"], f"tprs-{role}-{digest}.{suffix}")
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"Invalid Student {role} attachment descriptor: {error}") from error
    _need(refs["grid"].sha256 == recipe.get("definition_sha256"), "Student audit grid differs from its recipe pin")
    return refs


def _partition_schema_parents(edition: Mapping[str, Any], records: Mapping[str, Any]) -> tuple[str, str]:
    selected_digest = edition["column_schema_sha256"]
    references = [ref for ref in edition["metadata"]["references"]
                  if ref["role"] == "schema" and ref.get("reviewed_sha256") == selected_digest]
    _need(len(references) == 1 and references[0]["reference_id"] == "columns" and len(references[0]["provenance_ids"]) == 1,
          "Campus partition requires an exact selected schema observation")
    schema = records[references[0]["provenance_ids"][0]]
    _need(schema["dataset"] == "reference:tprs_student_selected_projection_schema" and schema["kind"] == "acquisition"
          and schema["method"] == "local_observation" and schema["requested_year"] == schema["resolved_year"] == EDITION
          and schema["artifact"]["sha256"] == selected_digest,
          "Campus partition requires an exact selected schema observation")
    parents = schema["source_metadata"]["provenance_ids"]
    intakes = [identifier for identifier in parents if records[identifier]["dataset"] == "reference:tprs_student_intake"]
    configs = [identifier for identifier in parents if records[identifier]["dataset"] == "reference:tprs_student_config"]
    _need(len(intakes) == len(configs) == 1, "Campus partition schema must bind one exact intake and configuration observation")
    return intakes[0], configs[0]


def student_report_ledger(document: Mapping[str, Any], key: EditionKey, roster: bytes,
                          processing: Mapping[str, Any]) -> tuple[ReportSelection, ...]:
    records = document["provenance"]
    inputs = processing["inputs"]
    intake_ids = [identifier for identifier in inputs if records[identifier]["dataset"] == "reference:tprs_student_intake"]
    intakes = [records[identifier] for identifier in intake_ids]
    recipe = processing["parameters"]["recipe"]
    pin = recipe.get("intake_sha256")
    _need(isinstance(pin, str) and _DIGEST.fullmatch(pin) is not None and len(intakes) == 1
          and intakes[0]["kind"] == "acquisition" and intakes[0]["method"] == "local_observation"
          and intakes[0]["requested_year"] == intakes[0]["resolved_year"] == EDITION
          and intakes[0]["source_metadata"].get("version") == str(EDITION)
          and intakes[0]["artifact"]["sha256"] == pin, "Student ledger differs from its direct exact intake receipt")
    body = json.loads(roster, object_pairs_hook=_unique_object)
    _need(isinstance(body, dict), "Invalid student roster attachment")
    partitioned = recipe["operation"] == CAMPUS_PARTITION_OPERATION
    campus = partitioned or recipe["operation"] == CAMPUS_COHORT_OPERATION
    partition = recipe["campus_partition"]["ordinal"] if partitioned else None
    members = roster_membership(body, records, scope="campus" if campus else "district", partition=partition)
    roster_receipts = {entry["receipt"] for entry in body.values()}
    _need(roster_receipts <= set(inputs), "Roster receipts are not direct processing inputs")
    receipts = document["edition"]["acquisition_ids"]
    _need(set(receipts) <= set(inputs), "Report receipts are not direct processing inputs")
    reports = tuple(ReportSelection(key.scope,
        dict(parse_qsl(urlsplit(records[receipt]["requested_url"]).query))["id"],
        receipt, records[receipt]["artifact"]["sha256"]) for receipt in receipts)
    validate_report_ledger(reports, members, records, scope=key.scope)
    if campus:
        intake = {"schema_version": CAMPUS_COHORT_VERSION, "dataset": DATASET, "edition": str(EDITION),
                  "rosters": {region: entry["receipt"] for region, entry in body.items()},
                  "reports": [asdict(report) for report in reports]}
        if partitioned:
            _need(set(body) == {recipe["campus_partition"]["region"]}, "Campus partition recipe differs from the exact roster region")
            intake_id, config_id = _partition_schema_parents(document["edition"], records)
            expected = {*roster_receipts, *receipts, intake_id, config_id}
            _need(intake_ids == [intake_id] and len(inputs) == len(expected) and set(inputs) == expected,
                  "Campus partition requires exact direct processing inputs bound by its selected schema")
            config = records[config_id]
            _need(config["kind"] == "acquisition" and config["method"] == "local_observation"
                  and config["requested_year"] == config["resolved_year"] == EDITION
                  and config["source_metadata"].get("version") == str(EDITION), "Invalid campus partition configuration observation")
            intake.update(schema_version=CAMPUS_PARTITION_VERSION, partition=partition)
        else:
            _need(sorted(body) == recipe["campus_regions"], "Campus cohort recipe differs from the exact roster regions")
        raw = (canonical_json(intake) + "\n").encode()
        _need(sha256(raw).hexdigest() == pin and len(raw) == intakes[0]["artifact"]["size_bytes"],
              "Campus ledger does not reproduce its exact canonical intake")
    return reports


def audit_rows(raw: bytes, reports: tuple[ReportSelection, ...]) -> Iterator[dict[str, Any]]:
    """Stream the exact scope's audited source coordinates with finite bounds."""
    receipts = {entry.receipt: number for number, entry in enumerate(reports)}
    fields = {"receipt", "row", "column", "section", "label", "population", "measure", "reason",
              "token_status", "token_reason", "raw"}
    statuses = {status.value for status in ValueStatus}
    decode = json.JSONDecoder(object_pairs_hook=_unique_object).decode
    last = (-1, 0, 0)
    total = 0
    with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
        for number in range(MAX_AUDIT_ROWS + 1):
            line = stream.readline(MAX_AUDIT_LINE_BYTES + 1)
            if not line:
                return
            total += len(line)
            _need(number < MAX_AUDIT_ROWS and total <= MAX_AUDIT_BYTES and len(line) <= MAX_AUDIT_LINE_BYTES
                  and line.endswith(b"\n"), "Student raw audit exceeds its bounds")
            # A quoted key immediately after the brace identifies the usual UTF-8
            # wire form. Keep json.loads encoding detection for other byte prefixes.
            row = (decode(line.decode("utf-8", "surrogatepass")) if line.startswith(b'{"')
                   else json.loads(line, object_pairs_hook=_unique_object))
            _need(isinstance(row, dict) and row.keys() == fields, "Invalid student raw audit record")
            receipt, source_row, column = row["receipt"], row["row"], row["column"]
            status, reason, token_reason = row["token_status"], row["reason"], row["token_reason"]
            report = receipts.get(receipt) if isinstance(receipt, str) else None
            _need(report is not None and type(source_row) is int and source_row > 0
                  and type(column) is int and column > 0
                  and isinstance(row["section"], str) and isinstance(row["label"], str)
                  and isinstance(row["population"], str) and isinstance(row["measure"], str)
                  and isinstance(row["raw"], str) and isinstance(status, str) and status in statuses
                  and (reason is None or isinstance(reason, str))
                  and (token_reason is None or isinstance(token_reason, str)), "Invalid student raw audit record")
            assert report is not None
            current = (report, source_row, column)
            _need(current > last, "Duplicate or unordered student raw audit coordinates")
            last = current
            yield row


def verify_audit_coverage(raw: bytes, reports: tuple[ReportSelection, ...], grid_bytes: bytes) -> int:
    """Every report must carry each unbound local slot in its reviewed grid."""
    grid = json.loads(grid_bytes, object_pairs_hook=_unique_object)
    scope = reports[0].scope
    fields = {(f["section"], f["label"], f["population"], f["quantity"]) for f in semantic_fields(grid)}
    expected = {}
    for spec in grid["rows"]:
        section, label = spec["section"], spec["label"]
        for index, (population, quantity, role) in enumerate(grid["column_roles"][scope], 2):
            quantity = spec.get("quantity_overrides", {}).get(scope, {}).get(role, quantity)
            if role == "local" and (section, label, population, quantity) not in fields:
                expected[section, label, index] = population, quantity, spec["hold"]
    rows = audit_rows(raw, reports)
    for report in reports:
        seen = set()
        for _ in expected:
            row = next(rows, None)
            _need(row is not None and row["receipt"] == report.receipt, "Student audit omits a required report slot")
            assert row is not None
            slot = row["section"], row["label"], row["column"]
            _need(slot not in seen and expected.get(slot) == (row["population"], row["measure"], row["reason"]),
                  "Student audit has an invalid or repeated report slot")
            seen.add(slot)
    _need(next(rows, None) is None, "Student audit contains extra report slots")
    return len(reports) * len(expected)
