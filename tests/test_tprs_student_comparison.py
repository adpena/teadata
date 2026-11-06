from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from teadata import tprs_students as students
from teadata.analytical import write_leaf_artifact
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset, STATEWIDE_SINGLETON_GRAIN
from teadata.entities import Campus, District
from teadata.manifest import BuildRecorder, ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease, publish_release
from teadata.statewide import _reviewed_students
from teadata.teadata_config import Config, load_config

from test_tprs_students import html, receipt, report_rows


@pytest.fixture(scope="module")
def prepared_students(tmp_path_factory):
    root = tmp_path_factory.mktemp("student-comparison")
    store = ProvenanceStore(root / "provenance")
    cfg = load_config()
    selected = cfg.resolve_dataset(students.DATASET, students.EDITION, reader_profile="source", exact=True)
    metadata = selected.metadata.to_dict()
    grid = students.definition()
    # Synthetic reference receipts qualify this fixture's contract only. Real
    # publisher parity is checked separately against the sealed native bundle.
    for index, ref in enumerate(metadata["references"]):
        path = root / f"reference-{index}.txt"
        path.write_text("Synthetic interpretation reference " + ref["reference_id"])
        rid = store.record_acquisition(path, ResolvedDataset("reference:synthetic", "reference", 2026, 2026, "", DatasetMetadata()))
        ref.update(path=path.name, url=None, reviewed_sha256=sha256(path.read_bytes()).hexdigest(), provenance_ids=[rid])
    reports, observations = {}, {}
    for scope, identifier in (("state", "S"), ("district", "001902"), ("campus", "001902001")):
        rows = report_rows(grid, scope, identifier, omitted=["Grade 12"] if scope == "campus" else [])
        roles = grid["column_roles"][scope]
        for row in rows:
            if row[0] == "Total Students":
                for index, (population, quantity, role) in enumerate(roles, 1):
                    if role == "local":
                        row[index] = ("1,000" if scope == "state" else "0") if quantity == "count" else "100.0%"
            elif row[0] == "Special Education":
                for index, (population, _, role) in enumerate(roles, 1):
                    if role == "local" and scope != "state":
                        row[index] = "*" if population == "membership" else "-"
        raw = html(rows)
        path = root / f"{scope}.html"
        path.write_bytes(raw)
        original = receipt(store, root, raw, scope=scope, identifier=identifier)
        reports[scope] = students.parse_student_report(raw, scope=scope, identifier=identifier, receipt=original)
        observations[scope] = store.record_acquisition(path, ResolvedDataset(students.DATASET, "data_sources", 2026, 2026, "",
            DatasetMetadata(version="2026", provenance_ids=(original,))),
            parameters={"scope": scope, "identifier": identifier, "original_download_receipt": original})
    schema = students._crosswalk({scope: [report.sha256] for scope, report in reports.items()})
    path = root / "selected.json"
    path.write_text(canonical_json(schema) + "\n")
    rid = store.record_acquisition(path, ResolvedDataset("reference:synthetic-schema", "reference", 2026, 2026, "", DatasetMetadata()))
    ref = next(ref for ref in metadata["references"] if ref["reference_id"] == "columns")
    ref.update(path=path.name, reviewed_sha256=sha256(path.read_bytes()).hexdigest(), provenance_ids=[rid])
    assert selected.metadata.value_policy is not None
    policy = cfg.value_policies[selected.metadata.value_policy]
    document = {"data_sources": {students.DATASET: {2026: {"path": "source-only/report.html", "metadata": metadata,
        "reader": {"profiles": {"source": {"column_schema": path.name}}}}}},
        "value_policies": {policy.policy_id: policy.to_dict()}}
    config = Config.from_dict(document)
    config.base_dir = root
    source = config.resolve_dataset(students.DATASET, 2026, exact=True, reader_profile="source")
    recorder = BuildRecorder(config, store)
    editions = {}
    for scope, report in reports.items():
        table, support, _ = students.projection([report], policy)
        keys = [] if scope == "state" else [scope + "_number"]
        key = EditionKey(students.DATASET, "2026", EditionKey.source_revision((report.sha256,)), scope, scope,
                         STATEWIDE_SINGLETON_GRAIN if not keys else "one row per " + keys[0])
        recipe = {"operation": "synthetic student comparison fixture"}
        contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
            columns=table.column_names, key_columns=keys, rows=1, inputs=[report.sha256], recipe=recipe,
            toolchain={"fixture": "synthetic"}, column_support=support)
        identity = sha256(canonical_json(contract).encode()).hexdigest()
        leaf = root / (identity + ".parquet")
        physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract)
        process = store.record_processing(dataset=students.DATASET, inputs=[observations[scope], rid], outputs={"leaf": leaf},
            processor="synthetic fixture", code_version=recorder.processor_sha256,
            parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": {"fixture": "synthetic"}})
        ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), process, "leaf")
        recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[observations[scope]])
        editions[scope] = key
    publish_release(root, recorder.catalog(schema_version=4), schema_version=4)
    return root, config, editions


@pytest.fixture
def comparison(prepared_students, monkeypatch):
    root, config, editions = prepared_students
    monkeypatch.setattr("teadata.statewide.load_config", lambda: config)
    return DataRelease(root / "release.json", offline=True), editions


def district(identifier="001902"):
    return District(uuid4(), "Synthetic district", district_number=identifier)


def campus():
    return Campus(uuid4(), None, "Synthetic campus", "", False, campus_number="001902001")


def test_entity_sources_discover_without_leaf_io_and_keep_entity_values_separate(comparison, monkeypatch):
    release, editions = comparison
    entity = district()
    entity.meta['membership_total_students_count'] = 999
    with monkeypatch.context() as guard:
        guard.setattr(release, 'artifact', lambda *a, **k: pytest.fail('Discovery opened a native table'))
        sources = release.for_entity(entity, edition='2026')
        assert sources.datasets() == (editions['district'],)
        fields = sources.tprs_student_membership_enrollment_reports
        variables = fields.variables(r'^membership_total_students_(?:count|percent)$')
        assert {value['field'] for value in variables} == {'membership_total_students_count', 'membership_total_students_percent'}
        with pytest.raises(ContractError, match='complete raw TAPR table'):
            release.to_csv(editions['district'], BytesIO(), restore_original_headers=True)
    assert fields.membership_total_students_count == 0
    assert entity.membership_total_students_count == 999
    selected = fields.select(*(row['field'] for row in variables)).to_pylist()
    assert selected[0]['tprs_raw_membership_total_students_count'] == '0'
    with pytest.raises(AttributeError):
        _ = fields.unpublished
    with pytest.raises(ContractError, match='exactly one published row'):
        release.for_entity(district('001903'), edition='2026')[students.DATASET]['membership_total_students_count']


def test_all_168_fields_use_the_direct_texas_leaf_and_exact_population(comparison):
    release, keys = comparison
    for scope, entity in (("district", district()), ("campus", campus())):
        rows = list(Query([entity], repo=None).compare_statewide(release=release, local=keys[scope], statewide=keys["state"]))
        assert len(rows) == 168 and {row.metric for row in rows} == {f["field"] for f in students.semantic_fields()}
        count = next(row for row in rows if row.metric == "membership_total_students_count")
        assert (count.identifier, count.local_raw, count.local_value, count.statewide_raw,
                count.statewide_value, count.difference, count.difference_pp) == (
                    "001902" if scope == "district" else "001902001", "0", 0, "1,000", 1000, -1000, None)
        assert count.period == students.PERIOD and count.local_edition == keys[scope]
        assert count.statewide_edition == keys["state"]
        percent = next(row for row in rows if row.metric == "membership_total_students_percent")
        assert (percent.unit, percent.difference_pp) == ("percent", 0)
        for population, status in (("membership", "suppressed"), ("enrollment", "not_applicable")):
            masked = next(row for row in rows if row.metric == f"{population}_special_education_count")
            assert masked.local_status == status and masked.local_value is None and masked.difference is None


def test_missing_grade_absent_entity_and_private_population_are_distinct(comparison):
    release, keys = comparison
    row, = Query([campus()], repo=None).compare_statewide(release=release, local=keys["campus"],
        statewide=keys["state"], metric="membership_grade_12_count")
    assert (row.local_raw, row.local_status, row.local_value, row.difference) == (None, "missing", None, None)
    absent, = Query([district("999999")], repo=None).compare_statewide(release=release, local=keys["district"],
        statewide=keys["state"], metric="membership_total_students_count")
    assert absent.local_status == "unavailable"
    private = Campus(uuid4(), None, "Private", "", False, is_private=True, source_namespace="tefa", source_id="local")
    result, = Query([private], repo=None).compare_statewide(release=release, local=keys["campus"],
        statewide=keys["state"], metric="membership_total_students_count")
    assert result.local_status == "not_applicable"


def test_entities_supply_only_identifiers_without_rebinding_scalar_catalogs(comparison):
    release, keys = comparison
    entity = district()
    entity._repo = SimpleNamespace(catalog=SimpleNamespace(build_id="different entity build"))
    entity.meta["membership_total_students_count"] = 999999
    row, = Query([entity], repo=entity._repo).compare_statewide(release=release, local=keys["district"],
        statewide=keys["state"], metric="membership_total_students_count")
    assert row.local_value == 0 and entity.meta["membership_total_students_count"] == 999999
    with pytest.raises(ContractError, match="another data release"):
        release.catalog._entity_build_id(entity)


@pytest.mark.parametrize("damage", ["period", "metric", "definition", "policy", "dimensions", "revision", "grain", "scope", "year"])
def test_unreviewed_contracts_refuse_before_observations(comparison, monkeypatch, damage):
    release, keys = comparison
    local, state = keys["district"], keys["state"]
    dimensions = None
    if damage in {"period", "metric", "definition"}:
        original = release.catalog.metadata
        def changed(dataset, *, edition):
            metadata = original(dataset, edition=edition)
            if edition.scope != "district":
                return metadata
            if damage == "period":
                return replace(metadata, period=replace(metadata.period, start_year=2024, end_year=2025))
            if damage == "definition":
                return replace(metadata, references=metadata.references[:-1])
            metrics = dict(metadata.metrics)
            name = "membership_total_students_count"
            metrics[name] = replace(metrics[name], field="enrollment_total_students_count")
            return replace(metadata, metrics=metrics)
        monkeypatch.setattr(release.catalog, "metadata", changed)
    elif damage == "policy":
        policy = release.catalog.policy(students.DATASET, edition=local)
        monkeypatch.setattr(release.catalog, "policy", lambda *a, **k: replace(policy, policy_id="wrong"))
    elif damage == "dimensions":
        dimensions = {"population": "membership"}
    elif damage == "revision":
        local = replace(local, revision_sha256="0" * 64)
    elif damage == "grain":
        local = replace(local, grain="invented grain")
    elif damage == "year":
        state = replace(state, edition="2025")
    else:
        local = replace(local, scope="region")
    monkeypatch.setattr(release, "analytical_dataset", lambda *_a, **_k: pytest.fail("Unreviewed contract opened observations"))
    with pytest.raises(ContractError):
        Query([district()], repo=None).compare_statewide(release=release, local=local, statewide=state, dimensions=dimensions)


@pytest.mark.parametrize("damage", ["inputs", "mapping", "policy", "coverage", "support", "lineage", "duplicate",
                                    "relabel", "observation-method", "parent-size", "parent-sha"])
def test_exact_native_and_lineage_bindings_refuse_drift(comparison, monkeypatch, damage):
    release, keys = comparison
    if damage in {"lineage", "duplicate", "relabel", "observation-method", "parent-size", "parent-sha"}:
        acquisitions = release.catalog.acquisitions
        def changed_acquisitions(dataset, *, edition):
            rows = deepcopy(acquisitions(dataset, edition=edition))
            if edition.scope == "district":
                if damage == "duplicate":
                    rows.append(rows[0])
                elif damage == "relabel":
                    rows[0]["parameters"]["identifier"] = "001903"
                elif damage == "observation-method":
                    rows[0]["method"] = "download"
                elif damage == "parent-size":
                    rows[0]["artifact"]["size_bytes"] += 1
                elif damage == "parent-sha":
                    rows[0]["artifact"]["sha256"] = "0" * 64
                else:
                    rows[0]["parameters"]["original_download_receipt"] = "0" * 64
            return rows
        monkeypatch.setattr(release.catalog, "acquisitions", changed_acquisitions)
    else:
        resolve = release.catalog.resolve_artifacts
        def changed_plans(fields, *, period=None):
            plans = resolve(fields, period=period)
            changed = []
            for plan in plans:
                artifact = plan["artifact"]
                contract = artifact.contract
                if artifact.edition.scope == "district":
                    if damage == "inputs":
                        contract["inputs"] = ["0" * 64]
                    elif damage == "coverage":
                        contract["rows"] = 2
                    elif damage == "support":
                        contract["column_support"]["fields"]["membership_total_students_count"]["auxiliary"].reverse()
                    else:
                        contract[damage + "_sha256"] = "0" * 64
                    artifact = replace(artifact, contract_json=canonical_json(contract))
                changed.append({**plan, "artifact": artifact})
            return tuple(changed)
        monkeypatch.setattr(release.catalog, "resolve_artifacts", changed_plans)
    monkeypatch.setattr(release, "analytical_dataset", lambda *_a, **_k: pytest.fail("Unreviewed binding opened observations"))
    with pytest.raises(ContractError):
        Query([district()], repo=None).compare_statewide(release=release, local=keys["district"], statewide=keys["state"])


@pytest.mark.parametrize("metric", ["membership_technical_education_count", "membership_disability_count", "district_2026_enrollment"])
def test_raw_holds_and_aliases_do_not_become_metrics(comparison, monkeypatch, metric):
    release, keys = comparison
    monkeypatch.setattr(release, "analytical_dataset", lambda *_a, **_k: pytest.fail("Unreviewed metric opened observations"))
    with pytest.raises(ContractError, match="reviewed numeric"):
        Query([district()], repo=None).compare_statewide(release=release, local=keys["district"], statewide=keys["state"], metric=metric)


@pytest.mark.parametrize("change", [
    {"raw": None, "status": "observed", "value": 0},
    {"raw": None, "status": "missing", "reason": "absent"},
    {"raw": None, "status": "missing", "reason": "row_not_published", "row": 12},
    {"raw": "", "status": "missing", "reason": "absent"},
    {"raw": ".", "status": "missing"},
    {"raw": "**", "status": "suppressed"},
    {"raw": "1", "status": "observed", "value": True, "reason": None},
    {"raw": "1", "status": "observed", "value": 0, "reason": None},
])
def test_null_grade_and_typed_tokens_fail_closed(comparison, change):
    release, keys = comparison
    typed = _reviewed_students(release, keys["campus"])[-1]
    field = "membership_grade_12_count"
    row = {"campus_number": "001902001", "tprs_report_receipt": typed.sources["001902001"][0],
           "tprs_report_sha256": typed.sources["001902001"][1], field: None}
    values: dict[str, Any] = dict(raw=None, status="missing", row=None, column=None, reason="row_not_published", value=None)
    if change.get("raw") is not None:
        values.update(row=12, column=typed.local_columns["membership", "count"])
    values.update(change)
    row[field] = values.pop("value")
    row.update({f"tprs_{kind}_{field}": value for kind, value in values.items()})
    with pytest.raises(ContractError):
        typed.value(row, field, "count")


def test_state_and_non_grade_rows_cannot_inherit_campus_absence_exception(comparison):
    release, keys = comparison
    for scope, field in (("state", "membership_grade_12_count"), ("campus", "membership_total_students_count")):
        typed = _reviewed_students(release, keys[scope])[-1]
        identifier = "S" if scope == "state" else "001902001"
        row = {"campus_number": identifier, "tprs_report_receipt": typed.sources[identifier][0],
               "tprs_report_sha256": typed.sources[identifier][1], field: None,
               **{f"tprs_{kind}_{field}": value for kind, value in dict(raw=None, status="missing", row=None,
                       column=None, reason="row_not_published").items()}}
        with pytest.raises(ContractError, match="missing campus grade"):
            typed.value(row, field, "count")


def test_row_context_cannot_substitute_another_reports_receipt(comparison):
    release, keys = comparison
    field = "membership_total_students_count"
    typed = _reviewed_students(release, keys["district"])[-1]
    plan, = release.catalog.resolve_artifacts([(keys["district"], field)])
    row, = release.analytical_dataset(plan["artifact"].identity).to_table(columns=list(plan["columns"])).to_pylist()
    assert typed.value(row, field, "count") == ("0", "observed", 0)
    row["tprs_report_receipt"] = "0" * 64
    with pytest.raises(ContractError, match="source binding"):
        typed.value(row, field, "count")
