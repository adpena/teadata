from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZipFile

import pyarrow as pa
import pytest
import yaml

from teadata import tprs_students as students
from teadata.column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import load_config
from teadata.values import ValueStatus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grid():
    return students.definition()


@pytest.fixture(scope="module")
def policy():
    cfg = load_config()
    return cfg.value_policies["tprs_student_selected_2026"]


def report_rows(grid, scope="district", identifier="001902", *, omitted=()):
    rows = [["", "Texas Education Agency", ""], ["2025-26 Student Information"],
            ["State" if scope == "state" else f"SYNTHETIC SCHOOL ({identifier})"], [""],
            ["", "Membership", "Enrollment"]]
    if scope != "state":
        rows.append(grid["geography_headers"][scope])
    rows.append(grid["headers"][scope])
    section = ""
    for spec in grid["rows"]:
        if spec["section"] != section:
            section = spec["section"]
            rows.append([section])
        if spec["label"] not in omitted:
            rows.append([spec["label"], *["1" if spec.get("quantity_overrides", {}).get(scope, {}).get(role[2], role[1]) == "count" else "1.0%"
                         for role in grid["column_roles"][scope]]])
    return [*rows, ["Mobility (2024-25)"], *grid["legend"]]


def html(rows):
    return ("<table>" + "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
                                for row in rows) + "</table>").encode("cp1252")


def parse(grid, **kwargs):
    scope, identifier = kwargs.pop("scope", "district"), kwargs.pop("identifier", "001902")
    return students.parse_student_report(html(report_rows(grid, scope, identifier, **kwargs)),
                                         scope=scope, identifier=identifier)


def receipt(store, tmp_path, raw, *, identifier="001902", scope="district", overrides=None, legacy=False):
    controls = {"_service": "marykay", "_program": "perfrept.perfmast.sas", "_debug": "0", "ccyy": "2026",
                "id": identifier, "lev": students.SCOPES[scope], "prgopt": "reports/tapr/student.sas"}
    url = students.BROKER + "?" + urlencode(controls)
    options = {"downloaded_at": datetime(2026, 11, 4, tzinfo=timezone.utc), "requested_url": url,
               "resolved_url": url, "response_headers": {"content-type": "text/html; charset=windows-1252"},
               "parameters": {"http_status": 200, **({"controls": controls} if legacy else {"http_method": "GET"})}}
    options.update(overrides or {})
    source = tmp_path / "synthetic-student.html"
    source.write_bytes(raw)
    return store.record_acquisition(source, ResolvedDataset("reference:synthetic-student", "reference",
        None if legacy else 2026, None if legacy else 2026, "",
        DatasetMetadata(title="Synthetic student report", version=None if legacy else "2026")), **options)


def test_packaged_student_contract_is_explicit_and_source_only(grid):
    cfg = load_config()
    assert cfg.is_source_only(students.DATASET, 2026)
    assert cfg.column_schema(students.DATASET, 2026) is None
    source = cfg.resolve_dataset(students.DATASET, 2026, reader_profile="source", exact=True)
    fields = students.semantic_fields(grid)
    assert len(fields) == len(source.metadata.metrics) == 168
    assert set(source.metadata.metrics) == {field["field"] for field in fields}
    assert all(metric.period == students.PERIOD for metric in source.metadata.metrics.values())
    assert not any(any(token in field for token in ("technical", "disabilit", "disciplinary", "mobility"))
                   for field in source.metadata.metrics)
    assert source.metadata.metrics["membership_special_education_count"].unit == "count"
    assert source.metadata.metrics["enrollment_special_education_percent"].unit == "percent"
    assert "." not in cfg.value_policies["tprs_student_selected_2026"].tokens
    assert "overlap" in grid["definition"]["overlap_note"]
    resource = cfg.column_contract(students.DATASET, 2026, reader_profile="source")
    assert resource is not None
    tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), students.DATASET, "2026")["tables"]
    assert {table["table"] for table in tables} == set(students.SCOPES)
    for table in tables:
        assert len(table["fields"]) == 168
        assert table["header_identity"] == "position"
        assert all(field["code"] == field["label"] and field["dictionary_status"] == "not_documented"
                   for field in table["fields"])


@pytest.mark.parametrize("scope,identifier", [("state", "S"), ("district", "001902"), ("campus", "001902001")])
def test_all_raw_current_cells_and_lagged_period_remain_visible(grid, policy, scope, identifier):
    report = parse(grid, scope=scope, identifier=identifier)
    assert len(report.cells) == len(grid["rows"]) * len(grid["column_roles"][scope])
    lagged = [cell for cell in report.cells if "Disciplinary" in cell.label]
    assert lagged and all(cell.period.end_year == 2025 and cell.field is None for cell in lagged)
    assert all(cell.period == students.PERIOD for cell in report.cells if cell not in lagged)
    assert all(cell.row > 0 and cell.column > 1 for cell in report.cells)
    local = [cell for cell in report.cells if cell.geography_role == "local"]
    assert sum(cell.field is not None for cell in local) == 168
    assert all(cell.field is None for cell in report.cells if cell.geography_role != "local")
    table, support, audit = students.projection([report], policy)
    assert len(table) == 1 and audit["projection_qualified"] is True
    assert audit["whole_report_semantics_qualified"] is False and audit["raw_field_holds"]
    assert len(support["fields"]) == 168
    assert table["membership_total_students_count"].type == pa.int64()
    assert table["membership_total_students_percent"].type == pa.float64()
    if scope != "state":
        assert table[scope + "_number"].to_pylist() == [identifier]


def test_masks_zero_and_undefined_dot_are_distinct(grid, policy):
    rows = report_rows(grid)
    row = next(row for row in rows if row[0] == "Total Students")
    row[1:3] = ["0", "0.0%"]
    row = next(row for row in rows if row[0] == "Special Education")
    row[1:3] = ["*", "*"]
    row[5:7] = ["-", "-"]
    row = next(row for row in rows if row[0] == "Career and Technical Education")
    row[1] = "."
    next(row for row in rows if row[0] == "Students with Autism")[1] = "**"
    report = students.parse_student_report(html(rows), scope="district", identifier="001902")
    table, _, audit = students.projection([report], policy)
    assert table["membership_total_students_count"].to_pylist() == [0]
    assert table["tprs_status_membership_total_students_count"].to_pylist() == ["observed"]
    assert table["membership_special_education_count"].to_pylist() == [None]
    assert table["tprs_raw_membership_special_education_count"].to_pylist() == ["*"]
    assert table["tprs_status_membership_special_education_percent"].to_pylist() == ["suppressed"]
    assert table["tprs_status_enrollment_special_education_count"].to_pylist() == ["not_applicable"]
    dot, = [cell for cell in report.cells if cell.raw == "."]
    assert dot.field is None and dot.hold and dot.measurement(policy).status == ValueStatus.INVALID
    assert any(item["row"] == dot.row and item["token_status"] == "invalid" for item in audit["raw_field_holds"])
    complementary, = [cell for cell in report.cells if cell.raw == "**"]
    assert complementary.field is None and complementary.measurement(policy).status == ValueStatus.SUPPRESSED


@pytest.mark.parametrize("token", [".", "", "-1", "1.0", "01", "NaN", "1e2", "**"])
def test_unknown_qualified_count_token_cannot_publish(grid, policy, token):
    rows = report_rows(grid)
    next(row for row in rows if row[0] == "Total Students")[1] = token
    report = students.parse_student_report(html(rows), scope="district", identifier="001902")
    with pytest.raises(ContractError, match="Unreviewed token"):
        students.projection([report], policy)


@pytest.mark.parametrize("token", ["1", "101.0%", "1.00%", "-1.0%"])
def test_percent_unit_range_and_publisher_spelling_are_not_inferred(grid, policy, token):
    rows = report_rows(grid)
    next(row for row in rows if row[0] == "Total Students")[2] = token
    report = students.parse_student_report(html(rows), scope="district", identifier="001902")
    with pytest.raises(ContractError, match="Unreviewed token"):
        students.projection([report], policy)


def test_missing_campus_grade_is_missing_not_zero(grid, policy):
    report = parse(grid, scope="campus", identifier="001902001", omitted=["Grade 12"])
    table, _, _ = students.projection([report], policy)
    field = "membership_grade_12_count"
    assert table[field].to_pylist() == [None]
    assert table["tprs_raw_" + field].to_pylist() == [None]
    assert table["tprs_status_" + field].to_pylist() == ["missing"]
    assert table["tprs_reason_" + field].to_pylist() == ["row_not_published"]
    with pytest.raises(ContractError, match="grade coverage"):
        parse(grid, omitted=["Grade 12"])


def test_campus_total_benchmarks_have_explicit_row_specific_count_role(grid, policy):
    rows = report_rows(grid, "campus", "001902001")
    total = next(row for row in rows if row[0] == "Total Students")
    total[3:5] = ["123,456", "5,000,000"]
    report = students.parse_student_report(html(rows), scope="campus", identifier="001902001")
    benchmarks = [cell for cell in report.cells if cell.label == "Total Students" and cell.geography_role != "local"]
    assert len(benchmarks) == 4 and all(cell.measure == "count" and cell.field is None and cell.hold for cell in benchmarks)
    assert [cell.measurement(policy).value for cell in benchmarks[:2]] == [123456, 5000000]
    other = [cell for cell in report.cells if cell.label == "Grade 1" and cell.geography_role != "local"]
    assert all(cell.measure == "percent" and cell.raw == "1.0%" and cell.field is None for cell in other)
    assert benchmarks[0].column == other[0].column == 4


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "header", "population", "period", "geography", "legend", "boundary"])
def test_layout_and_source_meaning_drift_refuse(grid, damage):
    rows = report_rows(grid)
    if damage == "duplicate":
        rows.insert(8, rows[7])
    elif damage == "unknown":
        rows[7] = ["New total", *rows[7][1:]]
    elif damage == "header":
        rows[6] = ["Student Information", *reversed(rows[6][1:])]
    elif damage == "population":
        rows[4] = ["", "Enrollment", "Membership"]
    elif damage == "period":
        rows[1] = ["2024-25 Student Information"]
    elif damage == "geography":
        rows[2] = ["SYNTHETIC SCHOOL (001902) (001903)"]
    elif damage == "legend":
        rows[-1] = ["?", "Unknown"]
    else:
        rows[-5] = ["Mobility (2025-26)"]
    with pytest.raises(ContractError):
        students.parse_student_report(html(rows), scope="district", identifier="001902")


@pytest.mark.parametrize("scope,identifier", [("district", "01902"), ("district", "000000"),
    ("campus", "00190201"), ("campus", "００１９０２００１"), ("region", "13"), ("state", "0")])
def test_literal_geography_contract(scope, identifier):
    with pytest.raises(ContractError):
        students.parse_student_report(b"unused", scope=scope, identifier=identifier)


def test_duplicate_entities_and_semantic_aliases_cannot_collapse(grid, policy):
    report = parse(grid)
    with pytest.raises(ContractError, match="Duplicate"):
        students.projection([report, report], policy)
    altered = deepcopy(grid)
    altered["rows"][1]["name"] = altered["rows"][0]["name"]
    with pytest.raises(ContractError, match="Duplicate"):
        students.semantic_fields(altered)


@pytest.mark.parametrize("scope,identifier", [("state", "S"), ("district", "001902"), ("campus", "001902001")])
def test_workbook_witness_population_and_benchmark_headers_cannot_swap(grid, scope, identifier):
    cells = {(1, "A"): "Texas Education Agency", (2, "A"): "2025-26 Student Information",
             (3, "A"): "State" if scope == "state" else f"SYNTHETIC ({identifier})"}
    for index, (population, quantity, role) in enumerate(grid["column_roles"][scope], 1):
        if quantity == "count" and role == "local":
            cells[5, chr(ord("A") + index)] = population.title()
    header = 6 if scope == "state" else 7
    for index, value in enumerate(grid["headers"][scope]):
        cells[header, chr(ord("A") + index)] = value
    if scope != "state":
        for index, value in enumerate(grid["geography_headers"][scope][1:]):
            cells[6, chr(ord("B") + 2 * index)] = value
    students._workbook_headers(cells, grid, scope, identifier)
    altered = dict(cells)
    altered[5, "B"] = "Enrollment"
    with pytest.raises(ContractError, match="population headers"):
        students._workbook_headers(altered, grid, scope, identifier)
    altered = dict(cells)
    altered[header, "B"] = "Percent"
    with pytest.raises(ContractError, match="quantity/benchmark"):
        students._workbook_headers(altered, grid, scope, identifier)
    if scope != "state":
        altered = dict(cells)
        altered[6, "B"] = "State"
        with pytest.raises(ContractError, match="geography role"):
            students._workbook_headers(altered, grid, scope, identifier)


@pytest.mark.parametrize("legacy", [False, True])
def test_receipt_pins_original_download_and_truthful_legacy_limits(tmp_path, grid, legacy):
    store = ProvenanceStore(tmp_path / "store")
    raw = html(report_rows(grid))
    rid = receipt(store, tmp_path, raw, legacy=legacy)
    original = store.read_record(rid)
    report = students.read_student_report(store, rid, scope="district", identifier="001902", expected_sha256=sha256(raw).hexdigest())
    assert report.identifier == "001902" and bool(report.acquisition_notes) is legacy
    assert store.read_record(rid) == original
    if legacy:
        assert original["requested_year"] is None and original["parameters"].get("http_method") is None


@pytest.mark.parametrize("damage", ["host", "duplicate-control", "year", "redirect", "method", "status", "mime", "hash"])
def test_wrong_acquisition_identity_refuses_before_projection(tmp_path, grid, damage):
    store = ProvenanceStore(tmp_path / "store")
    raw = html(report_rows(grid))
    valid = students.BROKER + "?" + urlencode({"_service": "marykay", "_program": "perfrept.perfmast.sas",
        "_debug": "0", "ccyy": "2026", "id": "001902", "lev": "D", "prgopt": "reports/tapr/student.sas"})
    overrides = {}
    if damage in {"host", "duplicate-control", "year"}:
        changed = valid.replace("rptsvr1.tea.texas.gov", "example.com") if damage == "host" else (
            valid + "&id=001902" if damage == "duplicate-control" else valid.replace("ccyy=2026", "ccyy=2025"))
        overrides = {"requested_url": changed, "resolved_url": changed}
    elif damage == "redirect":
        overrides = {"resolved_url": valid + "&extra=1"}
    elif damage in {"method", "status"}:
        overrides = {"parameters": {"http_method": "POST" if damage == "method" else "GET",
                                     "http_status": 200 if damage == "method" else 500}}
    elif damage == "mime":
        overrides = {"response_headers": {"content-type": "text/plain"}}
    rid = receipt(store, tmp_path, raw, overrides=overrides)
    with pytest.raises(ContractError, match="exact public request"):
        students.read_student_report(store, rid, scope="district", identifier="001902",
                                     expected_sha256="0" * 64 if damage == "hash" else sha256(raw).hexdigest())


def test_report_size_refusal_precedes_artifact_read(tmp_path, grid, monkeypatch):
    store = ProvenanceStore(tmp_path / "store")
    raw = html(report_rows(grid)) + b" " * students.MAX_REPORT_BYTES
    rid = receipt(store, tmp_path, raw)
    monkeypatch.setattr(store, "verify_artifact", lambda *_: pytest.fail("Oversized report must not be opened"))
    with pytest.raises(ContractError, match="bounded HTML size"):
        students.read_student_report(store, rid, scope="district", identifier="001902", expected_sha256=sha256(raw).hexdigest())


def test_same_size_change_after_cas_verification_cannot_rebind_receipt(tmp_path, grid, monkeypatch):
    store = ProvenanceStore(tmp_path / "store")
    raw = html(report_rows(grid))
    rid = receipt(store, tmp_path, raw)
    original = store.verify_artifact
    calls = 0

    def changed_after_verification(digest):
        nonlocal calls
        path = original(digest)
        calls += 1
        if calls == 2:  # Closure and then the selected reader's artifact check.
            changed = raw.replace(b">1</td>", b">2</td>", 1)
            assert changed != raw and len(changed) == len(raw)
            path.write_bytes(changed)
        return path

    monkeypatch.setattr(store, "verify_artifact", changed_after_verification)
    with pytest.raises(ContractError, match="bytes changed"):
        students.read_student_report(store, rid, scope="district", identifier="001902", expected_sha256=sha256(raw).hexdigest())


def test_source_parse_buffer_is_bounded_and_pinned_even_if_file_grows(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    digest = sha256(path.read_bytes()).hexdigest()
    assert students._source_bytes(path, digest, 8) == b"original"
    path.write_bytes(b"changed!")
    with pytest.raises(ContractError, match="bytes changed"):
        students._source_bytes(path, digest, 8)
    monkeypatch.setattr(students, "MAX_REPORT_BYTES", 7)
    path.write_bytes(b"original")
    with pytest.raises(ContractError, match="bytes changed"):
        students._source_bytes(path, digest, 8)
    path.write_bytes(b"original" * 100)
    with pytest.raises(ContractError, match="bytes changed"):
        students._source_bytes(path, sha256(path.read_bytes()).hexdigest(), 800)


def test_workbook_exact_parse_buffer_refuses_after_verify_comment_change(tmp_path, grid, monkeypatch):
    store = ProvenanceStore(tmp_path / "store")
    rows = report_rows(grid, "state", "S")
    raw = html(rows)
    html_id = receipt(store, tmp_path, raw, identifier="S", scope="state")
    cells = {(1, "A"): "Texas Education Agency", (2, "A"): "2025-26 Student Information", (3, "A"): "State",
             (5, "B"): "Membership", (5, "D"): "Enrollment"}
    for index, value in enumerate(grid["headers"]["state"]):
        cells[6, chr(ord("A") + index)] = value
    for number, row in enumerate(rows[6:], 7):
        if row == ["Mobility (2024-25)"]:
            break
        for index, value in enumerate(row):
            cells[number, chr(ord("A") + index)] = value
    strings = list(cells.values())
    namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    sheet = '<worksheet xmlns="' + namespace + '"><sheetData>'
    for number in sorted({key[0] for key in cells}):
        sheet += f'<row r="{number}">'
        for index, ((r, column), _) in enumerate(cells.items()):
            if r == number:
                sheet += f'<c r="{column}{r}" t="s"><v>{index}</v></c>'
        sheet += '</row>'
    sheet += '</sheetData></worksheet>'
    path = tmp_path / "synthetic.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr('xl/workbook.xml', '<workbook xmlns="' + namespace + '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Student Information" sheetId="1" r:id="r1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr('xl/sharedStrings.xml', '<sst xmlns="' + namespace + '">' + ''.join('<si><t>' + escape(value) + '</t></si>' for value in strings) + '</sst>')
        archive.writestr('xl/worksheets/sheet1.xml', sheet)
        archive.comment = b'before'
    digest = sha256(path.read_bytes()).hexdigest()
    url = students.BROKER + '?' + urlencode({'_service': 'marykay', '_program': 'perfrept.perfmast.sas', '_debug': '0',
        'ccyy': '2026', 'id': 'S', 'lev': 'S', 'prgopt': 'reports/tapr/student.sas', 'dest': 'E'})
    xlsx_id = store.record_acquisition(path, ResolvedDataset('reference:synthetic-workbook', 'reference', 2026, 2026, '',
        DatasetMetadata(title='Synthetic workbook', version='2026', provenance_ids=(html_id,))),
        downloaded_at=datetime(2026, 11, 4, tzinfo=timezone.utc), requested_url=url, resolved_url=url,
        response_headers={'content-type': 'application/ms-excel'}, parameters={'http_method': 'GET', 'http_status': 200})
    contract = deepcopy(grid)
    contract['witnesses'] = {'state': {'identifier': 'S', 'html_receipt': html_id, 'html_sha256': sha256(raw).hexdigest(),
        'xlsx_receipt': xlsx_id, 'xlsx_sha256': digest}}
    monkeypatch.setattr(students, 'definition', lambda: contract)
    assert students.verify_workbook_witnesses(store)['comparisons'][0]['current_cells'] == 204
    verify = store.verify_artifact
    calls = 0

    def mutate_comment_after_verification(pin):
        nonlocal calls
        selected = verify(pin)
        if pin == digest:
            calls += 1
            if calls == 2:
                before = selected.stat().st_size
                with ZipFile(selected, 'a') as archive:
                    archive.comment = b'after!'
                assert selected.stat().st_size == before
        return selected

    monkeypatch.setattr(store, 'verify_artifact', mutate_comment_after_verification)
    with pytest.raises(ContractError, match='bytes changed'):
        students.verify_workbook_witnesses(store)


def test_selected_aggregate_size_refuses_before_parsing_or_staging(tmp_path, grid, monkeypatch):
    store = ProvenanceStore(tmp_path / "store")
    raw = html(report_rows(grid))
    rid = receipt(store, tmp_path, raw)
    config = ROOT / "teadata/teadata_sources.yaml"
    intake = tmp_path / "intake.json"
    intake.write_text(canonical_json({"schema_version": 1, "dataset": students.DATASET, "edition": "2026",
        "reports": [{"receipt": rid, "sha256": sha256(raw).hexdigest(), "scope": "district", "identifier": "001902"}]}))
    monkeypatch.setattr(students, "MAX_SELECTION_BYTES", len(raw) - 1)
    monkeypatch.setattr(students, "read_student_report", lambda *_args, **_kwargs: pytest.fail("Oversized selection must not parse"))
    output = tmp_path / "candidate"
    with pytest.raises(ContractError, match="bounded aggregate size"):
        students.materialize_students(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
            intake=intake, intake_sha256=sha256(intake.read_bytes()).hexdigest(), output=output)
    assert not output.exists()


def test_dynamic_crosswalk_changes_only_ordered_actual_input_identity():
    a, b = "a" * 64, "b" * 64
    forward = students._crosswalk({"district": [a, b]})
    reverse = students._crosswalk({"district": [b, a]})
    left, = forward["datasets"][0]["tables"]
    right, = reverse["datasets"][0]["tables"]
    assert table_input_sha256s(left) == (a, b)
    assert table_input_sha256s(right) == (b, a)
    assert {k: v for k, v in left.items() if k != "input_sha256s"} == {k: v for k, v in right.items() if k != "input_sha256s"}
    with pytest.raises(ValueError, match="distinct"):
        students._crosswalk({"district": [a, a]})


@pytest.mark.parametrize("damage", ["config", "semantics", "policy", "intake", "json-duplicate"])
def test_candidate_refuses_tampering_before_creating_output(tmp_path, damage):
    config = tmp_path / "source.yaml"
    body = (ROOT / "teadata/teadata_sources.yaml").read_bytes()
    if damage in {"semantics", "policy"}:
        document = yaml.safe_load(body)
        if damage == "semantics":
            document["data_sources"][students.DATASET][2026]["metadata"]["metrics"]["membership_total_students_count"]["unit"] = "percent"
        else:
            document["value_policies"]["tprs_student_selected_2026"]["tokens"]["."] = {"status": "missing", "reason": "guessed"}
        body = yaml.safe_dump(document).encode()
    config.write_bytes(body)
    intake = tmp_path / "intake.json"
    intake.write_bytes(b'{"schema_version":1,"schema_version":1}' if damage == "json-duplicate" else
                      canonical_json({"schema_version": 1, "dataset": students.DATASET, "edition": "2026", "reports": []}).encode())
    output = tmp_path / "candidate"
    with pytest.raises((ContractError, ValueError)):
        students.materialize_students(ProvenanceStore(tmp_path / "store"), config=config,
            config_sha256="0" * 64 if damage == "config" else sha256(body).hexdigest(), intake=intake,
            intake_sha256="0" * 64 if damage == "intake" else sha256(intake.read_bytes()).hexdigest(), output=output)
    assert not output.exists()


def test_thin_command_requires_explicit_pins_and_has_no_loader_default():
    from teadata.scripts.refresh_tprs_students import main
    with pytest.raises(SystemExit) as result:
        main([])
    assert result.value.code == 2
    assert students.EDITION == 2026
    assert json.loads(students.files("teadata").joinpath(students.RESOURCE).read_bytes())["edition"] == "2026"
