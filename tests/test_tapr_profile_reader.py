import csv
from dataclasses import replace
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest

from teadata.column_schema import source_schema_fingerprint, validate_header_table
from teadata.identifiers import canonical_source_identifier, canonical_source_identifiers
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.tapr import (
    TaprReadBudget, TaprSourceRequest, TaprSourceTable, merge_tapr_tables, read_tapr_table,
)


LABELS = ["9 Digit Campus Number", "Example Value"]
CODES = ["CAMPUS", "VALUE"]
FIELDS = ["campus_number", "example_value"]
DATASET = "campus_tapr_student_staff_profile"


def write_source(tmp_path, records):
    path = tmp_path / "source.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(records)
    return path


def mapping_for(path, *, labels=LABELS, codes=CODES, fields=FIELDS, layout=None):
    mapping = {"table": "student", "input_sha256": sha256(path.read_bytes()).hexdigest(),
        "source_schema_sha256": source_schema_fingerprint(labels, codes),
        "fields": [{"position": i, "code": code, "label": label, "field": field}
                   for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)]}
    if layout is not None:
        mapping["layout"] = layout
    return mapping


def read_source(path, *, mapping=None, key_field="campus_number", dataset=DATASET, edition="2025"):
    return read_tapr_table(path, mapping=mapping_for(path) if mapping is None else mapping,
                           key_field=key_field, dataset=dataset, edition=edition)


@pytest.mark.parametrize("token", ["-1", "-3", "-2", ".", "•", "n/a", "0", "0.0", "9007199254740993", "", " -3 ", "first\nsecond"])
def test_native_reader_preserves_exact_source_tokens_without_masking_or_whitespace_casts(tmp_path, token):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", token]])
    before = path.read_bytes()
    result = read_source(path)
    assert result.table.to_pylist() == [{"campus_number": "001902001", "example_value": token}]
    assert all(pa.types.is_string(field.type) for field in result.table.schema)
    assert path.read_bytes() == before


@pytest.mark.parametrize("override", ["CamelCase", "naïve", "2025_name", "class", ""])
def test_unsafe_canonical_names_fail_without_sanitizing_at_runtime(tmp_path, override):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    with pytest.raises(ValueError, match="Invalid canonical header"):
        read_source(path, mapping=mapping_for(path, fields=["campus_number", override]))


@pytest.mark.parametrize("codes", [["CAMPUS", ""], ["CAMPUS", " CAMPUS "]])
def test_duplicate_or_blank_publisher_codes_fail(tmp_path, codes):
    path = write_source(tmp_path, [LABELS, codes, ["001902001", "42"]])
    with pytest.raises(ValueError, match="(Invalid header crosswalk|Duplicate source code)"):
        read_source(path, mapping=mapping_for(path, codes=codes))


def test_canonical_header_collision_is_not_suffixed(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    with pytest.raises(ValueError, match="Canonical header collision"):
        read_source(path, mapping=mapping_for(path, fields=["campus_number", "campus_number"]))


@pytest.mark.parametrize("records", [[], [LABELS], [LABELS, ["CAMPUS"]], [LABELS, CODES + ["EXTRA"]],
    [LABELS, ["001902001", "42"], ["001902041", "99"]], [[LABELS[0], "Changed label"], CODES],
    [["Metadata title"], LABELS, CODES, ["001902001", "42"]]])
def test_unreviewed_header_preamble_or_code_record_changes_fail(tmp_path, records):
    path = write_source(tmp_path, records)
    with pytest.raises(ValueError) as error:
        read_source(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("row", [["001902001"], ["001902001", "42", "EXTRA"], []])
def test_bad_record_width_is_never_padded_truncated_or_skipped(tmp_path, row):
    path = write_source(tmp_path, [LABELS, CODES, row])
    with pytest.raises(ValueError, match=r"record 3 has .* fields; expected 2"):
        read_source(path)


@pytest.mark.parametrize("key", ["", "000000000", "-1", "True", "001902001.5", "1001902001", "1e3", "９９９９９９９９９", "Notes: suppression applies"])
def test_invalid_identity_or_footer_text_is_not_silently_removed(tmp_path, key):
    path = write_source(tmp_path, [LABELS, CODES, [key, "42"]])
    with pytest.raises(ValueError, match="invalid campus_number"):
        read_source(path)


@pytest.mark.parametrize("later", ["001902001", "1902001", "'001902001"])
def test_duplicate_entity_identity_is_not_overwritten(tmp_path, later):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], [later, "99"]])
    with pytest.raises(ValueError, match=r"duplicate campus_number at records \[3, 4\]"):
        read_source(path)


@pytest.mark.parametrize("district", ["", "227901"])
def test_campus_parent_claim_is_required_and_consistent_when_present(tmp_path, district):
    labels, codes = [LABELS[0], "District Number"], ["CAMPUS", "DISTRICT"]
    path = write_source(tmp_path, [labels, codes, ["001902001", district]])
    with pytest.raises(ValueError, match="campus and district identifiers disagree"):
        read_source(path, mapping=mapping_for(path, labels=labels, codes=codes, fields=["campus_number", "district_number"]))


def test_original_content_revision_is_not_replaced_under_one_contract(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    mapping = mapping_for(path)
    path.write_text(path.read_text().replace("42", "99"))
    with pytest.raises(ValueError, match="input differs"):
        read_source(path, mapping=mapping)


def test_tapr_singleton_list_keeps_the_same_physical_reader_contract(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    mapping = mapping_for(path)
    digest = mapping.pop("input_sha256")
    mapping["input_sha256s"] = [digest]
    result = read_source(path, mapping=mapping)
    assert result.input_sha256 == digest
    assert result.table.to_pylist() == [{"campus_number": "001902001", "example_value": "42"}]


@pytest.mark.parametrize("workflow", ["reader", "comparison", "merge", "record_count"])
def test_tapr_single_file_workflows_refuse_multi_input_tables(tmp_path, monkeypatch, workflow):
    from teadata.tapr import compare_tapr_tables, reviewed_tapr_data_record_count

    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    mapping = mapping_for(path)
    mapping["input_sha256s"] = [mapping.pop("input_sha256"), "b" * 64]
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Multi-input physical source opened"))
        with pytest.raises(ValueError, match="single raw input"):
            if workflow == "reader":
                TaprSourceRequest(path, mapping, "campus_number", DATASET, "2025").open(budget=TaprReadBudget())
            elif workflow == "record_count":
                reviewed_tapr_data_record_count(mapping)
            else:
                source = TaprSourceTable(pa.table({"campus_number": ["001902001"], "example_value": ["42"]}),
                    path, mapping["input_sha256s"][0], DATASET, "2025", (), mapping)
                if workflow == "comparison":
                    compare_tapr_tables(source, source, key_field="campus_number")
                else:
                    merge_tapr_tables([source], key_field="campus_number")


def test_csv_quoting_error_is_not_silently_accepted(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text('9 Digit Campus Number,Example Value\nCAMPUS,VALUE\n001902001,"unfinished\n')
    with pytest.raises(ValueError, match="invalid CSV near physical line"):
        read_source(path)


def test_reviewed_preamble_gap_and_footer_notes_have_exact_provenance(tmp_path):
    records = [["Publisher report"], CODES, ["001902001", " -3 "], [], ["001902041", "."], ["Notes:", "Unavailable is not zero"]]
    path = write_source(tmp_path, records)
    def note(number, kind):
        text = StringIO(newline="")
        csv.writer(text).writerow(records[number - 1])
        return {"first_record": number, "last_record": number, "kind": kind, "sha256": sha256(text.getvalue().encode()).hexdigest()}
    layout = {"code_record": 2, "label_record": None, "first_data_record": 3, "last_data_record": 5,
              "annotations": [note(1, "preamble"), note(4, "blank"), note(6, "footer")]}
    result = read_source(path, mapping=mapping_for(path, layout=layout))
    assert result.table.to_pylist() == [{"campus_number": "001902001", "example_value": " -3 "}, {"campus_number": "001902041", "example_value": "."}]
    assert result.annotations[-1]["records"] == [["Notes:", "Unavailable is not zero"]]
    assert [note["first_record"] for note in result.annotations] == [1, 4, 6]
    changed = mapping_for(path, layout=layout)
    changed["layout"]["annotations"][-1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="annotation range"):
        read_source(path, mapping=changed)


@pytest.mark.parametrize("kind", ["unreviewed", "overlap", "header_overlap"])
def test_ambiguous_or_unaccounted_note_ranges_fail(tmp_path, kind):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], ["Footer", "note"]])
    layout = {"code_record": 2, "label_record": 1, "first_data_record": 3, "last_data_record": 3, "annotations": []}
    if kind != "unreviewed":
        item = {"first_record": 1 if kind == "header_overlap" else 4, "last_record": 4, "kind": "footer", "sha256": "0" * 64}
        layout["annotations"] = [item] if kind == "header_overlap" else [item, item]
    with pytest.raises(ValueError, match="(unaccounted record 4|overlap)"):
        read_source(path, mapping=mapping_for(path, layout=layout))


def native_source(records, *, edition="2025", labels=None, table_name="student"):
    table = pa.Table.from_pylist(records)
    fields = table.column_names
    codes = ["CAMPUS" if name == "campus_number" else name.upper() for name in fields]
    labels = labels or fields
    mapping = {"table": table_name, "input_sha256": "0" * 64, "source_schema_sha256": source_schema_fingerprint(labels, codes),
               "fields": [{"position": i, "code": code, "label": label, "field": field} for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)]}
    return TaprSourceTable(table, Path(__file__), "0" * 64, DATASET, edition, (), mapping)


def test_same_edition_shared_identity_conflicts_fail_without_masking_assumptions():
    a = native_source([{"campus_number": "001902001", "example": "."}])
    b = native_source([{"campus_number": "001902001", "example": "42"}], table_name="staff")
    with pytest.raises(ValueError, match="Conflicting TAPR values.*example"):
        merge_tapr_tables([a, b], key_field="campus_number")


def test_one_sided_historical_entities_remain_and_absent_table_is_not_raw_blank():
    a = native_source([{"campus_number": "001902001", "student": "-1"}])
    b = native_source([{"campus_number": "001902041", "staff": "-3"}], table_name="staff")
    assert merge_tapr_tables([a, b], key_field="campus_number").to_pylist() == [
        {"campus_number": "001902001", "student": "-1", "staff": None},
        {"campus_number": "001902041", "student": None, "staff": "-3"}]


def test_blank_fill_and_nonblank_shared_values_have_explicit_rules():
    a = native_source([{"campus_number": "001902001", "campus_name": "Historical school", "empty": "", "student": "-1"}])
    b = native_source([{"campus_number": "001902001", "campus_name": "Historical school", "empty": "filled", "staff": "-3"}], table_name="staff")
    assert merge_tapr_tables([a, b], key_field="campus_number").to_pylist() == [
        {"campus_number": "001902001", "campus_name": "Historical school", "empty": "filled", "student": "-1", "staff": "-3"}]
    conflict = native_source([{"campus_number": "001902001", "campus_name": "A different school"}], table_name="staff")
    with pytest.raises(ValueError, match="campus_name"):
        merge_tapr_tables([a, conflict], key_field="campus_number")


def test_equivalent_identity_spellings_must_not_duplicate_or_overwrite_an_entity():
    a = native_source([{"campus_number": "001902001"}])
    b = native_source([{"campus_number": "1902001"}], table_name="staff")
    with pytest.raises(ValueError, match="Inconsistent TAPR campus_number representations"):
        merge_tapr_tables([a, b], key_field="campus_number")


def test_observations_from_different_editions_or_datasets_are_not_coalesced():
    original = native_source([{"campus_number": "001902001", "campus_name": "Earlier name"}], edition="2024")
    later = native_source([{"campus_number": "001902001", "campus_name": "Later name"}], edition="2025")
    with pytest.raises(ValueError, match="different qualified datasets or editions"):
        merge_tapr_tables([original, later], key_field="campus_number")
    foreign = replace(original, dataset="accountability")
    with pytest.raises(ValueError, match="different qualified datasets or editions"):
        merge_tapr_tables([original, foreign], key_field="campus_number")
    assert original.table["campus_name"][0].as_py() == "Earlier name"


def test_reissued_table_revision_is_not_silently_coalesced_with_retained_observation():
    original = native_source([{"campus_number": "001902001", "example": "42"}])
    reissued = replace(original, input_sha256="1" * 64)
    with pytest.raises(ValueError, match="Different revisions or repeated observations"):
        merge_tapr_tables([original, reissued], key_field="campus_number")


@pytest.mark.parametrize("token", [None, "", "  ", "'", "0", "000000000", "001902001", "1902001", "'001902001", "  ' 001902001.00 ", "\x1c001902001\x1f", "True", "-1", "1e3", "001902001.1", "９９９９９９９９９"])
def test_native_string_identifier_contract_matches_scalar_oracle(token):
    try:
        expected = canonical_source_identifier(token, width=9)
    except ValueError:
        with pytest.raises(ValueError):
            canonical_source_identifiers(pa.array([token], type=pa.string()), width=9)
    else:
        assert canonical_source_identifiers(pa.array([token], type=pa.string()), width=9).to_pylist() == [expected]


def test_bad_input_prevents_all_writers_and_output_activation(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42", "EXTRA"]])
    request = TaprSourceRequest(path, mapping_for(path), "campus_number", DATASET, "2025")
    cfg = SimpleNamespace(tapr_source_requests=lambda *a, **k: [request],
                          column_schema=lambda *a, **k: FIELDS)
    monkeypatch.setattr(pq, "ParquetWriter", lambda *a, **k: pytest.fail("Writer ran for invalid input"))
    out = tmp_path / "new" / "output"
    with pytest.raises(ValueError, match="record 3 has 3 fields"):
        prepare_profiles(cfg, edition="2025", scopes=["campus"], input_dir=tmp_path, output_dir=out)
    assert not out.parent.exists()


def test_preparation_never_overwrites_an_existing_directory(tmp_path):
    out = tmp_path / "preserved"
    out.mkdir()
    marker = out / "original.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        prepare_profiles(object(), edition="2024", scopes=["district"], input_dir=tmp_path, output_dir=out)
    assert marker.read_text() == "original"


def test_source_column_disposition_cannot_silently_exclude_a_publisher_field(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    mapping = mapping_for(path)
    mapping["fields"][1]["disposition"] = "ignore"
    with pytest.raises(ValueError, match="Unaccounted source column 2"):
        validate_header_table(mapping)


@pytest.mark.parametrize("large", [False, True])
def test_native_identifier_contract_supports_chunked_string_representations(large):
    kind = pa.large_string() if large else pa.string()
    values = pa.chunked_array([["1902001", None], ["'001902041"]], type=kind)
    assert canonical_source_identifiers(values, width=9).to_pylist() == ["'001902001", None, "'001902041"]


def test_publisher_dictionary_revision_is_checked_before_excel_decode(tmp_path, monkeypatch):
    import pandas as pd

    path = write_source(tmp_path, [CODES, ["001902001", "42"]])
    dictionary = tmp_path / "dictionary.xlsx"
    dictionary.write_bytes(b"changed publisher workbook")
    mapping = mapping_for(path, layout={"code_record": 1, "label_record": None,
        "first_data_record": 2, "last_data_record": 2, "annotations": []})
    mapping["dictionary"] = {"filename": dictionary.name, "sha256": "0" * 64, "sheet_name": "Sheet1"}
    for field in mapping["fields"]:
        field.update(publisher_type="Character", publisher_length="", disposition="fact")
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pytest.fail("Unreviewed dictionary decoded"))
    with pytest.raises(ValueError, match="publisher dictionary bytes differ"):
        read_source(path, mapping=mapping)


def test_dictionary_growth_after_stat_is_bounded_before_excel_decode(tmp_path, monkeypatch):
    import pandas as pd

    path = write_source(tmp_path, [CODES, ["001902001", "42"]])
    dictionary = tmp_path / "dictionary.xlsx"
    raw = b"x" * 129
    dictionary.write_bytes(raw)
    mapping = mapping_for(path, layout={"code_record": 1, "label_record": None,
        "first_data_record": 2, "last_data_record": 2, "annotations": []})
    mapping["dictionary"] = {"filename": dictionary.name, "sha256": sha256(raw).hexdigest(), "sheet_name": "Sheet1"}
    for field in mapping["fields"]:
        field.update(publisher_type="Character", publisher_length="", disposition="fact")
    original_stat = Path.stat
    def stat_before_growth(file, *args, **kwargs):
        return SimpleNamespace(st_size=0) if file == dictionary else original_stat(file, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat_before_growth)
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pytest.fail("Over-budget workbook decoded"))
    with pytest.raises(ValueError, match="dictionary exceeds metadata byte budget"):
        read_tapr_table(path, mapping=mapping, key_field="campus_number", dataset=DATASET,
                        edition="2025", budget=TaprReadBudget(metadata_bytes=128))


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_bounded_batches_preserve_multiline_headers_bom_notes_and_strings(tmp_path, ending):
    labels = [LABELS[0], "First\nsecond\nthird label"]
    records = [["Publisher\nreport"], labels, CODES,
               ["001902001", " -3 "], [], ["001902041", "a\nb"], ["Notes", "Do not infer"]]
    text = StringIO(newline="")
    # Quote embedded LF even when the file's physical separator is CR.
    writer = csv.writer(text, lineterminator=ending, quoting=csv.QUOTE_ALL)
    writer.writerows(records)
    raw = b"\xef\xbb\xbf" + text.getvalue().encode()
    path = tmp_path / "multiline.csv"
    path.write_bytes(raw)
    def note(number, kind):
        stream = StringIO(newline="")
        csv.writer(stream, lineterminator=ending, quoting=csv.QUOTE_ALL).writerow(records[number - 1])
        content = stream.getvalue().encode()
        if number == 1:
            content = b"\xef\xbb\xbf" + content
        return {"first_record": number, "last_record": number, "kind": kind,
                "sha256": sha256(content).hexdigest()}
    layout = {"label_record": 2, "code_record": 3, "first_data_record": 4, "last_data_record": 6,
              "annotations": [note(1, "preamble"), note(5, "blank"), note(7, "footer")]}
    mapping = mapping_for(path, labels=labels, layout=layout)
    budget = TaprReadBudget(batch_bytes=128, batch_rows=1)
    with TaprSourceRequest(path, mapping, "campus_number", DATASET, "2025").open(budget=budget) as reader:
        with pytest.raises(ValueError, match="validated EOF"):
            _ = reader.summary
        batches = list(reader)
        assert len(batches) == 2 and all(batch.num_rows == 1 and batch.nbytes <= 128 for batch in batches)
        assert reader.summary.rows == 2 and reader.summary.annotations[0]["records"] == [["Publisher\nreport"]]
    eager = read_tapr_table(path, mapping=mapping, key_field="campus_number", dataset=DATASET,
                            edition="2025", budget=budget)
    assert pa.Table.from_batches(batches).equals(eager.table)
    assert eager.table["example_value"].to_pylist() == [" -3 ", "a\nb"]
    assert path.read_bytes() == raw


def test_early_close_and_bad_tail_never_produce_a_validated_summary(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], ["001902041", "99"]])
    request = TaprSourceRequest(path, mapping_for(path), "campus_number", DATASET, "2025")
    with request.open(budget=TaprReadBudget(batch_rows=1)) as reader:
        assert next(reader).num_rows == 1
    with pytest.raises(ValueError, match="validated EOF"):
        _ = reader.summary
    mapping = mapping_for(path)
    mapping["input_sha256"] = "0" * 64
    with TaprSourceRequest(path, mapping, "campus_number", DATASET, "2025").open() as reader:
        with pytest.raises(ValueError, match="input differs"):
            list(reader)
        with pytest.raises(ValueError, match="validated EOF"):
            _ = reader.summary


@pytest.mark.parametrize("budget,match", [
    (TaprReadBudget(record_bytes=20), "record byte budget"),
    (TaprReadBudget(metadata_bytes=20), "metadata byte budget"),
    (TaprReadBudget(batch_bytes=16), "native batch byte budget"),
    (TaprReadBudget(key_bytes=30, batch_rows=1), "identity index byte budget"),
])
def test_structural_and_native_admission_budgets_reject_before_unbounded_storage(tmp_path, budget, match):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], ["001902041", "99"]])
    with pytest.raises(ValueError, match=match):
        read_tapr_table(path, mapping=mapping_for(path), key_field="campus_number",
                        dataset=DATASET, edition="2025", budget=budget)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 1024**3 + 1])
def test_budget_fields_fail_closed(value):
    with pytest.raises(ValueError, match="positive bounded integers"):
        TaprReadBudget(batch_bytes=value)


def test_wide_sparse_batch_admission_accounts_for_native_column_offsets(tmp_path, monkeypatch):
    import pyarrow.csv as pacsv

    width, rows = 2212, 12
    labels, codes = [LABELS[0]] + ["Value " + str(i) for i in range(width - 1)], ["CAMPUS"] + ["V" + str(i) for i in range(width - 1)]
    names = ["campus_number"] + ["value_" + str(i) for i in range(width - 1)]
    path = write_source(tmp_path, [labels, codes, *[[f"001902{i:03d}"] + [""] * (width - 1) for i in range(1, rows + 1)]])
    mapping = mapping_for(path, labels=labels, codes=codes, fields=names)
    budget = TaprReadBudget(batch_bytes=32768)
    original = pacsv.read_csv
    admitted = []
    def decode(*args, **kwargs):
        table = original(*args, **kwargs)
        admitted.append((table.num_rows, table.nbytes))
        return table
    monkeypatch.setattr(pacsv, "read_csv", decode)
    result = read_tapr_table(path, mapping=mapping, key_field="campus_number", dataset=DATASET,
                             edition="2025", budget=budget)
    assert result.table.num_rows == rows and len(admitted) > 1
    assert max(count for count, _ in admitted) <= 2
    assert all(size <= budget.batch_bytes for _, size in admitted)


def test_multiline_record_budget_is_cumulative_before_native_decode(tmp_path, monkeypatch):
    import pyarrow.csv as pacsv

    path = write_source(tmp_path, [CODES, ["001902001", "first\nsecond\nthird"]])
    layout = {"code_record": 1, "label_record": None, "first_data_record": 2,
              "last_data_record": 2, "annotations": []}
    mapping = mapping_for(path, layout=layout)
    monkeypatch.setattr(pacsv, "read_csv", lambda *a, **k: pytest.fail("Oversized record reached native decode"))
    with pytest.raises(ValueError, match="record byte budget"):
        read_tapr_table(path, mapping=mapping, key_field="campus_number", dataset=DATASET,
                        edition="2025", budget=TaprReadBudget(record_bytes=22))


def test_literal_bom_in_a_fact_cell_is_never_treated_as_a_new_chunk_signature(tmp_path):
    labels, codes, fields = ["Raw value", LABELS[0]], ["VALUE", "CAMPUS"], ["example_value", "campus_number"]
    path = write_source(tmp_path, [labels, codes, ["\ufefffirst", "001902001"], ["\ufeffsecond", "001902041"]])
    result = read_tapr_table(path, mapping=mapping_for(path, labels=labels, codes=codes, fields=fields),
        key_field="campus_number", dataset=DATASET, edition="2025", budget=TaprReadBudget(batch_rows=1))
    assert result.table["example_value"].to_pylist() == ["\ufefffirst", "\ufeffsecond"]


def test_metadata_budget_bounds_a_reviewed_footer_without_discarding_it(tmp_path):
    records = [LABELS, CODES, ["001902001", "42"], ["Notes", "x" * 256]]
    path = write_source(tmp_path, records)
    text = StringIO(newline="")
    csv.writer(text).writerow(records[-1])
    layout = {"label_record": 1, "code_record": 2, "first_data_record": 3, "last_data_record": 3,
        "annotations": [{"first_record": 4, "last_record": 4, "kind": "footer",
                         "sha256": sha256(text.getvalue().encode()).hexdigest()}]}
    with pytest.raises(ValueError, match="metadata byte budget"):
        read_tapr_table(path, mapping=mapping_for(path, layout=layout), key_field="campus_number",
            dataset=DATASET, edition="2025", budget=TaprReadBudget(metadata_bytes=512))


def test_utf8_errors_and_untyped_budgets_never_produce_a_summary(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"Campus,Value\nCAMPUS,VALUE\n001902001,\xff\n")
    mapping = mapping_for(path, labels=["Campus", "Value"])
    request = TaprSourceRequest(path, mapping, "campus_number", DATASET, "2025")
    bad_budget: Any = False  # Deliberately cross the typed boundary.
    with pytest.raises(ValueError, match="typed read budget"):
        request.open(budget=bad_budget)
    with request.open() as reader:
        with pytest.raises(UnicodeDecodeError):
            list(reader)
        with pytest.raises(ValueError, match="validated EOF"):
            _ = reader.summary


def test_global_duplicate_and_source_change_are_detected_across_batches(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], ["1902001", "99"]])
    request = TaprSourceRequest(path, mapping_for(path), "campus_number", DATASET, "2025")
    with request.open(budget=TaprReadBudget(batch_rows=1)) as reader:
        with pytest.raises(ValueError, match=r"duplicate campus_number at records \[3, 4\]"):
            list(reader)
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], ["001902041", "99"]])
    with TaprSourceRequest(path, mapping_for(path), "campus_number", DATASET, "2025").open(
            budget=TaprReadBudget(batch_rows=1)) as reader:
        next(reader)
        path.write_bytes(path.read_bytes())  # Same content, changed input during the open scan.
        with pytest.raises(ValueError, match="input changed"):
            list(reader)


def configured_sources(tmp_path, *, second_bad=False):
    """A complete tiny canonical Config/store; never load retained artifacts."""
    from teadata.manifest import canonical_json
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
    from teadata.provenance import ProvenanceStore
    from teadata.teadata_config import Config

    sources = {}
    paths = {}
    store = ProvenanceStore(tmp_path / "original")
    for scope, code, digits in [("campus", "CAMPUS", 9), ("district", "DISTRICT", 6)]:
        folder = tmp_path / scope
        folder.mkdir()
        labels, codes, names = [scope + " ID", "Raw token"], [code, "VALUE"], [scope + "_number", "value"]
        records = [labels, codes, *[[str(1902000 + i).zfill(digits), token]
            for i, token in enumerate([" -3 ", "first\nsecond", "", "•", "0"], 1)]]
        if scope == "district":
            for i, row in enumerate(records[2:], 1):
                row[0] = str(1900 + i).zfill(6)
        if second_bad and scope == "district":
            records[-1].append("unreviewed extra")
        path = write_source(folder, records)
        dataset = scope + "_tapr_student_staff_profile"
        mapping = mapping_for(path, labels=labels, codes=codes, fields=names)
        mapping["metric_period"] = {"kind": "unknown"}
        mapping["layout"] = {"label_record": 1, "code_record": 2, "first_data_record": 3, "last_data_record": 7, "annotations": []}
        for field in mapping["fields"]:
            identity = field["position"] == 1
            field.update(disposition="identity" if identity else "fact", publisher_type=None,
                publisher_description=None,
                dictionary_status="not_applicable" if identity else "not_documented")
        identifier = store.record_acquisition(path, ResolvedDataset(dataset, "data_sources", 2025, 2025,
            str(path), DatasetMetadata(version="2025")))
        schema = folder / "schema.json"
        schema_relative = str(schema.relative_to(tmp_path))
        schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1,
            "datasets": [{"dataset": dataset, "edition": "2025", "source_only": True, "tables": [mapping]}]}) + "\n")
        sources[dataset] = {2025: {"path": str(path), "metadata": {"version": "2025",
            "provenance_ids": [identifier],
            "references": [{"reference_id": "columns", "role": "schema", "path": schema_relative,
                "status": "available", "reviewed_sha256": sha256(schema.read_bytes()).hexdigest()}]},
            "reader": {"profiles": {"source": {"column_schema": schema_relative,
                "tapr_source": {"scope": scope, "tables": {"student": str(path)}}}}}}}
        paths[scope] = path
    cfg = Config.from_dict({"data_sources": sources})
    cfg.base_dir = tmp_path
    return cfg, paths, store


def test_all_scopes_validate_before_any_writer_or_archive(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from teadata.provenance import ProvenanceStore

    cfg, _, store = configured_sources(tmp_path, second_bad=True)
    monkeypatch.setattr(pq, "ParquetWriter", lambda *a, **k: pytest.fail("Invalid cohort opened writer"))
    monkeypatch.setattr(ProvenanceStore, "record_acquisition", lambda *a, **k: pytest.fail("Invalid cohort archived"))
    output = tmp_path / "new" / "cohort"
    with pytest.raises(ValueError, match="record 7 has 3 fields"):
        prepare_profiles(cfg, edition="2025", scopes=["campus", "district"], input_dir=None,
                          output_dir=output, reference_store=store, budget=TaprReadBudget(batch_rows=1))
    assert not output.parent.exists()


def test_source_requests_are_metadata_only_and_do_not_open_raw_paths(tmp_path, monkeypatch):
    cfg, paths, _ = configured_sources(tmp_path)
    original_open = Path.open
    def open_path(path, *args, **kwargs):
        if path in paths.values():
            pytest.fail("Metadata-only source selection opened raw rows")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_path)
    requests = cfg.tapr_source_requests(DATASET, 2025)
    assert len(requests) == 1 and requests[0].path == paths["campus"]


def test_bounded_producer_uses_exact_row_groups_and_original_processing_closure(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from teadata.provenance import ProvenanceStore

    cfg, paths, store = configured_sources(tmp_path)
    output = tmp_path / "cohort"
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("Whole Parquet readback"))
    monkeypatch.setattr(cfg, "read_tapr_sources", lambda *a, **k: pytest.fail("Eager source path"))
    result = prepare_profiles(cfg, edition="2025", scopes=["campus", "district"], input_dir=None,
        output_dir=output, reference_store=store, budget=TaprReadBudget(batch_rows=2, batch_bytes=256))
    for scope in paths:
        with pq.ParquetFile(output / (scope + ".parquet")) as native:
            assert native.metadata.num_row_groups == 3
            assert [native.metadata.row_group(i).num_rows for i in range(3)] == [2, 2, 1]
            table = pa.Table.from_batches(native.iter_batches(batch_size=2))
        assert table["value"].to_pylist() == [" -3 ", "first\nsecond", "", "•", "0"]
        receipt = result["scopes"][scope]
        assert receipt["raw_string_readback_equal"] and receipt["rows"] == 5
        graph = ProvenanceStore(output / "provenance").verify_record_closure([receipt["processing_record_id"]])
        processing = graph[receipt["processing_record_id"]]
        assert processing["parameters"]["source_read_budget"]["batch_bytes"] == 256
        assert receipt["qualified_metrics"] == {}
        assert all(record["downloaded_at"] is None for record in graph.values() if record["kind"] == "acquisition")


@pytest.mark.parametrize("problem", ["second_pass_source", "readback", "extra_table"])
def test_late_source_readback_failures_cleanup_and_multitable_has_no_eager_fallback(tmp_path, monkeypatch, problem):
    import teadata.scripts.build_tapr_profiles as producer
    from teadata.tapr import TaprSourceReader

    cfg, paths, store = configured_sources(tmp_path)
    output = tmp_path / "new" / "cohort"
    if problem == "extra_table":
        original_requests = cfg.tapr_source_requests
        monkeypatch.setattr(cfg, "tapr_source_requests", lambda *a, **k: original_requests(*a, **k) * 2)
        monkeypatch.setattr(TaprSourceReader, "_read", lambda *a: pytest.fail("Unsupported merge opened source"))
    elif problem == "second_pass_source":
        original_write = producer._write_raw_source
        def write(request, expected, path, budget):
            paths["campus"].write_text(paths["campus"].read_text().replace(" -3 ", " -2 "))
            return original_write(request, expected, path, budget)
        monkeypatch.setattr(producer, "_write_raw_source", write)
    else:
        import pyarrow.parquet as pq
        original_read_group = pq.ParquetFile.read_row_group
        def corrupt(self, *args, **kwargs):
            value = original_read_group(self, *args, **kwargs)
            return value.set_column(1, "value", pa.array(["changed"] * value.num_rows))
        monkeypatch.setattr(pq.ParquetFile, "read_row_group", corrupt)
    with pytest.raises(ValueError, match="physical table|input differs|changed|artifact differs"):
        prepare_profiles(cfg, edition="2025", scopes=["campus", "district"], input_dir=None,
            output_dir=output, reference_store=store, budget=TaprReadBudget(batch_rows=1))
    assert not output.exists() and not list(tmp_path.rglob(".tapr-preparation-*"))
    if problem == "extra_table":
        assert not output.parent.exists()


@pytest.fixture
def accepted_prepared_sources(tmp_path):
    from teadata.provenance import ProvenanceStore

    cfg, _, store = configured_sources(tmp_path)
    sources = tmp_path / "sources"
    result = prepare_profiles(cfg, edition="2025", scopes=["campus", "district"], input_dir=None,
        output_dir=sources, reference_store=store, budget=TaprReadBudget(batch_rows=2, batch_bytes=80))
    source_store = ProvenanceStore(sources / "provenance")
    # Bind the genuine synthetic schema observations already recorded by source preparation.
    from teadata.dataset_metadata import DatasetDefinition, DatasetMetadata
    from teadata.manifest import processor_identity
    for scope, prepared in result["scopes"].items():
        dataset = prepared["dataset"]
        definition = cfg.data_sources[dataset].entries[2025]
        metadata = definition.metadata.to_dict()
        schema_sha = prepared["column_schema_sha256"]
        metadata["references"][0]["provenance_ids"] = [identifier for identifier, record in
            source_store.verify_record_closure([prepared["processing_record_id"]]).items()
            if record["kind"] == "acquisition" and record["artifact"]["sha256"] == schema_sha]
        cfg.data_sources[dataset].entries[2025] = DatasetDefinition(definition.path, DatasetMetadata.from_dict(metadata), definition.reader)
    assert processor_identity() == source_store.read_record(result["scopes"]["campus"]["processing_record_id"])["code_version"]
    selected = {item["dataset"]: (item["processing_record_id"], (scope + "_number",)) for scope, item in result["scopes"].items()}
    return cfg, source_store, selected, result, sources


def test_prepared_leaf_preserves_admitted_groups_and_external_schema_bytes(tmp_path, monkeypatch, accepted_prepared_sources):
    import pyarrow.parquet as pq
    from teadata.release import DataRelease
    cfg, source_store, selected, result, sources = accepted_prepared_sources
    with pq.ParquetFile(sources / "campus.parquet") as source:
        assert next(source.iter_batches(batch_size=4096)).nbytes > 80
    output = tmp_path / "leaves"
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("Whole table read"))
    leaves = prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
        reference_store=source_store, prepared_inputs=selected)
    release = DataRelease(output / "release.json", offline=True)
    assert release.verify()
    for scope, item in result["scopes"].items():
        entry = leaves["datasets"][item["dataset"]]
        with pq.ParquetFile(release.artifact(entry["artifact_id"])) as native:
            assert [native.metadata.row_group(i).num_rows for i in range(native.metadata.num_row_groups)] == [2, 2, 1]
            assert all(native.read_row_group(i).nbytes <= 80 for i in range(native.metadata.num_row_groups))
        descriptor = next(ref for ref in release.catalog.column_schema_refs() if ref.sha256 == item["column_schema_sha256"])
        assert (output / descriptor.location).read_bytes() == cfg.column_contract(item["dataset"], 2025, reader_profile="source")["text"].encode()
    assert len(release.catalog.column_schema_refs()) == 2


@pytest.mark.parametrize("damage", ["missing_budget", "boolean_budget", "code", "script", "key_budget"])
def test_prepared_budget_and_code_admission_precede_every_writer(tmp_path, monkeypatch, accepted_prepared_sources, damage):
    import teadata.scripts.build_tapr_profiles as producer
    cfg, store, selected, result, _ = accepted_prepared_sources
    dataset = result["scopes"]["district"]["dataset"]
    record = store.read_record(selected[dataset][0])
    params = record["parameters"]
    code = record["code_version"]
    if damage == "missing_budget":
        params.pop("source_read_budget")
    elif damage == "boolean_budget":
        params["source_read_budget"]["batch_bytes"] = True
    elif damage == "code":
        code = "not-a-historical-digest"
    elif damage == "script":
        params["preparation_code_sha256"] = "not-a-historical-digest"
    else:
        params["source_read_budget"]["key_bytes"] = 1
    changed = store.record_processing(dataset=dataset, inputs=record["inputs"],
        outputs={name: store.verify_artifact(value["sha256"]) for name, value in record["outputs"].items()},
        processor=record["processor"], code_version=code, parameters=params, validation=record["validation"])
    selected[dataset] = changed, selected[dataset][1]
    monkeypatch.setattr(producer, "write_leaf_artifact", lambda *a, **k: pytest.fail("Writer before all source admissions"))
    output = tmp_path / "new" / "leaves"
    with pytest.raises(ValueError, match="bounded|budget"):
        prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
            reference_store=store, prepared_inputs=selected)
    assert not output.parent.exists()


def test_historical_preparation_reuses_exact_bytes_after_current_raw_readback(tmp_path, monkeypatch, accepted_prepared_sources):
    import pyarrow.parquet as pq
    import teadata.scripts.build_tapr_profiles as producer
    from teadata.provenance import ProvenanceStore

    cfg, store, selected, result, _ = accepted_prepared_sources
    dataset = result["scopes"]["district"]["dataset"]
    record = store.read_record(selected[dataset][0])
    record["parameters"]["preparation_code_sha256"] = "e" * 64
    historical = store.record_processing(dataset=dataset, inputs=record["inputs"],
        outputs={name: store.verify_artifact(value["sha256"]) for name, value in record["outputs"].items()},
        processor=record["processor"], code_version="f" * 64, parameters=record["parameters"], validation=record["validation"])
    selected[dataset] = historical, selected[dataset][1]
    before = store.verify_artifact(record["outputs"]["prepared_source"]["sha256"]).read_bytes()
    monkeypatch.setattr(producer, "_write_raw_source", lambda *a, **k: pytest.fail("Historical reuse must not write a new preparation"))
    output = tmp_path / "leaves"
    built = prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
                             reference_store=store, prepared_inputs=selected)
    archived = ProvenanceStore(output / "provenance")
    leaf = archived.read_record(built["datasets"][dataset]["processing_record_id"])
    recipe = leaf["parameters"]["recipe"]
    assert historical in leaf["inputs"]
    assert recipe["historical_processor_sha256"] == "f" * 64
    assert recipe["preparation_code_sha256"] == "e" * 64
    assert recipe["revalidation_code_sha256"] == sha256(Path(producer.__file__).read_bytes()).hexdigest()
    assert recipe["raw_source_revalidated"] is True and leaf["validation"]["raw_source_revalidated"] is True
    prepared = store.verify_artifact(record["outputs"]["prepared_source"]["sha256"])
    assert prepared.read_bytes() == before
    with pq.ParquetFile(prepared) as native:
        assert native.metadata.num_rows == 5


@pytest.mark.parametrize("problem", ["current_reader", "raw", "prepared"])
def test_reused_preparation_disagreement_blocks_every_writer(tmp_path, monkeypatch, accepted_prepared_sources, problem):
    import pyarrow.parquet as pq
    import teadata.scripts.build_tapr_profiles as producer
    from teadata.tapr import TaprSourceReader

    cfg, store, selected, result, _ = accepted_prepared_sources
    dataset = result["scopes"]["campus"]["dataset"]
    record = store.read_record(selected[dataset][0])
    if problem == "current_reader":
        original_next = TaprSourceReader.__next__
        def changed_reader(self):
            batch = original_next(self)
            return batch.set_column(1, batch.schema.field(1), pa.array(["changed dependency behavior"] * batch.num_rows))
        monkeypatch.setattr(TaprSourceReader, "__next__", changed_reader)
    elif problem == "raw":
        raw_sha = next(item["source_revision_sha256"] for item in result["scopes"]["campus"]["source_tables"])
        store.verify_artifact(raw_sha).write_bytes(b"changed original bytes")
    else:
        path = tmp_path / "different-preparation.parquet"
        source = store.verify_artifact(record["outputs"]["prepared_source"]["sha256"])
        with pq.ParquetFile(source) as original, pq.ParquetWriter(path, original.schema_arrow) as writer:
            for index in range(original.metadata.num_row_groups):
                table = original.read_row_group(index)
                if index == 0:
                    table = table.set_column(1, table.schema.field(1), pa.array(["wrong retained value"] * table.num_rows))
                writer.write_table(table, row_group_size=table.num_rows)
        changed = store.record_processing(dataset=dataset, inputs=record["inputs"],
            outputs={name: path if name == "prepared_source" else store.verify_artifact(value["sha256"])
                     for name, value in record["outputs"].items()}, processor=record["processor"],
            code_version=record["code_version"], parameters=record["parameters"], validation=record["validation"])
        selected[dataset] = changed, selected[dataset][1]
    monkeypatch.setattr(producer, "write_leaf_artifact", lambda *a, **k: pytest.fail("Writer before complete current raw readback"))
    output = tmp_path / "new" / "leaves"
    with pytest.raises(ValueError, match="checksum mismatch|differs from validated raw strings"):
        prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
                         reference_store=store, prepared_inputs=selected)
    assert not output.parent.exists()


@pytest.mark.parametrize("damage", ["readback", "source_changed"])
def test_prepared_leaf_late_failure_never_publishes_partial_cohort(tmp_path, monkeypatch, accepted_prepared_sources, damage):
    import teadata.scripts.build_tapr_profiles as producer
    cfg, store, selected, result, _ = accepted_prepared_sources
    dataset = result["scopes"]["campus"]["dataset"]
    record = store.read_record(selected[dataset][0])
    prepared_sha = record["outputs"]["prepared_source"]["sha256"]
    if damage == "readback":
        original = producer._prepared_batches
        calls = 0
        def changed_batches(native, budget):
            nonlocal calls
            calls += 1
            for index, batch in enumerate(original(native, budget)):
                if calls == 1 and index == 0:
                    yield batch.set_column(1, batch.schema.field(1), pa.array(["changed"] * batch.num_rows))
                else:
                    yield batch
        monkeypatch.setattr(producer, "_prepared_batches", changed_batches)
    else:
        original = producer.write_leaf_artifact
        def changed_source(*a, **k):
            output = original(*a, **k)
            # Perturb the active private-stage preparation, not the upstream fixture store.
            prepared = Path(a[0]).parent / "provenance" / "objects" / prepared_sha
            with prepared.open("ab") as stream:
                stream.write(b"injected late source change")
            return output
        monkeypatch.setattr(producer, "write_leaf_artifact", changed_source)
    output = tmp_path / "new" / "leaves"
    with pytest.raises(ValueError, match="readback differs|changed during materialization"):
        prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
            reference_store=store, prepared_inputs=selected)
    assert not output.exists() and list(output.parent.iterdir()) == []
