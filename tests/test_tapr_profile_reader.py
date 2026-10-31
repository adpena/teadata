import csv
from dataclasses import replace
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from teadata.column_schema import source_schema_fingerprint, validate_header_table
from teadata.identifiers import canonical_source_identifier, canonical_source_identifiers
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.tapr import TaprSourceTable, merge_tapr_tables, read_tapr_table


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
    def load(*args, **kwargs):
        return [read_source(path)]
    cfg = SimpleNamespace(read_tapr_sources=load)
    monkeypatch.setattr(pq, "write_table", lambda *a, **k: pytest.fail("Writer ran for invalid input"))
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
