import csv
from hashlib import sha256

import pytest

from teadata.scripts import build_tapr_2025_profiles as script
from teadata.column_schema import source_schema_fingerprint


LABELS = ["9 Digit Campus Number", "Example Value"]
CODES = ["CAMPUS", "VALUE"]


def write_source(tmp_path, records):
    path = tmp_path / "source.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(records)
    return path


def mapping_for(path, *, labels=LABELS, codes=CODES, fields=("campus_number", "example_value")):
    return {"table": "student", "input_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_schema_sha256": source_schema_fingerprint(labels, codes),
            "fields": [{"position": i, "code": code, "label": label, "field": field}
                       for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)]}


def read_source(path, *, mapping=None):
    return script._read_tapr_csv(path, mapping=mapping_for(path) if mapping is None else mapping, key_field="campus_number")


@pytest.mark.parametrize("token", ["-1", "-3", "-2", ".", "•", "n/a", "0", "0.0", "9007199254740993", ""])
def test_reader_preserves_source_tokens_without_interpreting_masking(tmp_path, token):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", token]])
    before = path.read_bytes()
    headers, rows, elements = read_source(path)
    assert headers == ["campus_number", "example_value"]
    assert rows == {"001902001": {"campus_number": "001902001", "example_value": token}}
    assert elements == [
        {"code": "CAMPUS", "label": LABELS[0], "sanitized_label": "campus_number"},
        {"code": "VALUE", "label": LABELS[1], "sanitized_label": "example_value"},
    ]
    assert path.read_bytes() == before


def test_metadata_retains_exact_human_labels_and_codes(tmp_path):
    labels = [LABELS[0], "  Campus 2025: Example (Value)  "]
    codes = ["CAMPUS", " VALUE "]
    path = write_source(tmp_path, [labels, codes, ["001902001", " -3 "]])
    headers, rows, elements = read_source(path, mapping=mapping_for(path, labels=labels, codes=codes,
                                                                  fields=("campus_number", "campus_2025_example_value")))
    assert headers == ["campus_number", "campus_2025_example_value"]
    assert elements[1] == {"code": " VALUE ", "label": labels[1], "sanitized_label": headers[1]}
    assert rows["001902001"][headers[1]] == "-3"  # Existing body whitespace behavior.


@pytest.mark.parametrize("override", ["CamelCase", "naïve", "2025_name", "class", ""])
def test_explicit_header_overrides_must_be_python_safe_ascii_snake_case(tmp_path, override):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    with pytest.raises(ValueError, match="(?i)invalid canonical header.*ASCII snake_case"):
        read_source(path, mapping=mapping_for(path, fields=("campus_number", override)))


def test_explicit_header_override_collision_names_the_column(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    with pytest.raises(ValueError, match="(?i)canonical header collision in column 2: 'campus_number'"):
        read_source(path, mapping=mapping_for(path, fields=("campus_number", "campus_number")))


@pytest.mark.parametrize("records, message", [
    ([], "header records"),
    ([LABELS], "header records"),
    ([LABELS, ["CAMPUS"]], "header width mismatch"),
    ([LABELS, CODES + ["EXTRA"]], "header width mismatch"),
    ([LABELS, ["001902001", "42"], ["001902041", "99"]], "expected source-code header 'CAMPUS'"),
    ([LABELS, ["CAMPUS", ""]], "headers differ"),
    ([LABELS, ["CAMPUS", "CAMPUS"]], "headers differ"),
    ([[LABELS[0], "Class"], CODES], "headers differ"),
    ([[LABELS[0], "2025 Value"], CODES], "headers differ"),
    ([[LABELS[0], "???"], CODES], "headers differ"),
    ([[LABELS[0], "Example-Value", "Example Value"], ["CAMPUS", "A", "B"]], "headers differ"),
    ([["Not The Campus Key", "Example"], CODES], "headers differ"),
    ([LABELS, CODES], "no TAPR data records"),
])
def test_ambiguous_or_incomplete_headers_fail_closed(tmp_path, records, message):
    path = write_source(tmp_path, records)
    with pytest.raises(ValueError, match=message) as error:
        read_source(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("codes", [["CAMPUS", ""], ["CAMPUS", " CAMPUS "]])
def test_crosswalk_rejects_empty_or_duplicate_source_codes(tmp_path, codes):
    path = write_source(tmp_path, [LABELS, codes, ["001902001", "42"]])
    with pytest.raises(ValueError, match="(?i)(invalid header crosswalk column|duplicate source code)"):
        read_source(path, mapping=mapping_for(path, codes=codes))


def test_reader_rejects_input_bytes_outside_the_reviewed_edition(tmp_path):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"]])
    mapping = mapping_for(path)
    path.write_text(path.read_text().replace("42", "99"))
    with pytest.raises(ValueError, match="input differs"):
        read_source(path, mapping=mapping)


@pytest.mark.parametrize("row", [["001902001"], ["001902001", "42", "EXTRA"], []])
def test_reader_rejects_bad_width_instead_of_padding_truncating_or_skipping(tmp_path, row):
    path = write_source(tmp_path, [LABELS, CODES, row])
    with pytest.raises(ValueError, match=r"record 3 has .* fields; expected 2"):
        read_source(path)


@pytest.mark.parametrize("key", ["", "000000000", "-1", "True", "001902001.5", "1001902001", "1e3", "９９９９９９９９９", "Notes: suppression applies"])
def test_invalid_identifiers_and_footer_text_are_not_data(tmp_path, key):
    path = write_source(tmp_path, [LABELS, CODES, [key, "42"]])
    with pytest.raises(ValueError, match="record 3 has invalid campus_number"):
        read_source(path)


@pytest.mark.parametrize("later", ["001902001", "1902001", "'001902001"])
def test_duplicate_entity_identity_is_never_overwritten(tmp_path, later):
    path = write_source(tmp_path, [LABELS, CODES, ["001902001", "42"], [later, "99"]])
    with pytest.raises(ValueError, match="duplicate campus_number at records 3 and 4"):
        read_source(path)


def test_campus_district_claim_is_checked_when_present(tmp_path):
    labels, codes = [LABELS[0], "6 Digit County District Number"], ["CAMPUS", "DISTRICT"]
    path = write_source(tmp_path, [
        labels, codes,
        ["001902001", "227901"],
    ])
    with pytest.raises(ValueError, match="campus and district identifiers disagree"):
        read_source(path, mapping=mapping_for(path, labels=labels, codes=codes,
                                              fields=("campus_number", "district_number")))


def test_district_reader_uses_six_digit_identity_and_preserves_metadata(tmp_path):
    labels, codes = ["6 Digit County District Number", "District Name"], ["DISTRICT", "DISTNAME"]
    path = write_source(tmp_path, [
        labels, codes,
        ["001902", "CAYUGA ISD"],
    ])
    headers, rows, elements = script._read_tapr_csv(path, mapping=mapping_for(path, labels=labels, codes=codes,
                                                 fields=("district_number", "district_name")), key_field="district_number")
    assert headers == ["district_number", "district_name"]
    assert rows["001902"]["district_name"] == "CAYUGA ISD"
    assert elements[0]["code"] == "DISTRICT"


def test_district_reader_refuses_a_data_record_as_the_code_header(tmp_path):
    path = write_source(tmp_path, [
        ["6 Digit County District Number", "District Name"],
        ["001902", "CAYUGA ISD"], ["001903", "FRANKSTON ISD"],
    ])
    with pytest.raises(ValueError, match="expected source-code header 'DISTRICT'"):
        script._read_tapr_csv(path, mapping=mapping_for(path,
            labels=["6 Digit County District Number", "District Name"], codes=["DISTRICT", "DISTNAME"],
            fields=("district_number", "district_name")), key_field="district_number")


def test_csv_quoting_error_is_not_silently_accepted(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text('9 Digit Campus Number,Example Value\nCAMPUS,VALUE\n001902001,"unfinished\n')
    with pytest.raises(ValueError, match="invalid CSV near physical line"):
        read_source(path)


def merge(primary, secondary):
    return script._merge_tables(
        list(next(iter(primary.values()))), primary,
        list(next(iter(secondary.values()))), secondary,
        key_field="campus_number", preferred_cols=("campus_number", "campus_name"),
    )


def test_merge_rejects_conflicting_shared_values_without_masking_assumptions():
    primary = {"001902001": {"campus_number": "001902001", "example": "."}}
    secondary = {"001902001": {"campus_number": "001902001", "example": "42"}}
    with pytest.raises(ValueError, match="Conflicting TAPR values.*example"):
        merge(primary, secondary)
    assert primary["001902001"]["example"] == "."


def test_merge_preserves_blank_fill_behavior_order_and_nonconflicting_tokens():
    primary = {"001902001": {"campus_number": "001902001", "empty": "", "student": "-1"}}
    secondary = {"001902001": {"campus_number": "001902001", "empty": "filled", "staff": "-3"}}
    headers, records = merge(primary, secondary)
    assert headers == ["campus_number", "empty", "student", "staff"]
    assert records == [{"campus_number": "001902001", "empty": "filled", "student": "-1", "staff": "-3"}]


def test_merge_retains_one_sided_entities_without_claiming_complete_coverage():
    primary = {"001902001": {"campus_number": "001902001", "student": "-1"}}
    secondary = {"001902041": {"campus_number": "001902041", "staff": "-3"}}
    headers, records = merge(primary, secondary)
    assert headers == ["campus_number", "student", "staff"]
    assert records == [primary["001902001"], secondary["001902041"]]
    assert script._rows_for_headers(headers, records) == [["001902001", "-1", ""], ["001902041", "", "-3"]]


def test_merge_refuses_two_spellings_of_the_same_source_identity():
    primary = {"001902001": {"campus_number": "001902001"}}
    secondary = {"1902001": {"campus_number": "1902001"}}
    with pytest.raises(ValueError, match="Inconsistent TAPR campus_number representations"):
        merge(primary, secondary)


def test_bad_input_prevents_cli_from_invoking_a_writer(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2025 Campus Student Information.csv").write_text(
        "9 Digit Campus Number,Example Value\nCAMPUS,VALUE\n001902001,42,EXTRA\n"
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("Writer must not run for invalid input")
    monkeypatch.setattr(script, "_write_workbook", forbidden)
    monkeypatch.setattr(script, "load_config", lambda *_: object())
    mapping = mapping_for(raw / "2025 Campus Student Information.csv")
    monkeypatch.setattr(script, "_header_tables", lambda *_: {"student": mapping, "staff": mapping})
    with pytest.raises(ValueError, match="record 3 has 3 fields"):
        script.main(["--input-dir", str(raw), "--output-dir", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()
