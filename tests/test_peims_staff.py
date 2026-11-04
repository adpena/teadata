from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from teadata.peims_leaves import _key_columns
from teadata.peims_staff import STAFF, staff_allows_unavailable_average, staff_employee_basis, staff_values, validate_staff_request
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.teadata_config import load_config
from teadata.values import ValuePolicy


@pytest.mark.parametrize(("year", "state_rows", "district_rows"), [(2023, 97, 53518), (2024, 99, 53836),
                                                                   (2025, 103, 53691), (2026, 100, 53638)])
def test_staff_config_binds_exact_publisher_year_scope_and_layout(year, state_rows, district_rows):
    config = load_config()
    source = config.resolve_dataset(STAFF, year, exact=True, reader_profile="source")
    assert config.is_source_only(STAFF, year)
    assert source.metadata.published_at == ("2026-03-19" if year == 2026 else None)
    assert source.metadata.period.label == f"{year - 1}-{year}"
    assert {name: entry.unit for name, entry in source.metadata.metrics.items()} == {
        "staff_fte": "fte", "staff_total_base_pay": "currency", "staff_average_base_pay": "currency",
        **({"staff_employee_count": "count"} if year == 2026 else {}),
    }
    options = reader_options(source.reader, "source")
    assert set(options["acquisition_ids"]) == {"state", "district"}
    store = ProvenanceStore(Path(__file__).resolve().parents[1] / "provenance")
    roots = [*options["acquisition_ids"].values(),
             *(identifier for ref in source.metadata.references for identifier in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    schema = config.column_contract(STAFF, year, reader_profile="source")
    assert schema is not None
    tables = crosswalk_dataset(parse_column_schema(schema["text"].encode()), STAFF, str(year))["tables"]
    for table, rows in zip(tables, (state_rows, district_rows), strict=True):
        scope = table["table"]
        record = graph[options["acquisition_ids"][scope]]
        validate_staff_request(record, year, scope)
        assert table_input_sha256s(table) == (record["artifact"]["sha256"],)
        layout = table["layout"]
        assert layout["last_data_record"] - layout["first_data_record"] + 1 == rows
        assert {item["field"] for item in table["fields"]} >= set(_key_columns(STAFF, scope))
        assert not any(item["field"] == "school_year" for item in table["fields"])
        with pytest.raises(ValueError, match="scope or edition"):
            validate_staff_request(record, year, "state" if scope == "district" else "district")


def test_staff_values_preserve_decimal_fte_and_unavailable_average():
    policy = ValuePolicy.from_dict({"policy_id": "staff", "tokens": {".": {"status": "missing", "reason": "publisher_unavailable"}}})
    fte, status = staff_values(pa.chunked_array([["0.00", "9.40", "767926.22"]]), policy, unit="fte", allow_dot=False)
    assert fte.type == pa.decimal128(16, 2)
    assert fte.to_pylist() == [Decimal("0.00"), Decimal("9.40"), Decimal("767926.22")]
    assert status.to_pylist() == ["observed"] * 3
    average, status = staff_values(pa.chunked_array([["0", ".", "57999"]]), policy, unit="currency", allow_dot=True)
    assert average.type == pa.int64() and average.to_pylist() == [0, None, 57999]
    assert status.to_pylist() == ["observed", "missing", "observed"]
    assert staff_allows_unavailable_average("staff_average_base_pay", 2026, "district")
    assert not staff_allows_unavailable_average("staff_total_base_pay", 2026, "district")
    assert not staff_allows_unavailable_average("staff_average_base_pay", 2026, "state")
    assert not staff_allows_unavailable_average("staff_average_base_pay", 2024, "district")


@pytest.mark.parametrize(("unit", "token"), [("fte", "7.0"), ("fte", "7.000"), ("fte", "1e0"),
                                               ("fte", "."), ("count", "3.00"), ("currency", "-999")])
def test_staff_value_projection_rejects_unreviewed_lexemes(unit, token):
    with pytest.raises(ValueError, match="outside its reviewed unit or status"):
        staff_values(pa.chunked_array([[token]]), ValuePolicy("staff"), unit=unit, allow_dot=False)


def test_staff_employee_count_basis_is_scoped_to_literal_total_roles():
    categories = pa.chunked_array([["TEACHING STAFF", "TEACHING STAFF", " "]])
    roles = pa.chunked_array([["KINDERGARTEN", "TOTAL TEACHING STAFF", "TOTAL PERSONNEL"]])
    assert staff_employee_basis(categories, roles).to_pylist() == ["positions", "distinct_employees", "distinct_employees"]
    with pytest.raises(ValueError, match="unreviewed employee-count total role"):
        staff_employee_basis(pa.chunked_array([["SUPPORT STAFF"]]), pa.chunked_array([["TOTAL TEACHING STAFF"]]))
