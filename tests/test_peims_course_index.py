import csv
from io import BytesIO, StringIO
import json
import os
from types import SimpleNamespace
from zipfile import ZipFile

import pyarrow as pa
import pytest

from teadata import peims_course_index as index
from teadata.manifest import ContractError, file_identity


@pytest.fixture
def course_release(monkeypatch):
    label = "2025-2026"
    records = [
        {"school_year": label, "course_code": "00123456", index.SOURCE_RECORD: 10,
         "district_number": "001903", index.FIELD: 0, index.RAW: "00", index.STATUS: "observed"},
        {"school_year": label, "course_code": "00123456", index.SOURCE_RECORD: 11,
         "district_number": "001902", index.FIELD: None, index.RAW: "-999", index.STATUS: "suppressed"},
    ]
    texas = {key: value for key, value in records[0].items() if key != "district_number"}
    texas.update({index.FIELD: 12, index.RAW: "12"})

    def observations(dataset, field, *, scope, school_year):
        assert (dataset, field, school_year) == (index.DATASET, index.FIELD, 2026)
        rows = records if scope == "district" else [texas]

        def batches():
            yield pa.RecordBatch.from_pylist(rows)

        return SimpleNamespace(measurement_period=SimpleNamespace(label=label), iter_batches=batches,
                               release=None, _entity_build=None, rows=rows)

    def source_projection(view, scope):
        source = SimpleNamespace(to_dict=lambda: {"title": scope})
        fields = ("measurement_school_year", *view.rows[0])
        return SimpleNamespace(artifact=SimpleNamespace(contract={"rows": len(view.rows)}),
                               provenance=lambda *args, **kwargs: source), fields

    monkeypatch.setattr(index, "_source_projection", source_projection)
    return SimpleNamespace(manifest_sha256="a" * 64, observation_years=lambda *args, **kwargs: {index.FIELD: (2026,)},
                           observations=observations)


def test_course_index_reads_publisher_order_and_produces_zip_receipt(tmp_path, course_release):
    database = tmp_path / "courses.sqlite3"
    index.build_course_index(course_release, database)
    pin = file_identity(database)["sha256"]
    result = index.select_course_enrollment(course_release, database, expected_sha256=pin,
                                            scope="district", years=[2026], course="00123456",
                                            districts=[1902, "'001903", 1904])
    assert [(row.district, row.enrollment, row.raw, row.status) for row in result.rows] == [
        ("001903", 0, "00", "observed"), ("001902", None, "-999", "suppressed"),
    ]
    direct = index.select_course_enrollment(course_release, database, expected_sha256=pin,
                                            scope="state", years=[2026], course="00123456")
    assert [(row.district, row.enrollment) for row in direct.rows] == [(None, 12)]
    output = BytesIO()
    result.to_csv_bundle(output)
    with ZipFile(output) as archive:
        rows = list(csv.DictReader(StringIO(archive.read("data.csv").decode())))
        assert [(row["district_number"], row[index.FIELD], row[index.RAW]) for row in rows] == [
            ("001903", "0", "00"), ("001902", "", "-999"),
        ]
        receipt = json.loads(archive.read("provenance.json"))
        assert receipt == result.to_dict()
        assert receipt["ids"] == ["001902", "001903", "001904"]


def test_course_index_rechecks_replaced_files_and_guards_release_pin(tmp_path, course_release):
    database = tmp_path / "courses.sqlite3"
    index.build_course_index(course_release, database)
    pin = file_identity(database)["sha256"]

    def select():
        return index.select_course_enrollment(course_release, database, expected_sha256=pin,
                                              scope="state", years=[2026], course="00123456")

    assert select().rows
    course_release.manifest_sha256 = "b" * 64
    with pytest.raises(ContractError, match="different data release"):
        select()
    course_release.manifest_sha256 = "a" * 64
    sidecar = database.with_name(database.name + "-wal")
    sidecar.touch()
    with pytest.raises(ValueError, match="checkpointed"):
        select()
    sidecar.unlink()
    original = database.stat()
    with database.open("r+b") as stream:
        stream.write(b"corrupted")
    os.utime(database, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="SHA-256"):
        select()


def test_course_index_preserves_existing_file_and_rejects_duplicate_ids(tmp_path, course_release):
    database = tmp_path / "courses.sqlite3"
    database.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        index.build_course_index(course_release, database)
    assert database.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="same ID"):
        index.select_course_enrollment(course_release, database, expected_sha256="a" * 64,
                                       scope="district", years=[2026], course="00123456", districts=[1901, "001901"])


def test_course_selection_captures_mutable_cohort_before_io(tmp_path, course_release):
    database = tmp_path / "courses.sqlite3"
    index.build_course_index(course_release, database)
    years = [2026]
    districts = [1902]
    original = course_release.observations

    def observations(*args, **kwargs):
        years.append(2025)
        districts.append(1903)
        return original(*args, **kwargs)

    course_release.observations = observations
    result = index.select_course_enrollment(course_release, database, expected_sha256=file_identity(database)["sha256"],
                                            scope="district", years=years, course="00123456", districts=districts)
    assert result.to_dict()["school_years"] == [2026]
    assert result.to_dict()["ids"] == ["001902"]
    assert len(result.rows) == 1 and result.rows[0].district == "001902"
