from csv import DictReader
from io import BytesIO, StringIO
import shutil

import pytest

from teadata.askted_leaves import DATASET, build_askted_archive
from teadata.dataset_metadata import EditionKey
from teadata.manifest import file_identity
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease

from test_askted_directory import archive, row


def test_archive_candidate_preserves_raw_order_and_provenance(tmp_path):
    district_only = list(row("244905", "", "", site=""))
    district_only[9] = ""
    originals = [
        row("'015915", "'015915001", "FIRST HIGH", site=" "),
        tuple(district_only),
        row("015915", "015915001", "FIRST HIGH", served="'08", site="ANOTHER CITY"),
    ]
    store, receipt, digest = archive(tmp_path, rows=originals)
    index_receipt = store.read_record(receipt)["parameters"]["archive_index_record"]
    release_path = build_askted_archive(store, receipt, year=2026, season="fall", expected_sha256=digest,
                                       output=tmp_path / "candidate")
    with pytest.raises(FileExistsError):
        build_askted_archive(store, receipt, year=2026, season="fall", expected_sha256=digest,
                            output=release_path.parent)
    shutil.rmtree(store.root)
    release = DataRelease(release_path, expected_sha256=file_identity(release_path)["sha256"], offline=True)
    assert len(release.verify()) == 2
    packaged = ProvenanceStore(release_path.parent / "provenance")
    for scope, expected in (("district", [2, 3, 4]), ("campus", [2, 4])):
        edition = EditionKey(DATASET, "2026-fall", EditionKey.source_revision((digest,)),
                             scope, scope, "one logical source record per row")
        selection = release.table(edition)
        native = selection.to_arrow()
        assert native["source_record"].to_pylist() == expected
        assert native["district_number"].to_pylist() == [originals[i - 2][0] for i in expected]
        provenance = selection.provenance().to_dict()
        assert provenance["artifact"]["id"] and provenance["source"]["report_period"] == {"kind": "as_of", "as_of": "2025-10-31"}
        processing = provenance["artifact"]["processing_receipt"]
        graph = packaged.verify_record_closure([processing])
        recipe = graph[processing]["parameters"]["recipe"]
        name = f"{scope}_name"
        assert recipe["data_origin"] == "publisher_reported_directory"
        assert recipe["publisher_column_ordinals"][name] == (2 if scope == "district" else 4)
        assert receipt in graph and index_receipt in graph
        assert {record["record_id"] for record in release.catalog.acquisitions(DATASET, edition=edition)} == {receipt}
        assert any(ref.reference_id == "archive_index" and ref.provenance_ids == (index_receipt,)
                   for ref in release.catalog.metadata(DATASET, edition=edition).references)
        csv = BytesIO()
        selection.to_csv(csv)
        assert [int(item["source_record"]) for item in DictReader(StringIO(csv.getvalue().decode()))] == expected
        if scope == "campus":
            assert native["campus_number"].to_pylist() == ["'015915001", "015915001"]
            assert native["campus_site_city"].to_pylist() == [" ", "ANOTHER CITY"]
        else:
            assert native["source_campus_number"].to_pylist() == ["'015915001", "", "015915001"]
