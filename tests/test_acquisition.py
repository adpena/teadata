from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from teadata.acquisition import AcquisitionClient, AcquisitionError
from teadata.dataset_metadata import ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.scripts.download_askted import (
    Archive, ArchivePage, validate_archive_columns, validate_archive_response,
)


class Response(BytesIO):
    status = 200
    url = "https://example.org/source.csv"

    def __init__(self, payload=b"id\n001\n", headers=None):
        super().__init__(payload)
        self.headers = headers or {}


def client(tmp_path, response, **options):
    return AcquisitionClient(ProvenanceStore(tmp_path / "archive"),
        opener=SimpleNamespace(open=lambda *args, **kwargs: response), **options)


def source():
    return ResolvedDataset("example", "data_sources", 2025, 2025, "")


def test_success_preserves_bytes_and_records_http_without_credentials(tmp_path):
    selected = client(tmp_path, Response(headers={"Content-Type":"text/csv", "Set-Cookie":"secret-cookie"}))
    result = selected.fetch("https://example.org/source.csv", source(), filename="source.csv",
        body=b"hidden=secret-token", headers={"Authorization":"secret-header"}, parameters={"year":2025})
    record = selected.store.read_record(result.record_id)
    assert result.path.read_bytes() == b"id\n001\n"
    assert record["downloaded_at"] is not None
    assert {key: value for key, value in record["parameters"].items() if key != "acquisition_client_sha256"} == {"year":2025, "http_method":"POST", "http_status":200}
    assert len(record["parameters"]["acquisition_client_sha256"]) == 64
    assert record["artifact"]["filename"] == "source.csv"
    assert "secret" not in json.dumps(record)


@pytest.mark.parametrize("payload,headers,limit", [
    (b"too large", {}, 2), (b"abc", {"Content-Length":"4"}, 20),
    (b"", {}, 20), (b"abc", {"Content-Length":"invalid"}, 20),
    (b"abc", {"Content-Length":"200"}, 20),
])
def test_invalid_transfer_never_gets_a_success_receipt(tmp_path, payload, headers, limit):
    selected = client(tmp_path, Response(payload, headers), max_bytes=limit)
    with pytest.raises(AcquisitionError):
        selected.fetch("https://example.org/source.csv", source(), filename="source.csv")
    assert not (selected.store.root / "records").exists()


def test_content_validation_fails_without_retry_or_success_receipt(tmp_path):
    selected = client(tmp_path, Response(b"<html>Unavailable</html>"))
    def validate(path):
        raise AcquisitionError("Expected CSV")
    with pytest.raises(AcquisitionError, match="Expected CSV"):
        selected.fetch("https://example.org/source.csv", source(), filename="source.csv", validate=validate)
    assert not (selected.store.root / "records").exists()


def test_retries_transient_http_error_but_not_permanent_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("teadata.acquisition.time.sleep", lambda delay: None)
    calls = []
    def opened(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("https://example.org", 503, "Unavailable", {}, None)
        return Response()
    selected = AcquisitionClient(ProvenanceStore(tmp_path / "archive"), opener=SimpleNamespace(open=opened))
    selected.fetch("https://example.org/source.csv", source(), filename="source.csv")
    assert len(calls) == 2
    def missing(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.org", 404, "Not found", {}, None)
    selected.opener = SimpleNamespace(open=missing)
    with pytest.raises(HTTPError):
        selected.fetch("https://example.org/missing.csv", source(), filename="source.csv")
    assert len(calls) == 3


def test_filename_cannot_escape_archive_staging(tmp_path):
    with pytest.raises(ValueError, match="portable"):
        client(tmp_path, Response()).fetch("https://example.org", source(), filename="../outside.csv")


def test_askted_archive_selection_uses_published_labels_not_control_numbers():
    page = ArchivePage()
    page.feed('''<form><input type="hidden" name="__VIEWSTATE" value="opaque-state">
      <table><tr><td>2024-2025 School Year</td>
      <td><a href="javascript:__doPostBack(&#39;changed-control&#39;,&#39;&#39;)">Fall (as of 10/31/2024)</a></td>
      <td><a href="javascript:__doPostBack(&#39;another-control&#39;,&#39;&#39;)">Spring (as of 03/31/2025)</a></td></tr></table></form>''')
    assert page.fields == {"__VIEWSTATE":"opaque-state"}
    assert [(item.end_year,item.season,item.declared_as_of,item.event_target) for item in page.archives] == [
        (2025,"fall","2024-10-31","changed-control"),
        (2025,"spring","2025-03-31","another-control"),
    ]


@pytest.mark.parametrize("retry_after", ["9", "Thu, 01 Jan 2099 00:00:00 GMT", "unparseable", "-1"])
def test_retry_after_never_retries_earlier_than_publisher_limit(tmp_path, monkeypatch, retry_after):
    sleeps = []
    monkeypatch.setattr("teadata.acquisition.time.sleep", sleeps.append)
    calls = []
    def opened(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.org", 429, "Limited", {"Retry-After": retry_after}, None)
    selected = AcquisitionClient(ProvenanceStore(tmp_path / "archive"), opener=SimpleNamespace(open=opened), max_retry_wait=5)
    with pytest.raises(AcquisitionError, match="deferred"):
        selected.fetch("https://example.org/source.csv", source(), filename="source.csv")
    assert len(calls) == 1
    assert not sleeps
    assert not (selected.store.root / "records").exists()


def test_retry_after_delay_is_honored(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("teadata.acquisition.time.sleep", sleeps.append)
    calls = []
    def opened(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("https://example.org", 503, "Unavailable", {"Retry-After": "4"}, None)
        return Response()
    selected = AcquisitionClient(ProvenanceStore(tmp_path / "archive"), opener=SimpleNamespace(open=opened))
    selected.fetch("https://example.org/source.csv", source(), filename="source.csv")
    assert sleeps == [4]


def test_response_identity_validation_precedes_receipt(tmp_path):
    selected = client(tmp_path, Response())
    with pytest.raises(AcquisitionError, match="selected year"):
        selected.fetch("https://example.org/source.csv", source(), filename="source.csv",
            validate_response=lambda headers: validate_archive_response(headers, Archive(2025, "fall", "2024-10-31", "control")))
    assert not (selected.store.root / "records").exists()


@pytest.mark.parametrize("filename,content_type", [
    ("ArchivedSchoolAndDistrictFall2024.csv", "text/csv"),
    ("ArchivedSchoolAndDistrictSpring2025.csv", "text/csv"),
    ("ArchivedSchoolAndDistrictFall2025.csv", "text/html"),
    ("../ArchivedSchoolAndDistrictFall2025.csv", "text/csv"),
])
def test_askted_wrong_attachment_vintage_or_type_rejected(filename, content_type):
    with pytest.raises(AcquisitionError):
        validate_archive_response({"content-disposition": f'attachment;filename="{filename}"', "content-type": content_type}, Archive(2025, "fall", "2024-10-31", "control"))


def test_askted_correct_attachment_and_columns(tmp_path):
    selected = Archive(2025, "fall", "2024-10-31", "control")
    validate_archive_response({"content-disposition": 'attachment;filename=ArchivedSchoolAndDistrictFall2025.csv', "content-type": "text/csv; charset=utf-8"}, selected)
    path = tmp_path / "archive.csv"
    columns = "District Number,District Name,District Type,School Number,School Name,School Status,Charter Type,District Enrollment as of Oct 2024,School Enrollment as of Oct 2024"
    path.write_text(columns + "\n")
    validate_archive_columns(path, selected)
    path.write_text(columns.replace("Oct 2024", "Oct 2023") + "\n")
    with pytest.raises(AcquisitionError, match="selected school year"):
        validate_archive_columns(path, selected)
