import csv
from hashlib import sha256
from io import BytesIO

import pytest

from teadata.source_csv import SourceLines


def test_default_utf8_bom_logical_records_and_raw_digest_are_preserved():
    raw = b'\xef\xbb\xbf"id","text"\r\n001,"a\r\nb"\r\n002,\xef\xbb\xbfvalue\n'
    lines = SourceLines(BytesIO(raw), 1024, "record limit")
    reader = csv.reader(lines, strict=True)
    rows, captured = [], []
    while True:
        lines.record.clear()
        try:
            rows.append(next(reader))
        except StopIteration:
            break
        captured.append(bytes(lines.record))
    assert rows == [["id", "text"], ["001", "a\r\nb"], ["002", "\ufeffvalue"]]
    assert b"".join(captured) == raw
    assert lines.digest.hexdigest() == sha256(raw).hexdigest()
    assert reader.line_num == 4


def test_crlf_split_at_existing_block_boundary_is_one_physical_line():
    raw = b"x" * 65535 + b"\r\nend\n"
    lines = SourceLines(BytesIO(raw), 100000, "record limit")
    assert next(lines) == "x" * 65535 + "\r\n"
    lines.record.clear()
    assert next(lines) == "end\n"
    with pytest.raises(StopIteration):
        next(lines)
    assert lines.digest.hexdigest() == sha256(raw).hexdigest()


def test_bound_applies_to_whole_multiline_logical_record():
    lines = SourceLines(BytesIO(b'"1234\n5678"\n'), 10, "record limit")
    with pytest.raises(ValueError, match="record limit"):
        next(csv.reader(lines, strict=True))


def test_explicit_single_byte_codec_and_strict_no_fallback():
    raw = "résumé\r\n".encode("cp1252")
    assert next(SourceLines(BytesIO(raw), 100, "limit", encoding="cp1252")) == "résumé\r\n"
    with pytest.raises(UnicodeDecodeError):
        next(SourceLines(BytesIO(raw), 100, "limit"))


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32", "shift_jis"])
def test_incompatible_newline_codecs_fail_before_reading(encoding):
    class NoRead(BytesIO):
        def read(self, *args):
            pytest.fail("Unsupported codec read source bytes")

    with pytest.raises(ValueError, match="ASCII-newline-compatible"):
        SourceLines(NoRead(b""), 100, "limit", encoding=encoding)
