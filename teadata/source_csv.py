"""Bounded original physical lines shared by source CSV structural readers."""

from __future__ import annotations

import codecs
from hashlib import sha256
import re
from typing import BinaryIO


class SourceLines:
    """Capture/hash original bytes while the C CSV decoder finds logical records.

    Only explicit ASCII-newline-compatible codecs are admitted. UTF-16/32 need
    different byte framing and are rejected rather than split incorrectly.
    The default retains the established first-line UTF-8 BOM behavior.
    """

    def __init__(self, stream: BinaryIO, limit: int, error: str, *, encoding: str = "utf-8"):
        encoding = codecs.lookup(encoding).name
        if encoding not in {"utf-8", "utf-8-sig", "ascii", "iso8859-1", "cp1252"}:
            raise ValueError("Source CSV requires a supported ASCII-newline-compatible encoding")
        self.stream, self.limit = stream, limit
        self.encoding = "utf-8" if encoding == "utf-8-sig" else encoding
        self.error = error
        self.pending = bytearray()
        self.record = bytearray()
        self.digest = sha256()
        self.eof = False
        self.first = True

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            match = re.search(rb"\r\n|\r|\n", self.pending)
            # A CR at the end of a block may be half of a CRLF.
            if match and (match.end() < len(self.pending) or self.eof or self.pending[-1] != 13):
                end = match.end()
                break
            if self.eof:
                if not self.pending:
                    raise StopIteration
                end = len(self.pending)
                break
            if len(self.record) + len(self.pending) > self.limit:
                raise ValueError(self.error)
            block = self.stream.read(min(65536, self.limit + 1))
            self.digest.update(block)
            self.pending.extend(block)
            self.eof = not block
        if len(self.record) + end > self.limit:
            raise ValueError(self.error)
        segment = bytes(self.pending[:end])
        del self.pending[:end]
        self.record.extend(segment)
        encoding = "utf-8-sig" if self.first and self.encoding == "utf-8" else self.encoding
        text = segment.decode(encoding, errors="strict")
        self.first = False
        return text
