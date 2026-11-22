"""Checksum scans for release artifacts and verified database readers."""

from hashlib import file_digest, sha256
from io import BufferedReader
import os


def sha256_stream(stream: BufferedReader, *, discard_cached_reads: bool = False) -> str:
    """Hash a binary file from its beginning with optional discard-behind advice."""
    advise = getattr(os, "posix_fadvise", None) if discard_cached_reads else None
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return file_digest(stream, "sha256").hexdigest()

    digest = sha256()
    discard_window = 16 * 1024 * 1024
    offset = position = 0
    while True:
        block = stream.read(1024 * 1024)
        if block:
            digest.update(block)
        position += len(block)
        if advise is not None and (position - offset >= discard_window or (not block and position > offset)):
            try:
                advise(stream.fileno(), offset, position - offset, dontneed)
            except OSError:
                # Cache advice is best effort. Hash every byte regardless.
                advise = None
            offset = position
        if not block:
            return digest.hexdigest()
