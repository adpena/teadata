"""Read-only SQLite readers with an explicit checkpointed-artifact policy."""

from collections.abc import Generator
from contextlib import closing, contextmanager
from hashlib import file_digest
import os
from pathlib import Path
import sqlite3
from stat import S_ISREG


def readonly_connection(path: str | Path, *, immutable: bool = False) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _plain_database_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    if not S_ISREG(info.st_mode) or any(os.path.lexists(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError(f"{label} requires a regular, checkpointed SQLite artifact")
    return _identity(info)


@contextmanager
def verified_database(path: Path, *, expected_sha256: str, application_id: int, schema_version: int,
                      label: str) -> Generator[sqlite3.Connection]:
    """Open one pinned, checkpointed database in a directory held immutable during use."""
    path = path.absolute()
    identity = _plain_database_identity(path, label)
    with path.open("rb") as stream:
        if (_identity(os.fstat(stream.fileno())) != identity
                or file_digest(stream, "sha256").hexdigest() != expected_sha256):
            raise ValueError(f"{label} file differs from its expected SHA-256")
        with closing(readonly_connection(path, immutable=True)) as connection:
            if (connection.execute("PRAGMA application_id").fetchone() != (application_id,)
                    or connection.execute("PRAGMA user_version").fetchone() != (schema_version,)):
                raise ValueError(f"Unsupported {label.lower()} file format")
            if (_identity(os.fstat(stream.fileno())) != identity or _plain_database_identity(path, label) != identity):
                raise ValueError(f"{label} file changed while opening")
            yield connection
