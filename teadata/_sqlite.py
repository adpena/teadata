"""Read-only SQLite readers with an explicit checkpointed-artifact policy."""

from pathlib import Path
import sqlite3


def readonly_connection(path: str | Path, *, immutable: bool = False) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True)
