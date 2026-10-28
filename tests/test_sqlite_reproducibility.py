"""Release database bytes must survive fresh interpreter processes."""

import hashlib
from pathlib import Path
import subprocess
import sys


def test_sqlite_schema_is_reproducible_across_processes(tmp_path):
    script = """
import sys
from teadata.persistence.sqlalchemy_store import create_engine, ensure_schema
engine = create_engine('sqlite:///' + sys.argv[1])
ensure_schema(engine)
ensure_schema(engine)
engine.dispose()
"""
    paths = [tmp_path / f"schema-{index}.sqlite" for index in range(3)]
    for path in paths:
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    assert len({hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}) == 1
