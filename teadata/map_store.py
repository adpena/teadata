import gzip
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .manifest import ContractError, read_sqlite_catalog
from .release import current_release
from .teadata_config import canonical_district_number
from .identifiers import normalize_campus_lookup_key

_GZIP_MAGIC = b"\x1f\x8b"

logger = logging.getLogger(__name__)


def map_store_path_for_snapshot(snapshot_path: Path) -> Path:
    base = snapshot_path.name.removeprefix("repo_").removesuffix(".gz").removesuffix(".pkl")
    return snapshot_path.with_name(f"map_payloads_{base}.sqlite")


def discover_map_store(explicit: str | Path | None = None) -> Path:
    path = Path(explicit) if explicit is not None else current_release().artifact("maps")
    read_sqlite_catalog(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(campus_profile)")}
    if "lookup_key" not in columns:
        raise ContractError("Legacy map campus identity schema requires a source-bound rebuild")
    return path


def _district_lookup_keys(district_number: Optional[str]) -> Iterable[str]:
    if not district_number:
        return ()
    keys = []
    canonical = canonical_district_number(district_number)
    if canonical:
        keys.append(canonical)
    raw = str(district_number).strip()
    if raw and raw not in keys:
        keys.append(raw)
    if raw.startswith("'") and len(raw) > 1:
        digits = raw[1:]
        if digits not in keys:
            keys.append(digits)
        if digits.isdigit():
            normalized = str(int(digits))
            if normalized not in keys:
                keys.append(normalized)
    return keys




def _decode_payload(payload: Any) -> Optional[dict]:
    if payload is None:
        return None
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytes):
        data = payload
        if data.startswith(_GZIP_MAGIC):
            try:
                data = gzip.decompress(data)
            except Exception:
                return None
        try:
            text = data.decode("utf-8")
        except Exception:
            return None
        return json.loads(text) if text else None
    if isinstance(payload, str):
        return json.loads(payload) if payload else None
    return None


def load_map_payload_parts(
    district_number: str, *, store_path: str | Path | None = None
) -> Tuple[Optional[dict], Optional[dict]]:
    path = discover_map_store(store_path)
    if not path:
        return None, None

    try:
        with sqlite3.connect(path) as conn:
            for key in _district_lookup_keys(district_number):
                row = conn.execute(
                    "SELECT payload_base, payload_transfers FROM closures_map WHERE district_number = ?",
                    (key,),
                ).fetchone()
                if row:
                    base = _decode_payload(row[0])
                    transfers = _decode_payload(row[1]) or {}
                    return base, transfers
    except Exception:
        logger.exception(
            "map_store.load_map_payload_parts_failed path=%s district_number=%s",
            path,
            district_number,
        )
        return None, None

    return None, None


def load_campus_profile_payload(
    campus_number: str, *, store_path: str | Path | None = None
) -> Optional[dict]:
    key = normalize_campus_lookup_key(campus_number)
    if key is None:
        return None
    path = discover_map_store(store_path)
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute("SELECT payload FROM campus_profile WHERE lookup_key = ?", (key,)).fetchone()
            return _decode_payload(row[0]) if row else None
    except sqlite3.OperationalError as error:
        raise ContractError("Legacy map campus identity schema requires a source-bound rebuild") from error
