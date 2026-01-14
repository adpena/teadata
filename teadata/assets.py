from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

try:
    from platformdirs import user_cache_dir as _platform_user_cache_dir
except Exception:  # pragma: no cover - optional dependency
    def _platform_user_cache_dir(*_args: object, **_kwargs: object) -> str:
        return str(Path(tempfile.gettempdir()) / "teadata")

user_cache_dir: Callable[..., str] = _platform_user_cache_dir

logger = logging.getLogger(__name__)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_LFS_POINTER_PREFIX))
    except Exception:
        return False
    return head == _LFS_POINTER_PREFIX


def _asset_cache_dir() -> Path:
    env = os.getenv("TEADATA_ASSET_CACHE_DIR")
    if env:
        return Path(env)
    if user_cache_dir is not None:
        return Path(user_cache_dir("teadata", "adpena"))
    return Path(tempfile.gettempdir()) / "teadata"


def ensure_local_asset(path: Path, *, url_env: str, label: str) -> Path:
    if path.exists() and not is_lfs_pointer(path):
        return path

    url = os.getenv(url_env, "").strip()
    if not url:
        if path.exists() and is_lfs_pointer(path):
            logger.warning(
                "%s asset is a git-lfs pointer; set %s to a download URL",
                label,
                url_env,
            )
        return path

    cache_dir = _asset_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / Path(url).name
    if target.exists() and not is_lfs_pointer(target):
        return target

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "teadata"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp_path.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(resp, handle)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    if is_lfs_pointer(target):
        raise RuntimeError(f"Downloaded {label} asset is a git-lfs pointer: {target}")

    logger.info("Downloaded %s asset to %s", label, target)
    return target
