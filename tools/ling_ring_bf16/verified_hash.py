"""Deterministic, bounded whole-file SHA-256 helpers for verified runners."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

MAX_HASH_WORKERS = 4


def require_hash_workers(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_HASH_WORKERS:
        raise ValueError(f"hash_workers must be an integer in 1..{MAX_HASH_WORKERS}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path], workers: int = 1) -> list[str]:
    """Hash each complete file in caller order; errors propagate unchanged."""
    values = [Path(path) for path in paths]
    require_hash_workers(workers)
    if workers == 1:
        return [sha256_file(path) for path in values]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="strata-sha256") as pool:
        return list(pool.map(sha256_file, values))
