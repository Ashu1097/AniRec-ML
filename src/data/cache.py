# -*- coding: utf-8 -*-
"""Disk-based JSON cache for API responses."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """
    Simple hash-addressed key-value store backed by individual JSON files.

    Keys are SHA-256 hashed; values must be JSON-serialisable.
    """

    def __init__(self, cache_dir: Path, prefix: str):
        self.root = Path(cache_dir) / prefix
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def set(self, key: str, value: Any) -> None:
        if value is None:
            raise ValueError(f"DiskCache.set({key!r}): refusing to cache None.")
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, separators=(",", ":"))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink(missing_ok=True)
