from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


class DiskCacheDataset(Dataset):
    """
    Simple on-disk cache keyed by dataset index.

    Designed for caching expensive *deterministic* preprocessing (e.g. resize/normalize)
    for repeated experiments. Do not use with stochastic augmentations unless you want
    them frozen after first access.
    """

    def __init__(self, base: Dataset, *, cache_dir: str | Path, cache_key: Optional[str] = None):
        self.base = base
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = cache_key or self._default_key()

        self._root = self.cache_dir / self.cache_key
        self._root.mkdir(parents=True, exist_ok=True)

    def _default_key(self) -> str:
        h = hashlib.sha1()
        h.update(self.base.__class__.__name__.encode("utf-8"))
        h.update(str(len(self.base)).encode("utf-8"))
        return h.hexdigest()[:16]

    def __len__(self):
        return len(self.base)

    def _path(self, idx: int) -> Path:
        return self._root / f"{idx:08d}.pt"

    def __getitem__(self, idx: int):
        p = self._path(idx)
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
        item = self.base[idx]
        try:
            torch.save(item, p)
        except Exception:
            # Cache is best-effort; fall back to uncached.
            pass
        return item

