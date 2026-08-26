"""Feature cache (Phase 1).

Embeddings are the expensive part of running an ablation sweep — the same clip is
tracked once per ablation stage, but its per-frame appearance features never change.
This is a tiny two-tier (in-memory + on-disk ``.npy``) cache keyed by an arbitrary
string (e.g. ``"MOT17-02/000001"``) so repeated runs recompute nothing.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np


class FeatureCache:
    """String-keyed array cache backed by an in-memory dict and optional disk store.

    ``root=None`` gives a purely in-memory cache. Keys may contain ``/`` to nest
    on disk. Stored arrays are returned as-is (no copy) — treat them as read-only.
    """

    def __init__(self, root: Optional[str] = None, memory: bool = True):
        self.root = root
        self._mem: Optional[Dict[str, np.ndarray]] = {} if memory else None
        if root:
            os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key + ".npy")

    def has(self, key: str) -> bool:
        if self._mem is not None and key in self._mem:
            return True
        return bool(self.root) and os.path.exists(self._path(key))

    def get(self, key: str) -> Optional[np.ndarray]:
        if self._mem is not None and key in self._mem:
            return self._mem[key]
        if self.root and os.path.exists(self._path(key)):
            arr = np.load(self._path(key))
            if self._mem is not None:
                self._mem[key] = arr
            return arr
        return None

    def put(self, key: str, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if self._mem is not None:
            self._mem[key] = arr
        if self.root:
            path = self._path(key)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.save(path, arr)
        return arr

    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the cached array for ``key``, or compute, store and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, compute())
