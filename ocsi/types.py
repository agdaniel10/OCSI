"""Core value types shared across OCSI.

Kept dependency-light (numpy only) so the Object Memory Bank and association
math can be unit-tested without importing torch / ultralytics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TrackState(Enum):
    """Lifecycle state ``s`` of an identity record (paper §3.3)."""

    TENTATIVE = "tentative"   # newly created, not yet confirmed
    CONFIRMED = "confirmed"   # reliably tracked
    LOST = "lost"             # unmatched confirmed track, retained for reactivation
    ARCHIVED = "archived"     # expired but kept for (future) cross-camera retrieval
    DELETED = "deleted"       # scheduled for removal


# --- bounding-box helpers (boxes are stored as tlwh: top-left x, y, width, height) ---

def tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=float)


def tlwh_to_cxcywh(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=float)


def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Center x, center y, aspect ratio (w/h), height — the Kalman state form."""
    x, y, w, h = tlwh
    h = max(float(h), 1e-6)
    return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=float)


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    """Inverse of :func:`tlwh_to_xyah`: (cx, cy, a, h) -> (x, y, w, h)."""
    cx, cy, a, h = xyah
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=float)


def center(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0], dtype=float)


@dataclass
class Detection:
    """A single frame-level person detection."""

    tlwh: np.ndarray                       # (4,) top-left x, y, width, height
    confidence: float
    class_id: int = 0                      # COCO person = 0
    frame_idx: int = -1
    embedding: Optional[np.ndarray] = None  # L2-normalized appearance feature (d,)
    keypoints: Optional[np.ndarray] = None  # (K, 3) optional pose (x, y, score)
    # --- behaviour cue (paper §3.5), attached by the Behaviour Intelligence layer
    #     over a track window; None until a window is long enough for HAR ---
    behaviour_embedding: Optional[np.ndarray] = None  # (d_b,) activity-window embedding b'
    activity_probs: Optional[np.ndarray] = None       # (num_classes,) activity class probs

    def __post_init__(self) -> None:
        self.tlwh = np.asarray(self.tlwh, dtype=float)
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=float)
        if self.keypoints is not None:
            self.keypoints = np.asarray(self.keypoints, dtype=float)
        if self.behaviour_embedding is not None:
            self.behaviour_embedding = np.asarray(self.behaviour_embedding, dtype=float)
        if self.activity_probs is not None:
            self.activity_probs = np.asarray(self.activity_probs, dtype=float)

    @property
    def xyxy(self) -> np.ndarray:
        return tlwh_to_xyxy(self.tlwh)

    @property
    def xyah(self) -> np.ndarray:
        return tlwh_to_xyah(self.tlwh)

    @property
    def center(self) -> np.ndarray:
        return center(self.tlwh)

    @property
    def area(self) -> float:
        return float(self.tlwh[2] * self.tlwh[3])
