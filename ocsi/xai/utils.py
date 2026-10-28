"""XAI utility functions."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_heatmap(hm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    hm = np.asarray(hm, dtype=float)
    if hm.size == 0:
        return hm
    mn = hm.min()
    mx = hm.max()
    if mx - mn < eps:
        return np.zeros_like(hm)
    return (hm - mn) / (mx - mn)


def resize_heatmap(hm: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for small heatmaps (no cv2 dependency)."""
    hm = np.asarray(hm, dtype=float)
    if hm.ndim != 2 or hm.size == 0:
        return np.zeros((size[0], size[1]), dtype=float)
    h, w = hm.shape
    th, tw = size
    if h == th and w == tw:
        return hm.copy()
    row_idx = (np.arange(th) * h / th).astype(int)
    col_idx = (np.arange(tw) * w / tw).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return hm[row_idx[:, None], col_idx[None, :]]


def bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    b1 = np.asarray(b1, dtype=float).reshape(4)
    b2 = np.asarray(b2, dtype=float).reshape(4)
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def heatmap_bbox_overlap(heatmap: np.ndarray, bbox: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """Soft overlap between heatmap energy and bbox (normalized)."""
    hm = normalize_heatmap(heatmap)
    if hm.size == 0:
        return 0.0
    H, W = img_shape[:2]
    x, y, w, h = np.asarray(bbox, dtype=float).reshape(4)
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    # Resize heatmap to image size if different
    if hm.shape != (H, W):
        hm_r = resize_heatmap(hm, (H, W))
    else:
        hm_r = hm
    roi = hm_r[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    total = hm_r.sum()
    if total < 1e-12:
        return 0.0
    return float(roi.sum() / total)


def mask_bbox(heatmap: np.ndarray, bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Create binary mask for bbox region."""
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=float)
    x, y, w, h = np.asarray(bbox, dtype=float).reshape(4)
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask
