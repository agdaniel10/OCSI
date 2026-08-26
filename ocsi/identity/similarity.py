"""Similarity cues for the unified association score (paper §4.2).

The unified score is ``S = w_a·S_app + w_m·S_motion + w_g·S_IoU + w_h·S_mem
(+ w_b·g·S_beh)``. Each cue is normalised to **[0, 1]** so the weights are
directly comparable, and :func:`weighted_score` renormalises the weights over
whichever cues are active — so the baseline (app+motion+IoU), +memory
(add S_mem) and +feedback (add gated S_beh) ablations all yield scores in [0, 1]
that a single ``s_min`` threshold can gate.

Time horizons are deliberately split so appearance and memory are complementary,
not redundant (design-review refinement):
  * ``S_app``  — cosine to the track's **short-term gallery** (best recent view).
  * ``S_mem``  — cosine to the track's **long-term EMA prototype** ``ā``.

Everything here is pure-numpy and model-free, so it is unit-testable in isolation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------- helpers
def _normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def cos_to_01(x: np.ndarray | float) -> np.ndarray:
    """Map a cosine similarity in [-1, 1] to [0, 1]."""
    return np.clip((1.0 + np.asarray(x, dtype=float)) / 2.0, 0.0, 1.0)


# ------------------------------------------------------------- cue matrices
def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes -> (T, D) in [0, 1]."""
    a = np.asarray(a_xyxy, dtype=float).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])   # (T, D, 2)
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between (T, d) and (D, d) -> (T, D) in [-1, 1]."""
    A = np.asarray(A, dtype=float).reshape(len(A), -1) if len(A) else np.zeros((0, 0))
    B = np.asarray(B, dtype=float).reshape(len(B), -1) if len(B) else np.zeros((0, 0))
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    return _normalize_rows(A) @ _normalize_rows(B).T


def gallery_similarity(
    galleries: Sequence[Optional[np.ndarray]], det_feats: np.ndarray
) -> np.ndarray:
    """Best cosine of each detection against each track's gallery -> (T, D) in [-1, 1].

    ``galleries[i]`` is a ``(G_i, d)`` array (or ``None``/empty). Empty galleries
    yield a row of ``-1`` (no appearance evidence).
    """
    det_feats = np.asarray(det_feats, dtype=float)
    D = len(det_feats)
    T = len(galleries)
    out = np.full((T, D), -1.0)
    if D == 0 or T == 0:
        return out.reshape(T, D)
    for i, G in enumerate(galleries):
        if G is None or len(G) == 0:
            continue
        out[i] = cosine_matrix(G, det_feats).max(axis=0)
    return out


def motion_similarity(d2: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance -> motion similarity ``exp(-0.5·d²)`` in (0, 1]."""
    return np.exp(-0.5 * np.asarray(d2, dtype=float))


# --------------------------------------------------------- score combination
def weighted_score(terms: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted sum over the *active* cues, with weights renormalised to sum to 1.

    Only cues present in both ``terms`` and ``weights`` (with weight > 0) count, so
    a caller can drop ``S_mem``/``S_beh`` for an ablation and still get a [0, 1]
    score. All term matrices must share a shape and lie in [0, 1].
    """
    active = {k: terms[k] for k in terms if weights.get(k, 0.0) > 0.0}
    if not active:
        raise ValueError("weighted_score: no active cues (empty terms/weights intersection)")
    wsum = sum(weights[k] for k in active)
    shape = next(iter(active.values())).shape
    out = np.zeros(shape, dtype=float)
    for k, mat in active.items():
        out = out + (weights[k] / wsum) * np.asarray(mat, dtype=float)
    return out


def gate_mask(
    iou_mat: np.ndarray,
    d2_mat: np.ndarray,
    iou_gate: float,
    maha_gate: float,
    track_classes: Optional[Sequence[int]] = None,
    det_classes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Boolean (T, D) mask of geometrically/kinematically plausible pairs.

    A pair is valid when it clears the IoU floor AND the Mahalanobis chi-square
    gate AND (if class ids are given) the classes match.
    """
    iou_mat = np.asarray(iou_mat, dtype=float)
    d2_mat = np.asarray(d2_mat, dtype=float)
    valid = (iou_mat >= iou_gate) & (d2_mat <= maha_gate)
    if track_classes is not None and det_classes is not None:
        tc = np.asarray(track_classes).reshape(-1, 1)
        dc = np.asarray(det_classes).reshape(1, -1)
        valid = valid & (tc == dc)
    return valid
