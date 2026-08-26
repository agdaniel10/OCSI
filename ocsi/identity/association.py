"""Confidence-gated data association (paper §4.3).

Turns the unified [0, 1] score matrix into det<->track matches:
  * cost = ``1 - score``; invalid pairs (failing :func:`~ocsi.identity.similarity.gate_mask`)
    are made unaffordable so the optimiser avoids them;
  * a global optimum is found with the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``);
  * matches scoring below ``s_min`` are rejected after the fact.

:func:`two_stage_associate` implements ByteTrack-style matching: high-confidence
detections are matched first, then low-confidence detections are matched against
whatever tracks remain — recovering occluded/low-score objects without letting
them steal high-quality matches.

Indices are positional (row = track slot, col = detection slot); the tracker maps
them back to identities. Pure-numpy/scipy and model-free.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Match = Tuple[int, int]                 # (track_index, detection_index)
_UNAFFORDABLE = 1e5                     # cost assigned to gated-out pairs


def _solve_subset(
    score: np.ndarray,
    valid: np.ndarray,
    rows: Sequence[int],
    cols: Sequence[int],
    s_min: float,
) -> Tuple[List[Match], List[int], List[int]]:
    """Hungarian match over a submatrix; returns (matches, unmatched_rows,
    unmatched_cols) with indices mapped back to the original matrix."""
    rows = list(rows)
    cols = list(cols)
    if not rows or not cols:
        return [], rows, cols

    sub_score = score[np.ix_(rows, cols)]
    sub_valid = valid[np.ix_(rows, cols)]
    cost = np.where(sub_valid, 1.0 - sub_score, _UNAFFORDABLE)

    r_idx, c_idx = linear_sum_assignment(cost)

    matches: List[Match] = []
    matched_r, matched_c = set(), set()
    for r, c in zip(r_idx, c_idx):
        if sub_valid[r, c] and sub_score[r, c] >= s_min:
            matches.append((rows[r], cols[c]))
            matched_r.add(r)
            matched_c.add(c)
    unmatched_rows = [rows[i] for i in range(len(rows)) if i not in matched_r]
    unmatched_cols = [cols[j] for j in range(len(cols)) if j not in matched_c]
    return matches, unmatched_rows, unmatched_cols


def solve_assignment(
    score: np.ndarray, valid: np.ndarray, s_min: float
) -> Tuple[List[Match], List[int], List[int]]:
    """Single-stage global assignment over all tracks and detections."""
    T, D = score.shape
    return _solve_subset(score, valid, range(T), range(D), s_min)


def two_stage_associate(
    score: np.ndarray,
    valid: np.ndarray,
    det_confidence: Sequence[float],
    s_min: float,
    high_conf_threshold: float,
) -> Tuple[List[Match], List[int], List[int]]:
    """ByteTrack-style two-stage matching.

    Stage 1 matches all tracks against high-confidence detections; stage 2 matches
    the still-unmatched tracks against low-confidence detections.
    """
    T, D = score.shape
    det_confidence = np.asarray(det_confidence, dtype=float)
    high_cols = [j for j in range(D) if det_confidence[j] >= high_conf_threshold]
    low_cols = [j for j in range(D) if det_confidence[j] < high_conf_threshold]

    m1, um_tracks, um_high = _solve_subset(score, valid, range(T), high_cols, s_min)
    m2, um_tracks, um_low = _solve_subset(score, valid, um_tracks, low_cols, s_min)

    matches = m1 + m2
    unmatched_dets = sorted(um_high + um_low)
    return matches, um_tracks, unmatched_dets
