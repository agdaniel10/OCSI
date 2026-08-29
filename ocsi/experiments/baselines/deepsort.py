"""DeepSORT baseline: Kalman filter + appearance + Hungarian association.

This is a minimal DeepSORT-style tracker that uses the same Kalman filter and
appearance embeddings as OCSI, but without the Object Memory Bank, behaviour
feedback, or contamination rollback. It serves as a controlled baseline.

The implementation follows the classic DeepSORT formulation:
  * Kalman filter for motion prediction (constant-velocity).
  * Appearance cosine similarity for association.
  * Hungarian algorithm for assignment.
  * No long-term memory, no behaviour feedback.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...config import OCSIConfig
from ...motion import KalmanFilter
from ...types import Detection, TrackState, tlwh_to_xyah
from ...identity.similarity import cosine_matrix, iou_matrix


class _Track:
    """Minimal track state for DeepSORT baseline."""

    def __init__(self, track_id: int, det: Detection, kf: KalmanFilter):
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(tlwh_to_xyah(det.tlwh))
        self.last_box = np.asarray(det.tlwh, dtype=float).copy()
        self.embedding = det.embedding
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE

    def predicted_center(self) -> np.ndarray:
        return np.array([self.mean[0], self.mean[1]], dtype=float)


class DeepSORTBaseline:
    """DeepSORT-style tracker without OCSI's memory/behaviour extensions."""

    def __init__(self, cfg: OCSIConfig):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.tracks: List[_Track] = []
        self._next_id = 1
        self.frame_idx = -1
        self.max_age = cfg.memory.max_age
        self.min_hits = cfg.memory.min_hits
        self.iou_gate = cfg.association.iou_gate

    def _new_track(self, det: Detection) -> _Track:
        t = _Track(self._next_id, det, self.kf)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def update(self, detections: List[Detection]) -> List[_Track]:
        self.frame_idx += 1

        # Predict
        for t in self.tracks:
            t.mean, t.covariance = self.kf.predict(t.mean, t.covariance)
            t.time_since_update += 1

        active = [t for t in self.tracks if t.time_since_update <= self.max_age]

        if detections:
            if active:
                T, D = len(active), len(detections)
                cost = np.full((T, D), 1e6, dtype=float)

                # IoU + appearance cost (convert tlwh to xyxy for iou_matrix)
                track_xyxy = np.stack([t.last_box for t in active]).copy()
                track_xyxy[:, 2] += track_xyxy[:, 0]
                track_xyxy[:, 3] += track_xyxy[:, 1]
                det_xyxy = np.stack([d.tlwh for d in detections]).copy()
                det_xyxy[:, 2] += det_xyxy[:, 0]
                det_xyxy[:, 3] += det_xyxy[:, 1]
                iou = iou_matrix(track_xyxy, det_xyxy)

                for i, t in enumerate(active):
                    for j, d in enumerate(detections):
                        app_sim = 0.0
                        if t.embedding is not None and d.embedding is not None:
                            app_sim = float(cosine_matrix(
                                np.asarray([t.embedding]), np.asarray([d.embedding])
                            )[0, 0])
                        # Combined cost: 0.5 * (1 - IoU) + 0.5 * (1 - app_sim)
                        cost[i, j] = 0.5 * (1.0 - iou[i, j]) + 0.5 * (1.0 - app_sim)

                # Hungarian assignment
                rows, cols = linear_sum_assignment(cost)
                matched_d = set()
                for r, c in zip(rows, cols):
                    if cost[r, c] < 1e5:
                        t = active[r]
                        d = detections[c]
                        t.mean, t.covariance = self.kf.update(t.mean, t.covariance, d.xyah)
                        t.last_box = np.asarray(d.tlwh, dtype=float).copy()
                        t.embedding = d.embedding
                        t.hits += 1
                        t.time_since_update = 0
                        if t.state == TrackState.TENTATIVE and t.hits >= self.min_hits:
                            t.state = TrackState.CONFIRMED
                        matched_d.add(c)
            else:
                matched_d = set()

            # Unmatched detections -> new tracks
            for j, d in enumerate(detections):
                if j not in matched_d:
                    self._new_track(d)

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks updated this frame
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED and t.time_since_update == 0]