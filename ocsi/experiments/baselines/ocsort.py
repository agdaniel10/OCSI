"""OC-SORT baseline: observation-centric SORT with re-centering.

This is a minimal OC-SORT-style tracker that uses the same Kalman filter as
OCSI but without the Object Memory Bank, behaviour feedback, or contamination
rollback. It implements OC-SORT's key ideas:
  * Observation-centric re-centering: when a track is re-associated after a gap,
    its Kalman state is re-initialised from the observation.
  * Velocity-weighted IoU for association.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...config import OCSIConfig
from ...motion import KalmanFilter
from ...types import Detection, TrackState, tlwh_to_xyah
from ...identity.similarity import iou_matrix


class _Track:
    """Minimal track state for OC-SORT baseline."""

    def __init__(self, track_id: int, det: Detection, kf: KalmanFilter):
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(tlwh_to_xyah(det.tlwh))
        self.last_box = np.asarray(det.tlwh, dtype=float).copy()
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.velocity = np.zeros(2, dtype=float)

    def predicted_center(self) -> np.ndarray:
        return np.array([self.mean[0], self.mean[1]], dtype=float)


class OCSORTBaseline:
    """OC-SORT-style tracker without OCSI's memory/behaviour extensions."""

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

    def _velocity_weighted_iou(self, track: _Track, det: Detection) -> float:
        """IoU weighted by velocity consistency (OC-SORT's observation-centric idea)."""
        # Convert tlwh to xyxy for iou_matrix
        track_box = np.asarray(track.last_box, dtype=float).copy()
        track_box[2] += track_box[0]
        track_box[3] += track_box[1]
        det_box = np.asarray(det.tlwh, dtype=float).copy()
        det_box[2] += det_box[0]
        det_box[3] += det_box[1]
        iou = float(iou_matrix(
            np.asarray([track_box]), np.asarray([det_box])
        )[0, 0])
        # Velocity consistency: if the track has a velocity, penalise large deviations
        if np.linalg.norm(track.velocity) > 0:
            det_center = det.center
            pred_center = track.predicted_center()
            dist = float(np.linalg.norm(det_center - pred_center))
            velocity_penalty = np.exp(-dist / 100.0)
            return iou * velocity_penalty
        return iou

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
                for i, t in enumerate(active):
                    for j, d in enumerate(detections):
                        sim = self._velocity_weighted_iou(t, d)
                        if sim >= self.iou_gate:
                            cost[i, j] = 1.0 - sim

                rows, cols = linear_sum_assignment(cost)
                matched_d = set()
                for r, c in zip(rows, cols):
                    if cost[r, c] < 1e5:
                        t = active[r]
                        d = detections[c]
                        # OC-SORT: re-centering — re-init Kalman from observation
                        t.mean, t.covariance = self.kf.initiate(d.xyah)
                        t.last_box = np.asarray(d.tlwh, dtype=float).copy()
                        # Update velocity
                        t.velocity = d.center - t.predicted_center()
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