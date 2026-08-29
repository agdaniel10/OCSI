"""ByteTrack baseline: two-stage association with high/low confidence detections.

This is a minimal ByteTrack-style tracker that uses the same Kalman filter as
OCSI but without the Object Memory Bank, behaviour feedback, or contamination
rollback. It implements ByteTrack's key idea: match high-confidence detections
first, then low-confidence detections against remaining tracks.
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
    """Minimal track state for ByteTrack baseline."""

    def __init__(self, track_id: int, det: Detection, kf: KalmanFilter):
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(tlwh_to_xyah(det.tlwh))
        self.last_box = np.asarray(det.tlwh, dtype=float).copy()
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE

    def predicted_center(self) -> np.ndarray:
        return np.array([self.mean[0], self.mean[1]], dtype=float)


class ByteTrackBaseline:
    """ByteTrack-style tracker without OCSI's memory/behaviour extensions."""

    def __init__(self, cfg: OCSIConfig):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.tracks: List[_Track] = []
        self._next_id = 1
        self.frame_idx = -1
        self.max_age = cfg.memory.max_age
        self.min_hits = cfg.memory.min_hits
        self.high_conf_threshold = cfg.association.high_conf_threshold
        self.iou_gate = cfg.association.iou_gate

    def _new_track(self, det: Detection) -> _Track:
        t = _Track(self._next_id, det, self.kf)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _match(self, active, detections, det_indices, iou_threshold):
        """Match active tracks to a subset of detections by IoU."""
        if not active or not det_indices:
            return [], set(), set()

        # Convert tlwh to xyxy for iou_matrix
        track_boxes = np.stack([t.last_box for t in active]).copy()
        track_boxes[:, 2] += track_boxes[:, 0]
        track_boxes[:, 3] += track_boxes[:, 1]
        det_boxes = np.stack([detections[j].tlwh for j in det_indices]).copy()
        det_boxes[:, 2] += det_boxes[:, 0]
        det_boxes[:, 3] += det_boxes[:, 1]
        iou = iou_matrix(track_boxes, det_boxes)

        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)

        matches = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_threshold:
                matches.append((r, det_indices[c]))
                matched_t.add(r)
                matched_d.add(det_indices[c])
        return matches, matched_t, matched_d

    def update(self, detections: List[Detection]) -> List[_Track]:
        self.frame_idx += 1

        # Predict
        for t in self.tracks:
            t.mean, t.covariance = self.kf.predict(t.mean, t.covariance)
            t.time_since_update += 1

        active = [t for t in self.tracks if t.time_since_update <= self.max_age]

        if detections:
            if active:
                # Split detections by confidence
                high_idx = [j for j, d in enumerate(detections) if d.confidence >= self.high_conf_threshold]
                low_idx = [j for j, d in enumerate(detections) if d.confidence < self.high_conf_threshold]

                # Stage 1: match high-confidence detections
                matches1, matched_t1, matched_d1 = self._match(active, detections, high_idx, self.iou_gate)

                # Stage 2: match low-confidence detections against remaining tracks
                remaining_t = [i for i in range(len(active)) if i not in matched_t1]
                remaining_d = [j for j in low_idx if j not in matched_d1]
                matches2, matched_t2, matched_d2 = self._match(
                    [active[i] for i in remaining_t], detections, remaining_d, self.iou_gate * 0.5
                )

                # Apply matches
                for r, c in matches1 + matches2:
                    t = active[r]
                    d = detections[c]
                    t.mean, t.covariance = self.kf.update(t.mean, t.covariance, d.xyah)
                    t.last_box = np.asarray(d.tlwh, dtype=float).copy()
                    t.hits += 1
                    t.time_since_update = 0
                    if t.state == TrackState.TENTATIVE and t.hits >= self.min_hits:
                        t.state = TrackState.CONFIRMED

                all_matched_d = matched_d1 | matched_d2
            else:
                all_matched_d = set()

            # Unmatched detections -> new tracks
            for j, d in enumerate(detections):
                if j not in all_matched_d:
                    self._new_track(d)

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks updated this frame
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED and t.time_since_update == 0]