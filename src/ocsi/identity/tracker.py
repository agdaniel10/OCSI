"""The OCSI tracker: the per-frame inference loop (paper §4).

Ties the pieces together each frame:

  1. Kalman-predict every existing track forward.
  2. Build the normalised cue matrices (IoU, motion, appearance, memory) between
     tracks (rows) and detections (cols).
  3. Gate implausible pairs (IoU floor, Mahalanobis chi-square, class).
  4. Combine cues into the unified [0,1] score and solve the assignment
     (two-stage, ByteTrack-style, by default).
  5. For matched pairs: Kalman-update + confidence-gated memory update.
     For unmatched tracks: mark missed (confirmed -> lost, tentative -> deleted).
     For unmatched detections: spawn a new tentative track.
  6. Retire expired records.

Behaviour feedback (the gated ``S_beh`` term, contradiction rejection and behaviour-
assisted reactivation) is layered on in :meth:`OCSITracker.update` /
:meth:`OCSITracker._reactivate` when ``cfg.behaviour.enabled`` — see
:mod:`ocsi.behaviour.gate` for the confidence gate that is the actual contribution.

Consumes ``List[Detection]`` (with embeddings already attached by the perception
layer), so it is testable with synthetic detections and no models.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..behaviour import behaviour_gate, blend_behaviour, contradiction_mask
from ..config import OCSIConfig
from ..memory import MemoryRecord, ObjectMemoryBank
from ..motion import KalmanFilter
from ..types import Detection, TrackState, tlwh_to_xyah, tlwh_to_xyxy, xyah_to_tlwh
from .association import solve_assignment, two_stage_associate
from .similarity import (
    cos_to_01,
    cosine_matrix,
    gallery_similarity,
    gate_mask,
    iou_matrix,
    motion_similarity,
    weighted_score,
)


class OCSITracker:
    """Memory-driven multi-object tracker (Perceptual + Identity + Memory layers)."""

    def __init__(self, cfg: OCSIConfig):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.bank = ObjectMemoryBank(cfg.memory)
        self.frame_idx = -1

    # ---------------------------------------------------------------- helpers
    def _ensure_kalman(self, rec: MemoryRecord) -> None:
        if rec.mean is None:
            rec.mean, rec.covariance = self.kf.initiate(tlwh_to_xyah(rec.last_box))

    def _predicted_xyxy(self, rec: MemoryRecord) -> np.ndarray:
        source = xyah_to_tlwh(rec.mean[:4]) if rec.mean is not None else rec.last_box
        return tlwh_to_xyxy(source)

    def _reliability(self, det: Detection, score: float) -> float:
        """Interim reliability: detection confidence blended with the multi-cue
        match score (a proxy for cross-cue consistency). The full design formula
        (adding the assignment margin) is a later refinement."""
        return float(np.clip(0.5 * det.confidence + 0.5 * score, 0.0, 1.0))

    def _behaviour_terms(self, tracks, dets):
        """Behaviour cue matrices for ``tracks x dets`` (paper §3.5), or ``None``.

        Returns ``None`` when feedback is disabled or the signals are missing (no
        detection carries an activity observation, or no track has a behaviour
        prototype yet). Otherwise returns ``(s_beh01, cos_bt, g)`` all shaped ``(T, D)``:

        * ``cos_bt``  — raw ``cos(b_track, b'_det)`` in ``[-1, 1]`` (0 where a track has
          no prototype yet, so it neither helps nor contradicts);
        * ``s_beh01`` — ``cos_bt`` mapped to ``[0, 1]`` for the score blend;
        * ``g``       — the confidence gate from per-detection ``p_max`` and per-track
          staleness ``frames_since_reliable`` (0 where a track has no prototype).
        """
        b = self.cfg.behaviour
        if not b.enabled or not tracks or not dets:
            return None
        if any(d.behaviour_embedding is None or d.activity_probs is None for d in dets):
            return None
        has_b = np.array([t.b is not None for t in tracks])
        if not has_b.any():
            return None
        dim = len(dets[0].behaviour_embedding)
        b_protos = np.stack([t.b if t.b is not None else np.zeros(dim) for t in tracks])
        det_b = np.stack([d.behaviour_embedding for d in dets])
        cos_bt = np.where(has_b[:, None], cosine_matrix(b_protos, det_b), 0.0)
        p_max = np.array([[float(d.activity_probs.max()) for d in dets]])   # (1, D)
        dt = np.array([[float(t.frames_since_reliable)] for t in tracks])   # (T, 1)
        g = behaviour_gate(p_max, dt, b.theta_b, b.tau_b, b.gate_enabled)   # (T, D)
        g = np.where(has_b[:, None], g, 0.0)
        return cos_to_01(cos_bt), cos_bt, g

    def _maybe_update_behaviour(self, track_id: int, det: Detection, reliability: float) -> None:
        """Update a track's behaviour prototype from a matched detection, but only when
        feedback is on and the window's activity confidence clears ``theta_b`` (paper
        step 13). Gating the *update* keeps uncertain HAR out of the prototype, so the
        evidence the gate later trusts was itself only ever built from reliable windows.
        Resets ``frames_since_reliable`` (the gate's staleness ``dt``)."""
        b = self.cfg.behaviour
        if not b.enabled or det.behaviour_embedding is None or det.activity_probs is None:
            return
        if float(det.activity_probs.max()) < b.theta_b:
            return
        self.bank.update_behaviour(track_id, det.activity_probs, det.behaviour_embedding, reliability)

    def _reactivate(self, tracks, detections, um_tracks, um_dets):
        """Appearance-only re-association of LOST tracks to leftover detections
        (occlusion recovery, paper §3.4).

        The main association gate requires an IoU overlap that a box which moved
        during a multi-frame occlusion no longer has, so a reappearing identity
        would spawn a fresh id. Here the still-unmatched LOST tracks are matched to
        the still-unmatched detections on **appearance alone** (best cosine against
        the track's gallery), gated by ``reactivation_app_gate`` and class. Returns
        ``(reactivations, um_tracks, um_dets)`` where each reactivation is
        ``(track_index, det_index, score01)``.
        """
        a = self.cfg.association
        lost_local = [
            i for i in um_tracks
            if tracks[i].state == TrackState.LOST and tracks[i].gallery_matrix() is not None
        ]
        if not lost_local or not um_dets:
            return [], um_tracks, um_dets

        det_local = list(um_dets)
        det_feats = np.stack([detections[j].embedding for j in det_local])
        galleries = [tracks[i].gallery_matrix() for i in lost_local]
        app = gallery_similarity(galleries, det_feats)          # (L, D') raw cosine in [-1, 1]
        tc = np.asarray([tracks[i].class_id for i in lost_local]).reshape(-1, 1)
        dc = np.asarray([detections[j].class_id for j in det_local]).reshape(1, -1)
        valid = (app >= a.reactivation_app_gate) & (tc == dc)
        aff = cos_to_01(app)
        # Behaviour can rescue or reject a borderline post-occlusion re-association
        # (paper step 14) — but only through the gate: with it shut, ``aff`` stays
        # exactly the appearance affinity and the veto is empty, so a stale/uncertain
        # activity guess never fabricates or blocks a recovery.
        bt = self._behaviour_terms(
            [tracks[i] for i in lost_local], [detections[j] for j in det_local]
        )
        if bt is not None:
            s_beh01, cos_bt, g = bt
            aff = blend_behaviour(aff, s_beh01, g, a.w_behaviour)
            valid &= ~contradiction_mask(
                cos_bt, g > 0.0, self.cfg.behaviour.contradiction_threshold
            )
        if not valid.any():
            return [], um_tracks, um_dets

        pairs, _, _ = solve_assignment(aff, valid, 0.0)
        reactivations, done_t, done_d = [], set(), set()
        for li, dj in pairs:
            reactivations.append((lost_local[li], det_local[dj], float(cos_to_01(app[li, dj]))))
            done_t.add(lost_local[li])
            done_d.add(det_local[dj])
        um_tracks = [i for i in um_tracks if i not in done_t]
        um_dets = [j for j in um_dets if j not in done_d]
        return reactivations, um_tracks, um_dets

    # ----------------------------------------------------------------- update
    def update(self, detections: List[Detection]) -> List[MemoryRecord]:
        self.frame_idx += 1
        f = self.frame_idx
        a = self.cfg.association

        tracks = self.bank.matchable()          # fixed row order for this frame
        for rec in tracks:                       # 1. predict
            self._ensure_kalman(rec)
            rec.mean, rec.covariance = self.kf.predict(rec.mean, rec.covariance)

        T, D = len(tracks), len(detections)
        score = np.zeros((T, D))
        if T > 0 and D > 0:
            det_xyxy = np.stack([d.xyxy for d in detections])
            det_meas = np.stack([d.xyah for d in detections])
            det_conf = np.array([d.confidence for d in detections], dtype=float)
            det_cls = [d.class_id for d in detections]

            track_xyxy = np.stack([self._predicted_xyxy(r) for r in tracks])
            track_cls = [r.class_id for r in tracks]

            iou_mat = iou_matrix(track_xyxy, det_xyxy)
            d2_mat = np.stack(
                [self.kf.gating_distance(r.mean, r.covariance, det_meas) for r in tracks]
            )

            terms = {"iou": iou_mat, "motion": motion_similarity(d2_mat)}
            weights = {"iou": a.w_iou, "motion": a.w_motion}

            if all(d.embedding is not None for d in detections):
                det_feats = np.stack([d.embedding for d in detections])
                galleries = [r.gallery_matrix() for r in tracks]
                terms["app"] = cos_to_01(gallery_similarity(galleries, det_feats))
                weights["app"] = a.w_app
                if a.use_memory:
                    has_proto = np.array([r.a_bar is not None for r in tracks])
                    protos = np.stack(
                        [r.a_bar if r.a_bar is not None else np.zeros(det_feats.shape[1]) for r in tracks]
                    )
                    mem = cos_to_01(cosine_matrix(protos, det_feats))
                    terms["mem"] = np.where(has_proto[:, None], mem, 0.0)
                    weights["mem"] = a.w_memory

            score = weighted_score(terms, weights)
            valid = gate_mask(
                iou_mat, d2_mat, a.iou_gate, a.mahalanobis_gate, track_cls, det_cls
            )
            # Behaviour feedback (contribution #3): gated refinement + contradiction
            # rejection. Both collapse to a no-op where the gate is shut (g=0), so
            # uncertain HAR can neither move the score nor veto a pair — the paper's
            # error-amplification guard, enforced by construction not by tuning.
            bt = self._behaviour_terms(tracks, detections)
            if bt is not None:
                s_beh01, cos_bt, g = bt
                score = blend_behaviour(score, s_beh01, g, a.w_behaviour)
                valid &= ~contradiction_mask(
                    cos_bt, g > 0.0, self.cfg.behaviour.contradiction_threshold
                )
            if a.two_stage:
                matches, um_tracks, um_dets = two_stage_associate(
                    score, valid, det_conf, a.s_min, a.high_conf_threshold
                )
            else:
                matches, um_tracks, um_dets = solve_assignment(score, valid, a.s_min)
        else:
            matches, um_tracks, um_dets = [], list(range(T)), list(range(D))

        # 4b. occlusion recovery: appearance-only reactivation of LOST tracks
        #     (bypasses the IoU floor that would otherwise reject a box that moved
        #      during the occlusion; this is where the memory bank earns its keep).
        reactivations: list = []
        if a.reactivation and D > 0 and T > 0 and all(d.embedding is not None for d in detections):
            reactivations, um_tracks, um_dets = self._reactivate(tracks, detections, um_tracks, um_dets)

        # 5. apply outcomes
        for ti, di in matches:
            rec, det = tracks[ti], detections[di]
            rec.mean, rec.covariance = self.kf.update(rec.mean, rec.covariance, det.xyah)
            rel = self._reliability(det, score[ti, di])
            self.bank.update(rec.track_id, det, rel, f)
            self._maybe_update_behaviour(rec.track_id, det, rel)
        for ti, di, appsc in reactivations:
            rec, det = tracks[ti], detections[di]
            rec.mean, rec.covariance = self.kf.initiate(det.xyah)   # reset stale post-occlusion Kalman
            rel = self._reliability(det, appsc)
            self.bank.update(rec.track_id, det, rel, f)
            self._maybe_update_behaviour(rec.track_id, det, rel)
        for ti in um_tracks:
            self.bank.mark_missed(tracks[ti].track_id)
        for di in um_dets:
            rec = self.bank.create(detections[di], f)
            rec.mean, rec.covariance = self.kf.initiate(detections[di].xyah)

        self.bank.step_end(f)                    # 6. retire
        return self.outputs()

    # ---------------------------------------------------------------- outputs
    def outputs(self) -> List[MemoryRecord]:
        """Confirmed tracks that were updated this frame (MOT-reportable)."""
        return [
            r for r in self.bank.records.values()
            if r.state == TrackState.CONFIRMED and r.time_since_update == 0
        ]

    def reset(self) -> None:
        self.bank = ObjectMemoryBank(self.cfg.memory)
        self.frame_idx = -1
