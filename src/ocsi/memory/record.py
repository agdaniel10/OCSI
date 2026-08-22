"""The per-identity memory record ``M`` of the Object Memory Bank (paper §3.3).

``M = { a_bar, G, Q_tau, Q_v, Q_p, Q_c, b, q, s }`` plus tracking bookkeeping and
a small history ring buffer used for contamination rollback (paper §3.4).

Depends only on numpy so it is unit-testable without any model/video.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..config import MemoryConfig
from ..types import Detection, TrackState, center


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class MemoryRecord:
    """A persistent, bounded, confidence-aware record for one identity."""

    def __init__(self, track_id: int, detection: Detection, cfg: MemoryConfig, frame_idx: int):
        self.cfg = cfg
        self.track_id = track_id
        self.state = TrackState.TENTATIVE

        # --- multimodal memory M ---
        self.a_bar: Optional[np.ndarray] = None          # appearance prototype (d,)
        self.gallery: Deque[np.ndarray] = deque(maxlen=cfg.gallery_size)  # G
        self.Q_tau: Deque[np.ndarray] = deque(maxlen=cfg.queue_size)      # trajectory (tlwh)
        self.Q_v: Deque[np.ndarray] = deque(maxlen=cfg.queue_size)        # velocity (dx, dy)
        self.Q_p: Deque[np.ndarray] = deque(maxlen=cfg.queue_size)        # pose
        self.Q_c: Deque[np.ndarray] = deque(maxlen=cfg.queue_size)        # context / interaction
        self.b: Optional[np.ndarray] = None              # behaviour prototype (embedding)
        self.b_probs: Optional[np.ndarray] = None        # last activity class probabilities
        self.q: float = float(cfg.confidence_init)       # memory confidence

        # --- bookkeeping ---
        self.hits: int = 0
        self.age: int = 0                    # frames since creation
        self.time_since_update: int = 0      # frames since last successful match
        self.frames_since_reliable: int = 0  # frames since last reliable behaviour obs
        self.last_box: np.ndarray = np.asarray(detection.tlwh, dtype=float)
        self.class_id: int = int(detection.class_id)
        self.start_frame: int = frame_idx
        self.last_frame: int = frame_idx

        # Kalman state (initiated/advanced by the tracker; None until then)
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        # rollback snapshots of (a_bar, b, q)
        self._history: Deque[Tuple[Optional[np.ndarray], Optional[np.ndarray], float]] = deque(
            maxlen=cfg.rollback_history
        )

        self._init_from_detection(detection)

    # ------------------------------------------------------------------ init
    def _init_from_detection(self, det: Detection) -> None:
        self.hits = 1
        self.Q_tau.append(np.asarray(det.tlwh, dtype=float))
        self.Q_v.append(np.zeros(2, dtype=float))
        if det.embedding is not None:
            emb = _l2_normalize(det.embedding)
            self.a_bar = emb.copy()
            self.gallery.append(emb.copy())
        if det.keypoints is not None:
            self.Q_p.append(np.asarray(det.keypoints, dtype=float))

    # ------------------------------------------------------- rollback support
    def snapshot(self) -> None:
        """Record pre-update memory state so it can be rolled back if a later
        conflict test flags this match as a likely contamination (paper §3.4)."""
        self._history.append(
            (
                None if self.a_bar is None else self.a_bar.copy(),
                None if self.b is None else self.b.copy(),
                self.q,
            )
        )

    def rollback(self) -> bool:
        """Revert appearance prototype, behaviour prototype and confidence to the
        most recent snapshot. Returns False if no snapshot is available."""
        if not self._history:
            return False
        a_bar, b, q = self._history.pop()
        self.a_bar = a_bar
        self.b = b
        self.q = q
        return True

    # --------------------------------------------------------------- updates
    def _effective_alpha(self, reliability: float) -> float:
        """EMA weight on the prototype. Higher reliability -> lower alpha (more
        weight on the new observation); the config value is the minimum alpha."""
        a0 = self.cfg.appearance_ema_alpha
        r = float(np.clip(reliability, 0.0, 1.0))
        return a0 + (1.0 - a0) * (1.0 - r)

    def update_appearance(self, embedding: np.ndarray, reliability: float) -> None:
        emb = _l2_normalize(embedding)
        if self.a_bar is None:
            self.a_bar = emb.copy()
        else:
            alpha = self._effective_alpha(reliability)
            self.a_bar = _l2_normalize(alpha * self.a_bar + (1.0 - alpha) * emb)
        if reliability >= self.cfg.gallery_reliability_min:
            self.gallery.append(emb.copy())

    def update_behaviour(self, probs: np.ndarray, embedding: np.ndarray, reliability: float) -> None:
        self.b_probs = np.asarray(probs, dtype=float)
        emb = _l2_normalize(embedding)
        if self.b is None:
            self.b = emb.copy()
        else:
            alpha = self._effective_alpha(reliability)
            self.b = _l2_normalize(alpha * self.b + (1.0 - alpha) * emb)
        self.frames_since_reliable = 0

    def update_confidence(self, reliability: float) -> None:
        d = self.cfg.confidence_decay
        r = float(np.clip(reliability, 0.0, 1.0))
        self.q = float(
            np.clip(d * self.q + (1.0 - d) * r, self.cfg.confidence_min, self.cfg.confidence_max)
        )

    # ---------------------------------------------------- per-frame outcomes
    def mark_matched(self, det: Detection, reliability: float, frame_idx: int) -> None:
        """Apply a confidence-gated update from a matched detection."""
        self.snapshot()
        box = np.asarray(det.tlwh, dtype=float)
        velocity = center(box) - center(self.last_box)
        self.Q_tau.append(box)
        self.Q_v.append(velocity)
        if det.embedding is not None:
            self.update_appearance(det.embedding, reliability)
        if det.keypoints is not None:
            self.Q_p.append(np.asarray(det.keypoints, dtype=float))
        self.update_confidence(reliability)

        self.last_box = box
        self.last_frame = frame_idx
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.frames_since_reliable += 1
        if self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED  # behaviour/appearance-driven reactivation

    def mark_missed(self) -> None:
        """No detection matched this frame: slowly decay confidence, advance counters.

        Uses a dedicated per-miss multiplicative decay (``confidence_decay_miss``,
        ~0.98) rather than the on-match EMA (``confidence_decay``). A lost track
        must retain enough confidence over ``max_age`` frames to be reactivated;
        decaying at the match rate (0.9) would crater it within ~10 frames.
        """
        lam = self.cfg.confidence_decay_miss
        self.q = float(np.clip(lam * self.q, self.cfg.confidence_min, self.cfg.confidence_max))
        self.time_since_update += 1
        self.age += 1
        self.frames_since_reliable += 1

    # --------------------------------------------------------- introspection
    def predicted_center(self) -> np.ndarray:
        """Cheap constant-velocity center prediction (superseded by the Kalman
        filter in Phase 3, but lets the record stand alone for now)."""
        v = self.Q_v[-1] if self.Q_v else np.zeros(2)
        return center(self.last_box) + v

    def gallery_matrix(self) -> Optional[np.ndarray]:
        return np.stack(list(self.gallery)) if self.gallery else None

    @property
    def is_matchable(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"MemoryRecord(id={self.track_id}, state={self.state.value}, hits={self.hits}, "
            f"q={self.q:.3f}, tsu={self.time_since_update}, |G|={len(self.gallery)})"
        )
