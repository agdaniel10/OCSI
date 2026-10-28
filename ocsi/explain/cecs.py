"""Cross-Stage Explanation Consistency (CECS).

Implements Eqs. (8)-(11) from X-OCSI:
- S_spatial = IoU_soft(E_det, E_act)
- S_temporal = (∑ s_t * r_t) / (∑ s_t + ε)
- S_identity = 1 - (∑ s_t * u_t) / (∑ s_t + ε)
- CECS = w_s*S_spatial + w_t*S_temporal + w_i*S_identity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..xai.utils import bbox_iou, heatmap_bbox_overlap, normalize_heatmap, resize_heatmap


def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s < eps:
        return np.ones_like(x) / len(x) if len(x) > 0 else x
    return x / s


@dataclass
class CECSResult:
    S_spatial: float = 0.0
    S_temporal: float = 0.0
    S_identity: float = 0.0
    CECS: float = 0.0
    weights: Tuple[float, float, float] = (0.34, 0.33, 0.33)
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class CECSCalculator:
    def __init__(
        self,
        w_s: float = 0.34,
        w_t: float = 0.33,
        w_i: float = 0.33,
        eps: float = 1e-9,
    ):
        w_sum = w_s + w_t + w_i
        if w_sum <= 0:
            self.w_s, self.w_t, self.w_i = 1.0 / 3, 1.0 / 3, 1.0 / 3
        else:
            self.w_s, self.w_t, self.w_i = w_s / w_sum, w_t / w_sum, w_i / w_sum
        self.eps = eps

    def s_spatial(
        self,
        det_heatmap: Optional[np.ndarray] = None,
        det_bbox: Optional[np.ndarray] = None,
        act_heatmap: Optional[np.ndarray] = None,
        act_bbox: Optional[np.ndarray] = None,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> float:
        # Eq. 8: IoU_soft(E_det, E_act). Heatmap evidence is checked before any
        # box-only fallback, otherwise a run that carries a real attribution map
        # for one stage and only a box for the other would silently degrade to
        # box IoU (which is trivially 1.0 when both stages refer to the same
        # detection box).
        if det_heatmap is not None and act_heatmap is not None:
            dh = normalize_heatmap(det_heatmap)
            ah = normalize_heatmap(act_heatmap)
            if dh.shape != ah.shape:
                ah = resize_heatmap(ah, dh.shape)
            inter = float(np.minimum(dh, ah).sum())
            union = float(np.maximum(dh, ah).sum())
            if union > self.eps:
                return inter / union
            return 0.0
        # One attribution map only: measure how much of its energy falls inside
        # the other stage's object box (spatial focus, paper R3 "energy-in-box").
        # A full-frame map supplies the reference shape when img_shape is absent.
        if det_heatmap is not None and act_bbox is not None:
            shape = self._reference_shape(det_heatmap, img_shape)
            if shape is not None:
                return heatmap_bbox_overlap(det_heatmap, act_bbox, shape)
        if act_heatmap is not None and det_bbox is not None:
            shape = self._reference_shape(act_heatmap, img_shape)
            if shape is not None:
                return heatmap_bbox_overlap(act_heatmap, det_bbox, shape)
        if det_bbox is not None and act_bbox is not None:
            return bbox_iou(det_bbox, act_bbox)
        return 0.0

    @staticmethod
    def _reference_shape(heatmap: np.ndarray, img_shape: Optional[Tuple[int, int]]):
        if img_shape is not None:
            return img_shape
        hm = np.asarray(heatmap)
        return hm.shape if hm.ndim == 2 and min(hm.shape) > 1 else None

    def s_temporal(self, saliency: Sequence[float], reliability: Sequence[float]) -> float:
        s = np.asarray(saliency, dtype=float)
        r = np.asarray(reliability, dtype=float)
        if len(s) == 0 or len(r) == 0:
            return 0.0
        if len(s) != len(r):
            m = min(len(s), len(r))
            s, r = s[:m], r[:m]
        s_n = _safe_norm(s, self.eps)
        val = float(np.sum(s_n * r))
        return float(np.clip(val, 0.0, 1.0))

    def s_identity(self, saliency: Sequence[float], uncertainty: Sequence[float]) -> float:
        s = np.asarray(saliency, dtype=float)
        u = np.asarray(uncertainty, dtype=float)
        if len(s) == 0 or len(u) == 0:
            return 1.0
        if len(s) != len(u):
            m = min(len(s), len(u))
            s, u = s[:m], u[:m]
        s_n = _safe_norm(s, self.eps)
        val = float(np.sum(s_n * u))
        return float(np.clip(1.0 - val, 0.0, 1.0))

    def compute(
        self,
        det_heatmap: Optional[np.ndarray] = None,
        det_bbox: Optional[np.ndarray] = None,
        act_heatmap: Optional[np.ndarray] = None,
        act_bbox: Optional[np.ndarray] = None,
        img_shape: Optional[Tuple[int, int]] = None,
        saliency: Optional[Sequence[float]] = None,
        reliability: Optional[Sequence[float]] = None,
        uncertainty: Optional[Sequence[float]] = None,
        allow_proxies: bool = True,
    ) -> CECSResult:
        """Weighted CECS.

        ``allow_proxies=True`` (the default) preserves the legacy behaviour of
        substituting spatial evidence for missing temporal/identity terms. X-OCSI
        callers must pass ``allow_proxies=False``: a partial chain then raises
        instead of producing a CECS that looks complete.
        """
        if not allow_proxies:
            missing = [
                name
                for name, value in (
                    ("saliency", saliency),
                    ("reliability", reliability),
                    ("uncertainty", uncertainty),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"CECS requires a complete evidence chain; missing {missing}")
        S_spatial = self.s_spatial(det_heatmap, det_bbox, act_heatmap, act_bbox, img_shape)
        if saliency is not None and reliability is not None:
            S_temporal = self.s_temporal(saliency, reliability)
        else:
            S_temporal = S_spatial  # legacy proxy (allow_proxies=True only)
        if saliency is not None and uncertainty is not None:
            S_identity = self.s_identity(saliency, uncertainty)
        else:
            S_identity = 1.0 - (1.0 - S_spatial) * 0.5  # legacy proxy
        CECS = self.w_s * S_spatial + self.w_t * S_temporal + self.w_i * S_identity
        CECS = float(np.clip(CECS, 0.0, 1.0))
        return CECSResult(
            S_spatial=float(S_spatial),
            S_temporal=float(S_temporal),
            S_identity=float(S_identity),
            CECS=CECS,
            weights=(self.w_s, self.w_t, self.w_i),
            details={},
        )


def compute_cecs(*args, **kwargs) -> CECSResult:
    calc = CECSCalculator(
        w_s=kwargs.pop("w_s", 0.34),
        w_t=kwargs.pop("w_t", 0.33),
        w_i=kwargs.pop("w_i", 0.33),
        eps=kwargs.pop("eps", 1e-9),
    )
    return calc.compute(*args, **kwargs)
