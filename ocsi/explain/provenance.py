"""Identity-persistent explanation provenance for X-OCSI (paper §3.1, §3.6).

The tracker calls :class:`ExplanationRecorder` after every accepted match. The
recorder appends three kinds of :class:`~ocsi.types.ExplanationEvent` to the same
identity's :class:`~ocsi.memory.record.MemoryRecord`:

* ``association`` — the auditable cue decomposition ``q_ij`` of Eq. (4),
* ``detection``   — spatial attribution of detector evidence (Eq. 2-3),
* ``temporal``    — HAR temporal saliency plus the window's tracking reliability
                    and identity uncertainty (Eq. 7).

CECS (Eq. 8-11) is computed only when the complete evidence chain is available
for that identity. A partially instrumented pipeline records what it has and
returns ``None`` for CECS instead of substituting a proxy term, so a reported
CECS always means that all three stages actually contributed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from ..config import XAIConfig
from ..types import ExplanationEvent, ExplanationInput
from .cecs import CECSCalculator, CECSResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.record import MemoryRecord


class ExplanationRecorder:
    """Build the per-identity explanation chain and compute CECS on demand."""

    def __init__(self, cfg: XAIConfig):
        self.cfg = cfg
        self.calculator = CECSCalculator(
            w_s=cfg.w_s,
            w_t=cfg.w_t,
            w_i=cfg.w_i,
            eps=cfg.temporal_epsilon,
        )
        # One row per CECS computation, in frame order.
        self.cecs_history: List[Dict[str, object]] = []
        self.event_counts: Dict[str, int] = {"association": 0, "detection": 0, "temporal": 0}

    # ------------------------------------------------------------- events
    def record_association(
        self,
        record: "MemoryRecord",
        frame_idx: int,
        cues: Dict[str, float],
        weights: Dict[str, float],
        score: float,
        reliability: float,
        margin: float = 0.0,
        extra: Optional[Dict[str, float]] = None,
    ) -> ExplanationEvent:
        """Store why this detection was assigned to this identity.

        ``cues`` are the normalised per-cue similarities (``q_ij``) and
        ``weights`` their effective weights, renormalised to sum to 1 across the
        final convex combination (including the gated behaviour cue when active),
        so the event can be re-scored after the fact (paper Eq. 4).
        """
        metadata: Dict[str, object] = {
            "cues": {str(k): float(v) for k, v in cues.items()},
            "weights": {str(k): float(v) for k, v in weights.items()},
            "margin": float(margin),
            "reliability": float(reliability),
        }
        if extra:
            metadata.update({str(k): float(v) for k, v in extra.items()})
        event = ExplanationEvent(
            frame_idx=int(frame_idx),
            kind="association",
            bbox=np.asarray(record.last_box, dtype=float).copy(),
            score=float(score),
            target="association",
            metadata=metadata,
        )
        record.add_explanation(event)
        self.event_counts["association"] += 1
        return event

    def record_detection(
        self,
        record: "MemoryRecord",
        frame_idx: int,
        evidence: ExplanationInput,
        bbox: np.ndarray,
        score: float,
    ) -> Optional[ExplanationEvent]:
        """Store the detector's spatial attribution ``E_det^y`` (Eq. 2-3)."""
        if evidence.det_heatmap is None:
            return None
        event = ExplanationEvent(
            frame_idx=int(frame_idx),
            kind="detection",
            bbox=np.asarray(bbox, dtype=float).copy(),
            # float32 keeps the 64-event history ~13 MB per track at 224x224;
            # CECS only uses overlap/energy ratios, so precision is irrelevant.
            heatmap=np.asarray(evidence.det_heatmap, dtype=np.float32).copy(),
            score=float(score),
            target="detector",
            metadata={"img_shape": evidence.img_shape},
        )
        record.add_explanation(event)
        self.event_counts["detection"] += 1
        return event

    def record_temporal(
        self,
        record: "MemoryRecord",
        frame_idx: int,
        evidence: ExplanationInput,
        bbox: np.ndarray,
        score: float,
    ) -> Optional[ExplanationEvent]:
        """Store HAR temporal saliency and its aligned reliability/uncertainty."""
        if evidence.saliency is None:
            return None
        event = ExplanationEvent(
            frame_idx=int(frame_idx),
            kind="temporal",
            bbox=np.asarray(bbox, dtype=float).copy(),
            heatmap=(
                None
                if evidence.act_heatmap is None
                else np.asarray(evidence.act_heatmap, dtype=float).copy()
            ),
            score=float(score),
            target="har",
            metadata={
                "saliency": np.asarray(evidence.saliency, dtype=float).copy(),
                "reliability": (
                    None
                    if evidence.reliability is None
                    else np.asarray(evidence.reliability, dtype=float).copy()
                ),
                "uncertainty": (
                    None
                    if evidence.uncertainty is None
                    else np.asarray(evidence.uncertainty, dtype=float).copy()
                ),
                "img_shape": evidence.img_shape,
            },
        )
        record.add_explanation(event)
        self.event_counts["temporal"] += 1
        return event

    def record_evidence(
        self,
        record: "MemoryRecord",
        frame_idx: int,
        evidence: ExplanationInput,
        bbox: np.ndarray,
        score: float,
    ) -> Optional[CECSResult]:
        """Record detection + temporal evidence, then compute CECS if complete."""
        self.record_detection(record, frame_idx, evidence, bbox, score)
        self.record_temporal(record, frame_idx, evidence, bbox, score)
        return self.compute_cecs(record, frame_idx, evidence)

    # --------------------------------------------------------------- CECS
    def compute_cecs(
        self,
        record: "MemoryRecord",
        frame_idx: int,
        evidence: Optional[ExplanationInput] = None,
    ) -> Optional[CECSResult]:
        """CECS for one identity at one frame, or ``None`` if evidence is partial.

        Requires detector attribution, temporal saliency and the aligned
        reliability/uncertainty vectors. Missing terms are *not* substituted.
        """
        det = record.last_detection_attribution
        temporal = record.last_temporal_saliency
        if det is None or temporal is None:
            return None

        saliency = temporal.metadata.get("saliency")
        reliability = temporal.metadata.get("reliability")
        uncertainty = temporal.metadata.get("uncertainty")
        if saliency is None or reliability is None or uncertainty is None:
            return None

        img_shape = temporal.metadata.get("img_shape")
        if img_shape is None and evidence is not None:
            img_shape = evidence.img_shape

        result = self.calculator.compute(
            det_heatmap=det.heatmap,
            det_bbox=det.bbox,
            act_heatmap=temporal.heatmap,
            act_bbox=temporal.bbox,
            img_shape=img_shape,
            saliency=saliency,
            reliability=reliability,
            uncertainty=uncertainty,
            allow_proxies=False,
        )
        record.last_cecs = result
        self.cecs_history.append(
            {
                "frame": int(frame_idx),
                "track_id": int(record.track_id),
                "S_spatial": float(result.S_spatial),
                "S_temporal": float(result.S_temporal),
                "S_identity": float(result.S_identity),
                "CECS": float(result.CECS),
            }
        )
        return result

    # ------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, object]:
        """Compact payload for experiment summaries (history stays separate)."""
        values = [float(row["CECS"]) for row in self.cecs_history]
        return {
            "enabled": True,
            "event_counts": dict(self.event_counts),
            "num_cecs": len(values),
            "cecs_mean": float(np.mean(values)) if values else None,
            "cecs_min": float(np.min(values)) if values else None,
            "cecs_max": float(np.max(values)) if values else None,
        }
