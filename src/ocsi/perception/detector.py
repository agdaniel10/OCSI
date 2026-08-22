"""Person detector adapter (Phase 1).

A thin wrapper over an Ultralytics YOLO model that yields :class:`~ocsi.types.Detection`
objects filtered to the person class. The Ultralytics import is deferred to the
constructor so the rest of OCSI runs without it; the pure post-processing step
(:func:`filter_detections`) is separated out so it can be unit-tested with no model,
no weights and no network.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import PerceptionConfig, resolve_device
from ..types import Detection


def filter_detections(
    boxes_tlwh: np.ndarray,
    scores: Sequence[float],
    classes: Sequence[int],
    person_class_id: int = 0,
    conf_threshold: float = 0.0,
    frame_idx: int = -1,
) -> List[Detection]:
    """Turn raw ``(box, score, class)`` triples into person ``Detection``s.

    ``boxes_tlwh`` is ``(N, 4)`` top-left-width-height. Keeps only rows whose class
    equals ``person_class_id`` and whose score >= ``conf_threshold``.
    """
    boxes = np.asarray(boxes_tlwh, dtype=float).reshape(-1, 4)
    dets: List[Detection] = []
    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) == person_class_id and float(score) >= conf_threshold:
            dets.append(
                Detection(
                    tlwh=np.asarray(box, dtype=float),
                    confidence=float(score),
                    class_id=int(cls),
                    frame_idx=frame_idx,
                )
            )
    return dets


class YOLODetector:
    """Ultralytics YOLO person detector -> ``List[Detection]`` per frame."""

    def __init__(self, cfg: Optional[PerceptionConfig] = None):
        self.cfg = cfg or PerceptionConfig()
        self.device = resolve_device(self.cfg.device)
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "YOLODetector needs the 'ultralytics' package: "
                "pip install -e '.[perception]'"
            ) from e
        self.model = YOLO(self.cfg.detector_weights)

    def __call__(self, frame_bgr: np.ndarray, frame_idx: int = -1) -> List[Detection]:
        """Detect people in a frame (BGR, as OpenCV loads it)."""
        result = self.model.predict(
            frame_bgr,
            classes=[self.cfg.person_class_id],
            conf=self.cfg.det_conf_threshold,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        tlwh = np.column_stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        return filter_detections(
            tlwh, scores, classes, self.cfg.person_class_id, self.cfg.det_conf_threshold, frame_idx
        )
