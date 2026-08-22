"""Sequence-level orchestration (paper §4 inference loop, applied over a clip).

Thin glue over the per-frame :class:`~ocsi.identity.tracker.OCSITracker`:

* :func:`run_sequence` — drive the tracker frame-by-frame over a clip and collect
  MOTChallenge result rows.
* :func:`run_and_evaluate` — run, then score against ground truth with the quick
  CLEAR-MOT/IDF1 metric.
* :func:`group_detections` — bridge public MOTChallenge ``det.txt`` rows into the
  per-frame ``Detection`` lists the tracker consumes (no models required, so a
  full sequence can be tracked and scored offline).

The perception adapters (Phase 1) will produce the same ``Detection`` lists with
learned embeddings/keypoints attached; nothing here changes when they land.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .config import OCSIConfig
from .eval.metrics import MOTMetrics, evaluate, rows_to_frames
from .eval.mot_io import ResultRow, tracker_rows
from .identity import OCSITracker
from .types import Detection


def group_detections(
    rows: Sequence[Tuple],
    person_class_id: int = 0,
    conf_threshold: float = 0.0,
) -> List[List[Detection]]:
    """Bridge MOTChallenge detection rows into dense per-frame ``Detection`` lists.

    ``rows`` are ``(frame, id, x, y, w, h, conf)`` (``id`` is ignored — public
    detections carry no identity). MOT frames are 1-indexed; the returned list is
    0-indexed and dense (frames with no detections become empty lists) so it feeds
    :func:`run_sequence` directly.
    """
    if not rows:
        return []
    max_frame = max(int(r[0]) for r in rows)
    per_frame: List[List[Detection]] = [[] for _ in range(max_frame)]
    for r in rows:
        frame = int(r[0])
        conf = float(r[6]) if len(r) > 6 else 1.0
        if conf < conf_threshold:
            continue
        tlwh = np.array([float(r[2]), float(r[3]), float(r[4]), float(r[5])])
        per_frame[frame - 1].append(
            Detection(tlwh=tlwh, confidence=conf, class_id=person_class_id, frame_idx=frame - 1)
        )
    return per_frame


def run_sequence(
    detections_per_frame: Sequence[Sequence[Detection]],
    cfg: OCSIConfig,
    conf: float = 1.0,
) -> List[ResultRow]:
    """Track a whole clip and return MOTChallenge result rows (1-indexed frames)."""
    tracker = OCSITracker(cfg)
    rows: List[ResultRow] = []
    for frame_idx, dets in enumerate(detections_per_frame):
        outputs = tracker.update(list(dets))
        rows.extend(tracker_rows(frame_idx, outputs, conf))
    return rows


def run_and_evaluate(
    detections_per_frame: Sequence[Sequence[Detection]],
    gt,
    cfg: OCSIConfig,
    iou_threshold: float = 0.5,
    conf: float = 1.0,
) -> Tuple[List[ResultRow], MOTMetrics]:
    """Run the tracker over a clip and score it against ``gt``.

    ``gt`` is the ``{frame: [(id, tlwh), ...]}`` dict from
    :func:`ocsi.eval.mot_io.read_gt`. Returns ``(result_rows, metrics)``.
    """
    rows = run_sequence(detections_per_frame, cfg, conf)
    metrics = evaluate(gt, rows_to_frames(rows), iou_threshold)
    return rows, metrics
