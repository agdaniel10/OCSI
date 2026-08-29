"""MOT20 sequence adapter for OCSI.

MOT20 uses the same MOTChallenge format as MOT17 but with denser crowds and
more occlusion. This module provides a thin adapter so the existing MOT17
pipeline can be reused for MOT20 with minimal changes.

MOT20 sequence layout:
    MOT20-01/
        img1/
        det/det.txt
        gt/gt.txt
        seqinfo.ini
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from ..config import OCSIConfig
from ..eval.mot_io import read_gt, read_results
from ..types import Detection
from .mot17_tracking import (
    _attach_embeddings_to_frames,
    _seq_name,
    array_to_detections,
    build_perception_cache,
    detections_to_array,
    load_cached_detections,
    run_mot17_sequence,
)


def mot20_public_detections(
    seq_dir: str,
    conf_threshold: float = 0.0,
    limit: Optional[int] = None,
) -> List[List[Detection]]:
    """Read MOT20 ``det/det.txt`` into dense per-frame public detections.

    MOT20 detection rows are the same MOTChallenge format as MOT17:
    ``frame,-1,x,y,w,h,conf,...``.
    """
    det_path = os.path.join(seq_dir, "det", "det.txt")
    rows = read_results(det_path)
    if limit is not None:
        rows = [r for r in rows if int(r[0]) <= int(limit)]
    if not rows:
        return []
    n_frames = max(int(r[0]) for r in rows)
    per_frame: List[List[Detection]] = [[] for _ in range(n_frames)]
    for row in rows:
        frame_number = int(row[0])
        conf = float(row[6]) if len(row) > 6 else 1.0
        if conf < conf_threshold:
            continue
        tlwh = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        per_frame[frame_number - 1].append(
            Detection(tlwh=tlwh, confidence=conf, class_id=0, frame_idx=frame_number - 1)
        )
    return per_frame


def build_mot20_public_detection_cache(
    seq_dir: str,
    cache_dir: str,
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    det_conf_threshold: float = 0.0,
) -> List[List[Detection]]:
    """Attach Re-ID embeddings to MOT20 public detections and cache them."""
    cfg = cfg or OCSIConfig()
    public = mot20_public_detections(seq_dir, det_conf_threshold, limit)

    def detections_for_frame(frame_idx: int, frame_bgr) -> Sequence[Detection]:
        del frame_bgr
        return public[frame_idx] if frame_idx < len(public) else []

    return _attach_embeddings_to_frames(
        seq_dir, cache_dir, detections_for_frame, cfg, "public", limit, det_conf_threshold
    )


def run_mot20_sequence(
    seq_dir: str,
    cache_dir: str,
    output_dir: str,
    stages: Sequence[str] = ("baseline", "memory"),
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    rebuild_cache: bool = False,
    detection_source: str = "public",
    det_conf_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    reactivation_app_gate: Optional[float] = None,
) -> dict:
    """Run OCSI on a MOT20 sequence.

    This is a thin wrapper around :func:`run_mot17_sequence` that uses the
    MOT20 public-detection cache builder instead of the MOT17 one.
    """
    base_cfg = cfg or OCSIConfig()
    if det_conf_threshold is not None:
        base_cfg = OCSIConfig.from_dict(base_cfg.to_dict())
        base_cfg.perception.det_conf_threshold = float(det_conf_threshold)

    # Build/load the MOT20 cache
    if rebuild_cache or not os.path.exists(cache_dir):
        if detection_source == "public":
            detections = build_mot20_public_detection_cache(
                seq_dir, cache_dir, base_cfg, limit, base_cfg.perception.det_conf_threshold
            )
        else:
            detections = build_perception_cache(seq_dir, cache_dir, base_cfg, limit)
    else:
        detections = load_cached_detections(
            seq_dir, cache_dir, limit, detection_source, base_cfg.perception.det_conf_threshold
        )

    # Reuse the MOT17 sequence runner with the pre-built detections
    # (we can't pass detections directly, so we call run_mot17_sequence with
    #  rebuild_cache=False and rely on the cache we just built).
    return run_mot17_sequence(
        seq_dir=seq_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        stages=stages,
        cfg=base_cfg,
        limit=limit,
        rebuild_cache=False,
        detection_source=detection_source,
        det_conf_threshold=det_conf_threshold,
        seed=seed,
        reactivation_app_gate=reactivation_app_gate,
    )