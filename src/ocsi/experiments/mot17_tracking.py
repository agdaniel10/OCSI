"""Run OCSI on a MOT17 sequence with cacheable detector/Re-ID features.

MOT17 has pedestrian boxes and identities, but no action labels. This runner is
therefore for contributions #1 and #2: memory-backed identity persistence and
unified association. Behaviour feedback (#3) remains covered by the synthetic
gating experiment in :mod:`ocsi.experiments.behaviour_feedback`.

Typical Colab usage::

    from ocsi.experiments.mot17_tracking import run_mot17_sequence
    payload = run_mot17_sequence(
        "/content/MOT17/train/MOT17-02-FRCNN",
        cache_dir="/content/drive/MyDrive/ocsi_cache/MOT17-02-FRCNN",
        output_dir="/content/ocsi_outputs",
        stages=("baseline", "memory"),
    )
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import OCSIConfig, apply_ablation
from ..eval.metrics import evaluate, rows_to_frames
from ..eval.mot17 import mot_image_files
from ..eval.mot_io import read_gt, tracker_rows, write_results
from ..identity import OCSITracker
from ..types import Detection


@dataclass
class MOT17StageResult:
    stage: str
    result_path: str
    metrics: Dict[str, float]
    summary: str


def detections_to_array(detections: Sequence[Detection]) -> np.ndarray:
    """Serialize detections as ``[x,y,w,h,conf,class,embedding...]`` rows."""
    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    dim = 0
    for det in detections:
        if det.embedding is not None:
            dim = int(det.embedding.shape[0])
            break
    rows = []
    for det in detections:
        emb = (
            np.asarray(det.embedding, dtype=np.float32)
            if det.embedding is not None
            else np.zeros((dim,), dtype=np.float32)
        )
        rows.append(
            np.concatenate(
                [
                    np.asarray(det.tlwh, dtype=np.float32),
                    np.array([det.confidence, det.class_id], dtype=np.float32),
                    emb,
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


def array_to_detections(array: np.ndarray, frame_idx: int) -> List[Detection]:
    """Deserialize cached detection rows for one 0-indexed frame."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, arr.shape[-1])
    dets: List[Detection] = []
    for row in arr:
        embedding = row[6:].astype(np.float32) if row.shape[0] > 6 else None
        dets.append(
            Detection(
                tlwh=row[:4].astype(float),
                confidence=float(row[4]),
                class_id=int(row[5]),
                frame_idx=frame_idx,
                embedding=embedding,
            )
        )
    return dets


def build_perception_cache(
    seq_dir: str,
    cache_dir: str,
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
) -> List[List[Detection]]:
    """Detect people, attach embeddings, and persist one ``.npy`` per frame."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MOT17 perception cache needs opencv-python") from exc
    try:
        from ..perception import FeatureCache, ReIDEmbedder, YOLODetector
    except Exception as exc:  # pragma: no cover
        raise ImportError("MOT17 perception cache needs the perception extra") from exc

    cfg = cfg or OCSIConfig()
    cache = FeatureCache(cache_dir)
    detector = YOLODetector(cfg.perception)
    embedder = ReIDEmbedder(cfg.perception)
    frames = mot_image_files(seq_dir, limit)
    seq_name = os.path.basename(os.path.normpath(seq_dir))

    detections_per_frame: List[List[Detection]] = []
    for frame_idx, path in enumerate(frames):
        key = f"{seq_name}/{frame_idx + 1:06d}"

        def compute() -> np.ndarray:
            frame_bgr = cv2.imread(path)
            if frame_bgr is None:
                raise FileNotFoundError(path)
            dets = detector(frame_bgr, frame_idx=frame_idx)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return detections_to_array(embedder.attach_embeddings(frame_rgb, dets))

        arr = cache.get_or_compute(key, compute)
        detections_per_frame.append(array_to_detections(arr, frame_idx))
    return detections_per_frame


def load_cached_detections(seq_dir: str, cache_dir: str, limit: Optional[int] = None) -> List[List[Detection]]:
    """Load cached detector/Re-ID outputs without importing perception models."""
    from ..perception.cache import FeatureCache

    cache = FeatureCache(cache_dir)
    frames = mot_image_files(seq_dir, limit)
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    out: List[List[Detection]] = []
    for frame_idx, _ in enumerate(frames):
        key = f"{seq_name}/{frame_idx + 1:06d}"
        arr = cache.get(key)
        if arr is None:
            raise FileNotFoundError(f"missing cached detections for {key!r} in {cache_dir!r}")
        out.append(array_to_detections(arr, frame_idx))
    return out


def track_cached_sequence(
    detections_per_frame: Sequence[Sequence[Detection]],
    cfg: OCSIConfig,
    output_path: str,
) -> List:
    """Replay cached detections through the tracker and write MOT results."""
    tracker = OCSITracker(cfg)
    rows = []
    for frame_idx, detections in enumerate(detections_per_frame):
        rows.extend(tracker_rows(frame_idx, tracker.update(list(detections))))
    write_results(output_path, rows)
    return rows


def run_mot17_sequence(
    seq_dir: str,
    cache_dir: str,
    output_dir: str,
    stages: Sequence[str] = ("baseline", "memory"),
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    rebuild_cache: bool = False,
) -> Dict:
    """Build/load MOT17 perception cache, run ablations, and report metrics."""
    base_cfg = cfg or OCSIConfig()
    os.makedirs(output_dir, exist_ok=True)
    if rebuild_cache or not os.path.exists(cache_dir):
        detections = build_perception_cache(seq_dir, cache_dir, base_cfg, limit)
    else:
        detections = load_cached_detections(seq_dir, cache_dir, limit)

    gt = read_gt(os.path.join(seq_dir, "gt", "gt.txt"))
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    results: List[MOT17StageResult] = []
    for stage in stages:
        stage_cfg = apply_ablation(base_cfg, stage)
        stage_cfg.behaviour.enabled = False
        result_path = os.path.join(output_dir, f"{seq_name}-{stage}.txt")
        rows = track_cached_sequence(detections, stage_cfg, result_path)
        metrics = evaluate(gt, rows_to_frames(rows))
        results.append(
            MOT17StageResult(
                stage=stage,
                result_path=result_path,
                metrics=metrics.as_dict(),
                summary=metrics.summary(),
            )
        )

    payload = {
        "sequence": seq_name,
        "frames": len(detections),
        "cache_dir": cache_dir,
        "results": [asdict(r) for r in results],
        "note": "MOT17 has no action labels; behaviour feedback is evaluated separately.",
    }
    with open(os.path.join(output_dir, f"{seq_name}-summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Import run_mot17_sequence from a notebook/script and pass seq_dir, cache_dir, output_dir."
    )
