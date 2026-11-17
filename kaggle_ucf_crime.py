#!/usr/bin/env python
"""Fast UCF-Crime testing with OCSI on Kaggle.

UCF-Crime has **video-level** anomaly labels (and frame intervals for the test
split), but no person boxes or person IDs.  This runner therefore does NOT
measure IDF1/MOTA.  It runs OCSI's ablations on sampled frames and derives an
anomaly score from the memory/behaviour signals the tracker already exposes:

    new IDs created, lost tracks, memory confidence q, (feedback) behaviour gate

It then reports frame-level AUC against the temporal annotations and video-level
normal-vs-abnormal AUC.  This tests the paper's cognitive/anomaly layer and
uncertainty claims, not the tracking claims (use MOT17/MOT20 for those).

Typical Kaggle session::

    !git clone https://github.com/agdaniel10/OCSI.git /kaggle/working/OCSI
    %cd /kaggle/working/OCSI
    !pip install -q ultralytics pyyaml opencv-python pandas matplotlib
    !python kaggle_ucf_crime.py --smoke          # 2 videos, 60 frames
    !python kaggle_ucf_crime.py --limit-videos 12 --frames 300 --stride 5 \
        --stages baseline,memory,feedback

Attach any Kaggle UCF-Crime dataset. The runner searches /kaggle/input by default.
Outputs go to /kaggle/working/ocsi_ucf_outputs/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ocsi.behaviour import behaviour_gate
from ocsi.config import OCSIConfig, apply_ablation
from ocsi.experiments.mot17_tracking import (
    _attach_behaviour_observations,
    _clone_frame_detections,
    _placeholder_behaviour_recognizer,
    _update_behaviour_windows,
)
from ocsi.identity import OCSITracker
from ocsi.types import Detection, TrackState

CLASSES = (
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
)
CLASS_ALIASES = {
    "roadaccident": "RoadAccidents", "road_accidents": "RoadAccidents",
    "accident": "RoadAccidents", "fighting": "Fighting", "normal": "Normal",
}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov")


# ------------------------------------------------------------------ discovery
def infer_class(path: Path) -> str:
    """Infer the UCF class from the parent folder or the filename prefix."""
    for parent in reversed(path.parents):
        text = parent.name.lower().replace("_", "").replace("-", "")
        if "normal" in text:
            return "Normal"
        for cls in CLASSES:
            if cls.lower().replace("_", "") in text:
                return cls
    name = path.stem.lower().replace("_", "").replace("-", "")
    for alias, cls in CLASS_ALIASES.items():
        if name.startswith(alias.lower().replace("_", "")):
            return cls
    for cls in sorted(CLASSES, key=len, reverse=True):
        if name.startswith(cls.lower().replace("_", "")):
            return cls
    return "Unknown"


def find_videos(root: Path) -> List[Path]:
    videos = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES]
    return sorted(videos, key=lambda p: str(p).lower())


def discover_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise SystemExit(f"--data-dir does not exist: {root}")
        return root
    for candidate in ("/kaggle/input", "/kaggle/input/ucf-crime-datasets", "."):
        path = Path(candidate)
        if path.is_dir() and find_videos(path):
            return path
    raise SystemExit("No UCF-Crime videos found. Pass --data-dir /kaggle/input/...")


def find_annotation_path(root: Path, explicit: Optional[str]) -> Optional[Path]:
    """Locate UCF-Crime's temporal annotation file, if present."""
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    matches = [
        p for p in root.rglob("*.txt")
        if "temporal" in p.name.lower() and "anomaly" in p.name.lower()
    ]
    return sorted(matches, key=lambda p: len(str(p)))[0] if matches else None


def find_annotations(path: Optional[Path]) -> Dict[str, List[Tuple[int, int]]]:
    """Load temporal anomaly intervals: {video stem: [(start, end), ...]}."""
    out: Dict[str, List[Tuple[int, int]]] = {}
    if path is None or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        stem = Path(parts[0]).stem
        intervals = []
        for start, end in zip(parts[2::2], parts[3::2]):
            try:
                s, e = int(start), int(end)
            except ValueError:
                continue
            if s >= 0 and e >= s:
                intervals.append((s, e))
        out[stem] = intervals
    return out


def select_videos(
    videos: Sequence[Path],
    names: Optional[str],
    classes: Optional[str],
    limit: Optional[int],
    smoke: bool,
) -> List[Path]:
    selected = list(videos)
    if names:
        wanted = {n.strip().lower() for n in names.split(",") if n.strip()}
        selected = [
            p for p in selected
            if p.stem.lower() in wanted or any(w in p.stem.lower() for w in wanted)
        ]
    if classes:
        wanted_cls = {c.strip().lower() for c in classes.split(",") if c.strip()}
        selected = [p for p in selected if infer_class(p).lower() in wanted_cls]
    if smoke:
        normal = [p for p in selected if infer_class(p) == "Normal"]
        abnormal = [p for p in selected if infer_class(p) not in ("Normal", "Unknown")]
        selected = normal[:1] + abnormal[:1]
        if len(selected) < 2:
            selected = selected[:2] or selected
    elif limit and limit < len(selected):
        by_class: Dict[str, List[Path]] = defaultdict(list)
        for p in selected:
            by_class[infer_class(p)].append(p)
        keys = sorted(by_class, key=lambda k: (k != "Normal", k))
        picked: List[Path] = []
        index = 0
        while len(picked) < limit and any(by_class[k] for k in keys):
            key = keys[index % len(keys)]
            if by_class[key]:
                picked.append(by_class[key].pop(0))
            index += 1
        selected = picked
    return selected


# ---------------------------------------------------------------- video / cache
def sample_frames(video: Path, n_frames: int, stride: int) -> Tuple[List[np.ndarray], List[int], float]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = min(n_frames, max(1, (total + stride - 1) // stride if total else n_frames))
    frames: List[np.ndarray] = []
    originals: List[int] = []
    for i in range(wanted):
        target = i * stride  # 0-based decode index
        if total:
            target = min(target, max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        originals.append(target + 1)  # annotations are 1-based
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    return frames, originals, fps


def cache_path(cache_dir: Path, video: Path, n_frames: int, stride: int,
               imgsz: int, det_conf: float, reid: str, device: str) -> Path:
    key = "|".join(map(str, (str(video), n_frames, stride, imgsz, det_conf, reid, device)))
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{video.stem}_{digest}.npz"


def build_detections(frames: Sequence[np.ndarray], cfg: OCSIConfig) -> Tuple[List[List[Detection]], int]:
    from ocsi.perception import ReIDEmbedder, YOLODetector

    detector = YOLODetector(cfg.perception)
    embedder = ReIDEmbedder(cfg.perception)
    dim = int(embedder.dim)
    per_frame: List[List[Detection]] = []
    for idx, frame in enumerate(frames):
        dets = detector(frame, frame_idx=idx)
        dets = embedder.attach_embeddings(cv2_cvt_rgb(frame), dets)
        per_frame.append(dets)
    return per_frame, dim


def cv2_cvt_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def save_cache(path: Path, per_frame: Sequence[Sequence[Detection]],
               originals: Sequence[int], fps: float, dim: int) -> None:
    rows = []
    for frame_idx, dets in enumerate(per_frame):
        for det in dets:
            emb = np.asarray(det.embedding, dtype=np.float32).reshape(-1)
            if emb.size != dim:
                emb = np.zeros(dim, dtype=np.float32)
            rows.append(np.concatenate(([frame_idx], det.tlwh, [det.confidence, det.class_id], emb)))
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 7 + dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows=array,
        dim=np.asarray(dim),
        originals=np.asarray(originals, dtype=np.int64),
        fps=np.asarray(fps),
    )


def load_cache(path: Path) -> Tuple[List[List[Detection]], List[int], float, int]:
    with np.load(path, allow_pickle=False) as data:
        rows = data["rows"]
        dim = int(data["dim"])
        originals = [int(v) for v in data["originals"]]
        fps = float(data["fps"])
    n_frames = len(originals)
    per_frame: List[List[Detection]] = [[] for _ in range(n_frames)]
    for row in rows:
        frame_idx = int(row[0])
        if frame_idx >= n_frames:
            continue
        per_frame[frame_idx].append(
            Detection(
                tlwh=np.asarray(row[1:5], dtype=float),
                confidence=float(row[5]),
                class_id=int(row[6]),
                frame_idx=frame_idx,
                embedding=np.asarray(row[7:], dtype=np.float32),
            )
        )
    return per_frame, originals, fps, dim


# --------------------------------------------------------------------- runner
def run_stage(
    detections: Sequence[Sequence[Detection]],
    stage: str,
    base_cfg: OCSIConfig,
    seed: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    cfg = apply_ablation(base_cfg, stage)
    tracker = OCSITracker(cfg)
    recognizer = None
    windows: Dict[int, Any] = {}
    max_window = int(cfg.behaviour.min_window)
    if cfg.behaviour.enabled:
        recognizer = _placeholder_behaviour_recognizer(detections, cfg, seed)

    signals: List[Dict[str, float]] = []
    prev_created = 0
    for frame_idx, frame_dets in enumerate(detections):
        dets = _clone_frame_detections(frame_dets)
        if recognizer is not None:
            _attach_behaviour_observations(tracker, dets, windows, recognizer)
        updated = tracker.update(dets)
        if recognizer is not None:
            _update_behaviour_windows(updated, dets, windows, max_window)

        created = int(getattr(tracker.bank, "_next_id", 1)) - 1
        new_ids = max(0, created - prev_created)
        prev_created = created
        confirmed = tracker.outputs()
        q_values = [float(r.q) for r in confirmed]
        records = list(tracker.bank.records.values())
        lost = sum(1 for r in records if r.state == TrackState.LOST)
        # Cheap, direct log of the same confidence gate the tracker applies. We
        # avoid enabling XAI just to read metadata: that would trigger Grad-CAM
        # saliency and slow the quick Kaggle smoke test down substantially.
        gates: List[float] = []
        if recognizer is not None:
            p_max = [float(np.max(d.activity_probs)) for d in dets if d.activity_probs is not None]
            if p_max:
                gates = behaviour_gate(
                    np.asarray(p_max, dtype=float),
                    0.0,
                    cfg.behaviour.theta_b,
                    cfg.behaviour.tau_b,
                    cfg.behaviour.gate_enabled,
                ).tolist()
        signals.append(
            {
                "frame": float(frame_idx),
                "n_detections": float(len(dets)),
                "n_active": float(len(confirmed)),
                "n_lost": float(lost),
                "new_ids": float(new_ids),
                "mean_q": float(np.mean(q_values)) if q_values else 0.0,
                "min_q": float(np.min(q_values)) if q_values else 0.0,
                "mean_gate": float(np.mean(gates)) if gates else 0.0,
                "gate_fraction": float(np.mean(np.asarray(gates) > 0.0)) if gates else 0.0,
                "contamination_flags": float(tracker.contamination_flags),
                "rollbacks": float(tracker.rollbacks_applied),
            }
        )

    totals = {
        "created_ids": float(prev_created),
        "contamination_flags": float(tracker.contamination_flags),
        "rollbacks": float(tracker.rollbacks_applied),
    }
    return signals, totals


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc_score(labels: Sequence[float], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    pos, neg = int((y > 0).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = _rankdata(s)
    return float((ranks[y > 0].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def anomaly_scores(signals: Sequence[Dict[str, float]]) -> np.ndarray:
    """Combine OCSI signals into one frame-level anomaly score (higher = odder)."""
    new_ids = np.array([s["new_ids"] for s in signals], dtype=float)
    lost = np.array([s["n_lost"] for s in signals], dtype=float)
    confidence = np.array([s["mean_q"] for s in signals], dtype=float)

    def norm(x: np.ndarray) -> np.ndarray:
        span = float(np.max(x) - np.min(x))
        return (x - np.min(x)) / span if span > 1e-9 else np.zeros_like(x)

    return 0.40 * norm(new_ids) + 0.30 * norm(lost) + 0.30 * (1.0 - norm(confidence))


def frame_labels(originals: Sequence[int], intervals: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.asarray(
        [any(start <= int(f) <= end for start, end in intervals) for f in originals],
        dtype=float,
    )


def save_plots(rows: Sequence[Dict[str, Any]], annotations: Dict[str, List[Tuple[int, int]]],
               out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_stage: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["frame_auc"] is not None:
            by_stage[row["stage"]].append(float(row["frame_auc"]))
    if by_stage:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        stages = sorted(by_stage)
        values = [np.mean(by_stage[s]) for s in stages]
        ax.bar(stages, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(stages)])
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("frame-level anomaly AUC")
        ax.set_title("UCF-Crime: OCSI signals as anomaly detectors")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "ucf_crime_auc.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Per-frame curves for the first annotated anomaly video.
    anomalies = sorted({r["video"] for r in rows if r["cls"] != "Normal" and r["frame_auc"] is not None})
    if anomalies:
        video = anomalies[0]
        subset = [r for r in rows if r["video"] == video]
        frames = json.loads(subset[0]["originals_json"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for row in subset:
            scores = json.loads(row["scores_json"])
            ax.plot(frames, scores, lw=1.2, label=row["stage"])
        for start, end in annotations.get(video, []):
            ax.axvspan(start, end, color="0.5", alpha=0.2)
        ax.set_xlabel("frame index")
        ax.set_ylabel("anomaly score")
        ax.set_title(f"UCF-Crime anomaly signals: {video}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "ucf_crime_scores.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fast UCF-Crime OCSI test (Kaggle).")
    parser.add_argument("--data-dir", default=None, help="root containing UCF-Crime videos")
    parser.add_argument("--labels", default=None, help="explicit temporal annotation txt")
    parser.add_argument("--out", default="/kaggle/working/ocsi_ucf_outputs")
    parser.add_argument("--cache", default="/kaggle/working/ocsi_ucf_cache")
    parser.add_argument("--videos", default=None, help="comma-separated video-name substrings")
    parser.add_argument("--classes", default=None, help="comma-separated UCF classes to include")
    parser.add_argument("--limit-videos", type=int, default=None, help="balanced subset size")
    parser.add_argument("--frames", type=int, default=300, help="sampled frames per video")
    parser.add_argument("--stride", type=int, default=5, help="frame step between samples")
    parser.add_argument("--stages", default="baseline,memory,feedback")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--reid-backend", default="auto", help="auto / torchvision / torchreid")
    parser.add_argument("--smoke", action="store_true", help="2 videos, 60 frames, baseline+memory")
    parser.add_argument("--refresh-cache", action="store_true", help="rebuild detection cache")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.frames = min(args.frames, 60)
        args.stages = "baseline,memory"
        args.limit_videos = None

    root = discover_root(args.data_dir)
    all_videos = find_videos(root)
    videos = select_videos(all_videos, args.videos, args.classes, args.limit_videos, args.smoke)
    if not videos:
        raise SystemExit(f"No videos selected under {root}")
    annotation_path = find_annotation_path(root, args.labels)
    annotations = find_annotations(annotation_path)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    cfg = OCSIConfig()
    cfg.perception.device = args.device
    cfg.perception.det_imgsz = int(args.imgsz)
    cfg.perception.det_conf_threshold = float(args.det_conf)
    cfg.perception.reid_backend = args.reid_backend

    print(f"data dir : {root}")
    print(f"videos   : {len(videos)} (found {len(all_videos)})")
    print(f"stages   : {', '.join(stages)}")
    print(f"frames   : {args.frames} per video, stride {args.stride}")
    print(f"device   : {cfg.perception.device}   Re-ID: {cfg.perception.reid_backend}")
    print(f"outputs  : {out_dir}")

    rows: List[Dict[str, Any]] = []          # one row per video/stage (for plots)
    frame_records: List[Dict[str, Any]] = [] # one row per sampled frame (CSV)
    summary_rows: List[Dict[str, Any]] = []
    started = time.time()

    for position, video in enumerate(videos, 1):
        cls = infer_class(video)
        print(f"\n[{position}/{len(videos)}] {video.name}  ({cls})")
        cache = cache_path(
            cache_dir, video, args.frames, args.stride, args.imgsz,
            args.det_conf, args.reid_backend, args.device,
        )
        if cache.exists() and not args.refresh_cache:
            detections, originals, fps, _ = load_cache(cache)
            print("  cache hit")
        else:
            frames, originals, fps = sample_frames(video, args.frames, args.stride)
            detections, dim = build_detections(frames, cfg)
            save_cache(cache, detections, originals, fps, dim)
            print(f"  cached {sum(len(f) for f in detections)} detections over {len(originals)} frames")
        intervals = annotations.get(video.stem, [])
        labels = frame_labels(originals, intervals) if cls != "Normal" else np.zeros(len(originals))

        for stage in stages:
            signals, totals = run_stage(detections, stage, cfg, args.seed)
            scores = anomaly_scores(signals)
            frame_auc = auc_score(labels, scores) if cls != "Normal" and intervals else None
            video_score = float(np.quantile(scores, 0.9)) if scores.size else 0.0
            last_signals = {k: float(v) for k, v in signals[-1].items()} if signals else {}
            row = {
                "video": video.stem,
                "cls": cls,
                "stage": stage,
                "frame_auc": frame_auc,
                "video_score": video_score,
                "created_ids": totals["created_ids"],
                "contamination_flags": totals["contamination_flags"],
                "rollbacks": totals["rollbacks"],
                "originals_json": json.dumps(originals),
                "scores_json": json.dumps([float(v) for v in scores]),
                **last_signals,
            }
            rows.append(row)
            for frame_idx, (signal, score) in enumerate(zip(signals, scores)):
                record: Dict[str, Any] = {
                    "video": video.stem,
                    "class": cls,
                    "stage": stage,
                    "original_frame": (
                        int(originals[frame_idx]) if frame_idx < len(originals) else frame_idx + 1
                    ),
                    "anomaly_score": float(score),
                }
                record.update({k: float(v) for k, v in signal.items()})
                frame_records.append(record)
            summary_rows.append(
                {
                    "video": video.stem,
                    "class": cls,
                    "stage": stage,
                    "frame_auc": "" if frame_auc is None else f"{frame_auc:.4f}",
                    "video_score": f"{video_score:.4f}",
                    "created_ids": int(totals["created_ids"]),
                    "contamination_flags": int(totals["contamination_flags"]),
                    "rollbacks": int(totals["rollbacks"]),
                }
            )
            tag = "" if frame_auc is None else f"  frame AUC={frame_auc:.3f}"
            print(f"    {stage:14s} score={video_score:.3f}  IDs={int(totals['created_ids']):3d}{tag}")

    # Video-level normal vs abnormal AUC from the per-video anomaly score.
    video_aucs: Dict[str, Optional[float]] = {}
    for stage in stages:
        stage_rows = [r for r in summary_rows if r["stage"] == stage]
        labels = [0 if r["class"] == "Normal" else 1 for r in stage_rows]
        score_values = [float(r["video_score"]) for r in stage_rows]
        video_aucs[stage] = (
            auc_score(labels, score_values) if len(set(labels)) == 2 else None
        )

    elapsed = time.time() - started
    payload: Dict[str, Any] = {
        "dataset_root": str(root),
        "videos": [str(v) for v in videos],
        "classes": [infer_class(v) for v in videos],
        "annotations_file": str(annotation_path) if annotation_path else "",
        "stages": stages,
        "config": {
            "frames": args.frames, "stride": args.stride, "imgsz": args.imgsz,
            "det_conf": args.det_conf, "reid_backend": args.reid_backend,
            "device": cfg.perception.device, "seed": args.seed,
        },
        "backend_name": cfg.perception.reid_backend,
        "video_auc": video_aucs,
        "elapsed_seconds": elapsed,
        "summary": summary_rows,
        "note": (
            "UCF-Crime has no person IDs or per-person action labels. These AUCs "
            "test the cognitive/anomaly layer only; they do not validate OCSI's "
            "IDF1/IDsw or behaviour-feedback claims."
        ),
    }
    with (out_dir / "ucf_crime_results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with (out_dir / "ucf_crime_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    frame_csv = out_dir / "ucf_crime_frame_scores.csv"
    base_cols = ["video", "class", "stage", "original_frame", "anomaly_score"]
    extra_cols = [k for k in frame_records[0] if k not in base_cols] if frame_records else []
    fieldnames = base_cols + extra_cols
    with frame_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in frame_records:
            writer.writerow(
                {
                    key: (f"{record[key]:.6f}" if isinstance(record.get(key), float) else record.get(key, ""))
                    for key in fieldnames
                }
            )

    if not args.no_plots:
        save_plots(rows, annotations, out_dir)

    print("\n=== Video-level normal-vs-abnormal AUC ===")
    for stage, value in video_aucs.items():
        print(f"  {stage:14s} {'-' if value is None else f'{value:.3f}'}")
    print(f"\nelapsed: {elapsed / 60.0:.1f} min")
    print(f"results: {out_dir / 'ucf_crime_results.json'}")
    print(f"summary: {out_dir / 'ucf_crime_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
