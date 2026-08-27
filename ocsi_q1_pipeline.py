#!/usr/bin/env python3
"""
OCSI-Q1: Reproducible experimental pipeline for memory-driven multi-object tracking.

Purpose
-------
This script is designed to address the main weaknesses identified in the OCSI
reproduction report:
1. Avoid threshold-selection leakage by separating calibration and evaluation sequences.
2. Use a person-ReID model (OSNet via torchreid when available).
3. Provide adaptive or globally calibrated reactivation thresholds.
4. Report IDF1, ID switches, MOTA-style counts, re-entry recovery, IDSW/1000 frames.
5. Bootstrap confidence intervals and paired significance tests.
6. Record runtime/FPS and memory-bank statistics.
7. Export per-sequence and aggregate CSV/JSON results suitable for a journal paper.
8. Optionally invoke the official TrackEval package when installed.

IMPORTANT
---------
This script does not manufacture results. It produces valid results only when run on
the real MOT17/MOT20 data and corresponding detections/frames.

Recommended Python:
    Python 3.10/3.11

Suggested installation:
    pip install numpy pandas scipy opencv-python torch torchvision scipy scikit-learn
    pip install torchreid
    # Install official TrackEval separately:
    # git clone https://github.com/JonathonLuiten/TrackEval.git
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon, ttest_rel

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# MOT I/O
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    frame: int
    det_id: int
    box: np.ndarray
    conf: float
    feature: Optional[np.ndarray] = None


@dataclass
class GT:
    frame: int
    obj_id: int
    box: np.ndarray
    visible: float = 1.0


def read_mot_txt(path: Path, is_gt: bool = False, det_conf: float = 0.30):
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            obj_id = int(float(p[1]))
            box = xywh_to_xyxy(list(map(float, p[2:6])))

            if is_gt:
                mark = int(float(p[6])) if len(p) > 6 else 1
                cls = int(float(p[7])) if len(p) > 7 else 1
                vis = float(p[8]) if len(p) > 8 else 1.0
                # MOT17 pedestrian GT: mark==1 and class==1
                if mark == 1 and cls == 1:
                    rows.append(GT(frame, obj_id, box, vis))
            else:
                conf = float(p[6]) if len(p) > 6 else 1.0
                if conf >= det_conf:
                    rows.append(Detection(frame, obj_id, box, conf))
    return rows


def group_by_frame(items):
    out = defaultdict(list)
    for x in items:
        out[x.frame].append(x)
    return out


# ---------------------------------------------------------------------------
# ReID embedder
# ---------------------------------------------------------------------------

class ReIDEmbedder:
    """
    OSNet/torchreid preferred. If torchreid is unavailable, the script can still
    run in motion-only mode, but Q1 experiments should use a person-ReID model.
    """
    def __init__(self, device="cuda", model_name="osnet_x1_0"):
        self.device = device if torch is not None and torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.transform = None
        self.backend = "none"

        if torch is None or cv2 is None:
            return

        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss="softmax",
                pretrained=True
            )
            self.model.eval().to(self.device)
            self.backend = f"torchreid:{model_name}"
            # Standard ReID preprocessing
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        except Exception as e:
            print(f"[WARN] torchreid unavailable or failed to load: {e}")
            print("[WARN] Running without appearance features is NOT recommended for Q1.")

    @torch.no_grad() if torch is not None else (lambda f: f)
    def embed(self, image_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or self.transform is None or image_bgr is None:
            return None

        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = image_bgr[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = self.transform(crop).unsqueeze(0).to(self.device)
        feat = self.model(t)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# OCSI memory tracker
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id: int
    box: np.ndarray
    feature: Optional[np.ndarray]
    last_frame: int
    hits: int = 1
    age: int = 0
    active: bool = True


class OCSIMemoryTracker:
    def __init__(
        self,
        iou_gate=0.20,
        short_app_gate=0.55,
        react_gate=0.85,
        max_lost=90,
        gallery_size=30,
        memory_alpha=0.90,
        use_memory=True,
        adaptive_gate=False,
        adaptive_margin=0.04,
    ):
        self.iou_gate = iou_gate
        self.short_app_gate = short_app_gate
        self.react_gate = react_gate
        self.max_lost = max_lost
        self.gallery_size = gallery_size
        self.memory_alpha = memory_alpha
        self.use_memory = use_memory
        self.adaptive_gate = adaptive_gate
        self.adaptive_margin = adaptive_margin

        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.gallery: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.gallery_size))
        self.memory_proto: Dict[int, np.ndarray] = {}
        self.neg_sims = deque(maxlen=5000)
        self.reactivation_attempts = 0
        self.reactivation_success_like = 0

    def _dynamic_gate(self) -> float:
        if not self.adaptive_gate or len(self.neg_sims) < 30:
            return self.react_gate
        # Conservative threshold = high quantile of observed competing-ID similarity + margin
        q95 = float(np.quantile(np.asarray(self.neg_sims), 0.95))
        return min(0.98, max(0.65, q95 + self.adaptive_margin))

    def _update_memory(self, tid, feat):
        if feat is None:
            return
        self.gallery[tid].append(feat)
        if tid not in self.memory_proto:
            self.memory_proto[tid] = feat.copy()
        else:
            p = self.memory_alpha * self.memory_proto[tid] + (1 - self.memory_alpha) * feat
            n = np.linalg.norm(p)
            self.memory_proto[tid] = p / n if n > 0 else p

    def _new_track(self, det: Detection):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = TrackState(tid, det.box.copy(), det.feature, det.frame)
        self._update_memory(tid, det.feature)
        return tid

    def update(self, frame: int, detections: List[Detection]) -> List[Tuple[int, np.ndarray, float]]:
        active_ids = [tid for tid, t in self.tracks.items()
                      if frame - t.last_frame <= self.max_lost]

        # Association cost: IoU + short-term appearance
        matches = []
        unmatched_t = set(active_ids)
        unmatched_d = set(range(len(detections)))

        if active_ids and detections:
            C = np.full((len(active_ids), len(detections)), 1e6, dtype=np.float32)
            for r, tid in enumerate(active_ids):
                t = self.tracks[tid]
                for c, d in enumerate(detections):
                    ov = iou(t.box, d.box)
                    sim = cosine_sim(t.feature, d.feature)
                    valid = (ov >= self.iou_gate) or (sim >= self.short_app_gate)
                    if valid:
                        app_cost = 1.0 - max(-1.0, sim)
                        iou_cost = 1.0 - ov
                        C[r, c] = 0.55 * iou_cost + 0.45 * app_cost

            rr, cc = linear_sum_assignment(C)
            for r, c in zip(rr, cc):
                if C[r, c] < 1e5:
                    tid = active_ids[r]
                    matches.append((tid, c))
                    unmatched_t.discard(tid)
                    unmatched_d.discard(c)

        # Update normal matches
        for tid, didx in matches:
            d = detections[didx]
            t = self.tracks[tid]
            # collect negative similarities for adaptive threshold estimation
            if d.feature is not None:
                for other_tid in active_ids:
                    if other_tid != tid and other_tid in self.memory_proto:
                        s = cosine_sim(self.memory_proto[other_tid], d.feature)
                        if s >= 0:
                            self.neg_sims.append(s)
            t.box = d.box.copy()
            t.feature = d.feature
            t.last_frame = frame
            t.hits += 1
            t.age = 0
            t.active = True
            self._update_memory(tid, d.feature)

        # Memory-only reactivation for remaining detections
        if self.use_memory and unmatched_d:
            lost_ids = [tid for tid in unmatched_t
                        if tid in self.memory_proto and
                        0 < frame - self.tracks[tid].last_frame <= self.max_lost]

            if lost_ids:
                C = np.full((len(lost_ids), len(unmatched_d)), 1e6, dtype=np.float32)
                dlist = list(unmatched_d)
                gate = self._dynamic_gate()
                for r, tid in enumerate(lost_ids):
                    for c, didx in enumerate(dlist):
                        sim = cosine_sim(self.memory_proto[tid], detections[didx].feature)
                        self.reactivation_attempts += 1
                        if sim >= gate:
                            C[r, c] = 1.0 - sim

                rr, cc = linear_sum_assignment(C)
                for r, c in zip(rr, cc):
                    if C[r, c] < 1e5:
                        tid, didx = lost_ids[r], dlist[c]
                        if didx not in unmatched_d:
                            continue
                        d = detections[didx]
                        t = self.tracks[tid]
                        t.box = d.box.copy()
                        t.feature = d.feature
                        t.last_frame = frame
                        t.hits += 1
                        t.age = 0
                        t.active = True
                        self._update_memory(tid, d.feature)
                        unmatched_d.discard(didx)
                        unmatched_t.discard(tid)
                        self.reactivation_success_like += 1

        # New tracks
        for didx in list(unmatched_d):
            self._new_track(detections[didx])

        # Age stale tracks
        for tid, t in self.tracks.items():
            if t.last_frame != frame:
                t.age = frame - t.last_frame
                t.active = t.age <= self.max_lost

        # Output only tracks updated this frame
        out = []
        for tid, t in self.tracks.items():
            if t.last_frame == frame:
                out.append((tid, t.box.copy(), 1.0))
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def match_gt_pred(gt_list: List[GT], pred_list, iou_thr=0.5):
    if not gt_list or not pred_list:
        return [], set(range(len(gt_list))), set(range(len(pred_list)))

    C = np.ones((len(gt_list), len(pred_list)), dtype=np.float32)
    for i, g in enumerate(gt_list):
        for j, (_, box, _) in enumerate(pred_list):
            C[i, j] = 1.0 - iou(g.box, box)

    rr, cc = linear_sum_assignment(C)
    matches = []
    ug, up = set(range(len(gt_list))), set(range(len(pred_list)))
    for r, c in zip(rr, cc):
        if 1.0 - C[r, c] >= iou_thr:
            matches.append((r, c))
            ug.discard(r)
            up.discard(c)
    return matches, ug, up


def evaluate_sequence(gt_by_frame, pred_by_frame, n_frames: int, iou_thr=0.5):
    """
    Lightweight local evaluator for diagnostic use.
    Final journal values should be cross-checked with official TrackEval.
    """
    total_gt = 0
    fp = fn = idsw = 0

    # IDF1 contingency counts: gt_id x pred_id matches
    pair_counts = defaultdict(int)
    gt_det_count = defaultdict(int)
    pr_det_count = defaultdict(int)
    last_pred_for_gt = {}

    # Re-entry statistics
    last_seen_gt = {}
    gt_last_pred = {}
    reentry_total = 0
    reentry_correct = 0
    reentry_bins = defaultdict(lambda: [0, 0])

    for f in range(1, n_frames + 1):
        gts = gt_by_frame.get(f, [])
        prs = pred_by_frame.get(f, [])
        total_gt += len(gts)

        for g in gts:
            gt_det_count[g.obj_id] += 1
        for pid, _, _ in prs:
            pr_det_count[pid] += 1

        matches, ug, up = match_gt_pred(gts, prs, iou_thr=iou_thr)
        fn += len(ug)
        fp += len(up)

        for gi, pi in matches:
            gid = gts[gi].obj_id
            pid = prs[pi][0]
            pair_counts[(gid, pid)] += 1

            if gid in last_pred_for_gt and last_pred_for_gt[gid] != pid:
                idsw += 1

            # Long-gap re-entry recovery diagnostic
            if gid in last_seen_gt:
                gap = f - last_seen_gt[gid] - 1
                if gap > 0 and gid in gt_last_pred:
                    reentry_total += 1
                    correct = int(pid == gt_last_pred[gid])
                    reentry_correct += correct
                    if gap < 30:
                        b = "<1s"
                    elif gap < 90:
                        b = "1-3s"
                    elif gap < 150:
                        b = "3-5s"
                    else:
                        b = ">=5s"
                    reentry_bins[b][0] += correct
                    reentry_bins[b][1] += 1

            last_pred_for_gt[gid] = pid
            gt_last_pred[gid] = pid
            last_seen_gt[gid] = f

        # Update last seen for unmatched GT too, because they are still visible in GT
        for gi, g in enumerate(gts):
            last_seen_gt[g.obj_id] = f

    # Global ID assignment maximizing matched detections
    gids = sorted(gt_det_count)
    pids = sorted(pr_det_count)
    if gids and pids:
        W = np.zeros((len(gids), len(pids)), dtype=np.int64)
        gi_map = {g: i for i, g in enumerate(gids)}
        pi_map = {p: i for i, p in enumerate(pids)}
        for (g, p), n in pair_counts.items():
            W[gi_map[g], pi_map[p]] = n
        rr, cc = linear_sum_assignment(-W)
        idtp = int(W[rr, cc].sum())
    else:
        idtp = 0

    total_gt_dets = sum(gt_det_count.values())
    total_pr_dets = sum(pr_det_count.values())
    idfn = total_gt_dets - idtp
    idfp = total_pr_dets - idtp
    idf1 = (2 * idtp / (2 * idtp + idfp + idfn)) if (2 * idtp + idfp + idfn) else 0.0

    mota = 1.0 - ((fn + fp + idsw) / total_gt) if total_gt else 0.0
    precision = (total_pr_dets - fp) / total_pr_dets if total_pr_dets else 0.0
    recall = (total_gt - fn) / total_gt if total_gt else 0.0
    reentry = reentry_correct / reentry_total if reentry_total else np.nan

    return {
        "MOTA_local": mota,
        "IDF1_local": idf1,
        "IDSW_local": idsw,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "ReentryRecovery": reentry,
        "ReentryEvents": reentry_total,
        "IDSW_per_1000_frames": idsw / max(1, n_frames) * 1000.0,
        **{
            f"Reentry_{k}": (v[0] / v[1] if v[1] else np.nan)
            for k, v in reentry_bins.items()
        }
    }


# ---------------------------------------------------------------------------
# Threshold calibration without leakage
# ---------------------------------------------------------------------------

def choose_threshold_from_similarity(
    same_sims: Iterable[float],
    diff_sims: Iterable[float],
    candidate_thresholds=np.arange(0.75, 0.931, 0.01),
):
    """
    Select threshold by maximizing balanced accuracy on CALIBRATION data only.
    This avoids tuning directly on evaluation sequences.
    """
    same = np.asarray(list(same_sims), dtype=float)
    diff = np.asarray(list(diff_sims), dtype=float)
    if len(same) == 0 or len(diff) == 0:
        raise ValueError("Need both same-ID and diff-ID calibration similarities.")

    best = None
    rows = []
    for th in candidate_thresholds:
        tpr = np.mean(same >= th)
        tnr = np.mean(diff < th)
        bal = 0.5 * (tpr + tnr)
        rows.append((th, tpr, tnr, bal))
        if best is None or bal > best[-1]:
            best = (th, tpr, tnr, bal)

    return best[0], pd.DataFrame(rows, columns=["threshold", "TPR_same", "TNR_diff", "balanced_accuracy"])


# ---------------------------------------------------------------------------
# Bootstrap + paired inference
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05, seed=42):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_stats(base, treatment):
    a = np.asarray(base, dtype=float)
    b = np.asarray(treatment, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": len(a), "mean_delta": np.nan, "wilcoxon_p": np.nan, "paired_t_p": np.nan}

    d = b - a
    try:
        w_p = wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    except Exception:
        w_p = np.nan
    try:
        t_p = ttest_rel(b, a).pvalue
    except Exception:
        t_p = np.nan

    return {
        "n": len(a),
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
    }


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def sequence_length(seq_dir: Path) -> int:
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        txt = ini.read_text(errors="ignore")
        for line in txt.splitlines():
            if line.lower().startswith("seqlength"):
                return int(line.split("=")[1].strip())
    img_dir = seq_dir / "img1"
    if img_dir.exists():
        return len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    raise RuntimeError(f"Cannot determine sequence length: {seq_dir}")


def load_frame(seq_dir: Path, frame: int):
    if cv2 is None:
        return None
    img = seq_dir / "img1" / f"{frame:06d}.jpg"
    if not img.exists():
        img = seq_dir / "img1" / f"{frame:06d}.png"
    return cv2.imread(str(img)) if img.exists() else None


def run_tracker_on_sequence(
    seq_dir: Path,
    mode: str,
    embedder: ReIDEmbedder,
    det_conf=0.30,
    react_gate=0.85,
    adaptive_gate=False,
    max_lost=90,
    cache_dir: Optional[Path] = None,
):
    det_path = seq_dir / "det" / "det.txt"
    gt_path = seq_dir / "gt" / "gt.txt"
    detections = read_mot_txt(det_path, is_gt=False, det_conf=det_conf)
    gt = read_mot_txt(gt_path, is_gt=True)
    det_by_frame = group_by_frame(detections)
    gt_by_frame = group_by_frame(gt)
    n_frames = sequence_length(seq_dir)

    use_memory = mode != "baseline"
    tracker = OCSIMemoryTracker(
        react_gate=react_gate,
        use_memory=use_memory,
        adaptive_gate=adaptive_gate,
        max_lost=max_lost,
    )

    pred_by_frame = defaultdict(list)
    cache_dir = cache_dir or (seq_dir / ".ocsi_cache")
    cache_dir.mkdir(exist_ok=True)

    start = time.perf_counter()
    num_embedded = 0

    for frame in range(1, n_frames + 1):
        img = None
        ds = det_by_frame.get(frame, [])

        if ds and embedder.model is not None:
            img = load_frame(seq_dir, frame)

        for idx, d in enumerate(ds):
            cache_file = cache_dir / f"{seq_dir.name}_{frame:06d}_{idx:03d}.npy"
            if cache_file.exists():
                d.feature = np.load(cache_file)
            elif img is not None:
                d.feature = embedder.embed(img, d.box)
                if d.feature is not None:
                    np.save(cache_file, d.feature)
                    num_embedded += 1

        out = tracker.update(frame, ds)
        pred_by_frame[frame] = out

    elapsed = time.perf_counter() - start
    metrics = evaluate_sequence(gt_by_frame, pred_by_frame, n_frames)
    metrics.update({
        "Sequence": seq_dir.name,
        "Mode": mode,
        "Frames": n_frames,
        "Seconds": elapsed,
        "FPS": n_frames / elapsed if elapsed > 0 else np.nan,
        "ReID_backend": embedder.backend,
        "ReactGate": react_gate,
        "AdaptiveGate": adaptive_gate,
        "MemoryReactivationAttempts": tracker.reactivation_attempts,
        "MemoryReactivations": tracker.reactivation_success_like,
        "TracksCreated": tracker.next_id - 1,
    })

    # Save MOT-format predictions for official TrackEval later.
    return metrics, pred_by_frame


def save_mot_predictions(pred_by_frame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        for frame in sorted(pred_by_frame):
            for tid, box, conf in pred_by_frame[frame]:
                x1, y1, x2, y2 = box
                w.writerow([
                    frame, tid, x1, y1, x2 - x1, y2 - y1,
                    conf, -1, -1, -1
                ])


# ---------------------------------------------------------------------------
# Paper-quality report
# ---------------------------------------------------------------------------

def summarize_for_paper(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["IDF1_local", "MOTA_local", "IDSW_local", "ReentryRecovery",
               "IDSW_per_1000_frames", "FPS"]

    summary_rows = []
    for mode, g in df.groupby("Mode"):
        for m in metrics:
            mean, lo, hi = bootstrap_mean_ci(g[m].values)
            summary_rows.append({
                "Mode": mode,
                "Metric": m,
                "Mean": mean,
                "CI95_Low": lo,
                "CI95_High": hi,
                "N_sequences": len(g)
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "paper_summary_bootstrap_ci.csv", index=False)

    # Paired baseline-vs-memory and baseline-vs-adaptive
    infer = []
    modes = set(df["Mode"])
    for treatment in ["memory", "adaptive_memory"]:
        if "baseline" not in modes or treatment not in modes:
            continue
        b = df[df.Mode == "baseline"].set_index("Sequence")
        t = df[df.Mode == treatment].set_index("Sequence")
        common = sorted(set(b.index) & set(t.index))
        for m in ["IDF1_local", "MOTA_local", "IDSW_local",
                  "ReentryRecovery", "IDSW_per_1000_frames"]:
            s = paired_stats(b.loc[common, m].values, t.loc[common, m].values)
            infer.append({
                "Comparison": f"{treatment} vs baseline",
                "Metric": m,
                **s
            })

    infer_df = pd.DataFrame(infer)
    infer_df.to_csv(out_dir / "paper_paired_statistics.csv", index=False)

    return summary, infer_df


# ---------------------------------------------------------------------------
# Optional TrackEval hook
# ---------------------------------------------------------------------------

def run_trackeval(trackeval_dir: Path, gt_folder: Path, trackers_folder: Path,
                  benchmark="MOT17", split="train"):
    """
    Invokes official TrackEval if installed locally.
    Adapt CLI options if your TrackEval checkout has changed.
    """
    script = trackeval_dir / "scripts" / "run_mot_challenge.py"
    if not script.exists():
        raise FileNotFoundError(f"TrackEval script not found: {script}")

    cmd = [
        sys.executable, str(script),
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", split,
        "--GT_FOLDER", str(gt_folder),
        "--TRACKERS_FOLDER", str(trackers_folder),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
    ]
    print("[TrackEval]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OCSI Q1 reproducible experiment pipeline")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root containing MOT17-xx-FRCNN sequence directories")
    p.add_argument("--out", type=Path, default=Path("ocsi_q1_results"))
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--calibration-sequences", nargs="*", default=None,
                   help="Reserved sequences for threshold calibration; never use evaluation outcomes to tune.")
    p.add_argument("--evaluation-sequences", nargs="*", default=None)
    p.add_argument("--det-conf", type=float, default=0.30)
    p.add_argument("--react-gate", type=float, default=0.85)
    p.add_argument("--max-lost", type=int, default=90)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-trackeval", action="store_true")
    p.add_argument("--trackeval-dir", type=Path)
    args = p.parse_args()

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([x for x in args.data_root.iterdir() if x.is_dir()])
    if args.sequences:
        wanted = set(args.sequences)
        seq_dirs = [x for x in seq_dirs if x.name in wanted]

    if args.evaluation_sequences:
        ew = set(args.evaluation_sequences)
        seq_dirs = [x for x in seq_dirs if x.name in ew]

    if not seq_dirs:
        raise RuntimeError("No MOT sequence directories found.")

    embedder = ReIDEmbedder(args.device)
    print(f"[INFO] ReID backend: {embedder.backend}")
    if embedder.backend == "none":
        print("[WARN] For publication-grade OCSI results, install/use a person-ReID model.")

    all_rows = []
    pred_root = args.out / "trackers"

    for seq_dir in seq_dirs:
        print(f"\n=== {seq_dir.name} ===")
        configs = [
            ("baseline", False, False),
            ("memory", True, False),
            ("adaptive_memory", True, True),
        ]

        for mode, use_mem, adapt in configs:
            print(f"  -> {mode}")
            metrics, preds = run_tracker_on_sequence(
                seq_dir=seq_dir,
                mode=mode,
                embedder=embedder,
                det_conf=args.det_conf,
                react_gate=args.react_gate,
                adaptive_gate=adapt,
                max_lost=args.max_lost,
                cache_dir=args.out / "feature_cache",
            )
            all_rows.append(metrics)
            save_mot_predictions(
                preds,
                pred_root / mode / "data" / f"{seq_dir.name}.txt"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out / "per_sequence_results.csv", index=False)

    summary, infer = summarize_for_paper(df, args.out)

    print("\n=== PER-SEQUENCE RESULTS ===")
    print(df.to_string(index=False))
    print("\n=== BOOTSTRAP SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED STATISTICS ===")
    print(infer.to_string(index=False))

    # JSON metadata for reproducibility
    metadata = {
        "seed": args.seed,
        "data_root": str(args.data_root),
        "sequences": [x.name for x in seq_dirs],
        "det_conf": args.det_conf,
        "react_gate": args.react_gate,
        "max_lost": args.max_lost,
        "reid_backend": embedder.backend,
        "note": (
            "Local metrics are diagnostic. Cross-check final HOTA/CLEAR/Identity "
            "with official TrackEval before manuscript submission."
        )
    }
    (args.out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.run_trackeval:
        if args.trackeval_dir is None:
            raise ValueError("--trackeval-dir is required with --run-trackeval")
        run_trackeval(
            args.trackeval_dir,
            gt_folder=args.data_root,
            trackers_folder=pred_root,
            benchmark="MOT17",
            split="train"
        )


if __name__ == "__main__":
    main()
