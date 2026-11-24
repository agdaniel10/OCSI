#!/usr/bin/env python
"""AVA behaviour-feedback experiment for OCSI (Kaggle-ready).

This runner validates the paper's *behaviour* layer with real person-centred
action labels. It does NOT measure tracking metrics. AVA's ``person_id`` links
boxes of the same person in adjacent annotated frames (per the official AVA
v2.2 format), so short pseudo-tracklets are formed primarily from that ID, with
IoU/box continuity as a fallback. These are used to train/evaluate the lightweight
temporal HAR recognizer and to test the confidence gate under real HAR
confidences with injected low-confidence behaviour noise.

Outputs (default ``/kaggle/working/ocsi_ava_outputs``)::

    ava_har_results.json
    ava_action_metrics.csv
    ava_gate_ablation.csv

Typical Kaggle session::

    !git clone https://github.com/agdaniel10/OCSI.git /kaggle/working/OCSI
    %cd /kaggle/working/OCSI
    !python kaggle_ava_har.py \
        --annotations-dir /kaggle/input/ava/annotations \
        --video-root /kaggle/input/ava/videos \
        --max-videos 40 --max-timestamps-per-video 300
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ocsi.behaviour import behaviour_gate
from ocsi.behaviour.temporal_har import TemporalHARRecognizer
from ocsi.config import OCSIConfig, PerceptionConfig

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".webm")


# ---------------------------------------------------------------------- data
@dataclass
class Observation:
    video_id: str
    ts: float
    person_idx: int
    box_xyxy: np.ndarray
    actions: np.ndarray


@dataclass
class Tracklet:
    video_id: str
    tid: int
    obs: List[Observation]


def _find_csv(root: Path, kind: str) -> Optional[Path]:
    matches: List[Path] = []
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if "excluded" in name:
            continue
        if kind == "train" and ("train" in name or "atiku" in name):
            matches.append(path)
        elif kind in ("val", "valid") and ("val" in name or "valid" in name):
            matches.append(path)
    return sorted(matches, key=lambda p: (len(str(p)), str(p).lower()))[0] if matches else None


def _find_excluded_csvs(root: Path) -> List[Path]:
    matches = [p for p in root.rglob("*.csv") if "excluded" in p.name.lower()]
    return sorted(matches, key=lambda p: (len(str(p)), str(p).lower()))


def load_excluded(path: Any) -> set:
    """Load one excluded-timestamp CSV or a sequence of them (train + val)."""
    if path is None:
        return set()
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    out: set = set()
    for item in paths:
        csv_path = Path(item)
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            for parts in csv.reader(fh):
                if len(parts) < 2:
                    continue
                try:
                    out.add((parts[0].strip(), int(round(float(parts[1])))))
                except ValueError:
                    continue
    return out


def load_ava_frames(
    path: Path,
    num_classes: int,
    excluded: set,
) -> Dict[Tuple[str, float], List[Observation]]:
    """Parse AVA rows into ``(video_id, timestamp) -> [Observation]``."""
    grouped: Dict[Tuple[str, float], Dict[int, Observation]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        for parts in csv.reader(fh):
            if len(parts) < 8:
                continue
            if parts[0].strip().lower().startswith("video"):
                continue
            try:
                video_id = parts[0].strip()
                ts = float(parts[1])
                x1, y1, x2, y2 = (float(v) for v in parts[2:6])
                action_id = int(float(parts[6]))
                person_idx = int(float(parts[7]))
            except ValueError:
                continue
            if (video_id, int(round(ts))) in excluded:
                continue
            if not (1 <= action_id <= num_classes):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            key = (video_id, ts)
            obs = grouped[key].get(person_idx)
            if obs is None:
                obs = Observation(
                    video_id=video_id,
                    ts=ts,
                    person_idx=person_idx,
                    box_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    actions=np.zeros(num_classes, dtype=np.float32),
                )
                grouped[key][person_idx] = obs
            obs.actions[action_id - 1] = 1.0

    return {key: list(people.values()) for key, people in grouped.items()}


def _limit_timestamps(
    frames: Dict[Tuple[str, float], List[Observation]],
    max_per_video: int,
) -> Dict[Tuple[str, float], List[Observation]]:
    if max_per_video <= 0:
        return frames
    by_video: Dict[str, List[float]] = defaultdict(list)
    for video_id, ts in frames:
        by_video[video_id].append(ts)
    keep = set()
    for video_id, stamps in by_video.items():
        stamps = sorted(set(stamps))
        if len(stamps) <= max_per_video:
            chosen = stamps
        else:
            idx = np.linspace(0, len(stamps) - 1, max_per_video).astype(int)
            chosen = [stamps[int(i)] for i in sorted(set(idx))]
        keep.update((video_id, float(ts)) for ts in chosen)
    return {key: value for key, value in frames.items() if key in keep}


def _box_to_pixels(box: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    """Convert AVA's normalised [0,1] boxes to xyxy pixels when needed."""
    box = np.asarray(box, dtype=np.float32).copy()
    if box.size == 4 and float(np.max(box)) <= 1.0 and float(np.min(box)) >= 0.0:
        box *= np.array([frame_width, frame_height, frame_width, frame_height], dtype=np.float32)
    return box


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_pseudo_tracklets(
    frames: Dict[Tuple[str, float], List[Observation]],
    video_id: str,
    iou_threshold: float = 0.3,
    max_gap_seconds: float = 2.0,
) -> List[Tracklet]:
    """Link AVA boxes into short pseudo-tracklets.

    In AVA v2.2, ``person_id`` links boxes of the same person in adjacent
    annotated frames. Use that ID first, then fall back to box IoU continuity
    for annotations where the ID link is missing or ambiguous.
    """
    stamps = sorted(ts for vid, ts in frames if vid == video_id)
    tracks: List[Tracklet] = []
    previous: List[Tuple[Tracklet, Observation]] = []

    for ts in stamps:
        current = frames[(video_id, ts)]
        matched_current: set[int] = set()
        used_previous: set[int] = set()

        # Pass 1: official AVA person_id continuity across adjacent timestamps.
        for ci, obs in enumerate(current):
            candidates: List[Tuple[float, int]] = []
            for pi, (_, last) in enumerate(previous):
                if pi in used_previous:
                    continue
                gap = float(obs.ts) - float(last.ts)
                if gap <= 0 or gap > max_gap_seconds:
                    continue
                if last.person_idx == obs.person_idx:
                    candidates.append((gap, pi))
            if candidates:
                _, pi = min(candidates)
                previous[pi][0].obs.append(obs)
                matched_current.add(ci)
                used_previous.add(pi)

        # Pass 2: IoU fallback for the remaining boxes.
        pairs: List[Tuple[float, int, int]] = []
        for ci, obs in enumerate(current):
            if ci in matched_current:
                continue
            for pi, (_, last) in enumerate(previous):
                if pi in used_previous:
                    continue
                gap = float(obs.ts) - float(last.ts)
                if gap <= 0 or gap > max_gap_seconds:
                    continue
                overlap = _iou(obs.box_xyxy, last.box_xyxy)
                if overlap >= iou_threshold:
                    pairs.append((overlap, ci, pi))

        for _, ci, pi in sorted(pairs, key=lambda item: item[0], reverse=True):
            if ci in matched_current or pi in used_previous:
                continue
            previous[pi][0].obs.append(current[ci])
            matched_current.add(ci)
            used_previous.add(pi)

        for ci, obs in enumerate(current):
            if ci not in matched_current:
                track = Tracklet(video_id=video_id, tid=len(tracks), obs=[obs])
                tracks.append(track)
                previous.append((track, obs))

        previous = [(t, t.obs[-1]) for t in tracks if abs(float(t.obs[-1].ts) - float(ts)) < 1e-6]

    return tracks


def obs_key(obs: Observation) -> str:
    box = ",".join(f"{v:.3f}" for v in obs.box_xyxy)
    return f"{obs.ts:.3f}|{obs.person_idx}|{box}"


# ------------------------------------------------------------------ features
def _index_videos(video_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in video_root.rglob("*"):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def _read_frame(cap, ts: float, fps: float):
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
    ok, frame = cap.read()
    if not ok and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(ts) * fps)))
        ok, frame = cap.read()
    return frame if ok else None


def extract_video_features(
    video_id: str,
    frames: Dict[Tuple[str, float], List[Observation]],
    video_index: Dict[str, Path],
    cache_dir: Path,
    embedder: Any,
    refresh_cache: bool,
) -> Dict[str, np.ndarray]:
    """Embed one GT crop per AVA observation; cache as npz."""
    import cv2

    cache_path = cache_dir / f"{video_id}.npz"
    if cache_path.exists() and not refresh_cache:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"].tolist()]
                embs = np.asarray(data["embeddings"], dtype=np.float32)
            return dict(zip(keys, embs))
        except Exception:
            pass

    video_path = video_index.get(video_id)
    if video_path is None:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    keys: List[str] = []
    embeddings: List[np.ndarray] = []

    for ts in sorted(ts for vid, ts in frames if vid == video_id):
        frame_bgr = _read_frame(cap, ts, fps)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_rgb.shape[:2]
        people = frames[(video_id, ts)]
        boxes_tlwh = []
        for obs in people:
            box = _box_to_pixels(obs.box_xyxy, frame_w, frame_h)
            boxes_tlwh.append(
                np.array(
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    dtype=np.float32,
                )
            )
        try:
            embs = embedder(frame_rgb, boxes_tlwh)
        except Exception:
            continue
        for obs, emb in zip(people, embs):
            keys.append(obs_key(obs))
            embeddings.append(np.asarray(emb, dtype=np.float32))
    cap.release()

    if embeddings:
        stack = np.stack(embeddings).astype(np.float32)
        np.savez_compressed(cache_path, keys=np.array(keys), embeddings=stack)
        return dict(zip(keys, stack))
    return {}


def make_windows(
    video_ids: Sequence[str],
    frames: Dict[Tuple[str, float], List[Observation]],
    tracks: Dict[str, List[Tracklet]],
    features: Dict[str, Dict[str, np.ndarray]],
    window: int,
    min_window: int,
    num_classes: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    windows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    for video_id in video_ids:
        feature_map = features.get(video_id, {})
        for track in tracks.get(video_id, []):
            seq = track.obs
            if len(seq) <= window:
                chunks = [seq] if len(seq) >= min_window else []
            else:
                chunks = [seq[i:i + window] for i in range(0, len(seq) - window + 1)]
            for chunk in chunks:
                feats: List[np.ndarray] = []
                for obs in chunk:
                    emb = feature_map.get(obs_key(obs))
                    if emb is None:
                        continue
                    feats.append(emb)
                if len(feats) < min_window:
                    continue
                target = np.zeros(num_classes, dtype=np.float32)
                for obs in chunk:
                    target[np.flatnonzero(obs.actions >= 0.5)] = 1.0
                if target.sum() == 0:
                    continue
                windows.append(np.stack(feats).astype(np.float32))
                targets.append(target)
                metadata.append(
                    {
                        "video": video_id,
                        "pseudo_track": track.tid,
                        "start_ts": float(chunk[0].ts),
                        "end_ts": float(chunk[-1].ts),
                        "length": len(feats),
                        "actions": np.flatnonzero(target >= 0.5).astype(int).tolist(),
                    }
                )
    return windows, targets, metadata


# -------------------------------------------------------------------- metrics
def average_precision(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / positives
    prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev) * precision))


def macro_f1(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, List[float], List[int]]:
    f1s: List[float] = []
    supports: List[int] = []
    for c in range(targets.shape[1]):
        y = targets[:, c].astype(bool)
        p = preds[:, c].astype(bool)
        support = int(y.sum())
        if support == 0 and not p.any():
            f1s.append(0.0)
            supports.append(0)
            continue
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom else 0.0)
        supports.append(support)
    return (float(np.mean(f1s)) if f1s else 0.0), f1s, supports


def top1_multilabel_preds(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert 80-way softmax scores to one-positive-per-window multi-label rows.

    A fixed 0.5 threshold is inappropriate for an 80-class multi-label problem:
    the max probability is often below 0.5 even when the ranking is useful.
    Reporting top-1 F1 gives a non-degenerate complementary metric while mAP
    remains the primary AVA-style ranking metric.
    """
    idx = np.argmax(np.asarray(probs, dtype=float), axis=1)
    preds = np.zeros((len(idx), num_classes), dtype=bool)
    preds[np.arange(len(idx)), idx] = True
    return preds


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i + 1 < bins else conf <= edges[i + 1])
        if not mask.any():
            continue
        total += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return total


def evaluate_recognizer(
    recognizer: TemporalHARRecognizer,
    windows: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    num_classes: int,
) -> Dict[str, Any]:
    probs_matrix: List[np.ndarray] = []
    conf: List[float] = []
    correct: List[float] = []
    for window, target in zip(windows, targets):
        obs = recognizer.observe(window)
        if obs is None:
            continue
        probs_matrix.append(obs.probs)
        conf.append(obs.p_max)
        top1 = int(np.argmax(obs.probs))
        correct.append(1.0 if float(target[top1]) >= 0.5 else 0.0)

    if not probs_matrix:
        return {
            "map": 0.0,
            "top1_macro_f1": 0.0,
            "macro_f1_at_half": 0.0,
            "ece": 0.0,
            "n_windows": 0,
            "per_class": [],
        }

    probs = np.stack(probs_matrix).astype(float)
    tgt = np.stack(targets).astype(float)[:len(probs)]
    aps: List[Optional[float]] = []
    per_class: List[Dict[str, Any]] = []
    for c in range(num_classes):
        ap = average_precision(probs[:, c], tgt[:, c])
        aps.append(ap)
        per_class.append(
            {
                "action_id": c + 1,
                "ap": "" if ap is None else round(ap, 6),
                "support": int(tgt[:, c].sum()),
            }
        )
    valid_aps = [v for v in aps if v is not None]
    map_score = float(np.mean(valid_aps)) if valid_aps else 0.0
    f1_half, f1s_half, supports = macro_f1(probs >= 0.5, tgt)
    pred_top1 = top1_multilabel_preds(probs, num_classes)
    f1_top1, f1s_top1, _ = macro_f1(pred_top1, tgt)
    class_rows = [
        row for row, sup in zip(per_class, supports) if row["support"] > 0
    ]
    for idx, row in enumerate(per_class):
        if row["support"] > 0:
            row["f1"] = round(f1s_top1[idx], 6)
            row["f1_at_half"] = round(f1s_half[idx], 6)

    return {
        "map": map_score,
        "top1_macro_f1": f1_top1,
        "macro_f1_at_half": f1_half,
        "ece": expected_calibration_error(np.asarray(conf), np.asarray(correct)),
        "p_max": {
            "mean": float(np.mean(conf)),
            "median": float(np.median(conf)),
            "max": float(np.max(conf)),
            "min": float(np.min(conf)),
        },
        "n_windows": len(probs),
        "per_class": class_rows,
    }


def gate_ablation(
    recognizer: TemporalHARRecognizer,
    windows: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    metadata: Sequence[Dict[str, Any]],
    noise_fraction: float,
    theta_b: Any,
    theta_quantile: float,
    tau_b: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Inject low-confidence, action-mismatched behaviour observations.

    Four modes are compared:

    * ``no_feedback`` — prototype frozen at its first clean observation
    * ``clean_feedback`` — full-trust updates from clean observations only
    * ``ungated`` — full-trust updates including injected wrong observations
    * ``gated`` — ``behaviour_gate`` ignores the injected low-confidence rows

    Donors are chosen from windows with disjoint action labels, so contamination
    is semantically wrong rather than merely another similar ReID crop.
    """
    rng = np.random.default_rng(seed)
    if not windows:
        return []

    observations = []
    for window in windows:
        obs = recognizer.observe(window)
        if obs is not None:
            observations.append((obs.probs.copy(), obs.embedding.copy(), float(obs.p_max)))
    if len(observations) < 2:
        return []
    observations = observations[: len(windows)]
    targets = list(targets)[: len(observations)]
    metadata = list(metadata)[: len(observations)]

    p_max_values = np.asarray([item[2] for item in observations], dtype=float)
    if theta_b is None or str(theta_b).strip().lower() == "auto":
        theta_value = float(np.quantile(p_max_values, float(np.clip(theta_quantile, 0.0, 1.0))))
    else:
        theta_value = float(theta_b)

    action_sets = [
        frozenset(np.flatnonzero(np.asarray(t, dtype=float) >= 0.5).tolist())
        for t in targets
    ]

    by_track: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        by_track[(str(meta["video"]), int(meta["pseudo_track"]))].append(i)

    clean_embeddings = np.stack([item[1] for item in observations])
    # Precompute one paired noise plan so all modes see identical contamination.
    noise_plan: Dict[Tuple[str, int], List[Tuple[bool, int]]] = {}
    for key, indices in by_track.items():
        noisy = list(rng.random(len(indices)) < noise_fraction)
        if indices:
            noisy[0] = False  # the seed observation must stay clean
        donors: List[int] = []
        for pos, idx in enumerate(indices):
            if not noisy[pos]:
                donors.append(idx)
                continue
            target_actions = action_sets[idx]
            candidates = [
                j for j in range(len(observations))
                if action_sets[j].isdisjoint(target_actions)
            ]
            if candidates:
                cand = np.asarray(candidates, dtype=int)
                sims = clean_embeddings[cand] @ clean_embeddings[idx]
                donors.append(int(cand[int(np.argmin(sims))]))
            else:
                donors.append(int(rng.integers(0, len(clean_embeddings))))
        noise_plan[key] = list(zip(noisy, donors))

    rows: List[Dict[str, Any]] = []

    for mode in ("no_feedback", "clean_feedback", "ungated", "gated"):
        probs_out: List[np.ndarray] = []
        cosines: List[float] = []
        label_indices: List[int] = []
        for key, indices in by_track.items():
            if len(indices) < 2:
                continue
            plan = noise_plan[key]
            clean_idx = [idx for idx, (is_noisy, _) in zip(indices, plan) if not is_noisy]
            clean_proto = clean_embeddings[clean_idx or indices].mean(axis=0)
            clean_proto = clean_proto / (np.linalg.norm(clean_proto) + 1e-12)
            proto = clean_embeddings[indices[0]].copy()

            for pos, (idx, (is_noisy, donor)) in enumerate(zip(indices, plan)):
                noisy_now = is_noisy and mode != "clean_feedback"
                if noisy_now:
                    emb = clean_embeddings[donor].copy()
                    p_max = min(0.2, theta_value * 0.5)
                else:
                    _, emb, p_max = observations[idx]

                if mode == "no_feedback":
                    reliability = 0.0
                elif mode in ("ungated", "clean_feedback"):
                    reliability = 1.0
                else:
                    reliability = float(
                        behaviour_gate(p_max, 0.0, theta_value, tau_b, gate_enabled=True).reshape(-1)[0]
                    )

                alpha = 0.7 + 0.3 * (1.0 - reliability)
                proto = alpha * proto + (1.0 - alpha) * emb
                proto = proto / (np.linalg.norm(proto) + 1e-12)

                if pos == len(indices) - 1:
                    cosines.append(float(np.dot(proto, clean_proto)))
                    logits = recognizer._class_protos @ proto
                    logits = logits - logits.max()
                    exp = np.exp(logits)
                    probs_out.append(exp / exp.sum())
                    label_indices.append(idx)

        if not probs_out:
            continue
        probs = np.stack(probs_out)
        labels = np.stack([targets[i] for i in label_indices])
        aps = [average_precision(probs[:, c], labels[:, c]) for c in range(probs.shape[1])]
        valid = [v for v in aps if v is not None]
        rows.append(
            {
                "mode": mode,
                "map": float(np.mean(valid)) if valid else 0.0,
                "mean_cosine_to_clean": float(np.mean(cosines)),
                "n_tracks": len(cosines),
                "noise_fraction": noise_fraction,
                "theta_b": theta_value,
            }
        )
    return rows


# ----------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AVA behaviour-feedback experiment for OCSI.")
    parser.add_argument("--annotations-dir", default="/kaggle/input",
                        help="directory containing AVA train/val CSV files")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--excluded-csv", default=None)
    parser.add_argument("--video-root", default="/kaggle/input",
                        help="directory containing AVA video clips")
    parser.add_argument("--out", default="/kaggle/working/ocsi_ava_outputs")
    parser.add_argument("--cache", default="/kaggle/working/ocsi_ava_cache")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--max-videos", type=int, default=40)
    parser.add_argument("--max-timestamps-per-video", type=int, default=300)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-window", type=int, default=4)
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument(
        "--theta-b",
        type=str,
        default="auto",
        help="fixed threshold or 'auto' (default: quantile of clean p_max)",
    )
    parser.add_argument(
        "--theta-quantile",
        type=float,
        default=0.50,
        help="p_max quantile used when --theta-b auto",
    )
    parser.add_argument("--tau-b", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reid-backend", default="auto",
                        help="auto / torchvision / torchreid")
    parser.add_argument("--reid-backbone", default="resnet50")
    parser.add_argument("--reid-batch-size", type=int, default=64)
    parser.add_argument("--allow-random-reid", action="store_true",
                        help="permit random-init Re-ID weights (diagnostic only, not paper evidence)")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)

    ann_root = Path(args.annotations_dir).expanduser()
    video_root = Path(args.video_root).expanduser()
    out_dir = Path(args.out).expanduser()
    cache_dir = Path(args.cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv) if args.train_csv else _find_csv(ann_root, "train")
    val_csv = Path(args.val_csv) if args.val_csv else _find_csv(ann_root, "val")
    excluded_paths = (
        [Path(args.excluded_csv)]
        if args.excluded_csv
        else _find_excluded_csvs(ann_root)
    )
    if train_csv is None or not train_csv.exists():
        raise SystemExit(f"Could not find AVA train CSV under {ann_root}; pass --train-csv")
    if val_csv is None or not val_csv.exists():
        raise SystemExit(f"Could not find AVA val CSV under {ann_root}; pass --val-csv")

    excluded = load_excluded(excluded_paths)
    train_frames = load_ava_frames(train_csv, args.num_classes, excluded)
    val_frames = load_ava_frames(val_csv, args.num_classes, excluded)
    train_frames = _limit_timestamps(train_frames, args.max_timestamps_per_video)
    val_frames = _limit_timestamps(val_frames, args.max_timestamps_per_video)

    # The AVA annotation CSVs cover all 235 train / 64 val videos, but the Kaggle
    # video subset covers only ~65 clips. Filter BEFORE selecting --max-videos,
    # otherwise selected videos can have zero crops and contribute no windows.
    video_index = _index_videos(video_root)
    cache_stems = {path.stem for path in cache_dir.glob("*.npz")}
    available_videos = set(video_index) | cache_stems
    annotated_videos = {vid for vid, _ in train_frames} | {vid for vid, _ in val_frames}
    train_frames = {k: v for k, v in train_frames.items() if k[0] in available_videos}
    val_frames = {k: v for k, v in val_frames.items() if k[0] in available_videos}
    print(
        "video availability: "
        f"{len(available_videos & annotated_videos)}/{len(annotated_videos)} "
        "annotated videos have a clip or cached crops"
    )

    if args.max_videos > 0:
        # Keep the limit per split. Limiting the union of sorted IDs previously
        # left almost no validation videos (e.g. 23 train / 2 val for 25 total).
        train_keep = set(sorted({vid for vid, _ in train_frames})[: args.max_videos])
        val_keep = set(sorted({vid for vid, _ in val_frames})[: args.max_videos])
        train_frames = {k: v for k, v in train_frames.items() if k[0] in train_keep}
        val_frames = {k: v for k, v in val_frames.items() if k[0] in val_keep}

    train_videos = sorted({vid for vid, _ in train_frames})
    val_videos = sorted({vid for vid, _ in val_frames})
    if not train_videos or not val_videos:
        raise SystemExit("Empty train or val split after filtering; check AVA CSVs and --max-videos")

    cfg = OCSIConfig()
    cfg.perception = PerceptionConfig(
        device=args.device,
        reid_backend=args.reid_backend,
        reid_backbone=args.reid_backbone,
        reid_batch_size=args.reid_batch_size,
        reid_pretrained=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from ocsi.perception.embedder import ReIDEmbedder
        embedder = ReIDEmbedder(cfg.perception)
    random_fallback = any("random init" in str(w.message).lower() for w in caught)
    if random_fallback and not args.allow_random_reid:
        raise SystemExit(
            "Re-ID weights fell back to random init. Attach pretrained weights or "
            "enable internet; use --allow-random-reid only for pipeline debugging."
        )

    print(f"train CSV : {train_csv}")
    print(f"val CSV   : {val_csv}")
    print(f"excluded  : {', '.join(p.name for p in excluded_paths) or 'none'}")
    print(f"videos    : {len(train_videos)} train / {len(val_videos)} val")
    print(f"Re-ID     : {embedder.backend_name}  dim={embedder.dim}")
    print(f"cache     : {cache_dir}")

    all_frames = {**train_frames, **val_frames}
    all_tracks: Dict[str, List[Tracklet]] = {}
    features: Dict[str, Dict[str, np.ndarray]] = {}
    for video_id in sorted(set(train_videos) | set(val_videos)):
        tracks = build_pseudo_tracklets(
            all_frames, video_id, iou_threshold=args.track_iou,
            max_gap_seconds=args.max_gap_seconds,
        )
        all_tracks[video_id] = tracks
        features[video_id] = extract_video_features(
            video_id, all_frames, video_index, cache_dir, embedder, args.refresh_cache
        )
        print(f"  {video_id}: {len(tracks)} pseudo-tracklets, {len(features[video_id])} crops")

    train_windows, train_targets, _ = make_windows(
        train_videos, train_frames, all_tracks, features,
        args.window, args.min_window, args.num_classes,
    )
    val_windows, val_targets, val_meta = make_windows(
        val_videos, val_frames, all_tracks, features,
        args.window, args.min_window, args.num_classes,
    )
    if not train_windows or not val_windows:
        raise SystemExit(
            "No usable windows. Check --max-timestamps-per-video, --window, "
            "--min-window, and whether AVA videos are present."
        )
    print(f"windows   : {len(train_windows)} train / {len(val_windows)} val")

    recognizer = TemporalHARRecognizer(
        feature_dim=int(train_windows[0].shape[-1]),
        num_classes=args.num_classes,
        min_window=args.min_window,
    )
    prototype_counts = recognizer.train_on_multilabel_windows(train_windows, train_targets)
    metrics = evaluate_recognizer(recognizer, val_windows, val_targets, args.num_classes)
    ablation = gate_ablation(
        recognizer, val_windows, val_targets, val_meta,
        noise_fraction=args.noise_fraction,
        theta_b=args.theta_b,
        theta_quantile=args.theta_quantile,
        tau_b=args.tau_b,
        seed=args.seed,
    )
    if str(args.theta_b).strip().lower() == "auto":
        theta_effective = ablation[0]["theta_b"] if ablation else None
    else:
        theta_effective = float(args.theta_b)

    with (out_dir / "ava_action_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["action_id", "ap", "support", "f1", "f1_at_half"]
        )
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow(row)
    with (out_dir / "ava_gate_ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["mode", "map", "mean_cosine_to_clean", "n_tracks", "noise_fraction", "theta_b"],
        )
        writer.writeheader()
        writer.writerows(ablation)

    payload = {
        "dataset": "AVA (person-centred atomic visual actions)",
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "excluded_csv": [str(p) for p in excluded_paths],
        "config": {
            "max_videos": args.max_videos,
            "max_timestamps_per_video": args.max_timestamps_per_video,
            "window": args.window,
            "min_window": args.min_window,
            "track_iou": args.track_iou,
            "max_gap_seconds": args.max_gap_seconds,
            "noise_fraction": args.noise_fraction,
            "theta_b": theta_effective,
            "theta_b_requested": args.theta_b,
            "theta_quantile": args.theta_quantile,
            "tau_b": args.tau_b,
            "seed": args.seed,
            "reid_backend": embedder.backend_name,
            "reid_dim": int(embedder.dim),
        },
        "splits": {
            "train_videos": train_videos,
            "val_videos": val_videos,
            "train_windows": len(train_windows),
            "val_windows": len(val_windows),
            "prototype_counts": {str(k): v for k, v in prototype_counts.items() if v},
        },
        "har": {
            "map": metrics["map"],
            "top1_macro_f1": metrics["top1_macro_f1"],
            "macro_f1_at_half": metrics["macro_f1_at_half"],
            "ece": metrics["ece"],
            "p_max": metrics["p_max"],
            "n_windows": metrics["n_windows"],
        },
        "gate_ablation": ablation,
        "note": (
            "AVA v2.2 person_id links boxes of the same person in adjacent annotated frames. "
            "Short pseudo-tracklets are built from that ID first, with IoU box continuity as a "
            "fallback; they are not long-term trajectories. AVA boxes are auto-scaled from "
            "normalised [0,1] coordinates to pixels before cropping. The behaviour-noise "
            "injection is synthetic and uses action-disjoint donors, but the action labels and "
            "HAR confidences are real. This validates the AVA behaviour layer, not MOT tracking "
            "metrics."
        ),
    }
    with (out_dir / "ava_har_results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== AVA behaviour-layer results ===")
    print(f"  action mAP     {metrics['map']:.4f}")
    print(f"  top-1 macro F1 {metrics['top1_macro_f1']:.4f}")
    print(f"  macro F1 @0.5  {metrics['macro_f1_at_half']:.4f}")
    print(f"  confidence ECE {metrics['ece']:.4f}")
    print(
        "  p_max          "
        f"mean={metrics['p_max']['mean']:.3f} median={metrics['p_max']['median']:.3f} "
        f"max={metrics['p_max']['max']:.3f}"
    )
    print(f"  gate theta_b   {theta_effective}")
    for row in ablation:
        print(
            f"  {row['mode']:12s} mAP={row['map']:.4f}  "
            f"cos_to_clean={row['mean_cosine_to_clean']:.4f}"
        )
    print(f"\nresults: {out_dir / 'ava_har_results.json'}")
    print(f"metrics: {out_dir / 'ava_action_metrics.csv'}")
    print(f"gate   : {out_dir / 'ava_gate_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
