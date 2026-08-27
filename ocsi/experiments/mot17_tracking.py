"""Run OCSI on MOT17 sequence(s) with cacheable detector/Re-ID features.

MOT17 has pedestrian boxes and identities, but no action labels. This runner
therefore uses a clearly marked placeholder behaviour recognizer for the
``feedback`` ablation: it pools each track's recent Re-ID embeddings as a stand-in
temporal feature so the confidence-gated feedback path is exercised end-to-end.

Typical Colab usage::

    from ocsi.experiments.mot17_tracking import run_mot17_sequence
    payload = run_mot17_sequence(
        "/content/MOT17/train/MOT17-02-FRCNN",
        cache_dir="/content/drive/MyDrive/ocsi_cache/MOT17-02-FRCNN",
        output_dir="/content/ocsi_outputs",
        stages=("baseline", "memory"),
        detection_source="public",
    )
"""
from __future__ import annotations

from collections import deque
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..behaviour import PrototypeBehaviourRecognizer
from ..config import OCSIConfig, apply_ablation
from ..eval.metrics import evaluate, rows_to_frames
from ..eval.mot17 import mot_image_files
from ..eval.mot_io import read_gt, read_results, tracker_rows, write_results
from ..identity.similarity import cosine_matrix, iou_matrix
from ..identity import OCSITracker
from ..memory import MemoryRecord
from ..types import Detection


@dataclass
class MOT17StageResult:
    stage: str
    result_path: str
    metrics: Dict[str, float]
    summary: str


def _seq_name(seq_dir: str) -> str:
    return os.path.basename(os.path.normpath(seq_dir))


def _cache_key(seq_dir: str, detection_source: str, frame_idx: int) -> str:
    return f"{_seq_name(seq_dir)}/{detection_source}/{frame_idx + 1:06d}"


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


def _clone_detection(det: Detection) -> Detection:
    """Copy a cached detection so stage-local behaviour annotations cannot leak."""
    return Detection(
        tlwh=np.asarray(det.tlwh, dtype=float).copy(),
        confidence=float(det.confidence),
        class_id=int(det.class_id),
        frame_idx=int(det.frame_idx),
        embedding=None if det.embedding is None else np.asarray(det.embedding, dtype=float).copy(),
        keypoints=None if det.keypoints is None else np.asarray(det.keypoints, dtype=float).copy(),
        behaviour_embedding=None,
        activity_probs=None,
    )


def _clone_frame_detections(detections: Sequence[Detection]) -> List[Detection]:
    return [_clone_detection(det) for det in detections]


def mot17_public_detections(
    seq_dir: str,
    conf_threshold: float = 0.0,
    limit: Optional[int] = None,
) -> List[List[Detection]]:
    """Read MOT17 ``det/det.txt`` into dense per-frame public detections.

    MOT17 detection rows are MOTChallenge rows: ``frame,-1,x,y,w,h,conf,...``.
    They do not include appearance embeddings; :func:`build_public_detection_cache`
    attaches Re-ID features from the sequence images.
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
        tlwh = np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        per_frame[frame_number - 1].append(
            Detection(tlwh=tlwh, confidence=conf, class_id=0, frame_idx=frame_number - 1)
        )
    return per_frame


def _attach_embeddings_to_frames(
    seq_dir: str,
    cache_dir: str,
    detections_for_frame: Callable[[int, str], Sequence[Detection]],
    cfg: OCSIConfig,
    detection_source: str,
    limit: Optional[int],
) -> List[List[Detection]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MOT17 perception cache needs opencv-python") from exc
    try:
        from ..perception import FeatureCache, ReIDEmbedder
    except Exception as exc:  # pragma: no cover
        raise ImportError("MOT17 perception cache needs the perception extra") from exc

    cache = FeatureCache(cache_dir)
    embedder = ReIDEmbedder(cfg.perception)
    frames = mot_image_files(seq_dir, limit)
    detections_per_frame: List[List[Detection]] = []
    for frame_idx, path in enumerate(frames):
        key = _cache_key(seq_dir, detection_source, frame_idx)

        def compute() -> np.ndarray:
            frame_bgr = cv2.imread(path)
            if frame_bgr is None:
                raise FileNotFoundError(path)
            dets = list(detections_for_frame(frame_idx, frame_bgr))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return detections_to_array(embedder.attach_embeddings(frame_rgb, dets))

        arr = cache.get_or_compute(key, compute)
        detections_per_frame.append(array_to_detections(arr, frame_idx))
    return detections_per_frame


def build_perception_cache(
    seq_dir: str,
    cache_dir: str,
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
) -> List[List[Detection]]:
    """Run YOLO, attach embeddings, and persist one ``.npy`` per frame."""
    try:
        from ..perception import YOLODetector
    except Exception as exc:  # pragma: no cover
        raise ImportError("MOT17 perception cache needs the perception extra") from exc

    cfg = cfg or OCSIConfig()
    detector = YOLODetector(cfg.perception)

    def detections_for_frame(frame_idx: int, frame_bgr: np.ndarray) -> Sequence[Detection]:
        return detector(frame_bgr, frame_idx=frame_idx)

    return _attach_embeddings_to_frames(seq_dir, cache_dir, detections_for_frame, cfg, "yolo", limit)


def build_public_detection_cache(
    seq_dir: str,
    cache_dir: str,
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    det_conf_threshold: float = 0.0,
) -> List[List[Detection]]:
    """Attach Re-ID embeddings to MOT17 public detections and cache them."""
    cfg = cfg or OCSIConfig()
    public = mot17_public_detections(seq_dir, det_conf_threshold, limit)

    def detections_for_frame(frame_idx: int, frame_bgr: np.ndarray) -> Sequence[Detection]:
        del frame_bgr
        return public[frame_idx] if frame_idx < len(public) else []

    return _attach_embeddings_to_frames(seq_dir, cache_dir, detections_for_frame, cfg, "public", limit)


def load_cached_detections(
    seq_dir: str,
    cache_dir: str,
    limit: Optional[int] = None,
    detection_source: str = "yolo",
) -> List[List[Detection]]:
    """Load cached detector/Re-ID outputs without importing perception models."""
    from ..perception.cache import FeatureCache

    cache = FeatureCache(cache_dir)
    frames = mot_image_files(seq_dir, limit)
    out: List[List[Detection]] = []
    for frame_idx, _ in enumerate(frames):
        key = _cache_key(seq_dir, detection_source, frame_idx)
        arr = cache.get(key)
        if arr is None:
            raise FileNotFoundError(f"missing cached detections for {key!r} in {cache_dir!r}")
        out.append(array_to_detections(arr, frame_idx))
    return out


def _first_embedding_dim(detections_per_frame: Sequence[Sequence[Detection]]) -> Optional[int]:
    for detections in detections_per_frame:
        for det in detections:
            if det.embedding is not None:
                return int(np.asarray(det.embedding).reshape(-1).shape[0])
    return None


def _sample_normalized_embeddings(
    detections_per_frame: Sequence[Sequence[Detection]],
    embedding_dim: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    embeddings = []
    for detections in detections_per_frame:
        for det in detections:
            if det.embedding is None:
                continue
            emb = np.asarray(det.embedding, dtype=float).reshape(-1)
            if emb.shape[0] != embedding_dim:
                continue
            norm = float(np.linalg.norm(emb))
            if norm > 1e-12:
                embeddings.append(emb / norm)

    if not embeddings:
        prototypes = rng.normal(size=(num_samples, embedding_dim))
    else:
        pool = np.stack(embeddings)
        replace = len(pool) < num_samples
        indices = rng.choice(len(pool), size=num_samples, replace=replace)
        prototypes = pool[indices].copy()

    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norms, 1e-12)


def _placeholder_behaviour_recognizer(
    detections_per_frame: Sequence[Sequence[Detection]],
    cfg: OCSIConfig,
    seed: Optional[int],
) -> Optional[PrototypeBehaviourRecognizer]:
    """Build a deterministic placeholder recognizer sized to the Re-ID features."""
    if not cfg.behaviour.enabled:
        return None
    embedding_dim = _first_embedding_dim(detections_per_frame)
    if embedding_dim is None:
        return None

    rng = np.random.default_rng(0 if seed is None else int(seed))
    # Placeholder only: MOT17 has no activity labels, so sampled Re-ID embeddings
    # stand in for activity prototypes until a trained HAR model is plugged in.
    prototypes = _sample_normalized_embeddings(
        detections_per_frame,
        embedding_dim,
        int(cfg.behaviour.num_classes),
        rng,
    )
    return PrototypeBehaviourRecognizer(
        prototypes=prototypes,
        min_window=cfg.behaviour.min_window,
    )


def _tlwh_iou_matrix(track_boxes: Sequence[np.ndarray], detections: Sequence[Detection]) -> np.ndarray:
    if not track_boxes or not detections:
        return np.zeros((len(track_boxes), len(detections)), dtype=float)
    track_xyxy = np.stack([np.asarray(box, dtype=float).copy() for box in track_boxes])
    track_xyxy[:, 2] = track_xyxy[:, 0] + track_xyxy[:, 2]
    track_xyxy[:, 3] = track_xyxy[:, 1] + track_xyxy[:, 3]
    det_tlwh = np.stack([np.asarray(det.tlwh, dtype=float).copy() for det in detections])
    det_xyxy = det_tlwh.copy()
    det_xyxy[:, 2] = det_tlwh[:, 0] + det_tlwh[:, 2]
    det_xyxy[:, 3] = det_tlwh[:, 1] + det_tlwh[:, 3]
    return iou_matrix(track_xyxy, det_xyxy)


def _assign_detections_to_tracks(
    tracks: Sequence[MemoryRecord],
    detections: Sequence[Detection],
    iou_threshold: float = 0.1,
) -> Dict[int, int]:
    """Associate experiment-side track records with frame detections by box overlap."""
    if not tracks or not detections:
        return {}
    iou = _tlwh_iou_matrix([track.last_box for track in tracks], detections)
    rows, cols = linear_sum_assignment(-iou)
    out: Dict[int, int] = {}
    for row, col in zip(rows, cols):
        if iou[row, col] >= iou_threshold:
            out[int(tracks[row].track_id)] = int(col)
    return out


def _attach_behaviour_observations(
    tracker: OCSITracker,
    detections: Sequence[Detection],
    windows: Dict[int, Deque[np.ndarray]],
    recognizer: Optional[PrototypeBehaviourRecognizer],
) -> None:
    if recognizer is None or not detections:
        return

    candidates = [
        track
        for track in tracker.bank.matchable()
        if track.track_id in windows and len(windows[track.track_id]) >= recognizer.min_window
    ]
    assignments = _assign_detections_to_tracks(candidates, detections)
    for track in candidates:
        det_idx = assignments.get(track.track_id)
        if det_idx is None:
            continue
        observation = recognizer.observe(windows[track.track_id])
        if observation is None:
            continue
        detections[det_idx].activity_probs = observation.probs
        detections[det_idx].behaviour_embedding = observation.embedding

    for det in detections:
        if det.activity_probs is not None and det.behaviour_embedding is not None:
            continue
        # The tracker consumes behaviour cues as a dense per-frame matrix. Detections
        # without a full per-track window get an explicit low-confidence placeholder:
        # the gate remains closed for them, so they cannot affect association or
        # update a behaviour prototype.
        det.activity_probs = np.full(recognizer.num_classes, 1.0 / recognizer.num_classes)
        if det.embedding is None:
            det.behaviour_embedding = np.zeros(recognizer.embedding_dim, dtype=float)
        else:
            emb = np.asarray(det.embedding, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(emb))
            det.behaviour_embedding = emb / norm if norm > 1e-12 else emb


def _update_behaviour_windows(
    updated_tracks: Sequence[MemoryRecord],
    detections: Sequence[Detection],
    windows: Dict[int, Deque[np.ndarray]],
    max_window: int,
) -> None:
    if not updated_tracks or not detections:
        return
    assignments = _assign_detections_to_tracks(updated_tracks, detections)
    for track in updated_tracks:
        det_idx = assignments.get(track.track_id)
        if det_idx is None:
            continue
        embedding = detections[det_idx].embedding
        if embedding is None:
            continue
        window = windows.setdefault(track.track_id, deque(maxlen=max_window))
        # MOT17 has no action signal. Re-ID embeddings are used here only as a
        # temporary per-frame feature to validate behaviour-feedback plumbing.
        window.append(np.asarray(embedding, dtype=float).reshape(-1).copy())


def track_cached_sequence(
    detections_per_frame: Sequence[Sequence[Detection]],
    cfg: OCSIConfig,
    output_path: str,
    behaviour_seed: Optional[int] = None,
) -> List:
    """Replay cached detections through the tracker and write MOT results."""
    tracker = OCSITracker(cfg)
    recognizer = _placeholder_behaviour_recognizer(detections_per_frame, cfg, behaviour_seed)
    behaviour_windows: Dict[int, Deque[np.ndarray]] = {}
    max_window = int(cfg.behaviour.min_window)
    rows = []
    for frame_idx, detections in enumerate(detections_per_frame):
        frame_detections = _clone_frame_detections(detections)
        _attach_behaviour_observations(tracker, frame_detections, behaviour_windows, recognizer)
        updated_tracks = tracker.update(frame_detections)
        _update_behaviour_windows(updated_tracks, frame_detections, behaviour_windows, max_window)
        rows.extend(tracker_rows(frame_idx, updated_tracks))
    write_results(output_path, rows)
    return rows


def embedding_diagnostics(
    detections_per_frame: Sequence[Sequence[Detection]],
    gt: Dict[int, List],
    iou_threshold: float = 0.5,
    max_samples_per_id: int = 250,
) -> Dict[str, object]:
    """Estimate whether cached Re-ID features separate MOT ground-truth identities.

    Public/Yolo detections do not carry identity labels, so this first assigns each
    detection to the best same-frame GT box by IoU. It then compares each embedding to
    its identity prototype and to other identity prototypes. If ``same`` and ``diff``
    cosines are close, appearance memory cannot reliably support reactivation.
    """
    per_id: Dict[int, List[np.ndarray]] = {}
    norms: List[float] = []
    total_embeddings = 0
    assigned_embeddings = 0

    for frame_idx, detections in enumerate(detections_per_frame):
        dets = [d for d in detections if d.embedding is not None and np.asarray(d.embedding).size]
        if not dets:
            continue
        total_embeddings += len(dets)
        for det in dets:
            norms.append(float(np.linalg.norm(det.embedding)))

        gt_items = gt.get(frame_idx + 1, [])
        if not gt_items:
            continue
        gt_boxes = np.stack([box for _, box in gt_items]).astype(float)
        det_boxes = np.stack([d.tlwh for d in dets]).astype(float)
        gt_xyxy = gt_boxes.copy()
        gt_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
        gt_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
        det_xyxy = det_boxes.copy()
        det_xyxy[:, 2] = det_boxes[:, 0] + det_boxes[:, 2]
        det_xyxy[:, 3] = det_boxes[:, 1] + det_boxes[:, 3]

        iou = iou_matrix(gt_xyxy, det_xyxy)
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] < iou_threshold:
                continue
            emb = np.asarray(dets[c].embedding, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(emb))
            if norm <= 1e-12:
                continue
            gid = int(gt_items[r][0])
            bucket = per_id.setdefault(gid, [])
            if len(bucket) < max_samples_per_id:
                bucket.append(emb / norm)
            assigned_embeddings += 1

    usable = {gid: embs for gid, embs in per_id.items() if embs}
    out: Dict[str, object] = {
        "total_embeddings": total_embeddings,
        "assigned_embeddings": assigned_embeddings,
        "num_identity_prototypes": len(usable),
        "embedding_dim": int(next(iter(usable.values()))[0].shape[0]) if usable else 0,
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "std_norm": float(np.std(norms)) if norms else 0.0,
    }
    if len(usable) < 2:
        out.update(
            same_id_proto_cosine=None,
            different_id_proto_cosine=None,
            separation_margin=None,
            note="Need embeddings assigned to at least two GT identities for separation diagnostics.",
        )
        return out

    prototypes = {
        gid: np.mean(np.stack(embs), axis=0)
        for gid, embs in usable.items()
    }
    prototypes = {
        gid: proto / max(float(np.linalg.norm(proto)), 1e-12)
        for gid, proto in prototypes.items()
    }

    same, diff = [], []
    proto_ids = sorted(prototypes)
    proto_mat = np.stack([prototypes[gid] for gid in proto_ids])
    for gid, embs in usable.items():
        sims = cosine_matrix(np.stack(embs), proto_mat)
        own_col = proto_ids.index(gid)
        same.extend(float(v) for v in sims[:, own_col])
        if len(proto_ids) > 1:
            diff.extend(float(v) for v in np.delete(sims, own_col, axis=1).reshape(-1))

    same_mean = float(np.mean(same)) if same else 0.0
    diff_mean = float(np.mean(diff)) if diff else 0.0
    out.update(
        same_id_proto_cosine=same_mean,
        different_id_proto_cosine=diff_mean,
        separation_margin=same_mean - diff_mean,
        note=(
            "Healthy Re-ID has same-id cosine clearly above different-id cosine; "
            "a small margin means memory/reactivation will be fragile."
        ),
    )
    return out


def recommended_reactivation_gate(
    diagnostics: Dict[str, object],
    default_gate: float = 0.84,
    diff_margin: float = 0.03,
    same_margin: float = 0.02,
) -> float:
    """Choose a conservative lost-track appearance gate from embedding diagnostics.

    The gate should sit above the diff-ID range, but still leave a little headroom
    below the same-ID mean. If the observed gap is too narrow for both constraints,
    prefer avoiding false reactivations and report the squeezed gate in the payload.
    """
    same = diagnostics.get("same_id_proto_cosine")
    diff = diagnostics.get("different_id_proto_cosine")
    gate = float(default_gate)
    if diff is not None:
        gate = max(gate, float(diff) + diff_margin)
    if same is not None:
        upper = float(same) - same_margin
        if upper > 0.0:
            gate = min(gate, upper)
    return float(np.clip(gate, 0.0, 1.0))


def run_mot17_sequence(
    seq_dir: str,
    cache_dir: str,
    output_dir: str,
    stages: Sequence[str] = ("baseline", "memory"),
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    rebuild_cache: bool = False,
    detection_source: str = "yolo",
    det_conf_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    reactivation_app_gate: Optional[float] = None,
) -> Dict:
    """Build/load MOT17 perception cache, run ablations, and report metrics."""
    base_cfg = cfg or OCSIConfig()
    if seed is not None:
        np.random.seed(int(seed))
    os.makedirs(output_dir, exist_ok=True)
    if det_conf_threshold is not None:
        base_cfg = OCSIConfig.from_dict(base_cfg.to_dict())
        base_cfg.perception.det_conf_threshold = float(det_conf_threshold)
    if detection_source not in ("yolo", "public"):
        raise ValueError("detection_source must be 'yolo' or 'public'")

    if rebuild_cache or not os.path.exists(cache_dir):
        if detection_source == "public":
            detections = build_public_detection_cache(
                seq_dir, cache_dir, base_cfg, limit, base_cfg.perception.det_conf_threshold
            )
        else:
            detections = build_perception_cache(seq_dir, cache_dir, base_cfg, limit)
    else:
        detections = load_cached_detections(seq_dir, cache_dir, limit, detection_source)

    gt = read_gt(os.path.join(seq_dir, "gt", "gt.txt"))
    diagnostics = embedding_diagnostics(detections, gt)
    applied_reactivation_app_gate = (
        float(reactivation_app_gate)
        if reactivation_app_gate is not None
        else recommended_reactivation_gate(diagnostics, base_cfg.association.reactivation_app_gate)
    )
    diagnostics["recommended_reactivation_app_gate"] = applied_reactivation_app_gate
    seq_name = _seq_name(seq_dir)
    results: List[MOT17StageResult] = []
    for stage in stages:
        stage_cfg = apply_ablation(base_cfg, stage)
        stage_cfg.association.reactivation_app_gate = applied_reactivation_app_gate
        suffix = f"{detection_source}-{stage}"
        if seed is not None:
            suffix = f"{suffix}-seed{int(seed)}"
        result_path = os.path.join(output_dir, f"{seq_name}-{suffix}.txt")
        rows = track_cached_sequence(detections, stage_cfg, result_path, behaviour_seed=seed)
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
        "detection_source": detection_source,
        "seed": seed,
        "reactivation_app_gate": applied_reactivation_app_gate,
        "embedding_diagnostics": diagnostics,
        "results": [asdict(r) for r in results],
        "note": (
            "MOT17 has no action labels; feedback uses placeholder activity prototypes over "
            "rolling Re-ID embeddings to exercise the behaviour gate plumbing."
        ),
    }
    summary_name = f"{seq_name}-{detection_source}-summary.json"
    if seed is not None:
        summary_name = f"{seq_name}-{detection_source}-seed{int(seed)}-summary.json"
    with open(os.path.join(output_dir, summary_name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def run_mot17_dataset(
    seq_dirs: Sequence[str],
    cache_root: str,
    output_dir: str,
    stages: Sequence[str] = ("baseline", "memory", "feedback"),
    cfg: Optional[OCSIConfig] = None,
    limit: Optional[int] = None,
    rebuild_cache: bool = False,
    detection_source: str = "public",
    seeds: Sequence[Optional[int]] = (None,),
    reactivation_app_gate: Optional[float] = None,
) -> Dict:
    """Run a MOT17 ablation grid across sequences and seeds."""
    runs = []
    for seed in seeds:
        for seq_dir in seq_dirs:
            seq_name = _seq_name(seq_dir)
            cache_dir = os.path.join(cache_root, seq_name)
            runs.append(
                run_mot17_sequence(
                    seq_dir=seq_dir,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    stages=stages,
                    cfg=cfg,
                    limit=limit,
                    rebuild_cache=rebuild_cache,
                    detection_source=detection_source,
                    seed=seed,
                    reactivation_app_gate=reactivation_app_gate,
                )
            )
    payload = {
        "detection_source": detection_source,
        "sequences": [_seq_name(s) for s in seq_dirs],
        "seeds": list(seeds),
        "runs": runs,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"mot17-{detection_source}-dataset-summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Import run_mot17_sequence from a notebook/script and pass seq_dir, cache_dir, output_dir."
    )
