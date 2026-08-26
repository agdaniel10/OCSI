"""Real occlusion-recovery experiment — the honest replacement for the source
notebook's *fabricated* recovery numbers (paper §6.3 Recovery metric; §6.4 item 23
sanctions "synthetic or annotated occlusion levels").

The original Colab notebook reported OCSI-vs-baseline occlusion recovery from two
hardcoded formulas (``max(70, 98 - gap*0.12)`` and ``max(10, 85 - gap*0.48)``) that
its own comments admit are "not dynamically influenced by the trained model". This
module MEASURES the curve instead:

  * K identities with distinct textures move across a canvas;
  * each is hidden for a controlled number of frames L and *maneuvers while hidden*
    (changes heading), so a constant-velocity Kalman prediction drifts — motion
    alone cannot recover it, which is precisely the "long occlusion + abrupt
    motion" failure mode OCSI targets;
  * REAL pretrained Re-ID features (torchvision resnet18) feed the REAL OCSI
    tracker; we measure how often the pre-occlusion track id is restored after the
    gap (identity recovery), stratified by L, for the `baseline` vs `+memory`
    ablations.

Everything computed is real: real CNN features, real Kalman/Hungarian association,
the real Object Memory Bank, real CLEAR-MOT/IDF1 metrics. Only the imagery is
synthetic, and it is fully deterministic (fixed seed, no wall-clock/RNG globals).

Run::

    python -m ocsi.experiments.occlusion_recovery
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import OCSIConfig, apply_ablation
from ..eval.metrics import MOTMetrics, evaluate, rows_to_frames
from ..identity.similarity import iou_matrix
from ..pipeline import run_sequence
from ..types import Detection

# ----------------------------------------------------------------- scene spec
CANVAS_W, CANVAS_H = 640, 360
BOX_W, BOX_H = 36, 72
SPEED = 6.0                       # px/frame — tuned so a multi-frame gap + maneuver
#                                   moves the box off its straight-line prediction
LENGTHS = (1, 2, 4, 8, 16, 24, 40)   # occlusion lengths to sweep (40 > max_age=30: the
#                                       retention cliff, where even memory cannot recover)
N_REP = 6                         # occlusion events per length
SEED = 20260821


# --------------------------------------------------------------- appearance
def _make_texture(rng: np.random.Generator, idx: int) -> np.ndarray:
    """A distinct, mildly-textured BOX_H x BOX_W x3 uint8 sprite per identity."""
    # widely-separated base hues so different identities are genuinely distinguishable
    bases = np.array([
        [220, 60, 60], [60, 200, 90], [70, 110, 230], [230, 190, 50],
        [200, 70, 210], [60, 200, 210], [240, 130, 40], [150, 150, 150],
    ], dtype=float)
    base = bases[idx % len(bases)]
    img = np.tile(base, (BOX_H, BOX_W, 1))
    yy, xx = np.mgrid[0:BOX_H, 0:BOX_W]
    kind = idx % 3
    if kind == 0:                                   # horizontal stripes
        img *= (0.6 + 0.4 * ((yy // 6) % 2))[..., None]
    elif kind == 1:                                 # vertical stripes
        img *= (0.6 + 0.4 * ((xx // 6) % 2))[..., None]
    else:                                           # checker
        img *= (0.6 + 0.4 * (((yy // 8) + (xx // 8)) % 2))[..., None]
    img += rng.normal(0, 6, img.shape)              # light per-identity grain
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class Identity:
    gid: int
    texture: np.ndarray
    pos: np.ndarray                                 # top-left (x, y), float
    vel: np.ndarray                                 # (vx, vy), float


@dataclass
class OcclusionEvent:
    gid: int
    start: int                                      # first hidden frame (0-indexed)
    length: int

    @property
    def before(self) -> int:                        # last visible frame pre-gap
        return self.start - 1

    @property
    def after(self) -> int:                         # first visible frame post-gap
        return self.start + self.length


# ------------------------------------------------------------- scene builder
def _schedule(n_ids: int) -> Tuple[List[OcclusionEvent], int]:
    """Lay non-overlapping occlusion events per identity with enough visible frames
    before/after each (so tracks confirm and can be scored). Deterministic."""
    lengths: List[int] = [L for L in LENGTHS for _ in range(N_REP)]
    # round-robin lengths onto identities
    per_id: List[List[int]] = [[] for _ in range(n_ids)]
    for i, L in enumerate(lengths):
        per_id[i % n_ids].append(L)

    warmup, gap_visible = 12, 10
    events: List[OcclusionEvent] = []
    max_end = 0
    for gid, Ls in enumerate(per_id):
        t = warmup
        for L in Ls:
            events.append(OcclusionEvent(gid=gid, start=t, length=L))
            t += L + gap_visible
        max_end = max(max_end, t)
    return events, max_end + warmup


def _trajectories(n_ids: int, events: List[OcclusionEvent], T: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Per-frame top-left positions, shape (T, n_ids, 2). Velocity flips at each of
    the identity's occlusion starts (an *unobserved* maneuver while hidden), and
    bounces off the canvas walls."""
    starts: Dict[int, set] = {g: set() for g in range(n_ids)}
    for e in events:
        starts[e.gid].add(e.start)

    ids: List[Identity] = []
    for g in range(n_ids):
        angle = 2 * np.pi * g / n_ids + 0.3
        vel = SPEED * np.array([np.cos(angle), np.sin(angle)])
        pos = np.array([
            80 + (CANVAS_W - 200) * ((g + 0.5) / n_ids),
            60 + (CANVAS_H - 160) * ((g * 0.37) % 1.0),
        ])
        ids.append(Identity(g, np.empty(0), pos.astype(float), vel.astype(float)))

    out = np.zeros((T, n_ids, 2))
    for k in range(T):
        for g, idn in enumerate(ids):
            if k in starts[g]:
                # maneuver while hidden: rotate heading ~150-210 deg (near reversal)
                theta = np.pi + rng.uniform(-0.5, 0.5)
                c, s = np.cos(theta), np.sin(theta)
                idn.vel = np.array([c * idn.vel[0] - s * idn.vel[1],
                                    s * idn.vel[0] + c * idn.vel[1]])
            idn.pos = idn.pos + idn.vel
            # bounce inside [0, W-BOX_W] x [0, H-BOX_H]
            for ax, hi in ((0, CANVAS_W - BOX_W), (1, CANVAS_H - BOX_H)):
                if idn.pos[ax] < 0:
                    idn.pos[ax] = -idn.pos[ax]; idn.vel[ax] *= -1
                elif idn.pos[ax] > hi:
                    idn.pos[ax] = 2 * hi - idn.pos[ax]; idn.vel[ax] *= -1
            out[k, g] = idn.pos
    return out


def _hidden_at(events: List[OcclusionEvent], gid: int, k: int) -> bool:
    return any(e.gid == gid and e.start <= k < e.start + e.length for e in events)


@dataclass
class Scene:
    frames: List[np.ndarray]                        # RGB uint8 (H, W, 3)
    detections: List[List[Detection]]               # 0-indexed, occluded ids omitted
    gt: Dict[int, List[Tuple[int, np.ndarray]]]     # 1-indexed {frame: [(gid, tlwh)]}
    events: List[OcclusionEvent]
    n_ids: int


def build_scene(n_ids: int = 4, rng: Optional[np.random.Generator] = None) -> Scene:
    rng = rng or np.random.default_rng(SEED)
    textures = [_make_texture(rng, g) for g in range(n_ids)]
    events, T = _schedule(n_ids)
    traj = _trajectories(n_ids, events, T, rng)

    frames: List[np.ndarray] = []
    detections: List[List[Detection]] = []
    gt: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for k in range(T):
        frame = np.full((CANVAS_H, CANVAS_W, 3), 30, np.uint8)   # dark background
        dets: List[Detection] = []
        gt_items: List[Tuple[int, np.ndarray]] = []
        for g in range(n_ids):
            if _hidden_at(events, g, k):
                continue                                          # occluded: no sprite, no det
            x, y = traj[k, g]
            xi, yi = int(round(x)), int(round(y))
            # paste sprite (clipped to canvas)
            sx0, sy0 = max(0, xi), max(0, yi)
            sx1, sy1 = min(CANVAS_W, xi + BOX_W), min(CANVAS_H, yi + BOX_H)
            if sx1 > sx0 and sy1 > sy0:
                tex = textures[g][sy0 - yi:sy1 - yi, sx0 - xi:sx1 - xi]
                frame[sy0:sy1, sx0:sx1] = tex
            box = np.array([x, y, BOX_W, BOX_H], float)
            gt_items.append((g + 1, box.copy()))                  # gt ids are 1-indexed
            jitter = rng.normal(0, 0.6, 4)                        # sub-pixel det noise
            dets.append(Detection(tlwh=box + jitter, confidence=0.92,
                                   class_id=0, frame_idx=k))
        frames.append(frame)
        detections.append(dets)
        if gt_items:
            gt[k + 1] = gt_items                                  # evaluate() is 1-indexed
    return Scene(frames, detections, gt, events, n_ids)


# --------------------------------------------------------- recovery metric
def _assigned_pred_id(gt_box: np.ndarray, pred_items: List[Tuple[int, np.ndarray]],
                      iou_min: float = 0.5) -> Optional[int]:
    """Pred track id whose box best overlaps ``gt_box`` (IoU >= iou_min), else None."""
    if not pred_items:
        return None
    boxes = np.stack([b for _, b in pred_items])
    gt_xyxy = np.array([[gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]])
    pr_xyxy = boxes.copy()
    pr_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    pr_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    iou = iou_matrix(gt_xyxy, pr_xyxy)[0]
    j = int(np.argmax(iou))
    return int(pred_items[j][0]) if iou[j] >= iou_min else None


def recovery_by_length(scene: Scene, pred_frames) -> Dict[int, Tuple[int, int]]:
    """For each occlusion length L, count (recovered, total) events. An event is
    'recovered' when the pred id on the last visible frame before the gap equals the
    pred id on the first visible frame after it (same identity restored)."""
    stats: Dict[int, List[int]] = {L: [0, 0] for L in LENGTHS}
    for e in scene.events:
        gt_before = dict(scene.gt.get(e.before + 1, []))
        gt_after = dict(scene.gt.get(e.after + 1, []))
        gid = e.gid + 1
        if gid not in gt_before or gid not in gt_after:
            continue                                              # event not fully observable
        id_before = _assigned_pred_id(gt_before[gid], pred_frames.get(e.before + 1, []))
        id_after = _assigned_pred_id(gt_after[gid], pred_frames.get(e.after + 1, []))
        ok = id_before is not None and id_after is not None and id_before == id_after
        stats[e.length][1] += 1
        stats[e.length][0] += int(ok)
    return {L: (r, t) for L, (r, t) in stats.items()}


# ------------------------------------------------------- feature diagnostics
def feature_separation(scene: Scene) -> Tuple[float, float]:
    """Mean same-identity vs mean different-identity cosine over the attached
    embeddings — the evidence that appearance can drive reactivation at all."""
    per_id: Dict[int, List[np.ndarray]] = {}
    for k, dets in enumerate(scene.detections):
        for d, (gid, _) in zip(dets, scene.gt.get(k + 1, [])):
            if d.embedding is not None:
                per_id.setdefault(gid, []).append(d.embedding)
    protos = {g: np.mean(v, 0) / (np.linalg.norm(np.mean(v, 0)) + 1e-9)
              for g, v in per_id.items() if v}
    same, diff = [], []
    for g, embs in per_id.items():
        for e in embs:
            for h, p in protos.items():
                (same if h == g else diff).append(float(e @ p))
    return float(np.mean(same)) if same else 0.0, float(np.mean(diff)) if diff else 0.0


# ----------------------------------------------------------------- runner
@dataclass
class StageResult:
    stage: str
    metrics: MOTMetrics
    recovery: Dict[int, Tuple[int, int]] = field(default_factory=dict)


def run(n_ids: int = 4, gate: Optional[float] = None, save_json: Optional[str] = None) -> Dict:
    """Build the scene, attach REAL embeddings once, run each ablation, report."""
    try:
        from ..perception import ReIDEmbedder
    except Exception as exc:                                       # pragma: no cover
        raise SystemExit(f"perception extra required (torch/torchvision/cv2): {exc}")

    print(f"building scene: {n_ids} identities, lengths={LENGTHS}, {N_REP} events each ...")
    scene = build_scene(n_ids)
    T = len(scene.frames)
    print(f"  {T} frames, {len(scene.events)} occlusion events")

    base_cfg = OCSIConfig()
    embedder = ReIDEmbedder(base_cfg.perception.__class__(reid_pretrained=True))
    print(f"attaching REAL pretrained Re-ID embeddings (dim={embedder.dim}) ...")
    for k in range(T):
        embedder.attach_embeddings(scene.frames[k], scene.detections[k])

    same, diff = feature_separation(scene)
    if gate is None:
        gate = round(0.5 * (same + diff), 3)                       # midpoint separator
    print(f"feature cosine: same-id={same:+.3f}  diff-id={diff:+.3f}  "
          f"-> reactivation gate={gate:+.3f}")

    results: List[StageResult] = []
    for stage in ("baseline", "memory"):
        cfg = apply_ablation(base_cfg, stage)
        cfg.association.reactivation_app_gate = gate
        rows = run_sequence(scene.detections, cfg)
        pred_frames = rows_to_frames(rows)
        metrics = evaluate(scene.gt, pred_frames)
        rec = recovery_by_length(scene, pred_frames)
        results.append(StageResult(stage, metrics, rec))

    _print_report(results)
    payload = {
        "config": {"n_ids": n_ids, "lengths": list(LENGTHS), "n_rep": N_REP,
                   "reactivation_gate": gate, "max_age": base_cfg.memory.max_age,
                   "feature_cos_same": same, "feature_cos_diff": diff},
        "stages": {r.stage: {"metrics": r.metrics.as_dict(),
                             "recovery": {str(L): r.recovery[L] for L in LENGTHS}}
                   for r in results},
    }
    if save_json:
        with open(save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nsaved: {save_json}")
    return payload


def _print_report(results: List[StageResult]) -> None:
    print("\n=== Identity recovery rate by occlusion length ===")
    header = "  L(frames) | " + " | ".join(f"{r.stage:>10}" for r in results)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for L in LENGTHS:
        cells = []
        for r in results:
            rec, tot = r.recovery[L]
            cells.append(f"{(rec / tot if tot else 0):>6.0%} ({rec}/{tot})".rjust(10))
        note = "  <- > max_age (retention cliff)" if L > 30 else ""
        print(f"  {L:>9} | " + " | ".join(cells) + note)

    print("\n=== Aggregate tracking metrics ===")
    for r in results:
        print(f"  {r.stage:>10}: {r.metrics.summary()}")


if __name__ == "__main__":                                         # pragma: no cover
    run(save_json="occlusion_recovery_results.json")
