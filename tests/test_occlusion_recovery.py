"""Tests for the occlusion-recovery experiment and the tracker's reactivation path.

These are model-free: they inject hand-crafted discriminative embeddings instead
of running the CNN, so they are fast and deterministic while still exercising the
real reactivation logic that makes the memory bank recover identity after a gap.
"""
import numpy as np

from ocsi.config import OCSIConfig, apply_ablation
from ocsi.eval.metrics import rows_to_frames
from ocsi.experiments.occlusion_recovery import (
    build_scene,
    recovery_by_length,
    LENGTHS,
)
from ocsi.identity import OCSITracker
from ocsi.pipeline import run_sequence
from ocsi.types import Detection


def _basis(i: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


# ------------------------------------------------------- tracker reactivation
def _drive(stage: str):
    """Establish a track, hide it for 6 frames with a displaced reappearance, and
    return (original_id, ids_after_reappearance)."""
    cfg = apply_ablation(OCSIConfig(), stage)
    cfg.association.reactivation_app_gate = 0.5
    tr = OCSITracker(cfg)
    emb = _basis(0)

    x = 100.0
    orig = None
    for k in range(3):                               # 3 hits -> CONFIRMED
        d = Detection(np.array([x, 100, 36, 72], float), 0.9, 0, k)
        d.embedding = emb.copy()
        out = tr.update([d])
        x += 5
    orig = out[0].track_id

    for k in range(3, 9):                            # occluded: no detections
        tr.update([])

    x = 360.0                                        # reappear far away (no IoU overlap)
    last = []
    for k in range(9, 12):
        d = Detection(np.array([x, 100, 36, 72], float), 0.9, 0, k)
        d.embedding = emb.copy()
        last = tr.update([d])
        x += 5
    return orig, [r.track_id for r in last]


def test_memory_reactivates_same_identity_after_occlusion():
    orig, after = _drive("memory")
    assert orig in after, "memory stage should restore the pre-occlusion id"


def test_baseline_cannot_recover_displaced_reappearance():
    orig, after = _drive("baseline")
    assert after and orig not in after, "baseline should assign a NEW id (no reactivation)"


# ------------------------------------------------------------- scene builder
def test_build_scene_structure():
    scene = build_scene(n_ids=3)
    T = len(scene.frames)
    assert len(scene.detections) == T and T > 0
    assert len(scene.events) == len(LENGTHS) * 6
    # every emitted detection has a matching gt entry that frame (same order)
    for k, dets in enumerate(scene.detections):
        assert len(dets) == len(scene.gt.get(k + 1, []))
    # during an occlusion the target contributes no detection
    e = scene.events[0]
    gids_mid = {gid for gid, _ in scene.gt.get(e.start + 1, [])}
    assert (e.gid + 1) not in gids_mid


# ----------------------------------------------------- end-to-end (fake feats)
def _attach_fake_features(scene, dim: int = 16):
    for k, dets in enumerate(scene.detections):
        for d, (gid, _) in zip(dets, scene.gt.get(k + 1, [])):
            d.embedding = _basis(gid - 1, dim)       # exact per-identity basis vector


def _mid_length_recovery(scene, stage: str) -> float:
    cfg = apply_ablation(OCSIConfig(), stage)
    cfg.association.reactivation_app_gate = 0.5
    rows = run_sequence(scene.detections, cfg)
    rec = recovery_by_length(scene, rows_to_frames(rows))
    r = t = 0
    for L, (rr, tt) in rec.items():
        if 4 <= L <= 24:                             # gaps too long for motion, within max_age
            r += rr
            t += tt
    return r / t if t else 0.0


def test_memory_beats_baseline_on_recovery():
    scene = build_scene(n_ids=3)
    _attach_fake_features(scene)
    baseline = _mid_length_recovery(scene, "baseline")
    memory = _mid_length_recovery(scene, "memory")
    assert baseline < 0.2, f"baseline recovery unexpectedly high: {baseline}"
    assert memory > 0.9, f"memory recovery unexpectedly low: {memory}"
