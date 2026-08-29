"""Tests for contamination-rollback integration in the tracker."""
import numpy as np

from ocsi.config import OCSIConfig
from ocsi.identity import OCSITracker
from ocsi.types import Detection


def _det(x, y=200.0, w=40.0, h=80.0, conf=0.9, emb=None, frame=0):
    return Detection(
        tlwh=np.array([x, y, w, h], dtype=float),
        confidence=conf,
        class_id=0,
        frame_idx=frame,
        embedding=None if emb is None else np.asarray(emb, dtype=float),
    )


def test_contamination_detection_flags_wrong_identity():
    """A track with high confidence that gets matched to a very different
    appearance should be flagged as a contamination."""
    cfg = OCSIConfig()
    cfg.contamination.enabled = True
    cfg.contamination.appearance_conflict_threshold = 0.30
    cfg.contamination.confidence_min = 0.0  # allow low-confidence tracks too
    cfg.memory.min_hits = 1
    tr = OCSITracker(cfg)

    # Track identity A (embedding [1,0,0])
    emb_a = np.array([1.0, 0.0, 0.0])
    for t in range(3):
        tr.update([_det(100, emb=emb_a, frame=t)])

    # Now feed a very different identity B (embedding [0,1,0])
    emb_b = np.array([0.0, 1.0, 0.0])
    tr.update([_det(100, emb=emb_b, frame=3)])

    assert tr.contamination_flags >= 1


def test_contamination_rollback_restores_prototype():
    """After enough consecutive conflicts, the track's prototype should be
    rolled back to its pre-contamination state."""
    cfg = OCSIConfig()
    cfg.contamination.enabled = True
    cfg.contamination.appearance_conflict_threshold = 0.30
    cfg.contamination.confidence_min = 0.0
    cfg.contamination.confirm_frames = 2
    cfg.memory.min_hits = 1
    tr = OCSITracker(cfg)

    # Track identity A
    emb_a = np.array([1.0, 0.0, 0.0])
    for t in range(3):
        tr.update([_det(100, emb=emb_a, frame=t)])

    rec = next(iter(tr.bank.records.values()))
    proto_before = rec.a_bar.copy()

    # Feed identity B twice (2 consecutive conflicts -> rollback)
    emb_b = np.array([0.0, 1.0, 0.0])
    tr.update([_det(100, emb=emb_b, frame=3)])
    tr.update([_det(100, emb=emb_b, frame=4)])

    assert tr.rollbacks_applied >= 1
    # After rollback, the prototype should be closer to identity A
    cos_after = float(np.dot(rec.a_bar, emb_a))
    assert cos_after > 0.9


def test_contamination_disabled_no_flags():
    """When contamination detection is disabled, no flags should be raised."""
    cfg = OCSIConfig()
    cfg.contamination.enabled = False
    cfg.memory.min_hits = 1
    tr = OCSITracker(cfg)

    emb_a = np.array([1.0, 0.0, 0.0])
    for t in range(3):
        tr.update([_det(100, emb=emb_a, frame=t)])

    emb_b = np.array([0.0, 1.0, 0.0])
    tr.update([_det(100, emb=emb_b, frame=3)])

    assert tr.contamination_flags == 0
    assert tr.rollbacks_applied == 0