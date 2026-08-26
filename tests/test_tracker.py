"""Integration tests for the OCSI tracker loop (Phase 3), using synthetic
detections — no detector/Re-ID models required.
"""
import numpy as np

from ocsi.config import OCSIConfig, apply_ablation
from ocsi.identity import OCSITracker
from ocsi.types import Detection, TrackState


def det(x, y=200.0, w=40.0, h=80.0, conf=0.9, emb=None, cls=0, frame=0):
    return Detection(
        tlwh=np.array([x, y, w, h], dtype=float),
        confidence=conf,
        class_id=cls,
        frame_idx=frame,
        embedding=None if emb is None else np.asarray(emb, dtype=float),
    )


def test_single_object_gets_stable_id():
    tr = OCSITracker(OCSIConfig())              # min_hits=3
    ids = []
    for t in range(6):
        out = tr.update([det(100 + 5 * t, frame=t)])
        ids.append([r.track_id for r in out])
    assert ids[0] == [] and ids[1] == []        # tentative -> not reported yet
    assert len(ids[2]) == 1                       # confirmed at hit 3
    conf_id = ids[2][0]
    for t in range(2, 6):
        assert ids[t] == [conf_id]                # one identity, held across frames
    assert len(tr.bank) == 1


def test_two_far_apart_objects_keep_distinct_ids():
    tr = OCSITracker(OCSIConfig())
    for t in range(5):
        tr.update([det(100 + 3 * t, frame=t), det(600 + 3 * t, frame=t)])
    out = tr.outputs()
    ids = sorted(r.track_id for r in out)
    assert len(ids) == 2 and ids[0] != ids[1]     # no cross-matching / id swap
    assert len(tr.bank) == 2


def test_lost_track_reactivates_with_same_id():
    tr = OCSITracker(OCSIConfig())                # max_age=30
    id_before = None
    for t in range(5):                             # present & stationary -> confirmed
        out = tr.update([det(100, frame=t)])
        if out:
            id_before = out[0].track_id
    assert id_before is not None

    for t in range(5, 8):                          # occluded: no detections
        tr.update([])
    assert len(tr.bank) == 1
    rec = next(iter(tr.bank.records.values()))
    assert rec.state == TrackState.LOST            # retained, not deleted

    out = tr.update([det(100, frame=8)])           # reappears at the same place
    assert len(out) == 1
    assert out[0].track_id == id_before            # reactivated, not a new identity
    assert len(tr.bank) == 1


def test_memory_populated_when_embeddings_present():
    tr = OCSITracker(OCSIConfig())
    e = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for t in range(4):
        tr.update([det(100, emb=e, frame=t)])
    recs = list(tr.bank.records.values())
    assert len(recs) == 1
    rec = recs[0]
    assert rec.state == TrackState.CONFIRMED
    assert rec.a_bar is not None                   # EMA prototype learned
    assert len(rec.gallery) >= 1                    # reliable views stored


def test_baseline_ablation_still_tracks():
    cfg = apply_ablation(OCSIConfig(), "baseline")  # use_memory=False (no S_mem term)
    assert cfg.association.use_memory is False
    tr = OCSITracker(cfg)
    e = np.array([1.0, 0.0, 0.0])
    ids = []
    for t in range(5):
        out = tr.update([det(100 + 3 * t, emb=e, frame=t)])
        ids.append([r.track_id for r in out])
    assert len(ids[4]) == 1                          # tracks fine without the memory cue
    assert len(tr.bank) == 1
