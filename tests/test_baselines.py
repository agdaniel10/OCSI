"""Tests for the controlled baseline trackers."""
import numpy as np

from ocsi.config import OCSIConfig
from ocsi.experiments.baselines import ByteTrackBaseline, DeepSORTBaseline, OCSORTBaseline
from ocsi.types import Detection


def _det(x, y=200.0, w=40.0, h=80.0, conf=0.9, emb=None, frame=0):
    return Detection(
        tlwh=np.array([x, y, w, h], dtype=float),
        confidence=conf,
        class_id=0,
        frame_idx=frame,
        embedding=None if emb is None else np.asarray(emb, dtype=float),
    )


def test_deepsort_tracks_single_object():
    cfg = OCSIConfig()
    cfg.memory.min_hits = 1
    tr = DeepSORTBaseline(cfg)
    ids = []
    for t in range(5):
        out = tr.update([_det(100 + 5 * t, frame=t)])
        ids.append([r.track_id for r in out])
    assert len(ids[-1]) == 1
    assert len(tr.tracks) == 1


def test_bytetrack_tracks_single_object():
    cfg = OCSIConfig()
    cfg.memory.min_hits = 1
    tr = ByteTrackBaseline(cfg)
    ids = []
    for t in range(5):
        out = tr.update([_det(100 + 5 * t, frame=t)])
        ids.append([r.track_id for r in out])
    assert len(ids[-1]) == 1
    assert len(tr.tracks) == 1


def test_ocsort_tracks_single_object():
    cfg = OCSIConfig()
    cfg.memory.min_hits = 1
    tr = OCSORTBaseline(cfg)
    ids = []
    for t in range(5):
        out = tr.update([_det(100 + 5 * t, frame=t)])
        ids.append([r.track_id for r in out])
    assert len(ids[-1]) == 1
    assert len(tr.tracks) == 1


def test_baselines_keep_distinct_ids_for_two_objects():
    cfg = OCSIConfig()
    cfg.memory.min_hits = 1
    for cls in (DeepSORTBaseline, ByteTrackBaseline, OCSORTBaseline):
        tr = cls(cfg)
        for t in range(5):
            tr.update([_det(100 + 3 * t, frame=t), _det(600 + 3 * t, frame=t)])
        out = tr.update([_det(112, frame=5), _det(615, frame=5)])
        ids = sorted(r.track_id for r in out)
        assert len(ids) == 2 and ids[0] != ids[1], f"{cls.__name__} failed to keep distinct ids"