"""End-to-end pipeline tests: drive the tracker over a synthetic clip and score it.

No models or external data — detections are constructed directly, exercising the
tracker's model-free (IoU + motion) path and the full run -> rows -> metric loop.
"""
import numpy as np

from ocsi.config import OCSIConfig, apply_ablation
from ocsi.pipeline import group_detections, run_and_evaluate, run_sequence
from ocsi.types import Detection


def _constant_velocity_clip(n_frames=20, dx=4.0, size=50.0):
    """Two well-separated objects moving at constant velocity.

    Returns ``(detections_per_frame, gt)`` where gt is the 1-indexed
    ``{frame: [(id, tlwh)]}`` dict the metric consumes.
    """
    dets_per_frame, gt = [], {}
    for k in range(n_frames):
        a = np.array([10 + dx * k, 100.0, size, size])
        b = np.array([400 + dx * k, 100.0, size, size])
        dets_per_frame.append([
            Detection(tlwh=a, confidence=0.9, class_id=0, frame_idx=k),
            Detection(tlwh=b, confidence=0.9, class_id=0, frame_idx=k),
        ])
        gt[k + 1] = [(1, a.copy()), (2, b.copy())]      # tracker frame k -> MOT frame k+1
    return dets_per_frame, gt


def test_group_detections_is_dense_and_one_indexed():
    rows = [
        (1, -1, 0.0, 0.0, 10.0, 10.0, 0.9),
        (3, -1, 5.0, 5.0, 10.0, 10.0, 0.4),             # note: frame 2 has no detections
    ]
    per_frame = group_detections(rows)
    assert len(per_frame) == 3                          # dense up to max frame
    assert len(per_frame[0]) == 1 and per_frame[1] == []
    d = per_frame[0][0]
    assert isinstance(d, Detection) and d.frame_idx == 0
    np.testing.assert_allclose(d.tlwh, [0.0, 0.0, 10.0, 10.0])


def test_group_detections_confidence_filter():
    rows = [(1, -1, 0, 0, 10, 10, 0.2), (1, -1, 50, 50, 10, 10, 0.8)]
    per_frame = group_detections(rows, conf_threshold=0.5)
    assert len(per_frame[0]) == 1 and per_frame[0][0].confidence == 0.8


def test_run_sequence_returns_sorted_one_indexed_rows():
    dets, _ = _constant_velocity_clip(n_frames=6)
    rows = run_sequence(dets, OCSIConfig())
    assert rows == sorted(rows, key=lambda r: (r[0], r[1]))
    assert min(r[0] for r in rows) >= 1                 # MOT frames are 1-indexed
    assert all(len(r) == 7 for r in rows)


def test_end_to_end_constant_velocity_tracks_cleanly():
    dets, gt = _constant_velocity_clip(n_frames=20)
    rows, m = run_and_evaluate(dets, gt, OCSIConfig())

    # min_hits=3 means each object is confirmed from tracker frame 2 (MOT frame 3):
    # 2 frames * 2 objects = 4 warm-up misses, the rest tracked with no errors.
    assert m.num_gt == 40
    assert m.idsw == 0 and m.fp == 0
    assert (m.tp, m.fn) == (36, 4)
    assert abs(m.mota - 0.9) < 1e-9
    assert m.precision == 1.0
    assert m.idf1 > 0.9


def test_ablation_presets_run_and_score():
    """Baseline / memory / feedback presets all execute and return valid metrics."""
    dets, gt = _constant_velocity_clip(n_frames=12)
    for stage in ("baseline", "memory", "feedback"):
        cfg = apply_ablation(OCSIConfig(), stage)
        _, m = run_and_evaluate(dets, gt, cfg)
        assert m.idsw == 0 and m.fp == 0
        assert -1.0 <= m.mota <= 1.0 and 0.0 <= m.idf1 <= 1.0
