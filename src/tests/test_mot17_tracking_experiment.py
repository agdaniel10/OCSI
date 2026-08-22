"""Tests for the cacheable MOT17 experiment runner."""
import numpy as np

from ocsi.config import OCSIConfig
from ocsi.experiments.mot17_tracking import (
    array_to_detections,
    detections_to_array,
    load_cached_detections,
    run_mot17_sequence,
)
from ocsi.perception.cache import FeatureCache
from ocsi.types import Detection


def _make_seq(tmp_path, n_frames=4):
    seq = tmp_path / "MOT17-02-FRCNN"
    img = seq / "img1"
    gt = seq / "gt"
    img.mkdir(parents=True)
    gt.mkdir()
    for i in range(1, n_frames + 1):
        (img / f"{i:06d}.jpg").write_text("")
    (gt / "gt.txt").write_text(
        "".join(f"{i},1,{10 + i},20,30,40,1,1,1.0\n" for i in range(1, n_frames + 1))
    )
    return seq


def test_detection_cache_roundtrip_preserves_frame_index():
    dets = [
        Detection(
            tlwh=np.array([1.0, 2.0, 3.0, 4.0]),
            confidence=0.8,
            class_id=0,
            frame_idx=7,
            embedding=np.array([0.1, 0.2], dtype=np.float32),
        )
    ]
    arr = detections_to_array(dets)
    out = array_to_detections(arr, frame_idx=7)

    assert arr.shape == (1, 8)
    assert out[0].frame_idx == 7
    np.testing.assert_allclose(out[0].tlwh, dets[0].tlwh)
    np.testing.assert_allclose(out[0].embedding, dets[0].embedding)


def test_load_cached_detections_uses_mot_frame_numbered_keys(tmp_path):
    seq = _make_seq(tmp_path, n_frames=2)
    cache_dir = str(tmp_path / "cache")
    cache = FeatureCache(cache_dir)
    cache.put("MOT17-02-FRCNN/000001", detections_to_array([Detection([1, 2, 3, 4], 0.9)]))
    cache.put("MOT17-02-FRCNN/000002", detections_to_array([Detection([5, 6, 7, 8], 0.8)]))

    dets = load_cached_detections(str(seq), cache_dir)

    assert [frame[0].frame_idx for frame in dets] == [0, 1]
    np.testing.assert_allclose(dets[1][0].tlwh, [5, 6, 7, 8])


def test_run_mot17_sequence_replays_cache_and_disables_behaviour(tmp_path):
    seq = _make_seq(tmp_path, n_frames=4)
    cache_dir = str(tmp_path / "cache")
    cache = FeatureCache(cache_dir)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    for i in range(1, 5):
        det = Detection([10 + i, 20, 30, 40], 0.95, embedding=emb)
        cache.put(f"MOT17-02-FRCNN/{i:06d}", detections_to_array([det]))

    cfg = OCSIConfig()
    cfg.memory.min_hits = 1
    cfg.behaviour.enabled = True
    payload = run_mot17_sequence(
        str(seq),
        cache_dir,
        str(tmp_path / "out"),
        stages=("memory",),
        cfg=cfg,
    )

    assert payload["note"].startswith("MOT17 has no action labels")
    assert payload["results"][0]["stage"] == "memory"
    assert payload["results"][0]["metrics"]["num_gt"] == 4
