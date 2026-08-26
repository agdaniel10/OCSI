"""Tests for the cacheable MOT17 experiment runner."""
import numpy as np

from ocsi.config import OCSIConfig
from ocsi.experiments.mot17_tracking import (
    array_to_detections,
    detections_to_array,
    embedding_diagnostics,
    load_cached_detections,
    mot17_public_detections,
    recommended_reactivation_gate,
    run_mot17_dataset,
    run_mot17_sequence,
)
from ocsi.perception.cache import FeatureCache
from ocsi.types import Detection


def _make_seq(tmp_path, n_frames=4, name="MOT17-02-FRCNN"):
    seq = tmp_path / name
    img = seq / "img1"
    gt = seq / "gt"
    img.mkdir(parents=True)
    gt.mkdir()
    det = seq / "det"
    det.mkdir()
    for i in range(1, n_frames + 1):
        (img / f"{i:06d}.jpg").write_text("")
    (gt / "gt.txt").write_text(
        "".join(f"{i},1,{10 + i},20,30,40,1,1,1.0\n" for i in range(1, n_frames + 1))
    )
    (det / "det.txt").write_text(
        "".join(f"{i},-1,{10 + i},20,30,40,0.95,-1,-1,-1\n" for i in range(1, n_frames + 1))
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


def test_embedding_diagnostics_reports_identity_separation():
    detections = [
        [
            Detection([10, 20, 30, 40], 0.95, embedding=np.array([1.0, 0.0])),
            Detection([100, 20, 30, 40], 0.95, embedding=np.array([0.0, 1.0])),
        ],
        [
            Detection([11, 20, 30, 40], 0.95, embedding=np.array([0.99, 0.01])),
            Detection([101, 20, 30, 40], 0.95, embedding=np.array([0.01, 0.99])),
        ],
    ]
    gt = {
        1: [
            (1, np.array([10, 20, 30, 40], dtype=float)),
            (2, np.array([100, 20, 30, 40], dtype=float)),
        ],
        2: [
            (1, np.array([11, 20, 30, 40], dtype=float)),
            (2, np.array([101, 20, 30, 40], dtype=float)),
        ],
    }

    diag = embedding_diagnostics(detections, gt)

    assert diag["total_embeddings"] == 4
    assert diag["assigned_embeddings"] == 4
    assert diag["num_identity_prototypes"] == 2
    assert diag["embedding_dim"] == 2
    assert diag["same_id_proto_cosine"] > 0.99
    assert diag["different_id_proto_cosine"] < 0.02
    assert diag["separation_margin"] > 0.97


def test_recommended_reactivation_gate_stays_above_diff_id_range():
    gate = recommended_reactivation_gate(
        {
            "same_id_proto_cosine": 0.92,
            "different_id_proto_cosine": 0.80,
        }
    )

    assert 0.83 <= gate <= 0.90


def test_load_cached_detections_uses_mot_frame_numbered_keys(tmp_path):
    seq = _make_seq(tmp_path, n_frames=2)
    cache_dir = str(tmp_path / "cache")
    cache = FeatureCache(cache_dir)
    cache.put("MOT17-02-FRCNN/yolo/000001", detections_to_array([Detection([1, 2, 3, 4], 0.9)]))
    cache.put("MOT17-02-FRCNN/yolo/000002", detections_to_array([Detection([5, 6, 7, 8], 0.8)]))

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
        cache.put(f"MOT17-02-FRCNN/yolo/{i:06d}", detections_to_array([det]))

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
    assert payload["detection_source"] == "yolo"
    assert payload["reactivation_app_gate"] == 0.84
    assert payload["embedding_diagnostics"]["recommended_reactivation_app_gate"] == 0.84
    assert payload["results"][0]["stage"] == "memory"
    assert payload["results"][0]["metrics"]["num_gt"] == 4


def test_mot17_public_detections_reads_det_txt_and_filters_confidence(tmp_path):
    seq = _make_seq(tmp_path, n_frames=3)
    (seq / "det" / "det.txt").write_text(
        "1,-1,10,20,30,40,0.95,-1,-1,-1\n"
        "2,-1,11,20,30,40,0.10,-1,-1,-1\n"
        "3,-1,12,20,30,40,0.80,-1,-1,-1\n"
    )

    dets = mot17_public_detections(str(seq), conf_threshold=0.5)

    assert len(dets) == 3
    assert [len(frame) for frame in dets] == [1, 0, 1]
    assert dets[2][0].frame_idx == 2


def test_run_mot17_dataset_scales_sequences_and_seeds_from_public_cache(tmp_path):
    seq_a = _make_seq(tmp_path, n_frames=2, name="MOT17-02-FRCNN")
    seq_b = _make_seq(tmp_path, n_frames=2, name="MOT17-04-FRCNN")
    cache_root = tmp_path / "cache"
    emb = np.array([1.0, 0.0], dtype=np.float32)
    for seq in (seq_a, seq_b):
        cache = FeatureCache(str(cache_root / seq.name))
        for i in range(1, 3):
            det = Detection([10 + i, 20, 30, 40], 0.95, embedding=emb)
            cache.put(f"{seq.name}/public/{i:06d}", detections_to_array([det]))

    payload = run_mot17_dataset(
        [str(seq_a), str(seq_b)],
        str(cache_root),
        str(tmp_path / "out"),
        stages=("baseline", "memory", "feedback"),
        detection_source="public",
        seeds=(1, 2),
    )

    assert payload["detection_source"] == "public"
    assert len(payload["runs"]) == 4
    assert {r["seed"] for r in payload["runs"]} == {1, 2}
    assert all([x["stage"] for x in r["results"]] == ["baseline", "memory", "feedback"] for r in payload["runs"])
