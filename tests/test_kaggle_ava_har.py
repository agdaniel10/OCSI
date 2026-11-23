from pathlib import Path

import numpy as np

from kaggle_ava_har import (
    average_precision,
    build_pseudo_tracklets,
    load_ava_frames,
    make_windows,
    obs_key,
)


def test_load_ava_frames_groups_actions_and_applies_exclusions(tmp_path: Path):
    csv_path = tmp_path / "ava_train.csv"
    csv_path.write_text(
        "vid1,10,0,0,10,10,1,0\n"
        "vid1,10,0,0,10,10,3,0\n"
        "vid1,12,5,5,15,15,2,0\n"
        "vid1,12,5,5,15,15,2,1\n",
        encoding="utf-8",
    )
    frames = load_ava_frames(csv_path, num_classes=3, excluded={("vid1", 12)})
    assert set(frames) == {("vid1", 10.0)}
    obs = frames[("vid1", 10.0)][0]
    assert obs.person_idx == 0
    assert obs.actions.tolist() == [1.0, 0.0, 1.0]


def test_build_pseudo_tracklets_links_overlapping_boxes():
    def obs(ts, box, idx=0):
        actions = np.array([1.0, 0.0], dtype=np.float32)
        return type("O", (), {
            "video_id": "vid1", "ts": ts, "person_idx": idx,
            "box_xyxy": np.array(box, dtype=np.float32), "actions": actions,
        })()

    frames = {
        ("vid1", 10.0): [obs(10.0, [0, 0, 10, 10])],
        ("vid1", 11.0): [obs(11.0, [1, 1, 11, 11])],
        ("vid1", 12.0): [obs(12.0, [50, 50, 60, 60])],
    }
    tracks = build_pseudo_tracklets(frames, "vid1", iou_threshold=0.3)
    assert len(tracks) == 2
    assert len(tracks[0].obs) == 2
    assert len(tracks[1].obs) == 1


def test_average_precision_perfect_ranking():
    scores = np.array([0.9, 0.1, 0.8])
    labels = np.array([1, 0, 1])
    assert average_precision(scores, labels) == 1.0


def test_make_windows_requires_features_and_collects_multilabel_targets():
    actions = np.array([1.0, 1.0], dtype=np.float32)
    o1 = type("O", (), {
        "video_id": "vid1", "ts": 10.0, "person_idx": 0,
        "box_xyxy": np.array([0, 0, 10, 10], dtype=np.float32), "actions": actions,
    })()
    o2 = type("O", (), {
        "video_id": "vid1", "ts": 11.0, "person_idx": 0,
        "box_xyxy": np.array([0, 0, 10, 10], dtype=np.float32), "actions": actions,
    })()
    frames = {("vid1", 10.0): [o1], ("vid1", 11.0): [o2]}
    tracks = build_pseudo_tracklets(frames, "vid1")
    features = {
        "vid1": {
            obs_key(o1): np.ones(4, dtype=np.float32),
            obs_key(o2): np.ones(4, dtype=np.float32),
        }
    }
    windows, targets, metadata = make_windows(
        ["vid1"], frames, {"vid1": tracks}, features,
        window=2, min_window=2, num_classes=3,
    )
    assert len(windows) == 1
    assert windows[0].shape == (2, 4)
    assert targets[0].tolist() == [1.0, 1.0, 0.0]
    assert metadata[0]["length"] == 2
