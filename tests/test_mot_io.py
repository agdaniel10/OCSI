"""Unit tests for MOTChallenge-format I/O (Phase 3.5 / 6 plumbing)."""
import os
import types

import numpy as np

from ocsi.eval import (
    format_row,
    frame_index_from_number,
    frame_number_from_index,
    mot_image_files,
    read_gt,
    read_results,
    tracker_rows,
    write_results,
)


def test_format_row_exact():
    assert format_row(1, 7, (10.0, 20.0, 30.0, 40.0), conf=0.9) == \
        "1,7,10.00,20.00,30.00,40.00,0.9000,-1,-1,-1"


def test_write_read_roundtrip(tmp_path):
    rows = [
        (2, 2, 12.0, 20.0, 30.0, 40.0, 0.9),
        (1, 2, 10.0, 20.0, 30.0, 40.0, 0.8),
        (1, 1, 50.0, 60.0, 25.0, 45.0, 1.0),
    ]
    path = str(tmp_path / "res.txt")
    write_results(path, rows)
    back = read_results(path)
    assert back == sorted(rows, key=lambda r: (r[0], r[1]))   # written sorted by (frame, id)


def test_read_results_ignores_blank_and_trailing(tmp_path):
    path = str(tmp_path / "res.txt")
    (tmp_path / "res.txt").write_text("\n1,3,1.00,2.00,3.00,4.00,1.0000,-1,-1,-1\n\n")
    back = read_results(path)
    assert back == [(1, 3, 1.0, 2.0, 3.0, 4.0, 1.0)]


def test_read_gt_filters_class_flag_visibility(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(
        "1,1,10,10,20,40,1,1,1.0\n"    # keep: pedestrian, considered, visible
        "1,2,50,50,20,40,1,7,1.0\n"    # drop: class 7 (not pedestrian)
        "1,3,90,90,20,40,0,1,1.0\n"    # drop: consider flag = 0
        "2,1,12,10,20,40,1,1,0.1\n"    # kept by default, dropped at min_visibility=0.5
    )
    gt = read_gt(str(path), valid_classes=(1,), min_visibility=0.0)
    assert set(gt.keys()) == {1, 2}
    assert len(gt[1]) == 1 and gt[1][0][0] == 1
    np.testing.assert_allclose(gt[1][0][1], [10.0, 10.0, 20.0, 40.0])

    gt_vis = read_gt(str(path), valid_classes=(1,), min_visibility=0.5)
    assert 2 not in gt_vis                                   # visibility 0.1 < 0.5


def test_tracker_rows_are_one_indexed():
    rec = types.SimpleNamespace(track_id=5, last_box=np.array([10.0, 20.0, 30.0, 40.0]))
    rows = tracker_rows(0, [rec])                             # 0-indexed frame -> MOT frame 1
    assert rows == [(1, 5, 10.0, 20.0, 30.0, 40.0, 1.0)]


def test_mot17_frame_helpers_and_image_order(tmp_path):
    img_dir = tmp_path / "MOT17-02-FRCNN" / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000002.jpg").write_text("")
    (img_dir / "000001.jpg").write_text("")
    (img_dir / "notes.txt").write_text("")

    assert frame_number_from_index(0) == 1
    assert frame_index_from_number(1) == 0
    files = mot_image_files(str(tmp_path / "MOT17-02-FRCNN"))
    assert [os.path.basename(p) for p in files] == ["000001.jpg", "000002.jpg"]
