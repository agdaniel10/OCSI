"""Unit tests for the quick CLEAR-MOT + IDF1 metrics (Phase 3.5)."""
import numpy as np

from ocsi.eval.metrics import MOTMetrics, evaluate, rows_to_frames


def box(x, y, w=10.0, h=10.0):
    return np.array([x, y, w, h], dtype=float)


def test_perfect_tracking_is_all_ones():
    gt = {
        1: [(1, box(0, 0)), (2, box(100, 100))],
        2: [(1, box(2, 0)), (2, box(102, 100))],
    }
    m = evaluate(gt, gt, iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn, m.idsw) == (4, 0, 0, 0)
    assert m.mota == 1.0 and m.idf1 == 1.0
    assert m.motp == 1.0 and m.precision == 1.0 and m.recall == 1.0
    assert (m.mostly_tracked, m.mostly_lost) == (2, 0)


def test_false_positive_and_miss():
    gt = {1: [(1, box(0, 0)), (2, box(100, 100))]}
    pred = {1: [(1, box(0, 0)), (9, box(500, 500))]}   # id2 missed; id9 spurious
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn, m.idsw) == (1, 1, 1, 0)
    assert m.mota == 0.0                                # 1 - (1+1+0)/2
    assert (m.mostly_tracked, m.mostly_lost) == (1, 1)


def test_id_switch_counted_once():
    # one gt tracked 3 frames; the predicted id flips on the last frame
    gt = {1: [(1, box(0, 0))], 2: [(1, box(2, 0))], 3: [(1, box(4, 0))]}
    pred = {1: [(1, box(0, 0))], 2: [(1, box(2, 0))], 3: [(2, box(4, 0))]}
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn, m.idsw) == (3, 0, 0, 1)
    assert abs(m.mota - 2.0 / 3.0) < 1e-9
    # IDF1: gt1 best-matches pred1 (2 frames) -> id_tp=2, id_fp=id_fn=1
    assert (m.id_tp, m.id_fp, m.id_fn) == (2, 1, 1)
    assert abs(m.idf1 - 2.0 / 3.0) < 1e-9


def test_consistent_relabel_no_switch_but_full_idf1():
    # a single, *consistent* predicted id that merely differs in label is not
    # penalised by either MOTA (no switch) or IDF1 (one-to-one global match).
    gt = {k: [(1, box(2 * k, 0))] for k in range(1, 5)}
    pred = {k: [(7, box(2 * k, 0))] for k in range(1, 5)}
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert m.idsw == 0 and m.mota == 1.0 and m.idf1 == 1.0


def test_asymmetric_frames_all_fp_and_all_fn():
    gt = {1: [(1, box(0, 0))], 2: []}                  # frame 2 gt empty
    pred = {1: [], 2: [(1, box(0, 0))]}                # frame 1 pred empty
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_mostly_tracked_partial_lost_buckets():
    gt, pred = {}, {}
    for f in range(1, 11):
        gt[f] = [(1, box(0, 0)), (2, box(100, 0)), (3, box(200, 0))]
        row = [(1, box(0, 0))]                          # id1 tracked every frame -> MT
        if f <= 5:
            row.append((2, box(100, 0)))                # id2 tracked 5/10 -> PT
        if f == 1:
            row.append((3, box(200, 0)))                # id3 tracked 1/10 -> ML
        pred[f] = row
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert (m.mostly_tracked, m.partially_tracked, m.mostly_lost) == (1, 1, 1)


def test_below_threshold_iou_is_not_a_match():
    gt = {1: [(1, box(0, 0, 10, 10))]}
    pred = {1: [(1, box(8, 0, 10, 10))]}               # IoU = 2/18 ≈ 0.11 < 0.5
    m = evaluate(gt, pred, iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_rows_to_frames_adapter():
    rows = [(1, 5, 10.0, 20.0, 30.0, 40.0, 0.9), (1, 6, 1.0, 2.0, 3.0, 4.0, 1.0)]
    frames = rows_to_frames(rows)
    assert set(frames) == {1}
    ids = sorted(t for t, _ in frames[1])
    assert ids == [5, 6]
    np.testing.assert_allclose(frames[1][0][1], [10.0, 20.0, 30.0, 40.0])


def test_empty_inputs_are_safe():
    m = evaluate({}, {}, iou_threshold=0.5)
    assert isinstance(m, MOTMetrics)
    assert m.mota == 0.0 and m.idf1 == 0.0 and m.num_frames == 0
