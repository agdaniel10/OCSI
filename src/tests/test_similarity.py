"""Unit tests for similarity cues + score combination (Phase 3, identity layer)."""
import numpy as np

from ocsi.identity import (
    cos_to_01,
    cosine_matrix,
    gallery_similarity,
    gate_mask,
    iou_matrix,
    motion_similarity,
    weighted_score,
)


def test_iou_identical_and_disjoint():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    same = np.array([[0.0, 0.0, 10.0, 10.0]])
    disjoint = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert iou_matrix(a, same)[0, 0] == 1.0
    assert iou_matrix(a, disjoint)[0, 0] == 0.0


def test_iou_known_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])       # area 100
    b = np.array([[5.0, 0.0, 15.0, 10.0]])       # area 100, overlap 5x10 = 50
    np.testing.assert_allclose(iou_matrix(a, b)[0, 0], 50.0 / 150.0)


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((2, 4))).shape == (0, 2)


def test_cosine_matrix_values():
    A = np.array([[1.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(cosine_matrix(A, B)[0], [1.0, 0.0, -1.0], atol=1e-9)


def test_cos_to_01():
    np.testing.assert_allclose(cos_to_01(np.array([-1.0, 0.0, 1.0])), [0.0, 0.5, 1.0])


def test_gallery_similarity_best_and_empty():
    galleries = [np.array([[1.0, 0.0], [0.0, 1.0]]), None]
    dets = np.array([[1.0, 0.0]])
    out = gallery_similarity(galleries, dets)
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out[0, 0], 1.0)   # best of {cos=1, cos=0}
    assert out[1, 0] == -1.0                      # empty gallery -> no evidence


def test_motion_similarity():
    d2 = np.array([0.0, 9.4877])
    s = motion_similarity(d2)
    assert s[0] == 1.0
    assert 0.0 < s[1] < 0.05                       # exp(-4.74) ~ 0.0087


def test_weighted_score_renormalizes_active_weights():
    ones = np.ones((2, 2))
    terms = {"app": ones, "motion": ones}
    weights = {"app": 0.3, "motion": 0.1, "iou": 0.2}   # only app+motion present
    # (0.3 + 0.1) renormalised to 1 over all-ones -> all ones
    np.testing.assert_allclose(weighted_score(terms, weights), ones)


def test_weighted_score_mix():
    terms = {"app": np.zeros((1, 1)), "motion": np.ones((1, 1))}
    weights = {"app": 0.3, "motion": 0.1}
    # 0.3/0.4 * 0 + 0.1/0.4 * 1 = 0.25
    np.testing.assert_allclose(weighted_score(terms, weights), [[0.25]])


def test_weighted_score_drops_zero_weight_cue():
    A = np.full((1, 1), 0.7)
    terms = {"app": A, "mem": np.zeros((1, 1))}
    weights = {"app": 0.3, "mem": 0.0}             # mem inactive -> ignored
    np.testing.assert_allclose(weighted_score(terms, weights), A)


def test_gate_mask_iou_floor_and_maha():
    iou = np.array([[0.5, 0.05]])
    d2 = np.array([[1.0, 1.0]])
    valid = gate_mask(iou, d2, iou_gate=0.1, maha_gate=9.4877)
    np.testing.assert_array_equal(valid, [[True, False]])   # col1 fails IoU floor

    d2_far = np.array([[100.0, 1.0]])
    valid2 = gate_mask(iou, d2_far, iou_gate=0.0, maha_gate=9.4877)
    np.testing.assert_array_equal(valid2, [[False, True]])  # col0 fails Maha gate


def test_gate_mask_class_mismatch():
    iou = np.array([[0.5, 0.5]])
    d2 = np.array([[1.0, 1.0]])
    valid = gate_mask(
        iou, d2, iou_gate=0.1, maha_gate=9.4877,
        track_classes=[0], det_classes=[0, 1],
    )
    np.testing.assert_array_equal(valid, [[True, False]])   # col1 is a different class
