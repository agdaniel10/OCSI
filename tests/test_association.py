"""Unit tests for confidence-gated association (Phase 3, identity layer)."""
import numpy as np

from ocsi.identity import solve_assignment, two_stage_associate

ALL = lambda t, d: np.ones((t, d), dtype=bool)  # noqa: E731


def test_solve_simple_2x2():
    score = np.array([[0.9, 0.1], [0.2, 0.8]])
    matches, um_t, um_d = solve_assignment(score, ALL(2, 2), s_min=0.3)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert um_t == [] and um_d == []


def test_reject_below_s_min():
    # Optimal assignment pairs (1,1) at score 0.25 < s_min -> that match is dropped.
    score = np.array([[0.9, 0.1], [0.2, 0.25]])
    matches, um_t, um_d = solve_assignment(score, ALL(2, 2), s_min=0.3)
    assert matches == [(0, 0)]
    assert um_t == [1] and um_d == [1]


def test_gating_changes_assignment():
    # Ungated optimum is the cross (0->1),(1->0); gating out (0,1) forces the diagonal.
    score = np.array([[0.5, 0.95], [0.9, 0.4]])
    valid = np.array([[True, False], [True, True]])
    matches, um_t, um_d = solve_assignment(score, valid, s_min=0.3)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert um_t == [] and um_d == []


def test_more_dets_than_tracks():
    score = np.array([[0.9, 0.1]])
    matches, um_t, um_d = solve_assignment(score, ALL(1, 2), s_min=0.3)
    assert matches == [(0, 0)]
    assert um_t == [] and um_d == [1]


def test_empty_inputs():
    m, um_t, um_d = solve_assignment(np.zeros((0, 3)), np.zeros((0, 3), bool), s_min=0.3)
    assert m == [] and um_t == [] and um_d == [0, 1, 2]

    m, um_t, um_d = solve_assignment(np.zeros((2, 0)), np.zeros((2, 0), bool), s_min=0.3)
    assert m == [] and um_t == [0, 1] and um_d == []


def test_two_stage_high_confidence_first():
    score = np.array([[0.9, 0.4, 0.2], [0.3, 0.35, 0.85]])
    det_conf = [0.9, 0.2, 0.95]           # det1 is low-confidence
    matches, um_t, um_d = two_stage_associate(
        score, ALL(2, 3), det_conf, s_min=0.3, high_conf_threshold=0.5
    )
    assert sorted(matches) == [(0, 0), (1, 2)]
    assert um_t == [] and um_d == [1]     # the low-conf det matched nobody


def test_two_stage_recovers_low_conf_track():
    # track0 grabs the high-conf det in stage 1; track1 is recovered by the
    # low-conf det in stage 2.
    score = np.array([[0.9, 0.1], [0.2, 0.8]])
    det_conf = [0.9, 0.2]
    matches, um_t, um_d = two_stage_associate(
        score, ALL(2, 2), det_conf, s_min=0.3, high_conf_threshold=0.5
    )
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert um_t == [] and um_d == []
