"""Tests for paper-facing statistical helpers."""
import math

import pytest

from ocsi.eval.statistics import (
    bootstrap_mean_ci,
    choose_threshold_from_similarity,
    paired_stats,
)


def test_bootstrap_mean_ci_ignores_non_finite_values():
    mean, lo, hi = bootstrap_mean_ci([1.0, 2.0, float("nan"), 3.0], n_boot=200, seed=1)

    assert mean == 2.0
    assert lo <= mean <= hi


def test_paired_stats_reports_delta_after_filtering():
    stats = paired_stats([1.0, 2.0, float("nan"), 4.0], [2.0, 4.0, 9.0, 7.0])

    assert stats["n"] == 3
    assert stats["mean_delta"] == pytest.approx(2.0)
    assert stats["median_delta"] == pytest.approx(2.0)
    assert 0.0 <= stats["wilcoxon_p"] <= 1.0
    assert 0.0 <= stats["paired_t_p"] <= 1.0


def test_paired_stats_needs_at_least_two_pairs():
    stats = paired_stats([1.0], [2.0])

    assert stats["n"] == 1
    assert math.isnan(stats["mean_delta"])
    assert math.isnan(stats["wilcoxon_p"])


def test_choose_threshold_from_similarity_uses_calibration_samples():
    threshold, row = choose_threshold_from_similarity(
        same_sims=[0.86, 0.90, 0.92],
        diff_sims=[0.50, 0.60, 0.70],
        candidate_thresholds=[0.75, 0.85, 0.95],
    )

    assert threshold == 0.75
    assert row["balanced_accuracy"] == 1.0


def test_choose_threshold_from_similarity_requires_both_classes():
    with pytest.raises(ValueError):
        choose_threshold_from_similarity([], [0.5, 0.6])
