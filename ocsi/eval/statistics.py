"""Statistical helpers for paper-facing OCSI experiment summaries."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return ``(mean, low, high)`` for a bootstrap confidence interval."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_stats(base: Sequence[float], treatment: Sequence[float]) -> Dict[str, float]:
    """Paired delta, Wilcoxon, and paired t-test after dropping non-finite pairs."""
    a = np.asarray(base, dtype=float)
    b = np.asarray(treatment, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {
            "n": int(len(a)),
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "wilcoxon_p": np.nan,
            "paired_t_p": np.nan,
        }

    d = b - a
    try:
        wilcoxon_p = wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    except Exception:
        wilcoxon_p = np.nan
    try:
        paired_t_p = ttest_rel(b, a).pvalue
    except Exception:
        paired_t_p = np.nan

    return {
        "n": int(len(a)),
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "wilcoxon_p": float(wilcoxon_p) if np.isfinite(wilcoxon_p) else np.nan,
        "paired_t_p": float(paired_t_p) if np.isfinite(paired_t_p) else np.nan,
    }


def choose_threshold_from_similarity(
    same_sims: Iterable[float],
    diff_sims: Iterable[float],
    candidate_thresholds: Sequence[float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Choose a reactivation threshold on calibration similarities only.

    The selected threshold maximizes balanced accuracy between same-ID and
    different-ID similarity samples. The returned row is the winning diagnostic.
    """
    same = np.asarray(list(same_sims), dtype=float)
    diff = np.asarray(list(diff_sims), dtype=float)
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if len(same) == 0 or len(diff) == 0:
        raise ValueError("Need both same-ID and different-ID calibration similarities.")

    thresholds = (
        np.arange(0.75, 0.931, 0.01)
        if candidate_thresholds is None
        else np.asarray(candidate_thresholds, dtype=float)
    )
    best: Dict[str, float] | None = None
    for threshold in thresholds:
        tpr = float(np.mean(same >= threshold))
        tnr = float(np.mean(diff < threshold))
        row = {
            "threshold": float(threshold),
            "TPR_same": tpr,
            "TNR_diff": tnr,
            "balanced_accuracy": 0.5 * (tpr + tnr),
        }
        if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
            best = row

    assert best is not None
    return best["threshold"], best
