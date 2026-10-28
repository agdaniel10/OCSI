"""Temporal saliency for identity-conditioned HAR (X-OCSI Eq. 7).

Saliency is defined by *perturbation*: replace frame ``t`` of the HAR window with
the window mean and measure how much the model's confidence in its chosen
activity drops. That requires actually calling the model, which the previous
implementation never did (it returned a distance-to-mean heuristic regardless of
``model_forward``). The heuristic is retained only for callers that pass
``model_forward=None``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def uniform_saliency(window_len: int) -> np.ndarray:
    if window_len <= 0:
        return np.array([])
    return np.ones(window_len, dtype=float) / float(window_len)


def har_score_fn(
    recognizer,
    class_index: Optional[int] = None,
    reduce: str = "max",
) -> Callable[[Sequence[np.ndarray]], float]:
    """Build a ``model_forward`` for :func:`frame_perturbation_saliency`.

    The returned callable maps a window of per-frame features to a scalar
    confidence: the probability of ``class_index`` (or of the window's own argmax
    class when ``None``) as produced by a
    :class:`~ocsi.behaviour.recognizer.BehaviourRecognizer`.
    """

    def _score(window: Sequence[np.ndarray]) -> float:
        observation = recognizer.observe(list(window))
        if observation is None:
            return 0.0
        probs = np.asarray(observation.probs, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if class_index is not None:
            idx = int(class_index)
            return float(probs[idx]) if 0 <= idx < probs.size else 0.0
        if reduce == "sum":
            return float(probs.sum())
        return float(probs.max())

    return _score


def _heuristic_saliency(feats: np.ndarray) -> np.ndarray:
    """Distance-to-window-mean heuristic used when no model is available."""
    baseline = feats.mean(axis=0)
    sal = np.array(
        [1.0 / (1.0 + float(np.linalg.norm(f - baseline))) for f in feats], dtype=float
    )
    total = sal.sum()
    return sal / total if total > 1e-12 else uniform_saliency(len(feats))


def frame_perturbation_saliency(
    model_forward: Optional[Callable[[Sequence[np.ndarray]], float]],
    window_features: Sequence[np.ndarray],
    baseline: str = "mean",
    normalize: str = "sum",
    eps: float = 1e-12,
) -> np.ndarray:
    """Model-based temporal saliency over a HAR window.

    Parameters
    ----------
    model_forward:
        Callable mapping a window of per-frame features to a scalar confidence
        (see :func:`har_score_fn`). When ``None``, the distance heuristic is used.
    window_features:
        ``L`` per-frame feature vectors.
    baseline:
        ``"mean"`` replaces the frame with the window mean (deletion), ``"zero"``
        with a zero vector.
    normalize:
        ``"sum"`` (default) makes ``s_t`` weights summing to one, matching Eqs.
        (9)-(10); ``"max"`` scales the peak to one.

    Returns
    -------
    np.ndarray
        ``(L,)`` non-negative saliency, normalised as requested.
    """
    feats = np.stack(
        [np.asarray(f, dtype=float).reshape(-1) for f in window_features]
    ) if len(window_features) > 0 else np.zeros((0, 0), dtype=float)
    if feats.size == 0:
        return np.array([])
    L = len(feats)
    if model_forward is None:
        sal = _heuristic_saliency(feats)
    else:
        base_score = float(model_forward(feats))
        ref = feats.mean(axis=0) if baseline == "mean" else np.zeros(feats.shape[1])
        sal = np.zeros(L, dtype=float)
        for t in range(L):
            perturbed = feats.copy()
            perturbed[t] = ref
            sal[t] = base_score - float(model_forward(perturbed))
        sal = np.clip(sal, 0.0, None)
        if sal.sum() <= eps:
            # No frame mattered: fall back to the model's own attention signal so
            # the identity term is not silently driven by a degenerate window.
            sal = _heuristic_saliency(feats)
            return sal

    if normalize == "max":
        peak = float(sal.max())
        return sal / peak if peak > eps else uniform_saliency(L)
    total = float(sal.sum())
    return sal / total if total > eps else uniform_saliency(L)


def gradient_temporal_saliency(grads: np.ndarray) -> np.ndarray:
    g = np.asarray(grads, dtype=float)
    if g.ndim == 0:
        return np.array([])
    if g.ndim == 1:
        s = np.abs(g)
    else:
        s = np.abs(g).mean(axis=tuple(range(1, g.ndim)))
    s_sum = s.sum()
    if s_sum > 1e-12:
        s = s / s_sum
    else:
        s = uniform_saliency(len(s))
    return s
