"""Confidence-gated behaviour feedback — the core of paper contribution #3 (§3.5, §4.2).

The Behaviour Intelligence layer produces, per track window, an activity class-
probability vector and a compact behaviour embedding. Behaviour never replaces
appearance or motion; it acts as a *gated* semantic cue that refines association only
when the activity evidence is reliable, so uncertain HAR cannot corrupt identity — the
whole point of the gate.

    Similarity:  S_beh = cos(b, b')        (track prototype b vs. window embedding b')
    Gate:        g = 1[p_max >= theta_b] * p_max * exp(-dt / tau_b)
    Blend:       S = (S0 + w_b * g * S_beh) / (1 + w_b * g)

with ``p_max`` the max activity probability of the current window and ``dt`` the number
of frames since the track last had a reliable behaviour observation. When ``g = 0``
(activity confidence below threshold, or evidence gone stale) the blend collapses to
``S0`` exactly — behaviour has *no* effect — which is precisely the "limit error
amplification from uncertain HAR" guarantee the paper claims for the gate.

Everything here is pure-numpy and model-free, so it is unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np


def behaviour_gate(
    p_max: "np.ndarray | float",
    dt: "np.ndarray | float",
    theta_b: float,
    tau_b: float,
    gate_enabled: bool = True,
) -> np.ndarray:
    """The confidence gate ``g = 1[p_max >= theta_b] * p_max * exp(-dt / tau_b)``.

    ``p_max`` and ``dt`` broadcast against each other, so passing a detection-indexed
    row ``(1, D)`` of window confidences and a track-indexed column ``(T, 1)`` of
    staleness yields the full ``(T, D)`` gate matrix. With ``gate_enabled=False`` the
    gate is bypassed (``g = 1`` everywhere) — the naive full-trust behaviour used as
    the experiment's 'ungated' strawman. Returns a float array in ``[0, 1]``.
    """
    p_max = np.asarray(p_max, dtype=float)
    dt = np.asarray(dt, dtype=float)
    shape = np.broadcast_shapes(p_max.shape, dt.shape)
    if not gate_enabled:
        return np.ones(shape, dtype=float)
    indicator = (p_max >= theta_b).astype(float)
    decay = np.exp(-np.clip(dt, 0.0, None) / max(float(tau_b), 1e-9))
    g = np.broadcast_to(indicator * p_max * decay, shape)
    return np.clip(g, 0.0, 1.0).astype(float)


def blend_behaviour(
    base_score: np.ndarray,
    s_beh: np.ndarray,
    g: np.ndarray,
    w_b: float,
) -> np.ndarray:
    """Fold the gated behaviour cue into the base association score, per cell.

    ``S = (S0 + w_b * g * S_beh) / (1 + w_b * g)`` — a per-cell convex combination of
    the base cues (collective weight 1, already renormalised by ``weighted_score``) and
    behaviour (weight ``w_b * g``). Where ``g = 0`` this returns ``S0`` unchanged; with
    all inputs in ``[0, 1]`` the result stays in ``[0, 1]``.
    """
    base_score = np.asarray(base_score, dtype=float)
    s_beh = np.asarray(s_beh, dtype=float)
    wg = float(w_b) * np.asarray(g, dtype=float)
    return (base_score + wg * s_beh) / (1.0 + wg)


def contradiction_mask(
    cos_bt: np.ndarray,
    reliable: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """``(T, D)`` bool mask: True where a *reliable* behaviour cue contradicts the pair.

    A pair is a behaviour contradiction when the activity evidence is reliable (the gate
    is open, ``g > 0``) yet the window embedding is strongly *anti*-correlated with the
    track's behaviour prototype (``cos(b, b') < threshold``). Such pairs are removed from
    the candidate set (paper §3.5 'contradiction rejection'). Where the cue is unreliable
    the mask is False — the gate protects here too, so uncertain HAR cannot veto a match.
    """
    cos_bt = np.asarray(cos_bt, dtype=float)
    reliable = np.asarray(reliable, dtype=bool)
    return reliable & (cos_bt < float(threshold))
