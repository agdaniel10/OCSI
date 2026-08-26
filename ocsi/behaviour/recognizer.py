"""Behaviour Intelligence layer: turn a track's temporal window into an activity
class-probability vector and a compact behaviour embedding (paper §3.2, §3.5).

The paper leaves the HAR *network* '[AUTHOR TO COMPLETE]', so the contribution OCSI
validates is the confidence-gated *feedback* (see :mod:`ocsi.behaviour.gate`), not a
particular classifier. This module therefore provides a clean recognizer interface plus
a light, honest default — a nearest-prototype (nearest-centroid) classifier over a
window's pooled features. It is model-free and unit-testable; a heavier temporal CNN can
be dropped in behind the same interface without touching the tracker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class ActivityObservation:
    """The Behaviour layer's output for one track window: a class-probability vector
    and a compact, L2-normalised behaviour embedding ``b'``."""

    probs: np.ndarray            # (num_classes,), sums to 1
    embedding: np.ndarray        # (embedding_dim,), L2-normalised

    def __post_init__(self) -> None:
        self.probs = np.asarray(self.probs, dtype=float)
        self.embedding = _l2_normalize(self.embedding)

    @property
    def p_max(self) -> float:
        return float(self.probs.max()) if self.probs.size else 0.0

    @property
    def label(self) -> int:
        return int(np.argmax(self.probs)) if self.probs.size else -1


class BehaviourRecognizer(ABC):
    """Maps a temporal window of per-frame features to an :class:`ActivityObservation`."""

    @abstractmethod
    def observe(self, window: Sequence[np.ndarray]) -> Optional[ActivityObservation]:
        """Return an observation for ``window`` (a sequence of per-frame feature
        vectors), or ``None`` if the window is too short to classify."""
        raise NotImplementedError


class PrototypeBehaviourRecognizer(BehaviourRecognizer):
    """Nearest-prototype activity classifier over pooled window features.

    Holds one L2-normalised prototype per activity class. A window is summarised by
    mean-pooling its per-frame features (then L2-normalising) into the behaviour
    embedding ``b'``; class probabilities are a softmax over cosine similarity to the
    class prototypes (temperature ``tau``). A genuine, if simple, classifier — no
    training loop, prototypes are supplied/estimated — suitable for CPU use and for
    isolating the gate as the component under test.
    """

    def __init__(self, prototypes: np.ndarray, min_window: int = 8, tau: float = 0.1):
        proto = np.asarray(prototypes, dtype=float)
        if proto.ndim != 2:
            raise ValueError("prototypes must be (num_classes, embedding_dim)")
        self.prototypes = np.stack([_l2_normalize(p) for p in proto])
        self.num_classes = int(proto.shape[0])
        self.embedding_dim = int(proto.shape[1])
        self.min_window = int(min_window)
        self.tau = float(tau)

    def observe(self, window: Sequence[np.ndarray]) -> Optional[ActivityObservation]:
        if window is None or len(window) < self.min_window:
            return None
        feats = np.stack([np.asarray(f, dtype=float).reshape(-1) for f in window])
        pooled = _l2_normalize(feats.mean(axis=0))
        sims = self.prototypes @ pooled                    # cosine in [-1, 1]
        logits = sims / max(self.tau, 1e-9)
        logits = logits - logits.max()                     # stabilise softmax
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return ActivityObservation(probs=probs, embedding=pooled)
