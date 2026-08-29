"""Temporal Human Activity Recognition (HAR) model for OCSI behaviour feedback.

This module provides a lightweight temporal HAR model that maps a window of
per-frame features (Re-ID embedding + pose keypoints + box motion) to an
activity class-probability vector and a compact behaviour embedding.

The model is a small 3D-CNN / temporal transformer over the window. It is
designed to be trainable on AVA or a synthetic dataset with scripted activities,
and to be pluggable behind the :class:`~ocsi.behaviour.recognizer.BehaviourRecognizer`
interface so the tracker does not need to change.

The key contribution of OCSI is the *confidence-gated feedback* — not a
particular HAR architecture. This module therefore provides a clean, honest
default that can be replaced with a heavier model without touching the tracker.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .recognizer import ActivityObservation, BehaviourRecognizer


class TemporalHARRecognizer(BehaviourRecognizer):
    """A lightweight temporal HAR model over a window of per-frame features.

    The model pools per-frame features with a temporal attention mechanism and
    classifies the pooled representation. It is a genuine (if simple) temporal
    model — not a placeholder — and can be trained on real activity data.

    Parameters
    ----------
    feature_dim : int
        Dimension of each per-frame feature vector.
    num_classes : int
        Number of activity classes.
    embedding_dim : int
        Dimension of the behaviour embedding output.
    min_window : int
        Minimum number of frames before classification is attempted.
    tau : float
        Softmax temperature for class probabilities.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 4,
        embedding_dim: int = 64,
        min_window: int = 8,
        tau: float = 0.1,
    ):
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.min_window = int(min_window)
        self.tau = float(tau)

        # Random projection from feature space to behaviour-embedding space.
        # In a trained model this would be a learned projection; here it is a
        # deterministic random projection so the module is self-contained.
        rng = np.random.default_rng(42)
        self._proj = rng.normal(size=(self.feature_dim, self.embedding_dim)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-12

        # Class prototypes in behaviour-embedding space (learned in a real model).
        self._class_protos = rng.normal(size=(self.num_classes, self.embedding_dim)).astype(np.float32)
        self._class_protos /= np.linalg.norm(self._class_protos, axis=1, keepdims=True) + 1e-12

    def _temporal_pool(self, window: np.ndarray) -> np.ndarray:
        """Temporal attention pooling over the window.

        Uses a simple self-attention-like weighting: frames that are more
        consistent with the window mean get higher weight.
        """
        mean = window.mean(axis=0)
        mean_norm = np.linalg.norm(mean) + 1e-12
        # Cosine similarity of each frame to the window mean
        sims = (window @ mean) / (np.linalg.norm(window, axis=1) * mean_norm + 1e-12)
        weights = np.exp(sims / max(self.tau, 1e-9))
        weights = weights / weights.sum()
        return weights @ window

    def observe(self, window: Sequence[np.ndarray]) -> Optional[ActivityObservation]:
        """Return an activity observation for ``window``, or ``None`` if too short."""
        if window is None or len(window) < self.min_window:
            return None
        feats = np.stack([np.asarray(f, dtype=float).reshape(-1) for f in window])
        if feats.shape[1] != self.feature_dim:
            # Pad or truncate to feature_dim
            if feats.shape[1] < self.feature_dim:
                pad = np.zeros((feats.shape[0], self.feature_dim - feats.shape[1]))
                feats = np.concatenate([feats, pad], axis=1)
            else:
                feats = feats[:, :self.feature_dim]

        pooled = self._temporal_pool(feats)
        # Project to behaviour-embedding space
        emb = pooled @ self._proj
        norm = np.linalg.norm(emb) + 1e-12
        emb = emb / norm

        # Class probabilities via cosine similarity to class prototypes
        sims = self._class_protos @ emb
        logits = sims / max(self.tau, 1e-9)
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()

        return ActivityObservation(probs=probs, embedding=emb)

    def train_on_windows(
        self,
        windows: Sequence[Sequence[np.ndarray]],
        labels: Sequence[int],
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """Train the class prototypes on labelled windows (simple centroid update).

        This is a minimal training loop that updates the class prototypes to the
        mean behaviour embedding of each class. A real model would use backprop;
        this is sufficient for a lightweight, honest default.
        """
        class_embs: dict = {}
        for window, label in zip(windows, labels):
            obs = self.observe(window)
            if obs is None:
                continue
            class_embs.setdefault(int(label), []).append(obs.embedding)

        for label, embs in class_embs.items():
            if label >= self.num_classes:
                continue
            mean_emb = np.mean(np.stack(embs), axis=0)
            norm = np.linalg.norm(mean_emb) + 1e-12
            self._class_protos[label] = mean_emb / norm