"""Tests for the temporal HAR recognizer."""
import numpy as np

from ocsi.behaviour import TemporalHARRecognizer


def test_observe_returns_none_for_short_window():
    rec = TemporalHARRecognizer(feature_dim=8, min_window=5)
    window = [np.zeros(8) for _ in range(3)]
    assert rec.observe(window) is None


def test_observe_returns_observation_for_long_enough_window():
    rec = TemporalHARRecognizer(feature_dim=8, min_window=5)
    window = [np.ones(8) for _ in range(5)]
    obs = rec.observe(window)
    assert obs is not None
    assert obs.probs.shape == (rec.num_classes,)
    assert np.isclose(obs.probs.sum(), 1.0)
    assert obs.embedding.shape == (rec.embedding_dim,)
    assert np.isclose(np.linalg.norm(obs.embedding), 1.0, atol=1e-5)


def test_train_on_windows_updates_prototypes():
    rec = TemporalHARRecognizer(feature_dim=8, num_classes=2, min_window=3)
    # Class 0: all-ones features
    windows0 = [[np.ones(8) for _ in range(3)] for _ in range(5)]
    # Class 1: alternating features (non-zero, distinct from class 0)
    windows1 = [[np.array([1.0, -1.0] * 4) for _ in range(3)] for _ in range(5)]
    labels = [0] * 5 + [1] * 5
    rec.train_on_windows(windows0 + windows1, labels)

    # After training, class 0 should be closer to all-ones windows
    obs0 = rec.observe([np.ones(8) for _ in range(3)])
    obs1 = rec.observe([np.array([1.0, -1.0] * 4) for _ in range(3)])
    assert obs0 is not None and obs1 is not None
    assert np.argmax(obs0.probs) == 0
    assert np.argmax(obs1.probs) == 1


def test_padding_for_different_feature_dim():
    rec = TemporalHARRecognizer(feature_dim=8, min_window=3)
    # Window with 4-dim features should be padded to 8
    window = [np.ones(4) for _ in range(3)]
    obs = rec.observe(window)
    assert obs is not None
    assert obs.embedding.shape == (rec.embedding_dim,)