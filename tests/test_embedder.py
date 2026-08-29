"""Unit tests for the Re-ID embedder (Phase 1).

Uses a random-initialised backbone (``reid_pretrained=False``) so the tests are
fully offline and deterministic — they check the plumbing (shapes, L2-norm, box
clamping, determinism), not the semantic quality of the features.
"""
import numpy as np
import pytest

from ocsi.config import PerceptionConfig
from ocsi.types import Detection

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from ocsi.perception import ReIDEmbedder  # noqa: E402


def _cfg():
    return PerceptionConfig(
        reid_pretrained=False,
        reid_backend="torchvision",
        reid_backbone="resnet18",
    )


def _frame(h=200, w=320, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_embedding_shape_and_l2_norm():
    emb = ReIDEmbedder(_cfg())
    assert emb.dim == 512
    frame = _frame()
    boxes = [np.array([10.0, 20.0, 40.0, 80.0]), np.array([100.0, 30.0, 50.0, 90.0])]
    feats = emb(frame, boxes)
    assert feats.shape == (2, 512) and feats.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)


def test_empty_boxes_returns_zero_by_dim():
    emb = ReIDEmbedder(_cfg())
    feats = emb(_frame(), [])
    assert feats.shape == (0, 512)


def test_box_outside_image_is_clamped_not_crashed():
    emb = ReIDEmbedder(_cfg())
    frame = _frame(h=100, w=100)
    # box straddling / beyond the image bounds
    feats = emb(frame, [np.array([90.0, 90.0, 60.0, 60.0]), np.array([-20.0, -20.0, 30.0, 30.0])])
    assert feats.shape == (2, 512)
    assert np.isfinite(feats).all()


def test_deterministic_for_same_input():
    emb = ReIDEmbedder(_cfg())
    frame = _frame(seed=3)
    boxes = [np.array([5.0, 5.0, 60.0, 120.0])]
    np.testing.assert_array_equal(emb(frame, boxes), emb(frame, boxes))


def test_batching_matches_single_pass():
    cfg = _cfg()
    cfg.reid_batch_size = 1               # force multiple batches
    emb = ReIDEmbedder(cfg)
    frame = _frame(seed=5)
    boxes = [np.array([float(i * 10), 10.0, 40.0, 80.0]) for i in range(5)]
    batched = emb(frame, boxes)
    singles = np.concatenate([emb(frame, [b]) for b in boxes], axis=0)
    np.testing.assert_allclose(batched, singles, atol=1e-5)


def test_attach_embeddings_sets_detection_embedding():
    emb = ReIDEmbedder(_cfg())
    frame = _frame()
    dets = [Detection(np.array([10.0, 10.0, 40.0, 80.0]), 0.9), Detection(np.array([80.0, 20.0, 30.0, 70.0]), 0.8)]
    out = emb.attach_embeddings(frame, dets)
    for d in out:
        assert d.embedding is not None and d.embedding.shape == (512,)
        np.testing.assert_allclose(np.linalg.norm(d.embedding), 1.0, atol=1e-5)


def test_from_bgr_matches_manual_rgb_conversion():
    import cv2
    emb = ReIDEmbedder(_cfg())
    bgr = _frame(seed=7)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = [np.array([10.0, 10.0, 50.0, 100.0])]
    np.testing.assert_allclose(emb.from_bgr(bgr, boxes), emb(rgb, boxes), atol=1e-6)
