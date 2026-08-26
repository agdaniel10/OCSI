"""Integration test: perception (Re-ID embedder) -> pipeline -> metric.

Proves the Phase 1 embedder composes with the tracker/pipeline: real embeddings
are attached to detections, flow through the appearance/memory cues, and the clip
is tracked and scored. Uses a random-initialised backbone, so this asserts the
plumbing composes — NOT that the (untrained) features are discriminative.
"""
import numpy as np
import pytest

from ocsi.config import OCSIConfig
from ocsi.pipeline import run_and_evaluate
from ocsi.types import Detection

pytest.importorskip("torch")
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from ocsi.perception import ReIDEmbedder  # noqa: E402


def _render(ax, bx, y=60, w=44, h=84, W=320, H=200):
    frame = np.zeros((H, W, 3), np.uint8)
    cv2.rectangle(frame, (ax, y), (ax + w, y + h), (0, 0, 200), -1)   # object A
    cv2.rectangle(frame, (bx, y), (bx + w, y + h), (200, 0, 0), -1)   # object B
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (w, h, y)


def test_embedder_feeds_pipeline_end_to_end():
    emb = ReIDEmbedder(OCSIConfig().perception.__class__(reid_pretrained=False))
    dets_per_frame, gt = [], {}
    for k in range(8):
        ax, bx = 20 + 8 * k, 220 + 8 * k                 # two well-separated objects
        frame_rgb, (w, h, y) = _render(ax, bx)
        a = np.array([ax, y, w, h], float)
        b = np.array([bx, y, w, h], float)
        dets = [Detection(a, 0.95, 0, k), Detection(b, 0.95, 0, k)]
        emb.attach_embeddings(frame_rgb, dets)
        dets_per_frame.append(dets)
        gt[k + 1] = [(1, a.copy()), (2, b.copy())]

    # every detection came out with a unit-norm embedding of the backbone's dim
    for frame_dets in dets_per_frame:
        for d in frame_dets:
            assert d.embedding is not None and d.embedding.shape == (emb.dim,)
            np.testing.assert_allclose(np.linalg.norm(d.embedding), 1.0, atol=1e-5)

    _, m = run_and_evaluate(dets_per_frame, gt, OCSIConfig())
    # well-separated objects: IoU+motion (with the appearance cue riding along) track cleanly
    assert m.num_gt == 16 and m.idsw == 0 and m.fp == 0
    assert m.precision == 1.0 and m.mota > 0.7
