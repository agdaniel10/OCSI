"""Tests for the real X-OCSI producers: detector Grad-CAM and HAR saliency."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from ocsi.config import OCSIConfig, apply_ablation  # noqa: E402
from ocsi.explain.cecs import CECSCalculator  # noqa: E402
from ocsi.experiments import mot17_tracking  # noqa: E402
from ocsi.experiments.mot17_tracking import (  # noqa: E402
    attach_explanation_cache,
    detections_to_array,
    explanation_cache_path,
    read_explanation_cache,
    run_mot17_sequence,
    write_explanation_cache,
)
from ocsi.perception.cache import FeatureCache  # noqa: E402
from ocsi.types import Detection, ExplanationInput  # noqa: E402
from ocsi.xai.det_attrib import (  # noqa: E402
    boxes_to_input,
    letterbox_image,
    unletterbox_heatmaps,
)
from ocsi.xai.gradcam import DetectorGradCAM  # noqa: E402
from ocsi.xai.utils import heatmap_bbox_overlap  # noqa: E402
from ocsi.xai.temporal_saliency import (  # noqa: E402
    frame_perturbation_saliency,
    har_score_fn,
)


# ------------------------------------------------------------------ fake detector


class _FakeHead(nn.Module):
    """YOLO-shaped head: feature maps -> (B, 4 + nc, A), via a 1x1 conv."""

    nc = 2

    def __init__(self):
        super().__init__()
        self.cv = nn.Conv2d(8, 4 + self.nc, 1)

    def forward(self, feats):
        out = self.cv(feats[0])
        b, c, h, w = out.shape
        return out.reshape(b, c, h * w), {"raw": out}


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.head = _FakeHead()

    def forward(self, x):
        return self.head([torch.relu(self.stem(x))])


def _frozen_detector(seed=0):
    torch.manual_seed(seed)
    model = _FakeDetector().eval()
    for p in model.parameters():
        p.requires_grad_(False)  # mimic an inference checkpoint
    return model


def test_detector_gradcam_returns_one_map_per_box():
    model = _frozen_detector()
    cam = DetectorGradCAM(model)
    x = torch.rand(1, 3, 32, 32)
    cams = cam.cam_for_boxes(x, [[4, 4, 12, 12], [20, 20, 28, 28]], class_ids=[0, 1])
    assert len(cams) == 2
    for c in cams:
        assert c.shape == (32, 32)
        assert np.isfinite(c).all()
        assert c.min() >= 0.0
        assert c.sum() > 0.0


def test_detector_gradcam_is_per_box_not_per_frame():
    """Non-overlapping boxes must not contaminate each other's attribution."""
    model = _frozen_detector()
    cam = DetectorGradCAM(model)
    x = torch.rand(1, 3, 32, 32)
    a, b = [2, 2, 10, 10], [22, 22, 30, 30]
    joint = cam.cam_for_boxes(x, [a, b])
    only_a = cam.cam_for_boxes(x, [a])[0]
    only_b = cam.cam_for_boxes(x, [b])[0]
    assert np.allclose(joint[0], only_a, atol=1e-5)
    assert np.allclose(joint[1], only_b, atol=1e-5)
    # Each box's backward contribution is confined to its own anchor footprint:
    # a third box elsewhere must not change the first box's map either.
    c = [14, 14, 18, 18]
    joint3 = cam.cam_for_boxes(x, [a, b, c])
    assert np.allclose(joint3[0], only_a, atol=1e-5)


def test_detector_gradcam_restores_requires_grad_flags():
    model = _frozen_detector()
    cam = DetectorGradCAM(model)
    cam.cam_for_boxes(torch.rand(1, 3, 32, 32), [[4, 4, 12, 12]])
    assert all(p.requires_grad is False for p in model.parameters())


def test_detector_gradcam_handles_objectness_layout():
    """YOLOv5-style heads expose 5 + nc channels (objectness at index 4)."""

    class _Head(nn.Module):
        nc = 3

        def __init__(self):
            super().__init__()
            self.cv = nn.Conv2d(4, 5 + self.nc, 1)

        def forward(self, feats):
            out = self.cv(feats[0])
            return out.reshape(1, out.shape[1], -1)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 4, 3, padding=1)
            self.head = _Head()

        def forward(self, x):
            return self.head([torch.relu(self.stem(x))])

    model = _Model().eval()
    cam = DetectorGradCAM(model)
    offset, nc, has_obj = cam._class_layout(torch.zeros(1, 8, 16))
    assert (offset, nc, has_obj) == (5, 3, True)
    cams = cam.cam_for_boxes(torch.rand(1, 3, 16, 16), [[2, 2, 8, 8]])
    assert cams[0].shape == (16, 16)


# -------------------------------------------------------------------- letterbox


def test_letterbox_maps_boxes_and_heatmaps_back():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    chw, scale, pad = letterbox_image(image, size=640)
    assert chw.shape == (3, 640, 640)
    assert scale == pytest.approx(3.2)
    assert pad == (0, 160)

    boxes = boxes_to_input([[10, 20, 50, 40]], scale, pad)
    assert np.allclose(boxes, [[32, 224, 192, 352]])

    cam = np.zeros((640, 640), dtype=float)
    cam[320, 320] = 1.0
    out = unletterbox_heatmaps([cam], scale, pad, (100, 200))[0]
    assert out.shape == (100, 200)
    assert np.unravel_index(int(np.argmax(out)), out.shape) == (50, 100)


def test_detection_heatmaps_downsamples_to_heatmap_size():
    """Cached evidence is downsampled; boxes stay in source pixels."""
    from ocsi.xai.det_attrib import detection_heatmaps

    model = _frozen_detector()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    maps = detection_heatmaps(
        model, image, [[10, 20, 40, 30]], size=32, heatmap_size=(8, 12)
    )
    assert len(maps) == 1
    assert maps[0].shape == (8, 12)
    assert np.isfinite(maps[0]).all()


@pytest.mark.skipif(not os.path.exists("yolov8s.pt"), reason="no local YOLO weights")
def test_detection_heatmaps_with_real_yolo_weights():
    from ocsi.xai.det_attrib import detection_heatmaps

    from ultralytics import YOLO

    model = YOLO("yolov8s.pt").model.eval()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    boxes = [[20, 40, 60, 120], [180, 30, 80, 160]]
    maps = detection_heatmaps(model, image, boxes, class_ids=[0, 0], size=320)
    assert len(maps) == 2
    for heatmap in maps:
        assert heatmap.shape == (240, 320)
        assert np.isfinite(heatmap).all()
        assert heatmap.min() >= 0.0
    # Per-detection attribution: two boxes must not receive the same evidence.
    assert not np.allclose(maps[0], maps[1], atol=1e-4)


# -------------------------------------------------------- temporal perturbation


class _MarkerModel:
    """Confidence is 1 iff the window still contains the marker frame."""

    def __init__(self, marker, calls):
        self.marker = np.asarray(marker, dtype=float)
        self.calls = calls

    def __call__(self, window):
        self.calls.append(1)
        feats = np.asarray(window, dtype=float)
        return 1.0 if np.any(np.all(np.isclose(feats, self.marker), axis=1)) else 0.0


def test_perturbation_saliency_actually_calls_the_model():
    window = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([9.0, 9.0]),  # the only frame the model responds to
        np.array([0.5, 0.5]),
    ]
    calls = []
    model = _MarkerModel([9.0, 9.0], calls)
    sal = frame_perturbation_saliency(model, window)
    assert len(calls) == len(window) + 1        # baseline + one deletion per frame
    assert int(np.argmax(sal)) == 2
    assert sal[2] > 0.9
    assert sal.sum() == pytest.approx(1.0)
    assert (sal >= 0).all()


def test_perturbation_saliency_without_model_is_heuristic():
    window = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    sal = frame_perturbation_saliency(None, window)
    assert sal.sum() == pytest.approx(1.0)


def test_perturbation_saliency_is_uniform_when_nothing_matters():
    class _Constant:
        def __call__(self, window):
            return 0.5

    sal = frame_perturbation_saliency(_Constant(), [np.zeros(3) for _ in range(4)])
    assert sal.sum() == pytest.approx(1.0)


def test_har_score_fn_reports_class_probability():
    from ocsi.behaviour.temporal_har import TemporalHARRecognizer

    recognizer = TemporalHARRecognizer(feature_dim=8, num_classes=3, min_window=2)
    rng = np.random.default_rng(1)
    window = [rng.normal(size=8) for _ in range(4)]
    observation = recognizer.observe(window)
    chosen = int(np.argmax(observation.probs))
    score = har_score_fn(recognizer, class_index=chosen)(window)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(float(observation.probs[chosen]))


# ----------------------------------------------------------------- cache format


def test_explanation_cache_roundtrip(tmp_path):
    heatmap = np.zeros((8, 10))
    heatmap[2:6, 3:7] = 1.0
    dets = [
        Detection([1, 2, 3, 4], 0.9, explanation=ExplanationInput(
            det_heatmap=heatmap,
            saliency=np.array([0.5, 0.3, 0.2]),
            img_shape=(64, 80),
        )),
        Detection([5, 6, 7, 8], 0.8),
    ]
    path = explanation_cache_path(str(tmp_path / "frame.npy"))
    assert write_explanation_cache(path, dets) is True

    loaded = read_explanation_cache(path, len(dets))
    assert loaded[1] is None
    assert np.allclose(loaded[0].det_heatmap, heatmap)
    assert np.allclose(loaded[0].saliency, [0.5, 0.3, 0.2])
    assert tuple(loaded[0].img_shape) == (64, 80)

    target = [Detection([1, 2, 3, 4], 0.9), Detection([5, 6, 7, 8], 0.8)]
    attach_explanation_cache(path, target)
    assert target[0].explanation is not None
    assert target[1].explanation is None


def test_explanation_cache_quantises_heatmaps(tmp_path):
    """Heatmaps are stored as uint8: the cache is per detection, not per frame."""
    heatmap = np.linspace(0.0, 0.5, 64, dtype=np.float32).reshape(8, 8)
    dets = [Detection([0, 0, 4, 4], 0.9, explanation=ExplanationInput(det_heatmap=heatmap))]
    path = explanation_cache_path(str(tmp_path / "frame.npy"))
    assert write_explanation_cache(path, dets) is True

    with np.load(path) as data:
        assert data["det_heatmap_0"].dtype == np.uint8
        assert data["det_heatmap_0"].shape == heatmap.shape

    loaded = read_explanation_cache(path, 1)[0]
    assert loaded.det_heatmap.dtype == np.float64
    assert np.allclose(loaded.det_heatmap, heatmap, atol=1.0 / 255.0 + 1e-6)


def test_explanation_cache_normalises_over_range_heatmaps(tmp_path):
    heatmap = np.array([[0.0, 2.0], [4.0, 0.0]])
    path = explanation_cache_path(str(tmp_path / "frame.npy"))
    write_explanation_cache(
        path, [Detection([0, 0, 1, 1], 0.9, explanation=ExplanationInput(det_heatmap=heatmap))]
    )
    loaded = read_explanation_cache(path, 1)[0]
    assert np.allclose(loaded.det_heatmap, heatmap / 4.0, atol=1.0 / 255.0 + 1e-6)


def test_explanation_cache_skips_frames_without_evidence(tmp_path):
    path = explanation_cache_path(str(tmp_path / "frame.npy"))
    assert write_explanation_cache(path, [Detection([1, 2, 3, 4], 0.9)]) is False
    assert not os.path.exists(path)
    assert read_explanation_cache(path, 3) == [None, None, None]


def test_attach_explanation_cache_keeps_existing_evidence(tmp_path):
    heatmap = np.ones((4, 4))
    path = explanation_cache_path(str(tmp_path / "frame.npy"))
    write_explanation_cache(path, [Detection([0, 0, 2, 2], 0.5, explanation=ExplanationInput(det_heatmap=heatmap))])
    kept = ExplanationInput(det_heatmap=np.zeros((4, 4)))
    det = Detection([0, 0, 2, 2], 0.5, explanation=kept)
    attach_explanation_cache(path, [det])
    assert det.explanation is kept


# ------------------------------------------------------------ CECS spatial rule


def test_spatial_term_uses_energy_in_box_before_box_iou():
    calc = CECSCalculator()
    heatmap = np.zeros((20, 20))
    heatmap[0:5, 0:5] = 1.0                    # evidence concentrated away from the box
    box = np.array([10, 10, 8, 8])
    value = calc.s_spatial(det_heatmap=heatmap, act_bbox=box, img_shape=(20, 20))
    assert value == pytest.approx(heatmap_bbox_overlap(heatmap, box, (20, 20)))
    assert value == pytest.approx(0.0)
    # The box-only fallback is untouched.
    assert calc.s_spatial(det_bbox=box, act_bbox=box) == pytest.approx(1.0)


def test_spatial_term_infers_shape_from_full_frame_map():
    calc = CECSCalculator()
    heatmap = np.zeros((20, 20))
    heatmap[10:18, 10:18] = 1.0
    box = np.array([10, 10, 8, 8])
    assert calc.s_spatial(det_heatmap=heatmap, act_bbox=box) == pytest.approx(1.0)


# ----------------------------------------------------- end to end: MOT17 + CECS


def _make_sequence(tmp_path, n_frames=12, name="MOT17-02-FRCNN"):
    seq = tmp_path / name
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "det").mkdir()
    for i in range(1, n_frames + 1):
        (seq / "img1" / f"{i:06d}.jpg").write_text("")
    (seq / "gt" / "gt.txt").write_text(
        "".join(f"{i},1,{10 + i},20,30,40,1,1,1.0\n" for i in range(1, n_frames + 1))
    )
    (seq / "det" / "det.txt").write_text(
        "".join(f"{i},-1,{10 + i},20,30,40,0.95,-1,-1,-1\n" for i in range(1, n_frames + 1))
    )
    return seq


def test_cached_attribution_reaches_cecs_in_mot17_runner(tmp_path):
    """Full path: cached E_det -> tracker r_t/u_t -> HAR saliency -> CECS."""
    n_frames = 12
    seq = _make_sequence(tmp_path, n_frames=n_frames)
    cache_dir = str(tmp_path / "cache")
    cache = FeatureCache(cache_dir)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    heatmap = np.zeros((1080, 1920))
    heatmap[20:60, 10:50] = 1.0
    for i in range(1, n_frames + 1):
        key = mot17_tracking._cache_key(str(seq), "yolo", i - 1, 0.15)
        det = Detection([10 + i, 20, 30, 40], 0.95, embedding=emb)
        cache.put(key, detections_to_array([det]))
        sidecar = cache.path(key, ".xai.npz")
        write_explanation_cache(
            sidecar,
            [Detection([10 + i, 20, 30, 40], 0.95, explanation=ExplanationInput(
                det_heatmap=heatmap, img_shape=(1080, 1920)
            ))],
        )

    payload = run_mot17_sequence(str(seq), cache_dir, str(tmp_path / "out"), stages=("x-ocsi",))

    metrics = payload["results"][0]["metrics"]
    assert metrics["cecs"]["enabled"] is True
    assert metrics["cecs"]["num_cecs"] > 0
    assert metrics["cecs"]["event_counts"]["detection"] > 0
    assert metrics["cecs"]["event_counts"]["temporal"] > 0
    assert 0.0 <= metrics["cecs"]["cecs_mean"] <= 1.0
    assert payload["attribution"]["coverage"] == pytest.approx(1.0)
    assert metrics["cecs"]["attribution"]["detections_with_det_heatmap"] == n_frames

    history = payload["results"][0]["metrics"]["cecs"]
    assert "history" not in history  # full trace stays in the sidecar
    sidecar = tmp_path / "out" / "MOT17-02-FRCNN-yolo-x-ocsi-cecs.json"
    assert sidecar.exists()


def test_xocsi_run_without_attribution_warns_and_reports_zero(tmp_path):
    """An empty CECS table must be loud, not silently absent."""
    seq = _make_sequence(tmp_path, n_frames=4)
    cache_dir = str(tmp_path / "cache")
    cache = FeatureCache(cache_dir)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    for i in range(1, 5):
        key = mot17_tracking._cache_key(str(seq), "yolo", i - 1, 0.15)
        cache.put(key, detections_to_array([Detection([10 + i, 20, 30, 40], 0.95, embedding=emb)]))

    with pytest.warns(RuntimeWarning, match="CECS unavailable"):
        payload = run_mot17_sequence(str(seq), cache_dir, str(tmp_path / "out"), stages=("x-ocsi",))

    metrics = payload["results"][0]["metrics"]
    assert metrics["cecs"]["enabled"] is True
    assert metrics["cecs"]["num_cecs"] == 0
    assert "no detector attribution" in metrics["cecs"]["warning"]
    assert payload["attribution"]["coverage"] == 0.0


def test_cache_config_requests_attribution_for_xocsi_stages():
    cfg = OCSIConfig()
    assert cfg.xai.enabled is False
    memory_only = mot17_tracking._cache_cfg_for_stages(cfg, ("baseline", "memory"))
    assert memory_only is cfg                      # untouched when nobody needs XAI
    with_xai = mot17_tracking._cache_cfg_for_stages(cfg, ("feedback", "x-ocsi"))
    assert with_xai.xai.enabled is True
    assert cfg.xai.enabled is False                # caller config is not mutated


def test_dataset_calibration_builds_cache_with_attribution(monkeypatch, tmp_path):
    """The calibration pass must not create an attribution-free cache."""
    seq = _make_sequence(tmp_path, n_frames=2)
    seq_name = seq.name
    seen = []

    def fake_build_perception_cache(seq_dir, cache_dir, cfg=None, limit=None):
        seen.append(bool(cfg.xai.enabled))
        cache = FeatureCache(cache_dir)
        det = Detection([10, 20, 30, 40], 0.95, embedding=np.array([1.0, 0.0], dtype=np.float32))
        frames = mot17_tracking.mot_image_files(seq_dir, limit)
        for frame_idx in range(len(frames)):
            key = mot17_tracking._cache_key(
                seq_dir, "yolo", frame_idx, cfg.perception.det_conf_threshold
            )
            cache.put(key, detections_to_array([det]))
        return [[det] for _ in frames]

    monkeypatch.setattr(mot17_tracking, "build_perception_cache", fake_build_perception_cache)

    mot17_tracking.run_mot17_dataset(
        [str(seq)],
        str(tmp_path / "cache"),
        str(tmp_path / "out"),
        stages=("x-ocsi",),
        detection_source="yolo",
        calibration_sequences=[seq_name],
        evaluation_sequences=[seq_name],
    )

    assert seen                                  # the cache was (re)built
    assert all(seen)


def test_build_perception_cache_passes_heatmap_size(monkeypatch):
    """The producer downsamples evidence before it is cached."""
    import ocsi.perception as perception

    captured = []

    class _Detector:
        def __init__(self, cfg):
            pass

        def __call__(self, frame, frame_idx=-1):
            return [Detection([1, 2, 3, 4], 0.9)]

        def explain(self, frame, dets, heatmap_size=None):
            captured.append(heatmap_size)
            return dets

    def fake_attach(seq_dir, cache_dir, fn, cfg, source, limit):
        fn(0, np.zeros((4, 4, 3), dtype=np.uint8))
        return []

    monkeypatch.setattr(perception, "YOLODetector", _Detector)
    monkeypatch.setattr(mot17_tracking, "_attach_embeddings_to_frames", fake_attach)
    cfg = OCSIConfig()
    cfg.xai.enabled = True

    mot17_tracking.build_perception_cache("seq", "cache", cfg)

    assert captured == [tuple(cfg.xai.heatmap_size)]
