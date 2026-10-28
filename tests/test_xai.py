"""Tests for X-OCSI cross-stage explainability: XAI utilities, CECS, provenance."""
from pathlib import Path

import numpy as np
import pytest

from ocsi.config import OCSIConfig, apply_ablation
from ocsi.explain import CECSCalculator, ExplanationRecorder
from ocsi.identity import OCSITracker
from ocsi.pipeline import track_sequence
from ocsi.types import Detection, ExplanationInput
from ocsi.xai import (
    bbox_iou,
    heatmap_bbox_overlap,
    mask_bbox,
    normalize_heatmap,
    resize_heatmap,
)
from ocsi.xai.det_attrib import feature_activation_cam
from ocsi.xai.temporal_saliency import (
    frame_perturbation_saliency,
    gradient_temporal_saliency,
    uniform_saliency,
)

# --------------------------------------------------------------------- utils


def test_normalize_heatmap_scales_to_unit_range():
    hm = np.array([[1.0, 2.0], [3.0, 5.0]])
    out = normalize_heatmap(hm)
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_heatmap_handles_constant_map():
    np.testing.assert_allclose(normalize_heatmap(np.ones((3, 3))), np.zeros((3, 3)))


def test_resize_heatmap_preserves_corner_values():
    hm = np.arange(16, dtype=float).reshape(4, 4)
    out = resize_heatmap(hm, (8, 8))
    assert out.shape == (8, 8)
    assert out[0, 0] == hm[0, 0] and out[-1, -1] == hm[-1, -1]


def test_bbox_iou_known_values():
    a = np.array([0, 0, 10, 10])
    assert bbox_iou(a, a) == pytest.approx(1.0)
    assert bbox_iou(a, np.array([20, 20, 10, 10])) == 0.0
    assert bbox_iou(a, np.array([5, 0, 10, 10])) == pytest.approx(50 / 150)


def test_heatmap_bbox_overlap_and_mask():
    hm = np.zeros((10, 10))
    hm[0:5, 0:5] = 1.0
    assert heatmap_bbox_overlap(hm, np.array([0, 0, 5, 5]), (10, 10)) == pytest.approx(1.0)
    assert heatmap_bbox_overlap(hm, np.array([5, 5, 5, 5]), (10, 10)) == 0.0
    mask = mask_bbox(hm, np.array([0, 0, 5, 5]), (10, 10))
    assert mask.sum() == 25


def test_feature_activation_cam_averages_channels():
    fm = np.zeros((2, 4, 4))
    fm[0, 1:3, 1:3] = 1.0
    fm[1, 0:2, 0:2] = 1.0
    cam = feature_activation_cam(fm)
    assert cam.shape == (4, 4)
    assert cam.min() == 0.0 and cam.max() == 1.0


def test_temporal_saliency_helpers_normalise():
    assert uniform_saliency(4).sum() == pytest.approx(1.0)
    grads = np.array([1.0, -3.0, 2.0, 0.0])
    assert gradient_temporal_saliency(grads).sum() == pytest.approx(1.0)
    feats = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    sal = frame_perturbation_saliency(None, feats)
    assert len(sal) == 3 and sal.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------- CECS


def test_cecs_weights_are_normalised():
    calc = CECSCalculator(w_s=2.0, w_t=1.0, w_i=1.0)
    assert (calc.w_s, calc.w_t, calc.w_i) == pytest.approx((0.5, 0.25, 0.25))


def test_cecs_spatial_prefers_heatmaps_over_boxes():
    calc = CECSCalculator()
    a = np.zeros((8, 8))
    a[0:4, 0:4] = 1.0
    b = np.zeros((8, 8))
    b[4:8, 4:8] = 1.0

    # Heatmark agreement wins even when the boxes disagree.
    assert calc.s_spatial(
        det_heatmap=a, act_heatmap=a, det_bbox=[0, 0, 10, 10], act_bbox=[100, 100, 10, 10]
    ) == pytest.approx(1.0)
    assert calc.s_spatial(det_heatmap=a, act_heatmap=b) == pytest.approx(0.0)
    # Box fallback is unchanged for box-only callers.
    assert calc.s_spatial(det_bbox=[0, 0, 10, 10], act_bbox=[0, 0, 10, 10]) == pytest.approx(1.0)


def test_cecs_temporal_and_identity_terms():
    calc = CECSCalculator()
    saliency = [0.1, 0.2, 0.3, 0.4]
    reliability = [1.0, 1.0, 1.0, 0.0]
    uncertainty = [0.0, 0.0, 0.0, 1.0]
    assert calc.s_temporal(saliency, reliability) == pytest.approx(0.6)
    assert calc.s_identity(saliency, uncertainty) == pytest.approx(0.6)


def test_cecs_compute_is_bounded_and_weighted():
    calc = CECSCalculator(w_s=0.5, w_t=0.25, w_i=0.25)
    hm = np.zeros((4, 4))
    hm[1:3, 1:3] = 1.0
    result = calc.compute(
        det_heatmap=hm,
        act_heatmap=hm,
        saliency=[0.5, 0.5],
        reliability=[1.0, 1.0],
        uncertainty=[0.0, 0.0],
    )
    assert result.S_spatial == pytest.approx(1.0)
    assert result.S_temporal == pytest.approx(1.0)
    assert result.S_identity == pytest.approx(1.0)
    assert result.CECS == pytest.approx(1.0)


# --------------------------------------------------------------- provenance


def _evidence(with_reliability: bool = True) -> ExplanationInput:
    det_hm = np.zeros((16, 16))
    det_hm[4:10, 4:10] = 1.0
    act_hm = np.zeros((16, 16))
    act_hm[5:11, 5:11] = 1.0
    return ExplanationInput(
        det_heatmap=det_hm,
        act_heatmap=act_hm,
        saliency=np.array([0.1, 0.2, 0.3, 0.4]),
        reliability=np.array([0.9, 0.9, 0.8, 1.0]) if with_reliability else None,
        uncertainty=np.array([0.1, 0.1, 0.2, 0.0]) if with_reliability else None,
        img_shape=(64, 64),
    )


def _det(x: float, frame: int, explanation=None) -> Detection:
    return Detection(
        tlwh=np.array([x, 100.0, 40.0, 80.0]),
        confidence=0.9,
        frame_idx=frame,
        explanation=explanation,
    )


def test_explanation_recorder_summary_is_serialisable():
    recorder = ExplanationRecorder(OCSIConfig().xai)
    summary = recorder.summary()
    assert summary["num_cecs"] == 0 and summary["cecs_mean"] is None
    assert summary["event_counts"] == {"association": 0, "detection": 0, "temporal": 0}


def test_xocsi_tracker_records_full_explanation_chain():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    assert cfg.xai.enabled
    tracker = OCSITracker(cfg)
    for t in range(5):
        tracker.update([_det(100 + 3 * t, t, _evidence())])

    assert len(tracker.bank) == 1
    rec = next(iter(tracker.bank.records.values()))
    kinds = {event.kind for event in rec.explanation_history}
    assert {"association", "detection", "temporal"} <= kinds
    # Frame 0 creates the identity; frames 1-4 are accepted matches.
    assert len(rec.get_explanations("association")) == 4
    assert rec.last_detection_attribution is not None
    assert rec.last_temporal_saliency is not None
    assert rec.last_cecs is not None
    assert 0.0 <= rec.last_cecs.CECS <= 1.0
    assert len(tracker.cecs_history) == 5

    rationale = rec.get_explanations("association")[-1]
    assert "iou" in rationale.metadata["cues"]
    assert set(rationale.metadata["weights"]) == set(rationale.metadata["cues"])
    assert sum(rationale.metadata["weights"].values()) == pytest.approx(1.0)
    assert 0.0 <= rationale.metadata["reliability"] <= 1.0


def test_xocsi_tracker_without_evidence_records_association_only():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    tracker = OCSITracker(cfg)
    for t in range(4):
        tracker.update([_det(100 + 2 * t, t)])

    rec = next(iter(tracker.bank.records.values()))
    assert len(rec.get_explanations("association")) >= 1
    assert rec.last_detection_attribution is None
    assert rec.last_temporal_saliency is None
    assert tracker.cecs_history == []


def test_partial_evidence_does_not_fabricate_cecs():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    # Saliency only: no detector attribution, so the spatial stage never fired
    # and CECS must not be reported even though the temporal stage is present.
    tracker = OCSITracker(cfg)
    for t in range(4):
        tracker.update([_det(100 + 2 * t, t, ExplanationInput(saliency=np.array([0.4, 0.3, 0.2, 0.1])))])

    rec = next(iter(tracker.bank.records.values()))
    assert rec.last_detection_attribution is None
    assert rec.last_temporal_saliency is not None
    assert rec.last_cecs is None
    assert tracker.cecs_history == []

    # Detector attribution only: no HAR saliency, so no temporal stage.
    tracker = OCSITracker(cfg)
    heatmap = np.zeros((16, 16))
    heatmap[4:10, 4:10] = 1.0
    for t in range(4):
        tracker.update([_det(100 + 2 * t, t, ExplanationInput(det_heatmap=heatmap))])

    rec = next(iter(tracker.bank.records.values()))
    assert rec.last_temporal_saliency is None
    assert rec.last_cecs is None
    assert tracker.cecs_history == []


def test_tracker_supplies_reliability_and_uncertainty_terms():
    """r_t/u_t belong to the tracker, not the perception layer."""
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    tracker = OCSITracker(cfg)
    for t in range(4):
        tracker.update([_det(100 + 2 * t, t, _evidence(with_reliability=False))])

    rec = next(iter(tracker.bank.records.values()))
    temporal = rec.last_temporal_saliency
    assert temporal.metadata["reliability"] is not None
    assert temporal.metadata["uncertainty"] is not None
    assert len(temporal.metadata["reliability"]) == 4
    assert len(temporal.metadata["uncertainty"]) == 4
    assert rec.last_cecs is not None
    assert 0.0 <= rec.last_cecs.CECS <= 1.0


def test_identity_uncertainty_tracks_assignment_ambiguity():
    ambiguous = np.array([[0.90, 0.10], [0.80, 0.20]])
    # Column gap 0.1 dominates the row gap 0.8 -> high identity uncertainty.
    assert OCSITracker._assignment_uncertainty(ambiguous, 0, 0) == pytest.approx(0.9)
    # A single candidate track is never ambiguous, however many detections exist.
    assert OCSITracker._assignment_uncertainty(ambiguous[:1], 0, 0) == 0.0
    # Motion/class-gated competitors must not count as competition.
    valid = np.array([[True, False], [False, True]])
    assert OCSITracker._assignment_uncertainty(ambiguous, 0, 0, valid) == 0.0


def test_association_rationale_includes_gated_behaviour_cue():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    tracker = OCSITracker(cfg)
    for t in range(4):
        det = _det(100 + 2 * t, t)
        det.activity_probs = np.array([0.9, 0.1])
        det.behaviour_embedding = np.array([1.0, 0.0])
        tracker.update([det])

    rec = next(iter(tracker.bank.records.values()))
    event = rec.get_explanations("association")[-1]
    assert "behaviour" in event.metadata["cues"]
    assert event.metadata["weights"]["behaviour"] > 0.0
    assert event.metadata["behaviour_gate"] > 0.0
    assert sum(event.metadata["weights"].values()) == pytest.approx(1.0)


def test_explanation_history_is_per_identity():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    tracker = OCSITracker(cfg)
    for t in range(4):
        tracker.update([
            _det(100 + 2 * t, t, _evidence()),
            _det(500 + 2 * t, t, _evidence()),
        ])

    recs = list(tracker.bank.records.values())
    assert len(recs) == 2
    assert all(len(rec.get_explanations("association")) >= 1 for rec in recs)
    assert {row["track_id"] for row in tracker.cecs_history} == {rec.track_id for rec in recs}


def test_xai_disabled_records_nothing():
    tracker = OCSITracker(OCSIConfig())
    tracker.update([_det(100, 0, _evidence())])
    rec = next(iter(tracker.bank.records.values()))
    assert len(rec.explanation_history) == 0 and rec.last_cecs is None
    assert tracker.cecs_history == []


def test_track_sequence_returns_live_tracker_with_cecs():
    cfg = apply_ablation(OCSIConfig(), "x-ocsi")
    dets = [[_det(100 + 2 * t, t, _evidence())] for t in range(5)]
    rows, tracker = track_sequence(dets, cfg)
    assert rows and tracker.cecs_history
    assert all(set(row) == {"frame", "track_id", "S_spatial", "S_temporal", "S_identity", "CECS"}
               for row in tracker.cecs_history)


def test_default_yaml_includes_xai_section():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = OCSIConfig.from_yaml(str(path))
    assert cfg.xai.w_s == 0.34
    assert cfg.xai.heatmap_size == (224, 224)
    assert cfg.xai.enabled is False


def test_gradcam_produces_map_for_small_classifier():
    torch = pytest.importorskip("torch")
    from ocsi.xai.gradcam import GradCAM

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            feats = torch.relu(self.conv(x))
            return self.fc(feats.mean(dim=(2, 3)))

    model = Tiny()
    cam = GradCAM(model, model.conv)
    with cam:
        heatmap = cam.generate(torch.rand(1, 3, 8, 8), target_class=0)
    assert heatmap.shape == (8, 8)
    assert np.isfinite(heatmap).all()
