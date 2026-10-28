"""X-OCSI explainability (XAI) utilities.

Detection attribution, temporal saliency, and helpers for cross-stage
explanation consistency (CECS).
"""

from .det_attrib import (
    boxes_to_input,
    detection_heatmaps,
    explanation_map_summary,
    feature_activation_cam,
    letterbox_image,
    spatial_attribution_from_box,
    unletterbox_heatmaps,
)
from .gradcam import DetectorGradCAM, GradCAM
from .temporal_saliency import (
    frame_perturbation_saliency,
    gradient_temporal_saliency,
    har_score_fn,
    uniform_saliency,
)
from .utils import (
    bbox_iou,
    heatmap_bbox_overlap,
    mask_bbox,
    normalize_heatmap,
    resize_heatmap,
)

__all__ = [
    "DetectorGradCAM",
    "GradCAM",
    "bbox_iou",
    "boxes_to_input",
    "detection_heatmaps",
    "explanation_map_summary",
    "feature_activation_cam",
    "frame_perturbation_saliency",
    "gradient_temporal_saliency",
    "har_score_fn",
    "heatmap_bbox_overlap",
    "letterbox_image",
    "mask_bbox",
    "normalize_heatmap",
    "resize_heatmap",
    "spatial_attribution_from_box",
    "uniform_saliency",
    "unletterbox_heatmaps",
]
