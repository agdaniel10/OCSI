"""Identity Intelligence layer: similarity terms, gating, association — Phase 3."""
from .association import Match, solve_assignment, two_stage_associate
from .similarity import (
    cos_to_01,
    cosine_matrix,
    gallery_similarity,
    gate_mask,
    iou_matrix,
    motion_similarity,
    weighted_score,
)
from .tracker import OCSITracker

__all__ = [
    "cos_to_01",
    "cosine_matrix",
    "gallery_similarity",
    "gate_mask",
    "iou_matrix",
    "motion_similarity",
    "weighted_score",
    "Match",
    "solve_assignment",
    "two_stage_associate",
    "OCSITracker",
]
