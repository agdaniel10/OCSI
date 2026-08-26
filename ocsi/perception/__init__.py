"""Perceptual Intelligence layer: detector / Re-ID / pose adapters (Phase 1).

Importing this package is cheap and dependency-light: torch/torchvision and
ultralytics are imported lazily inside :class:`ReIDEmbedder` / :class:`YOLODetector`
constructors, so the model-free core and its test suite never need them.
"""
from .cache import FeatureCache
from .detector import YOLODetector, filter_detections
from .embedder import ReIDEmbedder

__all__ = [
    "FeatureCache",
    "YOLODetector",
    "filter_detections",
    "ReIDEmbedder",
]
