"""Controlled baselines for OCSI comparison.

These are thin wrappers around well-known trackers so that OCSI can be compared
against them under identical conditions (same detector outputs, same Re-ID
embeddings where applicable).

Each baseline implements a common interface:

    class BaselineTracker:
        def __init__(self, cfg: OCSIConfig): ...
        def update(self, detections: List[Detection]) -> List[MemoryRecord]: ...

The ``update`` method consumes the same ``Detection`` objects as OCSI and returns
track records with ``track_id`` and ``last_box`` attributes, so the same evaluation
pipeline can be used for all methods.
"""
from .bytetrack import ByteTrackBaseline
from .deepsort import DeepSORTBaseline
from .ocsort import OCSORTBaseline

__all__ = ["ByteTrackBaseline", "DeepSORTBaseline", "OCSORTBaseline"]