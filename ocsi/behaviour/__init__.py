"""Behaviour Intelligence layer: HAR + confidence gate + feedback — Phases 4-5.

The validated contribution (#3) is the confidence-gated feedback in :mod:`.gate`; the
recognizer in :mod:`.recognizer` is a clean, light HAR head behind a stable interface.
"""
from .gate import behaviour_gate, blend_behaviour, contradiction_mask
from .recognizer import (
    ActivityObservation,
    BehaviourRecognizer,
    PrototypeBehaviourRecognizer,
)
from .temporal_har import TemporalHARRecognizer

__all__ = [
    "behaviour_gate",
    "blend_behaviour",
    "contradiction_mask",
    "ActivityObservation",
    "BehaviourRecognizer",
    "PrototypeBehaviourRecognizer",
    "TemporalHARRecognizer",
]
