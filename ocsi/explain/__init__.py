"""Cross-stage explanation consistency (X-OCSI)."""

from .cecs import CECSCalculator, CECSResult, compute_cecs
from .provenance import ExplanationRecorder

__all__ = [
    "CECSCalculator",
    "CECSResult",
    "ExplanationRecorder",
    "compute_cecs",
]
