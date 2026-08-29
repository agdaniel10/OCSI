"""OCSI experiments: real, reproducible studies that replace the source
notebook's simulated/fabricated results with measured ones."""

from .mot17_tracking import run_mot17_dataset, run_mot17_sequence
from .mot20 import run_mot20_sequence

__all__ = ["run_mot17_dataset", "run_mot17_sequence", "run_mot20_sequence"]
