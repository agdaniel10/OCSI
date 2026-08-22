"""Evaluation: MOT-format I/O and CLEAR-MOT/IDF1 metrics — Phases 3.5 / 6."""
from .metrics import MOTMetrics, evaluate, rows_to_frames
from .mot_io import (
    format_row,
    read_gt,
    read_results,
    tracker_rows,
    write_results,
)

__all__ = [
    "format_row",
    "read_gt",
    "read_results",
    "tracker_rows",
    "write_results",
    "MOTMetrics",
    "evaluate",
    "rows_to_frames",
]
