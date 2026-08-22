"""Evaluation: MOT-format I/O and CLEAR-MOT/IDF1 metrics — Phases 3.5 / 6."""
from .metrics import MOTMetrics, evaluate, rows_to_frames
from .mot17 import frame_index_from_number, frame_number_from_index, mot_image_files
from .mot_io import (
    format_row,
    read_gt,
    read_results,
    tracker_rows,
    write_results,
)

__all__ = [
    "format_row",
    "frame_index_from_number",
    "frame_number_from_index",
    "mot_image_files",
    "read_gt",
    "read_results",
    "tracker_rows",
    "write_results",
    "MOTMetrics",
    "evaluate",
    "rows_to_frames",
]
