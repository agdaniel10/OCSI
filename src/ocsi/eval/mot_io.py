"""MOTChallenge-format I/O (paper §6 evaluation plumbing).

Read ground truth, write tracker results, and bridge :class:`~ocsi.memory.record.MemoryRecord`
outputs into result rows — the substrate the Phase 3.5 quick metric and the Phase 6
TrackEval/ablation harness both build on.

MOTChallenge CSV, one detection per line, **1-indexed frames**::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Results set the last three to ``-1``. In ``gt/gt.txt`` the 7th field is a 0/1
"consider" flag, the 8th a class id (pedestrian = 1) and the 9th a visibility
fraction — used to filter which boxes count.

Pure stdlib + numpy, so it round-trips under test with no external data.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# (frame, track_id, x, y, w, h, conf)
ResultRow = Tuple[int, int, float, float, float, float, float]


def format_row(frame: int, track_id: int, tlwh: Sequence[float], conf: float = 1.0) -> str:
    """One MOTChallenge result line (trailing world coords set to -1)."""
    x, y, w, h = (float(v) for v in tlwh)
    return f"{int(frame)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"


def write_results(path: str, rows: Iterable[ResultRow]) -> None:
    """Write result rows, sorted by (frame, id) as MOT tooling expects."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for frame, tid, x, y, w, h, conf in rows:
            fh.write(format_row(frame, tid, (x, y, w, h), conf) + "\n")


def read_results(path: str) -> List[ResultRow]:
    """Read result/detection rows -> list of ``(frame, id, x, y, w, h, conf)``."""
    rows: List[ResultRow] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            rows.append((frame, tid, x, y, w, h, conf))
    return rows


def read_gt(
    path: str,
    valid_classes: Sequence[int] = (1,),
    min_visibility: float = 0.0,
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Read ``gt.txt`` into ``{frame: [(track_id, tlwh), ...]}``.

    Keeps only rows whose consider-flag is set, whose class is in
    ``valid_classes`` (pedestrian = 1) and whose visibility >= ``min_visibility``.
    Missing optional columns are treated as "keep".
    """
    per_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame, tid = int(float(p[0])), int(float(p[1]))
            tlwh = np.array([float(p[i]) for i in range(2, 6)], dtype=float)

            consider = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else valid_classes[0]
            vis = float(p[8]) if len(p) > 8 else 1.0
            if consider < 0.5 or cls not in valid_classes or vis < min_visibility:
                continue
            per_frame.setdefault(frame, []).append((tid, tlwh))
    return per_frame


def tracker_rows(frame_idx: int, records, conf: float = 1.0) -> List[ResultRow]:
    """Convert one frame's tracker outputs to MOT rows.

    The tracker is 0-indexed; MOT frames are 1-indexed, so we emit ``frame_idx + 1``.
    """
    rows: List[ResultRow] = []
    for r in records:
        x, y, w, h = (float(v) for v in r.last_box)
        rows.append((frame_idx + 1, int(r.track_id), x, y, w, h, conf))
    return rows
