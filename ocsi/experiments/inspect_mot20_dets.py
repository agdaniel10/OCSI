"""Inspect MOT20 public detection confidence distribution.

MOT20 detections are denser and have lower confidence scores than MOT17.
This script helps find the right confidence threshold for MOT20.

Usage:
    python -m ocsi.experiments.inspect_mot20_dets --seq-dir /path/to/MOT20-01
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

import numpy as np


MOT_COLUMNS = (
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
)


def _resolve_det_path(seq_dir: str | None, det_path: str | None) -> str:
    if det_path:
        return det_path
    if not seq_dir:
        raise ValueError("pass either --seq-dir or --det-path")
    return os.path.join(seq_dir, "det", "det.txt")


def _load_raw_det_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        return np.zeros((0, len(MOT_COLUMNS)), dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < 7:
        raise ValueError(
            f"{path!r} has {data.shape[1]} columns; expected at least 7 "
            "(frame,id,bb_left,bb_top,w,h,conf)"
        )
    return data


def _format_histogram(values: np.ndarray, bins: Sequence[float] | int) -> Iterable[str]:
    counts, edges = np.histogram(values, bins=bins)
    max_count = int(counts.max()) if counts.size else 0
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar_len = 0 if max_count == 0 else int(round(40 * int(count) / max_count))
        yield f"[{lo:8.3f}, {hi:8.3f}) {int(count):8d} {'#' * bar_len}"


def inspect_mot20_dets(det_path: str, bins: int = 20) -> None:
    rows = _load_raw_det_txt(det_path)
    row_count = int(rows.shape[0])
    conf = rows[:, 6] if row_count else np.array([], dtype=float)

    print(f"det_path: {det_path}")
    print(f"columns: {', '.join(MOT_COLUMNS)}")
    print(f"total_rows: {row_count}")
    print(f"frame min/max: {int(rows[:, 0].min()) if row_count else 0} / "
          f"{int(rows[:, 0].max()) if row_count else 0}")

    if row_count == 0:
        print("confidence: no rows")
        return

    print(
        f"confidence min/max/mean: "
        f"{float(conf.min()):.6g} / {float(conf.max()):.6g} / {float(conf.mean()):.6g}"
    )

    # How many detections survive at different thresholds?
    print("\nDetections surviving at various confidence thresholds:")
    for th in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        n = int(np.sum(conf >= th))
        print(f"  conf >= {th:.2f}: {n:8d} ({100.0 * n / row_count:.1f}%)")

    print("\nconfidence histogram:")
    for line in _format_histogram(conf, bins=bins):
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect MOT20 public det.txt confidence scale and row count."
    )
    parser.add_argument("--seq-dir", help="Path to a MOT20 sequence directory, e.g. MOT20-01.")
    parser.add_argument("--det-path", help="Path directly to det/det.txt.")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins.")
    args = parser.parse_args(argv)

    det_path = _resolve_det_path(args.seq_dir, args.det_path)
    inspect_mot20_dets(det_path, bins=args.bins)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())