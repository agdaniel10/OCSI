"""Inspect raw MOTChallenge public detections before tracker filtering.

Usage:
    python -m ocsi.experiments.inspect_public_dets --seq-dir /path/to/MOT17-02-FRCNN
    python -m ocsi.experiments.inspect_public_dets --det-path /path/to/det.txt
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
GT_DETS_REFERENCE = 18_581


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


def inspect_public_dets(det_path: str, bins: int = 20) -> None:
    rows = _load_raw_det_txt(det_path)
    row_count = int(rows.shape[0])
    conf = rows[:, 6] if row_count else np.array([], dtype=float)

    print(f"det_path: {det_path}")
    print(f"columns: {', '.join(MOT_COLUMNS)}")
    print(f"total_rows: {row_count}")
    print(f"GT_Dets_reference: ~{GT_DETS_REFERENCE}")
    if GT_DETS_REFERENCE:
        print(f"raw_det_to_GT_Dets_ratio: {row_count / GT_DETS_REFERENCE:.3f}")

    if row_count == 0:
        print("confidence: no rows")
        return

    print(
        "confidence min/max/mean: "
        f"{float(conf.min()):.6g} / {float(conf.max()):.6g} / {float(conf.mean()):.6g}"
    )
    print(f"detection_id unique values: {sorted(set(rows[:, 1].astype(int).tolist()))[:10]}")
    print(f"frame min/max: {int(rows[:, 0].min())} / {int(rows[:, 0].max())}")
    print()
    print("confidence histogram:")
    for line in _format_histogram(conf, bins=bins):
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect raw MOT17 public det.txt confidence scale and row count."
    )
    parser.add_argument("--seq-dir", help="Path to a MOT17 sequence directory, e.g. MOT17-02-FRCNN.")
    parser.add_argument("--det-path", help="Path directly to det/det.txt.")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins.")
    args = parser.parse_args(argv)

    det_path = _resolve_det_path(args.seq_dir, args.det_path)
    inspect_public_dets(det_path, bins=args.bins)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
