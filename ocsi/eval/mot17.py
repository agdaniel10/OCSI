"""MOT17 sequence helpers.

These utilities keep MOTChallenge's 1-indexed file/frame conventions at the I/O
boundary so experiment code can use the tracker's 0-indexed per-frame loop.
"""
from __future__ import annotations

import os
from typing import List, Optional


def mot_image_files(seq_dir: str, limit: Optional[int] = None) -> List[str]:
    """Return sorted image paths for a MOTChallenge sequence's ``img1`` folder."""
    img_dir = os.path.join(seq_dir, "img1")
    names = sorted(
        n for n in os.listdir(img_dir)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if limit is not None:
        names = names[:max(int(limit), 0)]
    return [os.path.join(img_dir, n) for n in names]


def frame_number_from_index(frame_idx: int) -> int:
    """Convert a 0-indexed tracker loop index to MOTChallenge's 1-indexed frame."""
    return int(frame_idx) + 1


def frame_index_from_number(frame_number: int) -> int:
    """Convert a MOTChallenge 1-indexed frame number to a 0-indexed loop index."""
    return int(frame_number) - 1
