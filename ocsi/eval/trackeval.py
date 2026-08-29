"""Official TrackEval integration for OCSI.

This module provides a thin wrapper around the official TrackEval toolkit
(https://github.com/JonathonLuiten/TrackEval) so that OCSI results can be
scored with the official HOTA/DetA/AssA/MOTA/IDF1 metrics.

TrackEval is installed separately (not a PyPI package):
    pip install git+https://github.com/JonathonLuiten/TrackEval.git

Usage:
    from ocsi.eval.trackeval import run_trackeval
    results = run_trackeval(
        gt_folder="/path/to/MOT17/train",
        trackers_folder="/path/to/ocsi_outputs/trackers",
        benchmark="MOT17",
        split="train",
    )
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


def run_trackeval(
    gt_folder: str,
    trackers_folder: str,
    benchmark: str = "MOT17",
    split: str = "train",
    trackeval_dir: Optional[str] = None,
    metrics: Sequence[str] = ("HOTA", "CLEAR", "Identity"),
) -> Dict:
    """Invoke the official TrackEval toolkit and parse results.

    Parameters
    ----------
    gt_folder : str
        Path to the ground-truth folder (e.g. ``MOT17/train``).
    trackers_folder : str
        Path to the folder containing tracker result files, structured as
        ``{tracker_name}/data/{sequence}.txt``.
    benchmark : str
        Benchmark name (``MOT17``, ``MOT20``, etc.).
    split : str
        Split to evaluate (``train``, ``test``).
    trackeval_dir : str, optional
        Path to the TrackEval checkout. If not provided, tries to find it
        on the Python path or in common locations.
    metrics : sequence of str
        Metric families to compute.

    Returns
    -------
    dict
        Parsed results keyed by ``(tracker, sequence, metric)``.
    """
    # Locate TrackEval
    if trackeval_dir is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.expanduser("~"), "TrackEval"),
            os.path.join(os.getcwd(), "TrackEval"),
            os.path.join(os.path.dirname(__file__), "..", "..", "TrackEval"),
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "scripts", "run_mot_challenge.py")):
                trackeval_dir = cand
                break
    if trackeval_dir is None:
        raise FileNotFoundError(
            "TrackEval not found. Install it with: "
            "pip install git+https://github.com/JonathonLuiten/TrackEval.git"
        )

    script = os.path.join(trackeval_dir, "scripts", "run_mot_challenge.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"TrackEval script not found: {script}")

    cmd = [
        sys.executable,
        script,
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", split,
        "--GT_FOLDER", gt_folder,
        "--TRACKERS_FOLDER", trackers_folder,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "False",
    ]
    print("[TrackEval]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"TrackEval failed:\n{result.stderr}")

    # TrackEval writes results to a JSON file in the trackers folder
    # (e.g. {trackers_folder}/../results/{benchmark}_{split}.json)
    results_dir = os.path.join(os.path.dirname(trackers_folder), "results")
    results_file = os.path.join(results_dir, f"{benchmark}_{split}.json")
    if not os.path.exists(results_file):
        # Try alternate naming
        results_file = os.path.join(results_dir, f"{benchmark}.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"TrackEval results file not found at {results_file}. "
            f"TrackEval output:\n{result.stdout}"
        )

    with open(results_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_trackeval_results(raw: Dict) -> Dict[str, Dict[str, float]]:
    """Flatten TrackEval JSON into a per-tracker, per-sequence metric dict.

    TrackEval's JSON structure is:
        {tracker: {sequence: {metric: value, ...}, ...}, ...}

    Returns:
        {tracker: {sequence: {metric: value, ...}, ...}}
    """
    out: Dict[str, Dict[str, float]] = {}
    for tracker, sequences in raw.items():
        out[tracker] = {}
        for seq, metrics in sequences.items():
            out[tracker][seq] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return out