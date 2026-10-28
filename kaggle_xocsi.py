#!/usr/bin/env python
"""Kaggle runner for X-OCSI on MOT17.

Why a separate script: CECS needs detector attribution ``E_det``, and only
``detection_source="yolo"`` produces it. The public MOT ``det.txt`` files carry
boxes but no detector to attribute, so an ``x-ocsi`` row built on them would be
structurally empty. This script therefore runs the *whole* ablation table on
YOLO detections, keeping the table internally comparable.

Typical Kaggle session::

    !git clone https://github.com/agdaniel10/OCSI.git /kaggle/working/OCSI
    %cd /kaggle/working/OCSI
    !pip install -q ultralytics pyyaml     # torch/torchvision/opencv are preinstalled
    !python kaggle_xocsi.py --smoke      # one sequence, 30 frames, ~2 min
    !python kaggle_xocsi.py              # full table, all sequences

Attach the MOT17 dataset (``wenhoujinjust/mot-17``) and enable a GPU. If
``yolov8s.pt`` is not next to this file, Ultralytics downloads it (needs
internet); the Re-ID backbone likewise downloads ImageNet weights once.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

from ocsi.config import OCSIConfig
from ocsi.experiments.mot17_tracking import run_mot17_dataset

FULL_STAGES = ("baseline", "memory", "adaptive_memory", "feedback", "x-ocsi")


def find_sequences(root: str) -> list[str]:
    """Locate MOT17 train sequences (one variant per video, FRCNN preferred)."""
    candidates: dict[str, str] = {}
    paths = glob.glob(os.path.join(root, "**", "MOT17-*"), recursive=True)
    paths += [
        os.path.dirname(p)
        for p in glob.glob(os.path.join(root, "**", "seqinfo.ini"), recursive=True)
    ]
    for seq_dir in paths:
        name = os.path.basename(seq_dir)
        if not name.startswith("MOT17-") or not os.path.isdir(seq_dir):
            continue
        if not os.path.isdir(os.path.join(seq_dir, "img1")):
            continue
        if not os.path.exists(os.path.join(seq_dir, "gt", "gt.txt")):
            continue                      # test split has no ground truth
        candidates.setdefault(name, seq_dir)

    # MOT17 ships the same video three times (DPM / FRCNN / SDP) with different
    # public detections. YOLO replaces all of them, so keep exactly one copy per
    # video, preferring FRCNN.
    by_video: dict[str, tuple[int, str, str]] = {}
    preference = {"-FRCNN": 0, "": 1, "-DPM": 2, "-SDP": 3}
    for name, seq_dir in candidates.items():
        base = "-".join(name.split("-")[:2])
        suffix = name[len(base):]
        rank = preference.get(suffix, 4)
        if base not in by_video or rank < by_video[base][0]:
            by_video[base] = (rank, name, seq_dir)
    return [by_video[base][2] for base in sorted(by_video)]


def discover_root(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in ("/kaggle/input", "."):
        if os.path.isdir(candidate) and find_sequences(candidate):
            return candidate
    raise SystemExit(
        "No MOT17 train sequences found. Pass --root (e.g. /kaggle/input/mot-17)."
    )


def stage_rows(payload: dict) -> list[dict]:
    rows = []
    for run in payload["runs"]:
        attribution = run.get("attribution", {})
        for result in run["results"]:
            metrics = result["metrics"]
            cecs = metrics.get("cecs") or {}
            rows.append(
                {
                    "sequence": run["sequence"],
                    "seed": "" if run.get("seed") is None else run["seed"],
                    "stage": result["stage"],
                    "IDF1": metrics.get("idf1"),
                    "MOTA": metrics.get("mota"),
                    "IDsw": metrics.get("idsw"),
                    "cecs_num": cecs.get("num_cecs"),
                    "cecs_mean": cecs.get("cecs_mean"),
                    "cecs_min": cecs.get("cecs_min"),
                    "cecs_max": cecs.get("cecs_max"),
                    "attribution_coverage": attribution.get("coverage"),
                    "warning": cecs.get("warning", ""),
                }
            )
    return rows


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    columns = (
        ("sequence", "sequence"),
        ("stage", "stage"),
        ("IDF1", "IDF1"),
        ("MOTA", "MOTA"),
        ("IDsw", "IDsw"),
        ("cecs_mean", "CECS"),
        ("cecs_num", "n"),
        ("attribution_coverage", "coverage"),
    )
    labels = [label for _, label in columns]
    widths = [max(len(label), *(len(_fmt(row[key])) for row in rows)) for key, label in columns]
    print("  ".join(label.ljust(w) for label, w in zip(labels, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        values = [_fmt(row[key]) for key, _ in columns]
        print("  ".join(value.ljust(w) for value, w in zip(values, widths)))
    for row in rows:
        if row["warning"]:
            print(f"[warn] {row['sequence']} [{row['stage']}]: {row['warning']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run X-OCSI ablations on MOT17 (Kaggle).")
    parser.add_argument("--root", default=None, help="directory containing MOT17 sequences")
    parser.add_argument("--sequences", default=None, help="comma-separated subset, e.g. MOT17-02-FRCNN,MOT17-04-FRCNN")
    parser.add_argument("--limit", type=int, default=None, help="frames per sequence (smoke test)")
    parser.add_argument("--smoke", action="store_true", help="one sequence, 30 frames, x-ocsi only")
    parser.add_argument("--stages", default=",".join(FULL_STAGES))
    parser.add_argument("--seeds", default="0", help="comma-separated behaviour seeds")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--imgsz", type=int, default=None, help="detector input size (default 640)")
    parser.add_argument("--heatmap-size", type=int, default=224, help="cached attribution edge (default 224)")
    parser.add_argument("--cache", default="/kaggle/working/ocsi_cache")
    parser.add_argument("--out", default="/kaggle/working/ocsi_outputs")
    parser.add_argument("--rebuild", action="store_true", help="recompute the detection cache")
    args = parser.parse_args(argv)

    root = discover_root(args.root)
    sequences = find_sequences(root)
    if args.sequences:
        wanted = {name.strip() for name in args.sequences.split(",") if name.strip()}
        sequences = [s for s in sequences if os.path.basename(s) in wanted]
    if not sequences:
        raise SystemExit(f"No usable MOT17 sequences under {root!r}")
    if args.smoke:
        sequences = sequences[:1]
        args.limit = args.limit or 30
        args.stages = "x-ocsi"

    cfg = OCSIConfig()
    if args.device:
        cfg.perception.device = args.device
    if args.imgsz:
        cfg.perception.det_imgsz = int(args.imgsz)
    cfg.xai.heatmap_size = (int(args.heatmap_size), int(args.heatmap_size))

    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    print(f"root        : {root}")
    print(f"sequences   : {len(sequences)}")
    for seq in sequences:
        print(f"  - {os.path.basename(seq)}")
    print(f"stages      : {', '.join(stages)}")
    print(f"detections  : yolo (required for CECS), imgsz={cfg.perception.det_imgsz}")
    print(f"cache       : {args.cache}")
    print(f"outputs     : {args.out}")
    if not os.path.exists(cfg.perception.detector_weights):
        print(
            f"[warn] {cfg.perception.detector_weights!r} not found; Ultralytics will "
            "download it (needs internet)."
        )

    started = time.time()
    payload = run_mot17_dataset(
        sequences,
        cache_root=args.cache,
        output_dir=args.out,
        stages=stages,
        cfg=cfg,
        limit=args.limit,
        rebuild_cache=args.rebuild,
        detection_source="yolo",
        seeds=seeds,
    )
    elapsed = time.time() - started

    rows = stage_rows(payload)
    print()
    print_table(rows)
    print(f"\nelapsed: {elapsed / 60.0:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "xocsi-metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv     : {csv_path}")
    print(f"sidecars: {args.out}/*-cecs.json  (full CECS trace per run)")

    xocsi_rows = [r for r in rows if r["stage"] == "x-ocsi"]
    sys.stdout.flush()
    if xocsi_rows and all(not r["cecs_num"] for r in xocsi_rows):
        coverage = max((r["attribution_coverage"] or 0.0) for r in xocsi_rows)
        if coverage == 0.0:
            print(
                "\nERROR: no detector attribution reached the tracker. Rebuild the "
                "cache with --rebuild and check the warnings above.",
                file=sys.stderr,
            )
            return 1
        if args.smoke:
            print(
                "\n[note] attribution is present (coverage %.2f) but no identity "
                "completed an evidence chain yet. The HAR window is %d frames; "
                "re-run with a larger --limit to exercise CECS."
                % (coverage, OCSIConfig().behaviour.min_window),
                file=sys.stderr,
            )
            return 0
        print(
            "\nERROR: x-ocsi produced no CECS despite full attribution coverage.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
