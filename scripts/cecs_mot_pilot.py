#!/usr/bin/env python3
"""CECS-on-MOT pilot: compute CECS on existing MOT17 runs.

Uses GT + tracker outputs to estimate cross-stage explanation consistency
using box-level proxies (spatial IoU consistency + track reliability).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from ocsi.explain import CECSCalculator


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=float)


def iou(b1, b2):
    a = xywh_to_xyxy(b1)
    b = xywh_to_xyxy(b2)
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, xx2 - xx1), max(0, yy2 - yy1)
    inter = iw * ih
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ub = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def read_gt(path):
    gt = defaultdict(list)
    if not os.path.exists(path):
        return gt
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            gid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            gt[frame].append((gid, np.array([x, y, w, h])))
    return gt


def read_results(path):
    res = defaultdict(list)
    if not os.path.exists(path):
        return res
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            tid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            conf = float(p[6]) if len(p) > 6 else 1.0
            res[frame].append((tid, np.array([x, y, w, h]), conf))
    return res


def main():
    parser = argparse.ArgumentParser(description="CECS-on-MOT pilot")
    parser.add_argument("--mot17-root", required=True, help="Path to MOT17/train or sequence root")
    parser.add_argument("--sequence", default="MOT17-02-FRCNN")
    parser.add_argument("--results-dir", default=None, help="Directory containing tracker results")
    parser.add_argument("--out", default="cecs_pilot_results")
    args = parser.parse_args()

    seq_dir = os.path.join(args.mot17_root, args.sequence)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt = read_gt(gt_path)

    results_dir = args.results_dir or os.path.join(args.mot17_root, "..", "mot17_outputs")
    results_dir = os.path.abspath(results_dir)

    calc = CECSCalculator(w_s=0.34, w_t=0.33, w_i=0.33)
    os.makedirs(args.out, exist_ok=True)

    # Look for result files
    modes = ["baseline", "memory", "feedback"]
    all_rows = []

    for mode in modes:
        # Try common naming patterns
        candidates = [
            os.path.join(results_dir, f"{args.sequence}-public-{mode}.txt"),
            os.path.join(results_dir, f"{args.sequence}-yolo-{mode}.txt"),
            os.path.join(results_dir, f"{args.sequence}-FRCNN-public-{mode}-seed0.txt"),
        ]
        res_path = None
        for c in candidates:
            if os.path.exists(c):
                res_path = c
                break
        if res_path is None:
            # search
            for fn in os.listdir(results_dir):
                if args.sequence in fn and mode in fn and fn.endswith('.txt') and 'summary' not in fn:
                    res_path = os.path.join(results_dir, fn)
                    break
        if res_path is None:
            print(f"[SKIP] {mode}: no results found")
            continue

        preds = read_results(res_path)
        last_gt_for_tid = {}
        last_pid_for_gid = {}
        idsw_events = set()

        # first pass: detect idsw
        for frame in sorted(preds.keys()):
            gts = gt.get(frame, [])
            prs = preds[frame]
            # match
            if gts and prs:
                C = np.ones((len(gts), len(prs)))
                for i, (_, gb) in enumerate(gts):
                    for j, (_, pb, _) in enumerate(prs):
                        C[i, j] = 1.0 - iou(gb, pb)
                from scipy.optimize import linear_sum_assignment
                rr, cc = linear_sum_assignment(C)
                for r, c in zip(rr, cc):
                    if 1.0 - C[r, c] >= 0.5:
                        gid = gts[r][0]
                        pid = prs[c][0]
                        if gid in last_pid_for_gid and last_pid_for_gid[gid] != pid:
                            idsw_events.add((frame, pid))
                        last_pid_for_gid[gid] = pid

        # second pass: compute CECS
        for frame in sorted(preds.keys()):
            gts = gt.get(frame, [])
            prs = preds[frame]
            if not gts or not prs:
                continue
            C = np.ones((len(gts), len(prs)))
            for i, (_, gb) in enumerate(gts):
                for j, (_, pb, _) in enumerate(prs):
                    C[i, j] = 1.0 - iou(gb, pb)
            from scipy.optimize import linear_sum_assignment
            rr, cc = linear_sum_assignment(C)
            for r, c in zip(rr, cc):
                if 1.0 - C[r, c] < 0.5:
                    continue
                gid = gts[r][0]
                pid, pbox, pconf = prs[c][0], prs[c][1], prs[c][2]
                is_switched = (frame, pid) in idsw_events
                # Proxies
                S_spatial = 1.0 - C[r, c]  # IoU as spatial consistency proxy
                reliability = float(np.clip(pconf, 0.0, 1.0))
                uncertainty = 1.0 - reliability
                res = calc.compute(
                    det_bbox=gts[r][1],
                    act_bbox=pbox,
                    reliability=[reliability],
                    uncertainty=[uncertainty],
                )
                all_rows.append({
                    "sequence": args.sequence,
                    "mode": mode,
                    "frame": frame,
                    "gt_id": gid,
                    "track_id": pid,
                    "iou": float(1.0 - C[r, c]),
                    "conf": reliability,
                    "switched": 1 if is_switched else 0,
                    "S_spatial": res.S_spatial,
                    "S_temporal": res.S_temporal,
                    "S_identity": res.S_identity,
                    "CECS": res.CECS,
                })

    out_csv = os.path.join(args.out, f"cecs_{args.sequence}.csv")
    with open(out_csv, 'w', newline='') as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)

    # summary
    summary = {}
    for mode in modes:
        rows_m = [r for r in all_rows if r['mode'] == mode]
        if not rows_m:
            continue
        correct = [r['CECS'] for r in rows_m if r['switched'] == 0]
        switched = [r['CECS'] for r in rows_m if r['switched'] == 1]
        summary[mode] = {
            'n_total': len(rows_m),
            'n_correct': len(correct),
            'n_switched': len(switched),
            'cecs_correct_mean': float(np.mean(correct)) if correct else None,
            'cecs_switched_mean': float(np.mean(switched)) if switched else None,
            'cecs_all_mean': float(np.mean([r['CECS'] for r in rows_m])),
        }
    out_json = os.path.join(args.out, f"cecs_{args.sequence}_summary.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_csv}, {out_json}")


if __name__ == '__main__':
    main()
