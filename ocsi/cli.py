"""Command-line entry points for OCSI experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from .config import ABLATION_STAGES, OCSIConfig
from .eval.statistics import bootstrap_mean_ci, paired_stats
from .experiments import run_mot17_dataset


DEFAULT_Q1_STAGES = ("baseline", "memory", "adaptive_memory")


def _sequence_dirs(data_root: Path, selected: Sequence[str] | None) -> List[str]:
    wanted = set(selected or [])
    dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if wanted:
        dirs = [p for p in dirs if p.name in wanted]
    if not dirs:
        raise RuntimeError(f"No MOT sequence directories found under {data_root}")
    return [str(p) for p in dirs]


def _flatten_results(payload: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for run in payload["runs"]:
        for result in run["results"]:
            row = {
                "Sequence": run["sequence"],
                "Seed": run["seed"],
                "Stage": result["stage"],
                "ResultPath": result["result_path"],
                "ReIDSeparation": run["embedding_diagnostics"].get("separation_margin"),
                "RecommendedGate": run.get("recommended_reactivation_app_gate"),
            }
            row.update(result["metrics"])
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_q1_summary(out_dir: Path, rows: Sequence[Dict]) -> None:
    metrics = ["idf1", "mota", "idsw", "precision", "recall"]
    summary_rows: List[Dict] = []
    stages = sorted({str(row["Stage"]) for row in rows})
    for stage in stages:
        stage_rows = [row for row in rows if row["Stage"] == stage]
        for metric in metrics:
            values = [row.get(metric) for row in stage_rows]
            mean, lo, hi = bootstrap_mean_ci(values)
            summary_rows.append(
                {
                    "Stage": stage,
                    "Metric": metric,
                    "Mean": mean,
                    "CI95_Low": lo,
                    "CI95_High": hi,
                    "N": len(stage_rows),
                }
            )
    _write_csv(out_dir / "paper_summary_bootstrap_ci.csv", summary_rows)

    infer_rows: List[Dict] = []
    baselines = {
        (row["Sequence"], row["Seed"]): row
        for row in rows
        if row["Stage"] == "baseline"
    }
    for treatment in ("memory", "adaptive_memory", "feedback"):
        treatment_rows = {
            (row["Sequence"], row["Seed"]): row
            for row in rows
            if row["Stage"] == treatment
        }
        common = sorted(set(baselines) & set(treatment_rows))
        if not common:
            continue
        for metric in metrics:
            stats = paired_stats(
                [baselines[key].get(metric) for key in common],
                [treatment_rows[key].get(metric) for key in common],
            )
            infer_rows.append(
                {
                    "Comparison": f"{treatment} vs baseline",
                    "Metric": metric,
                    **stats,
                }
            )
    _write_csv(out_dir / "paper_paired_statistics.csv", infer_rows)


def _add_mot17_q1_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "mot17-q1",
        help="Run leakage-aware MOT17 ablations and write paper-oriented summaries.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ocsi_q1_results"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--stages", nargs="*", default=list(DEFAULT_Q1_STAGES), choices=ABLATION_STAGES)
    parser.add_argument("--detection-source", choices=("public", "yolo"), default="public")
    parser.add_argument("--det-conf", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--reactivation-app-gate", type=float, default=None)
    parser.set_defaults(func=_run_mot17_q1)


def _run_mot17_q1(args) -> int:
    cfg = OCSIConfig()
    if args.det_conf is not None:
        cfg.perception.det_conf_threshold = float(args.det_conf)

    payload = run_mot17_dataset(
        seq_dirs=_sequence_dirs(args.data_root, args.sequences),
        cache_root=str(args.cache_root),
        output_dir=str(args.out),
        stages=tuple(args.stages),
        cfg=cfg,
        limit=args.limit,
        rebuild_cache=args.rebuild_cache,
        detection_source=args.detection_source,
        seeds=tuple(args.seeds),
        reactivation_app_gate=args.reactivation_app_gate,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = _flatten_results(payload)
    _write_csv(args.out / "per_sequence_results.csv", rows)
    _write_q1_summary(args.out, rows)
    (args.out / "q1_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote Q1 MOT17 outputs to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocsi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_mot17_q1_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
