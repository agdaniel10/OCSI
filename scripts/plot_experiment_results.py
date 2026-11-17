"""Plot the experiment results stored in this repository.

Usage (from the repository root)::

    python scripts/plot_experiment_results.py

Reads the JSON/CSV result files produced by ``ocsi.experiments.*`` and writes
PNG figures to ``figures/``.  Only numpy, pandas and matplotlib are needed
(no Seaborn), so it runs in the base environment.

Each figure is generated from the saved result files, not by re-running the
experiments, so the plots exactly reflect the numbers you already obtained.
"""
from __future__ import annotations

import json
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

COLORS = {
    "baseline": "#4C72B0",
    "memory": "#55A868",
    "feedback": "#C44E52",
    "clean": "#55A868",
    "contaminated": "#C44E52",
    "recovered": "#4C72B0",
    "gate": "#984EA3",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# -------------------------------------------------------------------- helpers
def load_json(rel_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON result file; skip missing/empty files with a clear message."""
    path = ROOT / rel_path
    if not path.exists():
        print(f"[skip] {rel_path}: file not found")
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text or text == "{}":
        print(f"[skip] {rel_path}: empty result")
        return None
    return json.loads(text)


def save_figure(fig: plt.Figure, filename: str, footnote: str = "") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if footnote:
        fig.text(0.012, 0.004, footnote, fontsize=8, color="0.35", ha="left", va="bottom")
    path = FIG_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {path.relative_to(ROOT)}")


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.3g}") -> None:
    """Write each bar's value just above it."""
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    series: Dict[str, list[float]],
    ylabel: str,
    title: str,
    fmt: str = "{:.3g}",
) -> None:
    """Small helper for a grouped bar chart with one series per ablation stage."""
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width, label=name, color=COLORS.get(name))
        label_bars(ax, bars, fmt)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


# ------------------------------------------------------------------ figures
def plot_occlusion_recovery(data: Dict[str, Any]) -> None:
    cfg, stages = data["config"], data["stages"]
    lengths = [int(x) for x in cfg["lengths"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for stage in ("baseline", "memory"):
        recovery = stages[stage]["recovery"]
        rates = []
        for length in lengths:
            recovered, total = recovery.get(str(length), [0, 0])
            rates.append(100.0 * recovered / total if total else 0.0)
        ax.plot(
            lengths,
            rates,
            "-o",
            lw=2.2,
            ms=6,
            color=COLORS[stage],
            label="baseline" if stage == "baseline" else "+memory",
        )
        for length, rate in zip(lengths, rates):
            ax.annotate(
                f"{rate:.0f}%",
                (length, rate),
                xytext=(0, 8 if rate < 90 else -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS[stage],
            )

    max_age = int(cfg["max_age"])
    ax.axvline(max_age, color="0.4", ls="--", lw=1.2)
    ax.text(
        max_age + 0.4,
        6,
        f"max_age = {max_age}\n(retention cliff)",
        fontsize=8,
        color="0.35",
    )
    ax.set_xticks(lengths)
    ax.set_xlabel("Occlusion length L (frames)")
    ax.set_ylabel("Identity recovery rate")
    ax.set_ylim(-6, 112)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Occlusion recovery: baseline vs. +memory")
    ax.legend(loc="lower left")
    save_figure(
        fig,
        "01_occlusion_recovery.png",
        f"{cfg['n_rep']} occlusion events per length; reactivation gate = "
        f"{cfg['reactivation_gate']:.3f}; same-id cosine = {cfg['feature_cos_same']:.3f}, "
        f"diff-id cosine = {cfg['feature_cos_diff']:.3f}.",
    )


def plot_occlusion_metrics(data: Dict[str, Any]) -> None:
    stages = data["stages"]
    present = [s for s in ("baseline", "memory") if s in stages]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    quality = [("mota", "MOTA"), ("motp", "MOTP"), ("idf1", "IDF1"),
               ("precision", "Precision"), ("recall", "Recall")]
    grouped_bars(
        axes[0],
        [label for _, label in quality],
        {
            s: [100.0 * stages[s]["metrics"][key] for key, _ in quality]
            for s in present
        },
        "score (%)",
        "Aggregate tracking quality",
        fmt="{:.1f}",
    )
    axes[0].set_ylim(0, 108)
    axes[0].yaxis.set_major_formatter(PercentFormatter())

    errors = [("idsw", "ID switches"), ("frag", "Fragmentations"),
              ("num_pred_ids", "Predicted IDs"), ("fn", "FN"), ("fp", "FP")]
    grouped_bars(
        axes[1],
        [label for _, label in errors],
        {s: [stages[s]["metrics"][key] for key, _ in errors] for s in present},
        "count",
        "Identity and detection errors (lower is better)",
        fmt="{:.0f}",
    )
    fig.suptitle("Occlusion study — aggregate metrics", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "02_occlusion_metrics.png")


def plot_mot17_ablation(data: Dict[str, Any]) -> None:
    results = {r["stage"]: r["metrics"] for r in data["results"]}
    present = [s for s in ("baseline", "memory", "feedback") if s in results]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    quality = [("mota", "MOTA"), ("idf1", "IDF1"), ("motp", "MOTP"),
               ("precision", "Prec."), ("recall", "Recall")]
    grouped_bars(
        axes[0, 0],
        [label for _, label in quality],
        {s: [results[s][key] for key, _ in quality] for s in present},
        "score",
        "(a) Tracking quality",
    )
    axes[0, 0].set_ylim(0, 1.05)

    id_errors = [("idsw", "ID switches"), ("frag", "Frag."),
                 ("num_pred_ids", "Pred. IDs")]
    grouped_bars(
        axes[0, 1],
        [label for _, label in id_errors],
        {s: [results[s][key] for key, _ in id_errors] for s in present},
        "count",
        "(b) Identity errors",
        fmt="{:.0f}",
    )

    det_errors = [("fp", "FP"), ("fn", "FN")]
    grouped_bars(
        axes[1, 0],
        [label for _, label in det_errors],
        {s: [results[s][key] for key, _ in det_errors] for s in present},
        "count",
        "(c) Detection errors",
        fmt="{:.0f}",
    )

    x = np.arange(len(present))
    bottom = np.zeros(len(present))
    for key, label, color in (
        ("mostly_tracked", "Mostly tracked", COLORS["memory"]),
        ("partially_tracked", "Partially tracked", "#DD8452"),
        ("mostly_lost", "Mostly lost", COLORS["feedback"]),
    ):
        values = np.asarray([results[s][key] for s in present], dtype=float)
        bars = axes[1, 1].bar(x, values, 0.6, bottom=bottom, label=label, color=color)
        for bar, base in zip(bars, bottom):
            axes[1, 1].annotate(
                f"{bar.get_height():.0f}",
                (bar.get_x() + bar.get_width() / 2, base + bar.get_height() / 2),
                ha="center",
                va="center",
                fontsize=8,
            )
        bottom = bottom + values
    axes[1, 1].set_xticks(x, present)
    axes[1, 1].set_ylabel("ground-truth identities")
    axes[1, 1].set_title("(d) Track coverage")
    axes[1, 1].legend()

    fig.suptitle(
        f"MOT17 ablation — {data['sequence']} "
        f"({data['detection_source']} detections, {data['frames']} frames, seed {data['seed']})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(
        fig,
        "03_mot17_ablation.png",
        "MOT17 has no action labels: the 'feedback' stage exercises the gating "
        "plumbing with placeholder activity prototypes only.",
    )


def plot_embedding_diagnostics(data: Dict[str, Any]) -> None:
    diag = data["embedding_diagnostics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    names = ["Same-ID cosine", "Different-ID cosine", "Reactivation gate"]
    values = [
        diag.get("same_id_proto_cosine"),
        diag.get("different_id_proto_cosine"),
        diag.get("recommended_reactivation_app_gate"),
    ]
    bars = ax1.barh(
        names[::-1],
        [0.0 if v is None else float(v) for v in values[::-1]],
        color=[COLORS["gate"], COLORS["baseline"], COLORS["memory"]][::-1],
        height=0.6,
    )
    for bar, value in zip(bars, values[::-1]):
        ax1.annotate(
            "n/a" if value is None else f"{float(value):.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("cosine similarity / gate")
    ax1.set_title("Re-ID embedding separation")
    ax1.grid(axis="y", visible=False)

    ax2.axis("off")
    margin = diag.get("separation_margin")
    facts = [
        f"sequence:              {data['sequence']}",
        f"embeddings (total):    {diag.get('total_embeddings')}",
        f"embeddings assigned:   {diag.get('assigned_embeddings')}",
        f"identity prototypes:   {diag.get('num_identity_prototypes')}",
        f"embedding dimension:   {diag.get('embedding_dim')}",
        f"mean norm:             {diag.get('mean_norm', 0):.6f}",
        f"separation margin:     {'n/a' if margin is None else f'{float(margin):.3f}'}",
        f"gate used in run:      {data.get('reactivation_app_gate')}",
        "",
        "Healthy Re-ID puts the gate above the different-ID",
        "cosine but below the same-ID cosine. A small margin",
        "makes memory reactivation fragile.",
    ]
    ax2.text(
        0.0,
        0.98,
        "\n".join(facts),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        transform=ax2.transAxes,
    )
    fig.suptitle("MOT17 — appearance feature diagnostics", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "04_embedding_diagnostics.png")


def plot_contamination_rollback(data: Dict[str, Any]) -> None:
    r = data["result"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    appearance = [
        ("clean", r["clean_appearance_cosine"]),
        ("contaminated", r["contaminated_appearance_cosine"]),
        ("recovered", r["recovered_appearance_cosine"]),
    ]
    bars = axes[0].bar(
        [name for name, _ in appearance],
        [value for _, value in appearance],
        color=[COLORS[name] for name, _ in appearance],
    )
    label_bars(axes[0], bars, "{:+.3f}")
    axes[0].set_ylim(-0.1, 1.15)
    axes[0].set_ylabel("cosine to true appearance")
    axes[0].set_title("Appearance prototype")

    behaviour = [
        ("contaminated", r["contaminated_behaviour_cosine"]),
        ("recovered", r["recovered_behaviour_cosine"]),
    ]
    bars = axes[1].bar(
        [name for name, _ in behaviour],
        [value for _, value in behaviour],
        color=[COLORS[name] for name, _ in behaviour],
    )
    label_bars(axes[1], bars, "{:+.3f}")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_ylabel("cosine to true behaviour")
    axes[1].set_title("Behaviour prototype")

    confidence = [
        ("clean", r["clean_confidence"]),
        ("contaminated", r["contaminated_confidence"]),
        ("recovered", r["recovered_confidence"]),
    ]
    bars = axes[2].bar(
        [name for name, _ in confidence],
        [value for _, value in confidence],
        color=[COLORS[name] for name, _ in confidence],
    )
    label_bars(axes[2], bars, "{:.3f}")
    axes[2].set_ylim(0, 1.15)
    axes[2].set_ylabel("memory confidence q")
    axes[2].set_title("Track confidence")

    status = "PASS" if r["passed"] else "FAIL"
    fig.suptitle(
        f"Memory contamination rollback — {status} "
        f"(conflict detected: {r['conflict_detected']}, rollback applied: {r['rollback_applied']})",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(
        fig,
        "05_contamination_rollback.png",
        "Synthetic, model-free experiment: a deliberately wrong appearance/behaviour "
        "update is applied, detected, then rolled back.",
    )


def plot_behaviour_feedback(data: Dict[str, Any]) -> None:
    scenarios = data["scenarios"]
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(4.4 * len(scenarios), 4.3),
        squeeze=False,
        layout="constrained",  # plays nicely with the shared colorbar
    )
    image = None
    for ax, scenario in zip(axes[0], scenarios):
        score = np.asarray(scenario["score"], dtype=float)
        gate = np.asarray(scenario["gate"], dtype=float)
        valid = np.asarray(scenario["valid"], dtype=bool)
        image = ax.imshow(score, cmap="RdYlGn", vmin=0.0, vmax=1.0)
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{score[i, j]:.2f}\ng={gate[i, j]:.2f} "
                    f"{'valid' if valid[i, j] else 'blocked'}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if 0.25 < score[i, j] < 0.85 else "white",
                )
        ax.set_xticks(range(score.shape[1]), [f"det {j}" for j in range(score.shape[1])])
        ax.set_yticks(range(score.shape[0]), [f"trk {i}" for i in range(score.shape[0])])
        ax.grid(visible=False)
        status = "PASS" if scenario["passed"] else "FAIL"
        title = "\n".join(textwrap.wrap(scenario["name"], width=30))
        ax.set_title(f"{title}\n[{status}]", fontsize=9)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label="blended association score")
    cfg = data["config"]
    fig.suptitle(
        "Behaviour feedback — confidence-gated association "
        f"($\\theta_b$={cfg['theta_b']}, $w_b$={cfg['w_behaviour']}, "
        f"contradiction $\\leq$ {cfg['contradiction_threshold']})",
        fontweight="bold",
    )
    save_figure(fig, "06_behaviour_feedback.png")


def plot_cecs_proxy(data: Dict[str, Any]) -> None:
    """Plot the early box-IoU proxy, clearly marked as *not* citable CECS."""
    present = [s for s in ("baseline", "memory", "feedback") if s in data]
    if not present:
        print("[skip] CECS pilot: no stages with data")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(present))
    width = 0.36
    for offset, key, label, color, hatch in (
        (-width / 2, "cecs_correct_mean", "correct matches", COLORS["memory"], "//"),
        (+width / 2, "cecs_switched_mean", "switched matches", COLORS["feedback"], "\\\\"),
    ):
        values = [data[s][key] for s in present]
        bars = ax.bar(x + offset, values, width, label=label, color=color, hatch=hatch)
        label_bars(ax, bars, "{:.3f}")
    ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
    ax.set_xticks(x, present)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Pilot score by association outcome (box-IoU proxy)")
    ax.legend()

    switched = [data[s]["n_switched"] for s in present]
    ax2 = ax.twinx()
    ax2.plot(x, switched, "o--", color="#333333", lw=1.2, label="switched count")
    ax2.set_ylabel("switched matches (count)")
    ax2.grid(visible=False)
    ax2.legend(loc="upper right")

    fig.text(
        0.5,
        0.01,
        "PRELIMINARY BOX-IoU PROXY — NOT CECS. Do not cite (see README §X-OCSI caveats).",
        ha="center",
        fontsize=9,
        color=COLORS["feedback"],
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "07_cecs_proxy_preliminary.png")


# ----------------------------------------------------------------------- main
def main() -> None:
    steps: list[tuple[str, str, Callable[[Dict[str, Any]], None]]] = [
        ("occlusion recovery", "occlusion_recovery_results.json", plot_occlusion_recovery),
        ("occlusion metrics", "occlusion_recovery_results.json", plot_occlusion_metrics),
        ("MOT17 ablation", "mot17_outputs/MOT17-02-FRCNN-public-seed0-summary.json", plot_mot17_ablation),
        ("embedding diagnostics", "mot17_outputs/MOT17-02-FRCNN-public-seed0-summary.json", plot_embedding_diagnostics),
        ("contamination rollback", "contamination_rollback_results.json", plot_contamination_rollback),
        ("behaviour feedback", "behaviour_feedback_results.json", plot_behaviour_feedback),
        ("CECS proxy", "cecs_pilot_results/cecs_MOT17-02-FRCNN_summary.json", plot_cecs_proxy),
    ]
    print(f"Reading results from {ROOT}")
    for title, rel_path, plot_fn in steps:
        data = load_json(rel_path)
        if data is None:
            continue
        try:
            plot_fn(data)
        except Exception:  # keep going if one result file has an unexpected shape
            print(f"[fail] {title}:")
            traceback.print_exc()
    print(f"\nDone. Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
