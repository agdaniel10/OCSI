# OCSI — Object-Centric Surveillance Intelligence

A memory-driven, closed-loop framework for multi-object tracking (MOT) and human
activity understanding, implementing the novel contributions of the OCSI paper on
top of proven, pretrained components.

The three contributions built here as new code:

1. **Object Memory Bank** (`ocsi/memory/`) — a persistent, bounded, confidence-aware
   per-identity record (appearance prototype + gallery, trajectory/velocity/pose/context
   queues, behaviour prototype, confidence, state) with EMA updates, decay, retention,
   and contamination rollback.
2. **Unified confidence-gated association** (`ocsi/identity/`) — a single similarity
   score over appearance, motion, IoU, memory and (gated) behaviour, solved with the
   Hungarian algorithm after gating.
3. **Behaviour-guided feedback loop** (`ocsi/behaviour/`) — high-confidence activity
   evidence, gated by `g = 1[p_max ≥ θ_b]·p_max·exp(−Δt/τ_b)`, feeds back into
   association to refine, reject, or reactivate identities.

## Scope (v1)

- **Novel core on a proven base**: pretrained YOLO (detection), a Re-ID embedder, and a
  Kalman/Hungarian base we extend. CPU-only; small models; everything pluggable.
- **Out of scope for now**: training from scratch, transformer-tracker variant, full
  multi-benchmark reproduction, cross-camera mode, real-time optimization.

See `../OCSI_paper_extracted.txt` for the source paper and the approved plan for design detail.

## Install

```bash
# core + tests (torch/opencv/numpy/scipy already present in this environment)
pip install -e ".[dev]"

# perception adapters (Phase 1+)
pip install -e ".[perception]"

# evaluation metrics (Phase 6)
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

## Run tests

```bash
pytest -q          # from the src/ directory
```

## MOT17 evaluation from Colab/Jupyter

MOT17 has tracking labels but no action labels, so it validates OCSI contributions
1 and 2 (memory + association). Contribution 3 (behaviour feedback) is measured
separately by `python -m ocsi.experiments.behaviour_feedback`.

```python
from ocsi.experiments.mot17_tracking import run_mot17_sequence

payload = run_mot17_sequence(
    seq_dir="/content/MOT17/train/MOT17-02-FRCNN",
    cache_dir="/content/drive/MyDrive/ocsi_cache/MOT17-02-FRCNN",
    output_dir="/content/ocsi_outputs",
    stages=("baseline", "memory"),
    detection_source="public",  # use MOT17 det.txt; use "yolo" to regenerate detections
)

for result in payload["results"]:
    print(result["stage"], result["summary"])
    print("MOT result file:", result["result_path"])
```

The first run builds cached YOLO + Re-ID detections under `cache_dir`; later runs
replay those cached features and only rerun the tracker/ablation logic. Use the
written result files with TrackEval for HOTA/AssA/DetA.

For a broader ablation pass:

```python
from ocsi.experiments import run_mot17_dataset

payload = run_mot17_dataset(
    seq_dirs=[
        "/content/MOT17/train/MOT17-02-FRCNN",
        "/content/MOT17/train/MOT17-04-FRCNN",
    ],
    cache_root="/content/drive/MyDrive/ocsi_cache",
    output_dir="/content/ocsi_outputs",
    stages=("baseline", "memory", "feedback"),
    detection_source="public",
    seeds=(1, 2, 3),
)
```

The `feedback` stage is included for consistent ablation bookkeeping, but MOT17
has no action labels; behaviour feedback remains a separate synthetic-gating
experiment.

## Layout

```
ocsi/
  config.py         # all thresholds/weights (fills the paper's blank table); ablation presets
  types.py          # Detection, TrackState, bbox helpers
  perception/       # Re-ID embedder + YOLO detector + feature cache       (Phase 1 ✓, pose TODO)
  memory/           # Object Memory Bank: record.py + bank.py             (Phase 2 ✓)
  motion/           # constant-velocity Kalman filter                     (Phase 3 ✓)
  identity/         # similarity, gating, Hungarian association, tracker  (Phase 3 ✓)
  behaviour/        # HAR (+ stub) and the confidence gate / feedback     (Phases 4-5)
  pipeline.py       # per-frame + full-sequence orchestration             (Phase 3.5 ✓)
  eval/             # MOT I/O + CLEAR-MOT/IDF1 metric + (TrackEval later)  (Phase 3.5 ✓)
configs/default.yaml
tests/
```

## Status

- **Phase 0 — scaffolding & config**: done.
- **Phase 1 — perception adapters**: Re-ID embedder (`ReIDEmbedder`, torchvision backbone),
  person detector (`YOLODetector`, lazy Ultralytics import; the pure `filter_detections` logic
  is tested without it), and a two-tier `FeatureCache` — done and tested. Pose is deferred (it
  feeds HAR in Phase 4). Caveat: the detector and *pretrained* Re-ID weights need the
  `perception` extra + a one-off download; **random-init features are non-discriminative**
  (verified: cross-identity cosine ≈ 0.98), so real weights are required before appearance/memory
  can beat the geometry-only baseline.
- **Phase 2 — Object Memory Bank + unit tests**: done.
- **Phase 3 — motion (Kalman), similarity/gating/association, tracker loop**: done.
- **Phase 3.5 — MOT I/O, CLEAR-MOT/IDF1 metric, sequence-runner pipeline**: done. The
  tracker runs end-to-end on synthetic or public detections (no models needed) and is
  scored offline; `apply_ablation` drives the baseline/memory/feedback presets.
- **Occlusion recovery (contributions #1 + #2), measured**: `ocsi/experiments/occlusion_recovery.py`
  replaces the source notebook's *fabricated* recovery formula with a real, deterministic
  study — distinct textured identities that maneuver while hidden, **real pretrained
  Re-ID features**, the real tracker, real metrics. It adds the paper's §3.4 lost-track
  **reactivation** step (appearance re-association bypassing the IoU floor). Result:

  | occlusion length L | 1 | 2 | 4 | 8 | 16 | 24 | 40* |
  |---|---|---|---|---|---|---|---|
  | baseline recovery | 100% | 100% | 0% | 0% | 0% | 0% | 0% |
  | +memory recovery  | 100% | 100% | 100% | 100% | 100% | 100% | 17% |

  Aggregate: baseline MOTA 0.837 / IDF1 0.348 / IDsw 30 → +memory MOTA 0.964 / IDF1 0.789 /
  IDsw 5. (*L=40 > `max_age`=30: the bounded-memory retention cliff.) Run:
  `python -m ocsi.experiments.occlusion_recovery`.
- **90 tests pass** (`python -m pytest -q` from `src/`).
- Next: the **behaviour** module (Phases 4-5). NOTE: real AVA HAR needs the annotation CSV +
  videos (the `ava_train_excluded_timestamps` file is only AVA's exclusion blacklist); the
  recommended real path is a pretrained action model feeding the confidence-gated feedback
  loop. Then optional Phase 6 (TrackEval/HOTA + CLI).

## Ethics

Persistent identity memory carries privacy obligations. This is a research prototype for
situational understanding, not autonomous decision-making. Follow the paper's §8.4 guidance
(data minimisation, short retention, access control, human review) before any deployment.
