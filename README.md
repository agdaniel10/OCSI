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
pytest -q          # from the repository root
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

## Running X-OCSI on Kaggle

CECS needs the detector attribution `E_det`, and only `detection_source="yolo"`
produces it: MOT17's public `det.txt` files contain boxes but no detector to
attribute. **Run the whole ablation table on YOLO detections** so the `x-ocsi`
row is comparable with the rows above it; mixing public rows with a YOLO
`x-ocsi` row compares different detections.

```bash
# in a Kaggle notebook cell, from the repository root
!pip install -q ultralytics pyyaml    # torch/torchvision/opencv are preinstalled
!python kaggle_xocsi.py --smoke      # 1 sequence, 30 frames, wiring check (~2 min on GPU)
!python kaggle_xocsi.py              # full table, all MOT17 train sequences
```

`kaggle_xocsi.py` discovers `MOT17-*/img1` sequences with `gt/gt.txt` (deduping
the DPM/FRCNN/SDP variants), runs
`baseline, memory, adaptive_memory, feedback, x-ocsi` with
`detection_source="yolo"`, prints IDF1/MOTA/IDsw plus CECS mean/count and
attribution coverage, and writes `ocsi_outputs/xocsi-metrics.csv` plus one
`*-cecs.json` trace per run. Useful flags: `--sequences`, `--limit`, `--seeds`,
`--imgsz`, `--heatmap-size`, `--rebuild`.

Practical notes:

- Enable a GPU. Grad-CAM adds one backward pass per frame; on a P100/T4 the
  full MOT17 train split is tens of minutes, on CPU it is hours.
- Attribution is cached per detection in `<key>.xai.npz` sidecars, downsampled
  to `cfg.xai.heatmap_size` (default 224×224) and quantised to `uint8`. A full
  MOT17 sequence becomes a few tens of MB instead of tens of GB.
- The cache key includes the detection source but **not** the weights, `imgsz`
  or `heatmap_size`; pass `--rebuild` after changing any of them. A partial cache
  (from `--smoke`, `--limit`, or an interrupted session) is detected and the
  missing frames are built on the next run.
- `/kaggle/working` is wiped between sessions. Copy `ocsi_outputs/` (or save it
  as a notebook output) before stopping; re-running detection on a fresh
  session rebuilds the cache from scratch.
- `yolov8s.pt` and the ImageNet Re-ID weights are downloaded on first use
  (needs internet). Without them Re-ID falls back to random init with a
  warning — the run completes, but identity metrics are not meaningful.
- A stage that enables X-OCSI but finds no attribution now reports
  `cecs.num_cecs = 0` with an explicit `warning` and a `RuntimeWarning`, instead
  of silently omitting the block.

MOT17 still has no activity labels, so the temporal term is structurally wired
but unvalidated there; report it as such.

## Layout

```
ocsi/
  config.py         # all thresholds/weights (fills the paper's blank table); ablation presets
  types.py          # Detection, TrackState, bbox helpers, ExplanationEvent/ExplanationInput
  perception/       # Re-ID embedder + YOLO detector + feature cache       (Phase 1 ✓, pose TODO)
  memory/           # Object Memory Bank: record.py + bank.py             (Phase 2 ✓)
  motion/           # constant-velocity Kalman filter                     (Phase 3 ✓)
  identity/         # similarity, gating, Hungarian association, tracker  (Phase 3 ✓)
  behaviour/        # HAR (+ stub) and the confidence gate / feedback     (Phases 4-5)
  explain/          # X-OCSI CECS + identity explanation provenance
  xai/              # Grad-CAM, detection attribution, temporal saliency
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
- **X-OCSI explanation wiring (engine + producers)**: `apply_ablation(cfg, "x-ocsi")`
  enables cross-stage provenance + CECS; the tracker records
  association/detection/temporal events per identity and `track_sequence` exposes
  the live tracker. Producers are real: per-detection head Grad-CAM (`E_det`) and
  perturbation-based HAR saliency (`s_t`), cached per frame. A trained HAR model
  and label-dependent CECS validation remain TODO (MOT17 has no activity labels).
- **179 tests pass** (`python -m pytest -q` from the repository root).

## Loophole Fixes (v1.1)

The following improvements address the weaknesses identified in the reproduction report
and the AI review:

1. **Contamination rollback integrated into the tracker** (`ocsi/identity/tracker.py`):
   The tracker now detects likely incorrect matches (appearance cosine below a threshold
   on a high-confidence track) and rolls back the memory record after consecutive conflicts.
   Config: `ocsi/config.py` → `ContaminationConfig`.

2. **Leakage-aware threshold calibration** (`ocsi/experiments/mot17_tracking.py`):
   `run_mot17_dataset` now supports `calibration_sequences` and `evaluation_sequences`
   parameters. The reactivation gate is calibrated on the calibration sequences only,
   then applied to the evaluation sequences — preventing threshold-selection leakage.

3. **Person-ReID backbone with auto fallback** (`ocsi/perception/embedder.py`):
   The default `reid_backend` is now `"auto"`, which tries `torchreid/OSNet` first,
   then `torchvision/resnet50`, then `torchvision/resnet18`. This ensures the strongest
   available appearance model is used.

4. **Pose and context cues** (`ocsi/identity/tracker.py`):
   The tracker now computes pose-similarity and context-similarity terms when pose
   keypoints and context features are available. The `Detection` type now carries a
   `context` field, and `MemoryRecord` populates the `Q_c` context queue.

5. **Adaptive weights** (`ocsi/identity/tracker.py`):
   When `cfg.association.adaptive_weights` is enabled, cue weights are modulated by
   occlusion ratio, motion uncertainty, and behaviour confidence (paper §4.3).

6. **Temporal HAR model** (`ocsi/behaviour/temporal_har.py`):
   A genuine temporal HAR recognizer with temporal attention pooling and trainable
   class prototypes. This replaces the placeholder recognizer for real behaviour
   feedback experiments.

7. **Official TrackEval integration** (`ocsi/eval/trackeval.py`):
   A wrapper around the official TrackEval toolkit for HOTA/DetA/AssA/MOTA/IDF1.

8. **Controlled baselines** (`ocsi/experiments/baselines/`):
   DeepSORT, ByteTrack, and OC-SORT baseline trackers that use the same Kalman filter
   and detection inputs as OCSI, enabling fair comparison.

9. **MOT20 support** (`ocsi/experiments/mot20.py`):
   An adapter for the MOT20 benchmark (denser crowds, more occlusion).

10. **Effect sizes in statistics** (`ocsi/eval/statistics.py`):
    `paired_stats` now reports Cohen's d and rank-biserial correlation alongside
    p-values.

11. **Efficiency reporting** (`ocsi/experiments/mot17_tracking.py`):
    `track_cached_sequence` now records per-frame timing, FPS, latency, and
    memory-bank statistics on the tracker instance.

12. **Improved reliability proxy** (`ocsi/identity/tracker.py`):
    `_reliability` now combines detection confidence, match score, assignment margin,
    and cross-cue consistency (paper §3.4 step 8).

## X-OCSI cross-stage explanations (paper §3.1–3.6)

X-OCSI adds identity-persistent explainability on top of OCSI. It is **off by
default** and enabled per run:

```python
from ocsi.config import OCSIConfig, apply_ablation

cfg = apply_ablation(OCSIConfig(), "x-ocsi")   # aliases: "xai", "xocs", "x_ocsi"
assert cfg.xai.enabled
```

The XAI/HAR layers attach evidence to each detection via `ExplanationInput`
(`det_heatmap`, `act_heatmap`, `saliency`, `reliability`, `uncertainty`). The
tracker then writes three event kinds to that identity's
`MemoryRecord.explanation_history`:

- `association` — the per-cue `q_ij` similarities, active weights, margin and
  reliability, i.e. why the identity was maintained (paper Eq. 4);
- `detection` — the detector attribution `E_det` stored with its box (Eq. 2–3);
- `temporal` — HAR temporal saliency `s_t` with aligned tracking reliability
  `r_t` and identity uncertainty `u_t` (Eq. 7).

### Producers

Both evidence producers are real models, not placeholders:

- **Detector attribution** — `ocsi.xai.DetectorGradCAM` runs Grad-CAM on the
  detection head itself, so `E_det^y` is per detection rather than per frame. It
  selects the anchor grid matching the box size, masks the backward gradient to
  that box's anchor footprint (plus one cell of receptive field), and
  un-letterboxes the map back to source resolution. `YOLODetector.explain(...)`
  wraps it; it works with YOLOv8-style (`4 + nc`) and YOLOv5-style (`5 + nc`)
  heads and re-enables gradients on inference checkpoints (Ultralytics loads
  weights with `requires_grad=False`).
- **Temporal saliency** — `frame_perturbation_saliency` replaces each frame of
  the HAR window with the window mean and re-runs the recogniser, so `s_t` is the
  measured drop in the chosen activity's probability. `har_score_fn(recognizer)`
  adapts any `BehaviourRecognizer`. The distance heuristic is used only when
  `model_forward=None`.

`r_t` and `u_t` are **owned by the tracker**, not the perception layer: `r_t` is
the per-match reliability and `u_t = 1 - gap` is the assignment ambiguity, where
`gap` is the smaller row/column score margin over gate-admissible pairs (zero for
a single candidate track). The tracker keeps a rolling window aligned to the HAR
window length. `CECSCalculator.compute(allow_proxies=False)` is therefore the
only call site; partial chains raise instead of silently substituting terms.

### Cache + reporting

Attribution is as expensive as detection, so it is cached with the detections in
per-frame `<key>.xai.npz` sidecars (compressed, detection-indexed). Maps are
downsampled to `cfg.xai.heatmap_size` (default 224×224) before storage and
quantised to `uint8`; boxes and `img_shape` stay in source pixels, so
energy-in-box and soft-IoU are unaffected. Rebuild the cache after enabling
X-OCSI to populate them. Reading the live state:

```python
from ocsi.pipeline import track_sequence

rows, tracker = track_sequence(detections_per_frame, cfg)
for row in tracker.cecs_history:
    print(row)                                  # frame, track_id, S_*, CECS
rec = next(iter(tracker.bank.records.values()))
print(rec.get_explanations("association")[-1].metadata)
```

`run_mot17_sequence` / `run_mot20_sequence` write a
`<result>-cecs.json` sidecar and add a compact `cecs` block to the stage metrics
when the `x-ocsi` stage is run. Attribution is requested automatically at
cache-build time whenever any requested stage enables it, including the
calibration pass of `run_mot17_dataset`. If a stage enables X-OCSI but no
attribution reached the cache, the block still appears with `num_cecs: 0`, a
`warning`, and the measured attribution coverage.

Caveats: `MOT17`/`MOT20` carry no activity labels, so the temporal term is
structural there (a recogniser runs, but its accuracy is unmeasured); the
`public` detection source has no detector to attribute, so only `yolo` runs
produce `E_det`; and `S_spatial` falls back to energy-in-box when only one
attribution map exists. The earlier `scripts/cecs_mot_pilot.py` numbers remain a
box-IoU proxy, not CECS — do not cite them.

## Ethics

Persistent identity memory carries privacy obligations. This is a research prototype for
situational understanding, not autonomous decision-making. Follow the paper's §8.4 guidance
(data minimisation, short retention, access control, human review) before any deployment.
