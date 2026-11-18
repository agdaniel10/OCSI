# UCF-Crime testing on Kaggle (OCSI)

Fast, reproducible commands for testing OCSI's anomaly/cognitive layer on UCF-Crime.
The runner is `kaggle_ucf_crime.py`; a ready-made notebook is `kaggle_ucf_crime.ipynb`.

> **Scope / honesty note.** UCF-Crime has video-level labels (and frame intervals for the
> test split), but **no person boxes or person IDs**. These runs can report frame-level
> and video-level anomaly AUC from OCSI's memory/uncertainty signals. They **cannot**
> validate IDF1, MOTA, ID switches, or per-person behaviour feedback. Use MOT17/MOT20
> for those claims.

## 1. One-time setup in a Kaggle notebook

```python
# Clone or pull the latest repo
!if [ -d /kaggle/working/OCSI/.git ]; then cd /kaggle/working/OCSI && git pull --ff-only; else git clone https://github.com/agdaniel10/OCSI.git /kaggle/working/OCSI; fi

# Dependencies (torch/torchvision/opencv are preinstalled on Kaggle)
!pip -q install ultralytics pyyaml opencv-python pandas matplotlib
```

Attach a UCF-Crime dataset with **Add Data**, or download one:

```python
import kagglehub
data_dir = kagglehub.dataset_download("sabbirahmad653/ucf-crime-datasets")
```

Other Kaggle dataset slugs that work with the auto-discovery:

- `sabbirahmad653/ucf-crime-datasets`
- `minhajuddinmeraj/anomalydetectiondatasetucf`
- `umongsain/ucf-crime-labels` (labels only; pair with a video dataset)
- `hiunguynththanh/ucfcrime-full`

Enable **GPU T4 x2** in notebook settings. The detector and Re-ID weights download
once on first run if internet is enabled. Use **Save & Run All** for long runs so Kaggle
keeps the session alive after you close the tab (12-hour session cap; ~30 GPU hours/week).

## 2. Quick commands

### Smoke test (2 videos, 60 frames, ~2–5 min)

```python
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py --smoke --data-dir /kaggle/input
```

### Balanced subset (recommended first real run)

```python
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py \
    --data-dir /kaggle/input \
    --limit-videos 12 --frames 300 --stride 5 \
    --stages baseline,memory,feedback
```

### Focus on specific classes / videos

```python
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py \
    --data-dir /kaggle/input \
    --classes Normal,Fighting,Assault,Abuse,Stealing,Shoplifting,Robbery \
    --frames 300 --stride 5 \
    --stages baseline,memory,feedback

# or by video-name substring
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py \
    --data-dir /kaggle/input --videos Abuse028,Fighting010,Normal001
```

### Full dataset run (all discovered videos)

Yes — just **omit `--limit-videos` and `--smoke`**. With default `--frames 300
--stride 5`, "full dataset" means all ~1,900 videos × 300 sampled frames =
**~570,000 YOLO + Re-ID inferences**, not every source frame.

```python
# Recommended full run: 320px + batch 32 keeps it inside a 12h Kaggle session
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py \
    --data-dir /kaggle/input \
    --frames 300 --stride 5 \
    --imgsz 320 --batch-size 32 \
    --stages baseline,memory,feedback \
    --resume \
    --out /kaggle/working/ocsi_ucf_outputs \
    --cache /kaggle/working/ocsi_ucf_cache
```

For maximum resolution (`--imgsz 640 --batch-size 16`), omit those two flags.
Expect longer runtime.

#### Time estimate (T4 x2, 4 CPU cores)

| Settings | Detections | Est. runtime |
|---|---:|---:|
| `300 frames`, `imgsz 320`, `batch 32`, 3 stages | ~570k | **~7–10 h** (fits 12 h cap, tight) |
| `300 frames`, `imgsz 640`, `batch 16`, 3 stages | ~570k | **~10–14 h** — likely exceeds one session |
| same, but `--stages baseline,memory` | ~570k | saves ~1–2 h |
| official test split only (290 videos), same settings | ~87k | **~1–1.5 h** |
| **every source frame** (`--frames 100000 --stride 1`) | ~13–14M | **~4–8 days** — not one session |

These figures include both perception and tracking. A local benchmark of the tracker
alone over 300 frames × 1,900 videos × 3 stages took **~3.8 h at 5 detections/frame**
and **~7.5 h at 10 detections/frame**; Kaggle's 4 CPU cores are in that range. Batched
inference keeps detection to ~3–4 h at 320 px, so the 640 px full run usually needs
chunking or a second session. Use `--resume` either way.

> If you literally need all ~13M frames, split the dataset into chunks with
> `--videos` or `--classes` across sessions, or raise `--frames`/`--stride` only
> for anomaly intervals you care about.

#### Resume / crash safety

The runner now appends one line per completed video/stage to
`ucf_crime_progress.jsonl` and streams per-frame rows to
`ucf_crime_frame_scores.csv`. Re-run with `--resume` to skip finished stages and
reuse the detection cache in `ocsi_ucf_cache/`. Checkpoints carry a settings
signature, so changing frames/stride/imgsz/stages starts a fresh set of rows
instead of silently skipping with incompatible settings.

`/kaggle/working` is wiped between sessions. To truly resume across sessions,
save the `outputs/` and `cache/` folders as a Kaggle dataset or mount Google Drive
and point `--out` / `--cache` there.

## 3. Flags

| Flag | Default | Meaning |
|---|---:|---|
| `--data-dir` | `/kaggle/input` (auto) | Root containing UCF videos |
| `--labels` | auto | Explicit `Temporal_Anomaly_Annotation*.txt` |
| `--out` | `/kaggle/working/ocsi_ucf_outputs` | Results directory |
| `--cache` | `/kaggle/working/ocsi_ucf_cache` | Detection cache |
| `--videos` | – | Comma-separated video-name substrings |
| `--classes` | – | Comma-separated UCF classes |
| `--limit-videos` | – | Balanced subset size (round-robin across classes) |
| `--frames` | `300` | Sampled frames per video |
| `--stride` | `5` | Original-frame step between samples |
| `--stages` | `baseline,memory,feedback` | OCSI ablations |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--imgsz` | `640` | YOLO input size |
| `--det-conf` | `0.15` | Person detection threshold |
| `--reid-backend` | `auto` | `auto` / `torchvision` / `torchreid` |
| `--batch-size` | `16` | Frames per YOLO/Re-ID batch (use 32 at `imgsz 320`) |
| `--smoke` | off | 2 videos, 60 frames, baseline+memory |
| `--refresh-cache` | off | Rebuild cached detections |
| `--resume` | off | Skip video/stage pairs already in the progress file |
| `--no-plots` | off | Skip PNG figures |

## 4. Outputs

In `/kaggle/working/ocsi_ucf_outputs/`:

- `ucf_crime_results.json` — config, video list, video-level AUC, per-video summary, caveat note
- `ucf_crime_summary.csv` — one row per video × stage (frame AUC, video score, IDs, flags)
- `ucf_crime_frame_scores.csv` — one row per sampled frame with all signals:
  `anomaly_score`, `mean_q`, `n_active`, `n_lost`, `new_ids`, `mean_gate`, `gate_fraction`, …
- `ucf_crime_progress.jsonl` — append-only checkpoint, one line per completed video/stage
- `ucf_crime_auc.png` — mean frame AUC by stage
- `ucf_crime_scores.png` — per-frame score curves for the first annotated anomaly video

## 5. How the anomaly score is built

Per sampled frame the runner logs tracker signals it already computes:

```
score = 0.40 * norm(new IDs created)
      + 0.30 * norm(lost tracks)
      + 0.30 * (1 - norm(mean memory confidence q))
```

`norm()` rescales to `[0, 1]` within each video. For `feedback`, it additionally logs
`mean_gate` from the same confidence gate `g = 1[p_max ≥ θ_b] · p_max · exp(-dt/τ_b)`
that OCSI applies to behaviour cues. Frame AUC uses the temporal anomaly intervals from
the annotation file; video AUC separates Normal from all anomaly classes using the 90th
percentile frame score per video.

## 6. Troubleshooting

- **No videos found** — pass the exact dataset path, e.g.
  `--data-dir /kaggle/input/ucf-crime-datasets/UCF_Crime_Videos`.
- **No frame AUC reported** — the annotation `.txt` was not found. Pass `--labels /path/to/Temporal_Anomaly_Annotation.txt`. Video-level AUC still works from class labels.
- **Slow first run** — one-off weight downloads and cache building; subsequent runs are fast.
- **CUDA OOM** — use `--imgsz 320 --frames 150 --stride 10`, or `--device cpu` for a tiny smoke test.
- **Class inferred as `Unknown`** — use `--classes`/`--videos` explicitly, or place videos in folders named after their UCF class.
