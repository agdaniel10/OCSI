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

Enable **GPU T4/P100** in notebook settings. The detector and Re-ID weights download
once on first run if internet is enabled.

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

### Full run (all discovered videos)

```python
!cd /kaggle/working/OCSI && python kaggle_ucf_crime.py \
    --data-dir /kaggle/input --frames 300 --stride 5 \
    --stages baseline,memory,feedback,adaptive_memory
```

Detections are cached under `/kaggle/working/ocsi_ucf_cache`, so reruns with different
stages are fast. Add `--refresh-cache` if you change detector/Re-ID settings.

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
| `--smoke` | off | 2 videos, 60 frames, baseline+memory |
| `--refresh-cache` | off | Rebuild cached detections |
| `--no-plots` | off | Skip PNG figures |

## 4. Outputs

In `/kaggle/working/ocsi_ucf_outputs/`:

- `ucf_crime_results.json` — config, video list, video-level AUC, per-video summary, caveat note
- `ucf_crime_summary.csv` — one row per video × stage (frame AUC, video score, IDs, flags)
- `ucf_crime_frame_scores.csv` — one row per sampled frame with all signals:
  `anomaly_score`, `mean_q`, `n_active`, `n_lost`, `new_ids`, `mean_gate`, `gate_fraction`, …
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
