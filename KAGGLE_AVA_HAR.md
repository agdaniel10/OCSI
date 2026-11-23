# AVA behaviour-feedback experiment on Kaggle

This runner validates the **behaviour layer** of OCSI with real person-centred
action labels. It replaces the "placeholder activity prototypes" limitation with
AVA-trained class prototypes and tests the confidence gate under injected
low-confidence behaviour noise.

It does **not** measure IDF1/IDsw/MOTA. Use MOT17/MOT20 for tracking claims.

## What AVA provides

- `video_id`, timestamp, person box
- multi-label atomic action IDs (up to 80 classes)

AVA's `person_id` is a **per-frame box index**, not a persistent identity.
The runner therefore links ground-truth boxes across neighbouring annotated
timestamps with IoU to form clearly-labelled *pseudo-tracklets*.

## Verified Kaggle datasets to attach

I checked the Kaggle dataset API. These are public and currently exist:

| Purpose | Kaggle link | Use as |
|---|---|---|
| Official AVA v2.2 annotations | <https://www.kaggle.com/datasets/mradul2/ava-official> | `--annotations-dir` |
| AVA video subset (65 clips, ~33 GB) | <https://www.kaggle.com/datasets/zamankhan1409/ava-65-videos> | `--video-root` |

Attach **both** to the same Kaggle notebook. The annotation dataset supplies
`ava_train*.csv` / `ava_val*.csv`; the video dataset supplies the `.mp4` clips
named by AVA `video_id`. With 65 videos, use `--max-videos 30` for a first
full-scale behaviour-layer run.

Other Kaggle AVA datasets found but **not** recommended for this runner:

- <https://www.kaggle.com/datasets/lucianobautista/human-actions-dataset-extracted-from-ava-google>
  — YOLO-format labels, not the AVA CSV format this script parses.
- <https://www.kaggle.com/datasets/jacopocrispolti/ava-dataset>
  — appears in search but returns 404 / is not reliably accessible.
- `Aesthetic Visual Analysis (AVA)` datasets — these are image-preference AVA,
  a completely different dataset.
- <https://www.kaggle.com/datasets/ahmedtahashiha/hajj-all-frames>
  — contains AVA-style `_2025` CSVs and frames, but is a renamed/modified
  Hajj-focused variant; use only after inspecting it.

The runner auto-discovers CSVs under `--annotations-dir` and videos under
`--video-root`, but explicit paths are safer. After attaching, run:

```python
!find /kaggle/input -maxdepth 5 -type f \( -name "*.csv" -o -name "*.mp4" \) | head -100
```

```python
!python kaggle_ava_har.py \
    --annotations-dir /kaggle/input/ava-official \
    --train-csv /kaggle/input/ava-official/ava_train_v2.2.csv \
    --val-csv /kaggle/input/ava-official/ava_val_v2.2.csv \
    --excluded-csv /kaggle/input/ava-official/ava_train_excluded_timestamps_v2.2.csv \
    --video-root /kaggle/input/ava-65-videos \
    --max-videos 30 \
    --max-timestamps-per-video 300 \
    --window 8 \
    --min-window 4 \
    --noise-fraction 0.25
```

Enable a Kaggle GPU. The script uses a pretrained Re-ID backbone to embed GT
person crops; if weights cannot be downloaded it stops with a random-init
warning. Use `--allow-random-reid` **only** for pipeline debugging, never for
paper results.

## What the script does

1. Parses AVA annotations and removes excluded timestamps.
2. Builds pseudo-tracklets by IoU-linking boxes at adjacent 1-second annotations.
3. Extracts one person crop embedding per annotated box and caches it under
   `/kaggle/working/ocsi_ava_cache/`.
4. Builds multi-label temporal windows and trains `TemporalHARRecognizer` class
   prototypes from the train split.
5. Evaluates on the val split:
   - action mAP
   - macro-F1
   - confidence expected calibration error (ECE)
6. Runs a real-data gate ablation:
   - `no_feedback`: ignore behaviour observations
   - `ungated`: full-trust behaviour, including low-confidence wrong observations
   - `gated`: `behaviour_gate()` ignores the low-confidence wrong observations

The noise injection is synthetic; the action labels and HAR confidences are real.

## Outputs

Under `/kaggle/working/ocsi_ava_outputs/`:

| File | Contents |
|---|---|
| `ava_har_results.json` | config, split sizes, HAR metrics, gate ablation |
| `ava_action_metrics.csv` | per-action AP, support, F1 |
| `ava_gate_ablation.csv` | mAP and cosine-to-clean for the three feedback modes |

## How to interpret the gate result

A supported gate result should look like:

```text
ungated  mAP lower, cosine_to_clean lower
gated    mAP close to no_feedback / clean, cosine_to_clean higher
```

If `gated` does not preserve prototype coherence better than `ungated`, the
behaviour-feedback claim is not supported by this run.

## Recommended paper wording

> AVA person boxes with atomic action labels were used to train and validate the
> behaviour layer. Because AVA does not provide persistent tracks, pseudo-tracklets
> were formed by IoU-linking ground-truth boxes across adjacent annotations.
> The confidence gate was tested by injecting low-confidence incorrect behaviour
> observations; gated feedback preserved action-prototype coherence better than
> ungated feedback. Tracking metrics were evaluated separately on MOT17/MOT20.

## Caveats

- Not an official AVA HAR submission; this validates OCSI's lightweight recognizer.
- Pseudo-tracklets are not ground-truth trajectories.
- GT boxes are used for HAR training/evaluation, so detector error is not included.
- No IDF1/IDsw/MOTA can be claimed from AVA.
- Save `/kaggle/working/ocsi_ava_outputs/` before the Kaggle session ends.
