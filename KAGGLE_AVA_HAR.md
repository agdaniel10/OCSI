# AVA behaviour-feedback experiment on Kaggle

This runner validates the **behaviour layer** of OCSI with real person-centred
action labels. It replaces the "placeholder activity prototypes" limitation with
AVA-trained class prototypes and tests the confidence gate under injected
low-confidence behaviour noise.

It does **not** measure IDF1/IDsw/MOTA. Use MOT17/MOT20 for tracking claims.

## What AVA provides

- `video_id`, timestamp, person box
- multi-label atomic action IDs (up to 80 classes)

AVA v2.2's `person_id` links boxes of the same person in **adjacent annotated
frames** (per the official format description). The runner uses that ID first to
form short *pseudo-tracklets*, with box IoU as a fallback where the ID link is
missing. These are still short spans, not long-term MOT trajectories.

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
!cd /kaggle/working/OCSI && git pull
!python kaggle_ava_har.py \
    --annotations-dir /kaggle/input/datasets/mradul2/ava-official/ava_files/annotations \
    --train-csv /kaggle/input/datasets/mradul2/ava-official/ava_files/annotations/ava_train_v2.2.csv \
    --val-csv /kaggle/input/datasets/mradul2/ava-official/ava_files/annotations/ava_val_v2.2.csv \
    --excluded-csv /kaggle/input/datasets/mradul2/ava-official/ava_files/annotations/ava_train_excluded_timestamps_v2.2.csv \
    --video-root /kaggle/input/datasets/zamankhan1409/ava-65-videos \
    --max-videos 25 \
    --max-timestamps-per-video 300 \
    --window 8 \
    --min-window 4 \
    --noise-fraction 0.25 \
    --theta-b auto \
    --theta-quantile 0.5
```

`--max-videos` now applies **per split**, so 25 gives up to 25 train and 25 val
videos instead of the previous 23 train / 2 val skew. The runner also filters to
videos that actually exist in the attached clip subset (or have cached crops)
**before** applying `--max-videos`; otherwise many selected videos print `0
crops` because the 65-clip Kaggle subset does not cover all AVA videos.

Enable a Kaggle GPU. The script uses a pretrained Re-ID backbone to embed GT
person crops; if weights cannot be downloaded it stops with a random-init
warning. Use `--allow-random-reid` **only** for pipeline debugging, never for
paper results.

## What the script does

1. Parses AVA annotations and removes excluded timestamps.
2. Builds short pseudo-tracklets from AVA `person_id` continuity, falling back to
   IoU linking of boxes at adjacent 1-second annotations.
3. Extracts one person crop embedding per annotated box and caches it under
   `/kaggle/working/ocsi_ava_cache/`.
4. Builds multi-label temporal windows and trains `TemporalHARRecognizer` class
   prototypes from the train split.
5. Evaluates on the val split:
   - action mAP (primary AVA-style ranking metric)
   - top-1 macro-F1
   - macro-F1 at the naive 0.5 threshold (usually degenerate for 80 classes)
   - confidence expected calibration error (ECE)
6. Runs a real-data gate ablation with four modes:
   - `no_feedback`: ignore behaviour observations (prototype frozen)
   - `clean_feedback`: full-trust updates from clean observations only
   - `ungated`: full-trust updates including injected wrong observations
   - `gated`: `behaviour_gate()` ignores the injected low-confidence rows

Wrong observations are drawn from windows with **disjoint action labels** and low
`p_max`, rather than from random crops that may be semantically similar. The gate
threshold is calibrated automatically as a quantile of the clean `p_max`
distribution (`--theta-b auto --theta-quantile 0.5` by default), because a fixed
`theta_b=0.6` is usually never reached by an 80-class softmax.

The noise injection is synthetic; the action labels and HAR confidences are real.

## Outputs

Under `/kaggle/working/ocsi_ava_outputs/`:

| File | Contents |
|---|---|
| `ava_har_results.json` | config, split sizes, HAR metrics, gate ablation |
| `ava_action_metrics.csv` | per-action AP, support, top-1 F1, F1 at 0.5 |
| `ava_gate_ablation.csv` | mAP and cosine-to-clean for the four feedback modes, with the effective `theta_b` |

## How to interpret the gate result

A supported gate result should show all of the following:

```text
clean_feedback  mAP / cosine_to_clean best or near-best
ungated         degraded relative to clean_feedback under injected noise
gated           close to clean_feedback, and clearly better than ungated
gated           cosine_to_clean > ungated cosine_to_clean
```

`gated == no_feedback` means the gate never opened; that is a threshold
calibration failure, not evidence for the gate. `ungated > gated` means the
injected observations were not harmful and the ablation cannot support the
claim.

A fixed high `theta_b` (e.g. 0.6 with 80 softmax classes) often produces exactly
that failure. Inspect `har.p_max` in the JSON and the effective `theta_b` in the
ablation CSV before drawing conclusions.

## Recommended paper wording

> AVA person boxes with atomic action labels were used to train and validate the
> behaviour layer. Short pseudo-tracklets were formed using AVA `person_id`
> continuity across adjacent annotated frames, with box overlap as a fallback;
> these are not long-term trajectories. The confidence gate was tested by
> injecting low-confidence, action-disjoint incorrect behaviour observations.
> The threshold was calibrated from the observed confidence distribution.
> Tracking metrics were evaluated separately on MOT17/MOT20. Only state that
> gated feedback preserved coherence better than ungated feedback if the
> ablation CSV actually shows it.

## Caveats

- Not an official AVA HAR submission; this validates OCSI's lightweight recognizer.
- Pseudo-tracklets from AVA are short, adjacent-frame links, not long-term trajectories.
- GT boxes are used for HAR training/evaluation, so detector error is not included.
- No IDF1/IDsw/MOTA can be claimed from AVA.
- Save `/kaggle/working/ocsi_ava_outputs/` before the Kaggle session ends.
