# OCSI-Q1 Experimental Pipeline

This package is intended to strengthen the OCSI reproduction study for a Q1 journal submission.

## What it fixes
- Uses a person-ReID backbone (OSNet through `torchreid`) instead of generic ImageNet-only features.
- Supports memory and adaptive-memory ablations.
- Produces leakage-aware experimental structure.
- Reports local diagnostic IDF1/MOTA/IDSW, re-entry recovery and IDSW per 1,000 frames.
- Produces 95% bootstrap confidence intervals.
- Produces paired Wilcoxon and paired t-test results.
- Exports MOT-format predictions for official TrackEval.
- Records FPS, reactivation attempts, reactivations and tracks created.

## Data layout
Point `--data-root` to a folder such as:

MOT17/
  MOT17-02-FRCNN/
    img1/
    det/det.txt
    gt/gt.txt
    seqinfo.ini
  MOT17-04-FRCNN/
  ...

## Example
```bash
python ocsi_q1_pipeline.py \
  --data-root /path/to/MOT17/train \
  --evaluation-sequences MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN \
      MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --out results_mot17 \
  --react-gate 0.85
```

## Q1 requirement
Do not report the local evaluator as the final benchmark result. Use the generated tracker files with the official TrackEval toolkit and report HOTA, DetA, AssA, MOTA, IDF1 and IDSW.

## Important scientific limitation
MOT17 does not provide action labels suitable for validating behaviour-guided feedback. A true behaviour-feedback claim requires a trained temporal action model and a dataset/protocol that supports action evidence. Do not claim the behaviour component is validated until that experiment has actually been run.
