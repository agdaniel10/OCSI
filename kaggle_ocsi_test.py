"""
OCSI Kaggle Notebook - MOT17 & MOT20 Testing Script
====================================================
Paste the cells below into a Kaggle Notebook (Python) to run OCSI on MOT17 and MOT20.

Prerequisites on Kaggle:
  1. Add the MOT17 dataset:  https://www.kaggle.com/datasets/wenhoujinjust/mot-17
  2. Add the MOT20 dataset:  https://www.kaggle.com/datasets/landlord/mot20challenge
  3. Enable GPU (P100 or T4) in the notebook settings.

The script will:
  - Clone the OCSI repository
  - Install dependencies
  - Run baseline / memory / adaptive_memory / feedback ablations on MOT17
  - Run baseline / memory ablations on MOT20
  - Report IDF1, MOTA, IDsw, and embedding diagnostics
  - Save results to /kaggle/working/ocsi_outputs/
"""

# =============================================================================
# CELL 1: Install dependencies and clone the repo
# =============================================================================
# In Kaggle, run these as separate cells:
#
# Cell 1a:
#   !git clone https://github.com/agdaniel10/OCSI.git
#   %cd /kaggle/working/OCSI
#
# Cell 1b:
#   !pip install -q numpy scipy opencv-python pandas pyyaml torch torchvision
#   !pip install -q ultralytics
#   !pip install -q -e ".[dev,perception]"
#
# =============================================================================
# CELL 2: Verify the dataset paths
# =============================================================================
import os

# MOT17 dataset path (adjust if your Kaggle dataset slug differs)
MOT17_ROOT = "/kaggle/input/mot-17/MOT17/train"
if not os.path.exists(MOT17_ROOT):
    # Try alternate paths
    for candidate in [
        "/kaggle/input/mot-17/MOT17/train",
        "/kaggle/input/wenhoujinjust/mot-17/MOT17/train",
        "/kaggle/input/mot17challenge/MOT17/train",
    ]:
        if os.path.exists(candidate):
            MOT17_ROOT = candidate
            break

# MOT20 dataset path (adjust if your Kaggle dataset slug differs)
MOT20_ROOT = "/kaggle/input/mot20challenge/MOT20/train"
if not os.path.exists(MOT20_ROOT):
    for candidate in [
        "/kaggle/input/mot20challenge/MOT20/train",
        "/kaggle/input/landlord/mot20challenge/MOT20/train",
        "/kaggle/input/mot20/MOT20/train",
    ]:
        if os.path.exists(candidate):
            MOT20_ROOT = candidate
            break

print("MOT17 root:", MOT17_ROOT, "exists:", os.path.exists(MOT17_ROOT))
print("MOT20 root:", MOT20_ROOT, "exists:", os.path.exists(MOT20_ROOT))

# List available sequences
if os.path.exists(MOT17_ROOT):
    print("\nMOT17 sequences:", sorted(os.listdir(MOT17_ROOT)))
if os.path.exists(MOT20_ROOT):
    print("MOT20 sequences:", sorted(os.listdir(MOT20_ROOT)))

# =============================================================================
# CELL 3: Run OCSI on MOT17 (all 7 training sequences)
# =============================================================================
import json
import numpy as np
from ocsi.config import OCSIConfig
from ocsi.experiments.mot17_tracking import run_mot17_sequence, run_mot17_dataset

# Configuration
CACHE_ROOT = "/kaggle/working/ocsi_cache"
OUTPUT_DIR = "/kaggle/working/ocsi_outputs"
DET_CONF = 0.30          # detection confidence threshold
REACT_GATE = 0.84        # reactivation appearance gate
SEEDS = (0, 1, 2)        # multiple seeds for statistical robustness

# All MOT17 training sequences (FRCNN public detections)
mot17_seqs = sorted([
    os.path.join(MOT17_ROOT, d)
    for d in os.listdir(MOT17_ROOT)
    if d.endswith("-FRCNN") and os.path.isdir(os.path.join(MOT17_ROOT, d))
])
print(f"Found {len(mot17_seqs)} MOT17 sequences")

# Split into calibration and evaluation sets (leakage-aware)
# Calibration: MOT17-02, MOT17-04, MOT17-05
# Evaluation:  MOT17-09, MOT17-10, MOT17-11, MOT17-13
calib_names = {"MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN"}
eval_names = {"MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"}

calib_seqs = [s for s in mot17_seqs if os.path.basename(s) in calib_names]
eval_seqs = [s for s in mot17_seqs if os.path.basename(s) in eval_names]

print(f"Calibration sequences: {[os.path.basename(s) for s in calib_seqs]}")
print(f"Evaluation sequences:  {[os.path.basename(s) for s in eval_seqs]}")

# Configure detection confidence threshold via the config object
cfg = OCSIConfig()
cfg.perception.det_conf_threshold = DET_CONF

# Run the full MOT17 ablation grid with leakage-aware calibration
mot17_payload = run_mot17_dataset(
    seq_dirs=mot17_seqs,
    cache_root=CACHE_ROOT,
    output_dir=OUTPUT_DIR,
    stages=("baseline", "memory", "adaptive_memory", "feedback"),
    detection_source="public",
    cfg=cfg,
    seeds=SEEDS,
    reactivation_app_gate=REACT_GATE,
    calibration_sequences=list(calib_names),
    evaluation_sequences=list(eval_names),
)

print("\n=== MOT17 Results ===")
for run in mot17_payload["runs"]:
    seq = run["sequence"]
    for result in run["results"]:
        print(f"  {seq:20s} {result['stage']:15s} {result['summary']}")

# =============================================================================
# CELL 4: Run OCSI on MOT20 (all 4 training sequences)
# =============================================================================
from ocsi.experiments.mot20 import run_mot20_sequence

mot20_seqs = sorted([
    os.path.join(MOT20_ROOT, d)
    for d in os.listdir(MOT20_ROOT)
    if os.path.isdir(os.path.join(MOT20_ROOT, d))
])
print(f"Found {len(mot20_seqs)} MOT20 sequences")

mot20_results = []
for seq_dir in mot20_seqs:
    seq_name = os.path.basename(seq_dir)
    print(f"\nRunning MOT20 sequence: {seq_name}")
    try:
        payload = run_mot20_sequence(
            seq_dir=seq_dir,
            cache_dir=os.path.join(CACHE_ROOT, f"mot20_{seq_name}"),
            output_dir=os.path.join(OUTPUT_DIR, "mot20"),
            stages=("baseline", "memory"),
            detection_source="public",
            det_conf_threshold=DET_CONF,
            seed=0,
            reactivation_app_gate=REACT_GATE,
        )
        mot20_results.append(payload)
        for result in payload["results"]:
            print(f"  {seq_name:20s} {result['stage']:15s} {result['summary']}")
    except Exception as e:
        print(f"  ERROR on {seq_name}: {e}")

# =============================================================================
# CELL 5: Run the occlusion recovery experiment (synthetic, no dataset needed)
# =============================================================================
import subprocess
import sys

print("Running occlusion recovery experiment...")
result = subprocess.run(
    [sys.executable, "-m", "ocsi.experiments.occlusion_recovery"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# =============================================================================
# CELL 6: Run the behaviour feedback experiment (synthetic)
# =============================================================================
print("Running behaviour feedback experiment...")
result = subprocess.run(
    [sys.executable, "-m", "ocsi.experiments.behaviour_feedback"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# =============================================================================
# CELL 7: Run the contamination rollback experiment (synthetic)
# =============================================================================
print("Running contamination rollback experiment...")
result = subprocess.run(
    [sys.executable, "-m", "ocsi.experiments.contamination_rollback"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# =============================================================================
# CELL 8: Summarize all results
# =============================================================================
import pandas as pd
from pathlib import Path

print("\n" + "=" * 80)
print("OCSI EXPERIMENT SUMMARY")
print("=" * 80)

# MOT17 summary
print("\n--- MOT17 Results ---")
for run in mot17_payload["runs"]:
    seq = run["sequence"]
    for result in run["results"]:
        m = result["metrics"]
        print(f"  {seq:20s} {result['stage']:15s} "
              f"IDF1={m['idf1']:.3f}  MOTA={m['mota']:.3f}  IDsw={m['idsw']:4d}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}")

# MOT20 summary
if mot20_results:
    print("\n--- MOT20 Results ---")
    for payload in mot20_results:
        seq = payload["sequence"]
        for result in payload["results"]:
            m = result["metrics"]
            print(f"  {seq:20s} {result['stage']:15s} "
                  f"IDF1={m['idf1']:.3f}  MOTA={m['mota']:.3f}  IDsw={m['idsw']:4d}  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}")

# Embedding diagnostics
print("\n--- Embedding Diagnostics (MOT17) ---")
for run in mot17_payload["runs"]:
    if run["seed"] == 0:  # only show seed 0 for brevity
        diag = run["embedding_diagnostics"]
        print(f"  {run['sequence']:20s} same-id={diag.get('same_id_proto_cosine', 'N/A'):.3f}  "
              f"diff-id={diag.get('different_id_proto_cosine', 'N/A'):.3f}  "
              f"margin={diag.get('separation_margin', 'N/A'):.3f}")

# Calibration info
print(f"\n--- Calibration ---")
print(f"  Calibration sequences: {mot17_payload.get('calibration_sequences')}")
print(f"  Evaluation sequences:  {mot17_payload.get('evaluation_sequences')}")
print(f"  Calibrated gate:       {mot17_payload.get('calibrated_reactivation_app_gate')}")

# Output files
print(f"\n--- Output Files ---")
print(f"  Results saved to: {OUTPUT_DIR}")
for f in sorted(Path(OUTPUT_DIR).rglob("*.json")):
    print(f"    {f}")

print("\nDone! Download the /kaggle/working/ocsi_outputs/ folder for full results.")