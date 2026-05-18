#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbone Ablation Experiment Runner (fully isolated from the main training pipeline).

Four experimental groups:
  [A] nf_daiml         -> PhiNetwork (2-layer, 64 units, SN)  + DAIML adversarial   [NF baseline]
  [B] ours_no_triplet  -> PhiNetworkFCML (4-layer)            + pure MSE (lambda=0)  [ablate triplet]
  [C] ours_nf_backbone -> PhiNetwork (2-layer, 64 units, SN)  + DTW-Triplet          [ablate backbone]
  [D] ours_full        -> PhiNetworkFCML (4-layer)            + DTW-Triplet          [full method]

Ablation chains:
  Backbone  : [D] vs [C]  -> proves 4-layer + constant-bias design is better for DTW-Triplet
  Loss      : [D] vs [B]  -> proves DTW-Triplet loss improves generalisation
  vs NF     : [D] vs [A]  -> full FCML vs Neural-Fly baseline

Output (fully isolated from main training):
  training_results/backbone_ablation/
    run_nf_daiml/              (training artefacts)
    run_ours_no_triplet/
    run_ours_nf_backbone/
    run_ours_full/
    curve_nf_daiml.csv         (plot data)
    curve_ours_no_triplet.csv
    curve_ours_nf_backbone.csv
    curve_ours_full.csv

Files that are never modified by this script:
  checkpoints/best_model.pth
  checkpoints/neural_fly_daiml_best.pth
  training_results/curve_original.csv
  training_results/curve_fcml.csv

Usage:
  python scripts/offline/run_backbone_ablation.py
"""

import os
import sys
import subprocess
import glob
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PROJECT_ROOT, TRAINING_DIR

BASE_DIR = PROJECT_ROOT
ABLATION_DIR = os.path.join(TRAINING_DIR, "backbone_ablation")
os.makedirs(ABLATION_DIR, exist_ok=True)


def run_triplet_group(scheme_name, lambda_val, backbone):
    """Train with DTW-Triplet loss (groups B, C, D). Output goes only to ABLATION_DIR."""
    curve_path = os.path.join(ABLATION_DIR, f"curve_{scheme_name}.csv")
    if os.path.exists(curve_path):
        print(f"\n  [SKIP]    {scheme_name:25s}  curve already exists, skipping training.")
        return

    out_dir = os.path.join(ABLATION_DIR, f"run_{scheme_name}")
    print(f"\n{'='*62}")
    print(f"  [Triplet] {scheme_name:25s}  backbone={backbone:<5s}  lambda={lambda_val}")
    print(f"{'='*62}")

    cmd = (
        f"python3 scripts/offline/train_offline_lightning.py"
        f" --backbone {backbone}"
        f" --lambda_triplet {lambda_val}"
        f" --output_dir {out_dir}"
    )
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for {scheme_name}")
        return

    _extract_curve(out_dir, scheme_name)


def run_daiml_group(scheme_name="nf_daiml"):
    """Train with DAIML adversarial loss (group A: Neural-Fly baseline). Output goes only to ABLATION_DIR."""
    curve_path = os.path.join(ABLATION_DIR, f"curve_{scheme_name}.csv")
    if os.path.exists(curve_path):
        print(f"\n  [SKIP]    {scheme_name:25s}  curve already exists, skipping training.")
        return

    out_dir = os.path.join(ABLATION_DIR, f"run_{scheme_name}")
    print(f"\n{'='*62}")
    print(f"  [DAIML]   {scheme_name:25s}  backbone=PhiNetwork (2-layer, SN)")
    print(f"{'='*62}")

    cmd = (
        f"python3 scripts/offline/train_original_nf_daiml.py"
        f" --output_dir {out_dir}"
    )
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for {scheme_name}")
        return

    _extract_curve(out_dir, scheme_name)


def _extract_curve(out_dir, scheme_name):
    """Copy the Lightning metrics.csv to ABLATION_DIR/curve_{name}.csv."""
    csv_files = glob.glob(os.path.join(out_dir, "lightning_logs", "version_*", "metrics.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        dest = os.path.join(ABLATION_DIR, f"curve_{scheme_name}.csv")
        shutil.copy(latest_csv, dest)
        print(f"  [Curve]   -> {dest}")
    else:
        print(f"  [WARNING] No metrics.csv found under {out_dir}")


def print_summary():
    print("\n" + "="*62)
    print("  Backbone Ablation Experiment Completed")
    print("="*62)
    print(f"  Output directory: {ABLATION_DIR}\n")
    for name in ["nf_daiml", "ours_no_triplet", "ours_nf_backbone", "ours_full"]:
        path = os.path.join(ABLATION_DIR, f"curve_{name}.csv")
        marker = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{marker:7s}]  curve_{name}.csv")
    print("\n  Next: python scripts/evaluation/plot_backbone_ablation.py")
    print("="*62)


if __name__ == "__main__":
    print("="*62)
    print("  Backbone Ablation Experiment  (4 groups)")
    print("  All outputs go to: training_results/backbone_ablation/")
    print("  checkpoints/ and main curve files will NOT be modified.")
    print("="*62)

    # [A] Neural-Fly baseline: DAIML adversarial training
    run_daiml_group("nf_daiml")

    # [B] Ablate triplet loss: our backbone + pure MSE
    run_triplet_group("ours_no_triplet", lambda_val=0.0, backbone="ours")

    # [C] Ablate backbone: NF backbone + DTW-Triplet loss
    run_triplet_group("ours_nf_backbone", lambda_val=1.0, backbone="nf")

    # [D] Full FCML method: our backbone + DTW-Triplet
    run_triplet_group("ours_full", lambda_val=1.0, backbone="ours")

    print_summary()
