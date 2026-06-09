#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始消融实验运行器（两组对比，用于主图训练曲线）:
  [1] original  - 纯 MSE 基线 (λ_triplet=0)
  [2] fcml      - 完整 FCML   (λ_triplet=1.0)

输出:
  training_results/curve_original.csv  → plot_training_curve.py 使用
  training_results/curve_fcml.csv      → plot_training_curve.py 使用
  checkpoints/best_model.pth           → online_mission_compare.py 加载

注意: 此脚本不影响 backbone_ablation 实验，也不影响 NF 权重。
"""

import os
import sys
import subprocess
import glob
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PROJECT_ROOT, TRAINING_DIR, CHECKPOINTS_DIR

BASE_DIR = PROJECT_ROOT
TRAIN_DIR = TRAINING_DIR
os.makedirs(TRAIN_DIR, exist_ok=True)


def run_training_and_extract(lambda_val, scheme_name):
    out_dir = os.path.join(TRAIN_DIR, f"run_{scheme_name}")
    print(f"\n{'='*50}\nStarting training: {scheme_name} (Triplet Lambda = {lambda_val})\n{'='*50}")

    cmd = (
        f"python3 scripts/offline/train_offline_lightning.py"
        f" --backbone ours"
        f" --lambda_triplet {lambda_val}"
        f" --output_dir {out_dir}"
    )
    subprocess.run(cmd, shell=True, check=True, cwd=BASE_DIR)

    csv_files = glob.glob(os.path.join(out_dir, "lightning_logs", "version_*", "metrics.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        dest_csv = os.path.join(TRAIN_DIR, f"curve_{scheme_name}.csv")
        shutil.copy(latest_csv, dest_csv)
        print(f"\n[{scheme_name}] Training curve extracted: {dest_csv}")

    if scheme_name == "fcml":
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        shutil.copy(os.path.join(out_dir, "best_model.pth"),
                    os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
        print(f"  [Deploy] FCML weights → {CHECKPOINTS_DIR}/best_model.pth")


if __name__ == "__main__":
    
    run_training_and_extract(0.0, "original")
    
    run_training_and_extract(1.0, "fcml")
    print("\nAll ablation experiments completed! Run plot_training_curve.py to view the comparison.")