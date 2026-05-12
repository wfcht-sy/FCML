#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import TRAINING_DIR, FIGURES_DIR

# Suppress Matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. Path Configuration =================
TRAIN_DIR = TRAINING_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_training_curves():
    csv_fcml = os.path.join(TRAIN_DIR, "curve_fcml.csv")
    csv_orig = os.path.join(TRAIN_DIR, "curve_original.csv")

    if not os.path.exists(csv_fcml) or not os.path.exists(csv_orig):
        print(f"ERROR: Training CSV data not found.")
        print(f"Please run first: python scripts/offline/run_ablations.py")
        return

    print("Generating training convergence comparison plot (Fig 0)...")

    # ================= 2. Data Cleaning =================
    # CSVLogger records at both step and epoch level. val_mse only exists
    # at epoch boundaries, so dropna extracts clean epoch-level curves.
    df_fcml = pd.read_csv(csv_fcml).dropna(subset=['val_mse'])
    df_orig = pd.read_csv(csv_orig).dropna(subset=['val_mse'])

    # ================= 3. Plotting =================
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Original Neural-Fly (DAIML) - blue dashed line
    ax.plot(df_orig['epoch'], df_orig['val_mse'], 
            color='#1f77b4', linestyle='--', linewidth=2.5, 
            label='Neural-Fly (DAIML)')
    
    # Our method (FCML) - red solid line
    ax.plot(df_fcml['epoch'], df_fcml['val_mse'], 
            color='#d62728', linestyle='-', linewidth=2.5, 
            label='FCML (DTW-Triplet Alignment)')

    # ================= 4. Figure Decoration =================
    ax.set_title('Validation MSE Convergence Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation MSE (Tracking Error)', fontsize=13, fontweight='bold')
    
    # Log scale to highlight late-stage performance gap
    ax.set_yscale('log') 
    
    ax.grid(True, which="major", ls="-", alpha=0.6, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.4, color='gray')
    
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')

    # ================= 5. Save Output =================
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig0_Training_Convergence.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_path}")

if __name__ == "__main__":
    plot_training_curves()