#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTW triplet generator for offline training data alignment.

Aligns no-wind and wind-affected flight trajectories using FastDTW on
velocity profiles, then constructs (Anchor, Positive, Negative) triplets
for metric learning.
"""

import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from pathlib import Path
import random
import time

# ================= Core Hyperparameters (tuned for 50Hz data) =================
DOWNSAMPLE_STEP = 1   # No downsampling at 50Hz (~8000 frames per flight)
DTW_RADIUS = 50       # Search radius (50 frames = 1.0s phase tolerance)
NEG_MARGIN = 150      # Negative sample offset (150 frames = 3.0s temporal distance)
# ==============================================================================

def smooth_and_normalize_3d(v_3d, window=11, poly=3):
    """Smooth and normalize a 3D velocity sequence for DTW alignment."""
    v_smooth = np.zeros_like(v_3d)
    for i in range(3):
        w = min(window, len(v_3d) if len(v_3d) % 2 != 0 else len(v_3d) - 1)
        if w > 3:
            v_smooth[:, i] = savgol_filter(v_3d[:, i], window_length=w, polyorder=poly)
        else:
            v_smooth[:, i] = v_3d[:, i]
    # Normalize to align waveform shape, not absolute magnitude
    return (v_smooth - np.mean(v_smooth, axis=0)) / (np.std(v_smooth, axis=0) + 1e-8)

def evaluate_alignment(v_a, v_p):
    """Compute alignment quality metrics (Pearson correlation and MSE)."""
    corr = np.corrcoef(v_a, v_p)[0, 1]
    mse = np.mean((v_a - v_p)**2)
    return corr, mse

def generate_triplets(csv_0, csv_w, output_path):
    print(f"\nAligning target file: {Path(csv_w).name}")
    
    df_0 = pd.read_csv(csv_0)
    df_w = pd.read_csv(csv_w)

    # 1. Extract full velocity data
    v0_3d = df_0[['v_x', 'v_y', 'v_z']].values
    vw_3d = df_w[['v_x', 'v_y', 'v_z']].values
    
    # 2. Generate smoothed guide sequences
    v0_guide = smooth_and_normalize_3d(v0_3d)[::DOWNSAMPLE_STEP]
    vw_guide = smooth_and_normalize_3d(vw_3d)[::DOWNSAMPLE_STEP]

    # 3. Execute FastDTW alignment
    print(f"   [Compute] Running FastDTW ({len(v0_guide)} frames, radius={DTW_RADIUS})...")
    start_time = time.time()
    distance, path_sub = fastdtw(v0_guide, vw_guide, radius=DTW_RADIUS, dist=euclidean)
    print(f"   [Done] Elapsed: {time.time()-start_time:.2f}s | Warping distance: {distance:.2f}")

    # 4. Construct triplets with self-validation
    triplet_data = []
    max_idx_0_sub = len(v0_guide) - 1
    
    feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
    label_cols = ['f_x', 'f_y', 'f_z']
    
    aligned_v_a, aligned_v_p = [], []

    for i_sub, j_sub in path_sub:
        # Map downsampled indices back to original DataFrame rows
        idx_0_real = i_sub * DOWNSAMPLE_STEP
        idx_w_real = j_sub * DOWNSAMPLE_STEP
        
        # Boundary check
        if idx_0_real >= len(df_0) or idx_w_real >= len(df_w): continue
        
        row_A = df_0.iloc[idx_0_real]
        row_P = df_w.iloc[idx_w_real]
        
        # Random negative sample (temporally distant from anchor)
        idx_N_sub = random.randint(0, max_idx_0_sub)
        while abs(idx_N_sub - i_sub) < NEG_MARGIN:
            idx_N_sub = random.randint(0, max_idx_0_sub)
        idx_N_real = idx_N_sub * DOWNSAMPLE_STEP
        row_N = df_0.iloc[idx_N_real]

        # Assemble triplet (A: no-wind anchor, P: wind-aligned positive, N: temporal negative)
        record = {'t_A': row_A['timestamp'], 't_P': row_P['timestamp'], 't_N': row_N['timestamp']}
        for c in feat_cols: record[f'A_{c}'] = row_A[c]; record[f'P_{c}'] = row_P[c]; record[f'N_{c}'] = row_N[c]
        for c in label_cols: record[f'A_{c}'] = row_A[c]; record[f'P_{c}'] = row_P[c]; record[f'N_{c}'] = row_N[c]
            
        triplet_data.append(record)
        
        aligned_v_a.append(row_A['v_x'])
        aligned_v_p.append(row_P['v_x'])

    # === Self-validation ===
    aligned_v_a = np.array(aligned_v_a)
    aligned_v_p = np.array(aligned_v_p)
    norm_v_a = (aligned_v_a - np.mean(aligned_v_a)) / (np.std(aligned_v_a) + 1e-8)
    norm_v_p = (aligned_v_p - np.mean(aligned_v_p)) / (np.std(aligned_v_p) + 1e-8)
    
    corr, mse = evaluate_alignment(norm_v_a, norm_v_p)
    print(f"   [Validation] Pearson r: {corr:.4f} | Warped MSE: {mse:.4f}")
    if corr >= 0.95: print("      [PASS] Peaks and valleys precisely aligned.")
    elif corr >= 0.85: print("      [WARN] Acceptable alignment quality; minor distortion from wind effects.")
    else: print("      [FAIL] Severe misalignment detected!")

    return pd.DataFrame(triplet_data)

def main():
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from config import PROCESSED_DIR, DTW_DATA_DIR
    
    INPUT_DIR = Path(PROCESSED_DIR)
    OUTPUT_DIR = Path(DTW_DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use no-wind flight as the anchor baseline
    base_file = INPUT_DIR / "processed_train_nowind.csv"
    if not base_file.exists():
        print(f"ERROR: Baseline file not found: {base_file}. Please check data processing.")
        return

    # Find all wind-affected training sets
    target_files = sorted(INPUT_DIR.glob("processed_train_*wind.csv"))
    target_files = [f for f in target_files if f.name != base_file.name]
    
    if not target_files:
        print("ERROR: No wind-affected data files found for alignment.")
        return

    all_triplets_dfs = []
    for target_file in target_files:
        suffix = target_file.stem.split('_')[-1]
        out_name = OUTPUT_DIR / f"dtw_triplet_nowind_to_{suffix}.csv"
        
        df_triplet = generate_triplets(base_file, target_file, out_name)
        df_triplet.to_csv(out_name, index=False)
        print(f"   Saved: {out_name.name} ({len(df_triplet)} triplets)\n")
        all_triplets_dfs.append(df_triplet)

    if all_triplets_dfs:
        combined_df = pd.concat(all_triplets_dfs, ignore_index=True)
        combined_out = OUTPUT_DIR / "dtw_triplet_combined_all.csv"
        combined_df.to_csv(combined_out, index=False)
        print(f"All wind conditions ({len(target_files)}) merged into: {combined_out.name}")
        print(f"   Total triplets: {len(combined_df)}. Ready for network training.")

if __name__ == "__main__":
    main()