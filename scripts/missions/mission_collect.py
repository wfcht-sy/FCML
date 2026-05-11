#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight data processor: converts raw PX4 ULG-derived CSV files into
standardized 50Hz training data with computed aerodynamic residuals.
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from pathlib import Path
import os
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==================== 1. Physical Constants ====================
G_VAL = 9.8066
MASS = 1.5
HOVER_THR = 0.705810

# Target frequency: 50 Hz (aligned with control loop rate)
TARGET_FREQ = 50.0
DT_TARGET = 1.0 / TARGET_FREQ

# Savitzky-Golay filter parameters (11-point window ~ 0.22s at 50Hz)
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import RAW_LOGS_DIR, PROCESSED_DIR

# Path configuration
ROOT_DIR = RAW_LOGS_DIR
OUTPUT_DIR = PROCESSED_DIR

# ==================== 2. Utility Functions ====================

def normalize_column_names(df):
    if df is None: return None
    new_columns = {}
    for col in df.columns:
        new_col = re.sub(r'_(\d+)$', r'[\1]', col)
        if new_col != col:
            new_columns[col] = new_col
    if new_columns:
        df.rename(columns=new_columns, inplace=True)
    return df

def read_csv_smart(path):
    if not path or not Path(path).exists(): return None
    try:
        df = pd.read_csv(path)
    except: return None

    df = normalize_column_names(df)
    t_cols = [c for c in df.columns if 'timestamp' in c]
    if not t_cols: return None
    t_col = t_cols[0]

    # Convert microseconds to seconds
    df['t'] = pd.to_numeric(df[t_col], errors='coerce') * 1e-6
    df = df.dropna(subset=['t']).sort_values('t').drop_duplicates('t', keep='last')
    return df.set_index('t')

def get_exact_file(case_dir, topic_name):
    pattern = f"*{topic_name}*.csv"
    files = list(Path(case_dir).glob(pattern))
    if not files: return None
    files.sort(key=lambda x: len(x.name))
    return str(files[0])

# ==================== 3. Core Processing Logic ====================

def process_one_case(case_dir, output_path):
    case_name = Path(case_dir).name
    print(f"Processing: [{case_name}] ...")

    path_pos = get_exact_file(case_dir, "vehicle_local_position")
    path_att = get_exact_file(case_dir, "vehicle_attitude")
    path_out = get_exact_file(case_dir, "actuator_outputs")

    df_pos  = read_csv_smart(path_pos)
    df_att  = read_csv_smart(path_att)
    df_out  = read_csv_smart(path_out)

    if df_pos is None or df_att is None or df_out is None:
        print(f"  [SKIP] Incomplete data files (missing pos/att/out)")
        return False

    # === Physical Event Alignment ===
    # Detect takeoff: altitude z first exceeds 0.3m (NED: z < -0.3)
    takeoff_mask = df_pos['z'] < -0.3
    if not takeoff_mask.any():
        print("  [FAIL] Takeoff not detected (altitude never exceeded 0.3m)")
        return False

    t_zero_epoch = df_pos[takeoff_mask].index[0]
    t_end_epoch = min(df_pos.index[-1], df_att.index[-1], df_out.index[-1])

    if t_end_epoch - t_zero_epoch < 5.0:
        print("  [FAIL] Effective flight duration too short (< 5s)")
        return False

    print(f"  -> Takeoff time: {t_zero_epoch:.2f}s, Duration: {t_end_epoch - t_zero_epoch:.2f}s")

    # Build 50Hz relative time axis
    new_t = np.arange(0, t_end_epoch - t_zero_epoch, DT_TARGET)
    df = pd.DataFrame(index=new_t)
    df.index.name = 't'

    # Resampling function
    def resample_channel(source_df, cols, prefix=""):
        if source_df is None: return
        t_source_relative = source_df.index - t_zero_epoch
        available_cols = [c for c in cols if c in source_df.columns]
        for c in available_cols:
            interpolated_data = np.interp(new_t, t_source_relative, source_df[c].values)
            df[f"{prefix}{c}"] = interpolated_data

    # Execute resampling
    resample_channel(df_pos, ['vx', 'vy', 'vz'])
    resample_channel(df_att, ['q[0]', 'q[1]', 'q[2]', 'q[3]'], prefix="att_")
    resample_channel(df_out, ['output[0]', 'output[1]', 'output[2]', 'output[3]'], prefix="pwm_")

    if 'vx' not in df.columns or 'att_q[0]' not in df.columns:
        print("  [FAIL] Missing velocity or attitude columns")
        return False

    # === Physical Computations ===
    # 1. Compute acceleration (first derivative of velocity, smoothed)
    for axis in ['x', 'y', 'z']:
        df[f'acc_{axis}'] = savgol_filter(df[f'v{axis}'], SAVGOL_WINDOW, SAVGOL_POLY, deriv=1, delta=DT_TARGET)

    # 2. Thrust computation (PWM to Newton mapping)
    pwm_cols = [f'pwm_output[{i}]' for i in range(4)]
    pwm_mean = df[pwm_cols].mean(axis=1)

    if pwm_mean.mean() > 100:
        thrust_norm = (pwm_mean - 1000.0) / 1000.0
    else:
        thrust_norm = pwm_mean

    thrust_newton = (thrust_norm / HOVER_THR) * MASS * G_VAL

    # 3. Residual force: f_residual = m(a - g) - R * T_body
    vec_ma = np.vstack((df['acc_x'], df['acc_y'], df['acc_z'])).T * MASS
    vec_mg = np.array([0, 0, MASS * G_VAL])

    quats = df[['att_q[1]', 'att_q[2]', 'att_q[3]', 'att_q[0]']].to_numpy()  # x,y,z,w
    rot_mats = R.from_quat(quats).as_matrix()

    vec_thrust_body = np.zeros_like(vec_ma)
    vec_thrust_body[:, 2] = -thrust_newton  # Thrust along body Z-up (NED: negative Z)

    vec_thrust_world = np.einsum('ijk,ik->ij', rot_mats, vec_thrust_body)
    vec_fa = vec_ma - vec_mg - vec_thrust_world

    # === Export and Trimming ===
    export_df = pd.DataFrame({
        'timestamp': df.index,
        'v_x': df['vx'], 'v_y': df['vy'], 'v_z': df['vz'],
        'q_w': df['att_q[0]'], 'q_x': df['att_q[1]'], 'q_y': df['att_q[2]'], 'q_z': df['att_q[3]'],
        'pwm_1': df['pwm_output[0]'], 'pwm_2': df['pwm_output[1]'],
        'pwm_3': df['pwm_output[2]'], 'pwm_4': df['pwm_output[3]'],
        'f_x': vec_fa[:, 0], 'f_y': vec_fa[:, 1], 'f_z': vec_fa[:, 2]
    })

    # Trim first 3.0s (takeoff transient) and last 2.0s (landing deceleration)
    export_df = export_df[(export_df['timestamp'] >= 3.0) & (export_df['timestamp'] <= df.index[-1] - 2.0)]

    output_path = str(Path(output_path).with_suffix('.csv'))
    export_df.to_csv(output_path, index=False)
    print(f"  [DONE] -> Saved: {Path(output_path).name} ({len(export_df)} frames at 50Hz)")
    return True

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"ERROR: Raw log directory not found: {ROOT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    subdirs = [f for f in Path(ROOT_DIR).iterdir() if f.is_dir()]
    subdirs.sort()

    print("\n" + "="*50)
    print(f"Processing logs: found {len(subdirs)} flight conditions")
    print("="*50 + "\n")

    success_count = 0
    for subdir in subdirs:
        output_name = f"processed_{subdir.name}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        if process_one_case(str(subdir), output_path):
            success_count += 1

    print("\n" + "="*50)
    print(f"Processing complete! Generated {success_count}/{len(subdirs)} high-quality 50Hz datasets.")
    print("You can now use these CSV files for DTW triplet alignment and network training.")
    print("="*50)

if __name__ == '__main__':
    main()
