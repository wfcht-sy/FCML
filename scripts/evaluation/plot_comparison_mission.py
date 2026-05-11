#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual waypoint mission comparison plotting script.

Generates Fig 1 (RMSE bar chart), Fig 2 (trajectory plots per wind condition),
and Fig 3 (force tracking plots per wind condition) for all 5 controllers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import EVAL_RESULTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore")

CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'Ours']
RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

# All 5 wind conditions
WIND_CONDITIONS = {
    'online_test_nowind': '0 m/s (Calm)', 
    'online_test_35wind': '4.2 m/s (Steady)',
    'online_test_70wind': '8.5 m/s (Strong)', 
    'online_test_70p20sint': 'Dynamic Gust (8.5+/-2.4)', 
    'online_test_100wind': '12.1 m/s (Extreme)'
}

LABEL_MAP = {
    'Baseline': 'Baseline PID',
    'INDI': 'INDI',
    'L1': 'L1 Adaptive',
    'Neural-Fly': 'Neural-Fly (DAIML)',
    'Ours': 'FCML (Ours)'
}

COLORS = {'Baseline': '#7f7f7f', 'INDI': '#1f77b4', 'L1': '#9467bd', 'Neural-Fly': '#ff7f0e', 'Ours': '#2ca02c'}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def calculate_cross_track_rmse(df_steady, ref_x, ref_y):
    """Compute spatial cross-track RMSE against a reference trajectory."""
    pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
    ref_pts = np.vstack((ref_x, ref_y)).T
    distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
    return np.sqrt(np.mean(distances**2))

def load_data(ctrl, wind):
    file_path = os.path.join(RESULTS_DIR, f"eval_data_VirtualMission_{ctrl}_{wind}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def plot_cross_track_rmse_bar():
    print("  -> Computing Cross-Track RMSE and generating bar chart...")
    theta_ref = np.linspace(0, 4 * np.pi, 2000)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

    rmse_data = {ctrl: [] for ctrl in CONTROLLERS}
    valid_winds = []
    
    for wind_key, wind_label in WIND_CONDITIONS.items():
        valid_winds.append(wind_label)
        for ctrl in CONTROLLERS:
            df = load_data(ctrl, wind_key)
            if df is not None and not df.empty:
                # Extract steady-state segment (15s to 85s)
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    rmse = calculate_cross_track_rmse(df_steady, ref_x, ref_y)
                    rmse_data[ctrl].append(rmse)
                else:
                    rmse_data[ctrl].append(np.nan)
            else: 
                rmse_data[ctrl].append(np.nan)

    x = np.arange(len(valid_winds))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, ctrl in enumerate(CONTROLLERS):
        ax.bar(x + i*width - width*2, rmse_data[ctrl], width, label=LABEL_MAP[ctrl], color=COLORS[ctrl], edgecolor='black')

    ax.set_ylabel('Cross-Track RMSE [m]', fontweight='bold')
    ax.set_title('Wind-Rejection Tracking Performance Comparison (Virtual Waypoint Mode)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_winds)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_Mission_RMSE_Bar.png'), dpi=300)
    plt.close(fig)

def plot_all_trajectories():
    print("  -> Generating figure-8 trajectory comparisons (5 wind conditions)...")
    theta_ref = np.linspace(0, 2 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

    for wind_key, wind_label in WIND_CONDITIONS.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=3, label='Reference Trajectory', zorder=10)

        has_data = False
        for ctrl in CONTROLLERS:
            df = load_data(ctrl, wind_key)
            if df is not None and not df.empty:
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    has_data = True
                    ax.plot(df_steady['p_x'], df_steady['p_y'], color=COLORS[ctrl], linewidth=2.5, label=LABEL_MAP[ctrl], alpha=0.8)

        if has_data:
            ax.set_xlabel('North (X) [m]', fontweight='bold', fontsize=12)
            ax.set_ylabel('East (Y) [m]', fontweight='bold', fontsize=12)
            ax.set_title(f'Waypoint Tracking Performance - {wind_label}', fontsize=16, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_aspect('equal', 'box')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'fig2_Mission_Trajectory_{wind_key}.png'), dpi=300)
        plt.close(fig)

def plot_all_force_tracking():
    print("  -> Generating force estimation comparison plots (5 wind conditions)...")
    
    for wind_key, wind_label in WIND_CONDITIONS.items():
        fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Aerodynamic Disturbance Estimation - {wind_label}', fontsize=16, fontweight='bold', y=0.97)
        
        has_data = False
        for i, ctrl in enumerate(CONTROLLERS):
            df = load_data(ctrl, wind_key)
            if df is not None and not df.empty:
                # Display window: 20s to 40s for detailed view
                df_plot = df[(df['time'] >= 20.0) & (df['time'] <= 40.0)]
                if not df_plot.empty:
                    has_data = True
                    axs[i].plot(df_plot['time'], df_plot['f_true_x'], color='black', linestyle='--', linewidth=2, label='Ground Truth (X-axis)', alpha=0.7)
                    axs[i].plot(df_plot['time'], df_plot['f_est_x'], color=COLORS[ctrl], linewidth=2.5, label=f'{LABEL_MAP[ctrl]} Estimate')
                    
                    rmse = np.sqrt(((df_plot['f_true_x'] - df_plot['f_est_x'])**2).mean())
                    axs[i].text(0.01, 0.85, f"Force RMSE: {rmse:.2f} N", transform=axs[i].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            axs[i].set_ylabel('Force [N]', fontweight='bold')
            axs[i].legend(loc='upper right')
            axs[i].grid(True, linestyle=':', alpha=0.6)
            
        if has_data:
            axs[-1].set_xlabel('Time [s]', fontweight='bold', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(FIGURES_DIR, f'fig3_Mission_Force_Tracking_{wind_key}.png'), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    print("Generating evaluation figure matrix...")
    plot_cross_track_rmse_bar()
    plot_all_trajectories()
    plot_all_force_tracking()
    print(f"All figures saved to {FIGURES_DIR}")