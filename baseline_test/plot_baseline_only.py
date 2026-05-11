#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waypoint Mission Mode Exclusive Subplot Plotting Script
Function: Generates an independent image for each wind condition, preventing 
trajectory overlap and clearly showing the 90s long-duration error accumulation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EVAL_RESULTS_DIR, FIGURES_DIR
RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

WIND_CONDITIONS = {
    'online_mission_nowind': ('0 m/s (Calm)', '#2ca02c'),       
    'online_mission_35wind': ('4.2 m/s (Steady)', '#1f77b4'),     
    'online_mission_70wind': ('8.5 m/s (Strong)', '#ff7f0e'),     
    'online_mission_70p20sint': ('Dynamic Gust (8.5+/-2.4)', '#9467bd'),
    'online_mission_100wind': ('12.1 m/s (Extreme)', '#d62728')   
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_separate_mission_trajectories():
    # Plot the theoretical perfect 4.0m reference trajectory (2 loops are enough to cover the display range)
    theta_ref = np.linspace(0, 4 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

    # Loop to generate a separate plot for each wind condition
    for wind_key, (wind_label, color) in WIND_CONDITIONS.items():
        file_path = os.path.join(RESULTS_DIR, f"eval_data_Mission_{wind_key}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Exclude takeoff and landing phases, keep the middle 80s long-duration segment
            df_steady = df[(df['time'] >= 5.0) & (df['time'] <= 85.0)]
            
            if not df_steady.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot reference line
                ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=3, label='Ideal Figure-8 Route (Reference)', zorder=10)
                
                # Plot actual waypoint trajectory
                ax.plot(df_steady['p_x'], df_steady['p_y'], color=color, linewidth=2.0, label=f'Actual Waypoint Flight Trajectory\n({wind_label})', alpha=0.85)
                
                # Calculate Cross-Track RMSE
                pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
                ref_pts = np.vstack((ref_x, ref_y)).T
                distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
                rmse = np.sqrt(np.mean(distances**2))
                
                print(f"[{wind_label}] Generated -> Cross-Track RMSE: {rmse:.3f} m")

                # Figure decorations
                ax.set_xlabel('North (X) [m]', fontweight='bold', fontsize=12)
                ax.set_ylabel('East (Y) [m]', fontweight='bold', fontsize=12)
                ax.set_title(f'PX4 Native Waypoint Mode Tracking Performance - {wind_label}\nCross-Track RMSE = {rmse:.3f} m', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=11)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_aspect('equal', 'box')
                
                # Save each image individually
                plt.tight_layout()
                out_path = os.path.join(FIGURES_DIR, f'fig_Mission_Baseline_{wind_key}.png')
                plt.savefig(out_path, dpi=300)
                plt.close(fig)

if __name__ == '__main__':
    print("Generating individual trajectory subplots...")
    plot_separate_mission_trajectories()
    print(f"All wind condition figures saved independently to {FIGURES_DIR} directory.")