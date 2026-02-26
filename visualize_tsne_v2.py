#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly Visualization V6 (User Requested Fixes)
- (a) Tolerance Corridor: Tightest bound (1.0x), no extra margin.
- (b) Physics Manifold: Cleaner look (smaller dots, lower alpha, downsampling).
- (c) Latent Dynamics: Text moved to top to avoid overlapping.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import os
import sys

# ================= Global Style =================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black'
})

# ================= Config =================
BASIS_DIM = 8
INPUT_DIM = 11
GLOBAL_LIMIT = None 
LOCAL_LIMIT = 400 

class PhiNetwork(nn.Module):
    def __init__(self):
        super(PhiNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, BASIS_DIM))
        )
    def forward(self, x): return self.net(x)

def load_data_raw(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
    return df, torch.FloatTensor(df[feat_cols].values)

# ================= Data Extraction =================
def get_features(model_path, anchor_csv, target_csv):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except:
        if not os.path.exists(model_path):
            print("Model file not found.")
            sys.exit(1)
        checkpoint = torch.load(model_path, map_location='cpu')

    model = PhiNetwork()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df_00, t_00 = load_data_raw(anchor_csv)
    df_XX, t_XX = load_data_raw(target_csv)
    
    # Align lengths
    min_len = min(len(t_00), len(t_XX))
    t_00_full = t_00[:min_len]
    t_XX_full = t_XX[:min_len]
    vel_XX_full = df_XX['v_x'].iloc[:min_len].values
    
    # Local Segment
    start = min_len // 3
    end = start + LOCAL_LIMIT
    if end > min_len: end = min_len
    seg_00 = t_00_full[start:end]
    
    with torch.no_grad():
        phi_full_00 = model(t_00_full).numpy()
        phi_full_XX = model(t_XX_full).numpy()
        phi_seg_00 = model(seg_00).numpy()
        
    return phi_full_00, phi_full_XX, phi_seg_00, vel_XX_full

# ================= Plotting Functions =================

def plot_a_dynamic_corridor_v6(phi_00, phi_XX):
    """(a) Tolerance Corridor: Tightest Bound"""
    print("Generating (a) Dynamic Tolerance Corridor V6...")
    
    norm_00 = np.linalg.norm(phi_00, axis=1)
    norm_XX = np.linalg.norm(phi_XX, axis=1)
    t = np.arange(len(norm_00)) * 0.02
    
    # Corridor Calculation: Tightest bound (1.0x)
    diff = np.abs(norm_XX - norm_00)
    max_deviation = np.max(diff) # No extra margin
    
    upper_bound = norm_XX + max_deviation
    lower_bound = norm_XX - max_deviation
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Draw Corridor
    ax.fill_between(t, lower_bound, upper_bound, color='#FF8C00', alpha=0.2, label='Admissible Tolerance Zone')
    ax.plot(t, upper_bound, color='#FF8C00', linewidth=0.5, alpha=0.4, linestyle='-')
    ax.plot(t, lower_bound, color='#FF8C00', linewidth=0.5, alpha=0.4, linestyle='-')
    
    # Center Line (Strong Wind)
    ax.plot(t, norm_XX, color='#FF8C00', linewidth=2.0, label='Strong Wind Feature')
    
    # Baseline (No Wind)
    ax.plot(t, norm_00, color='#003366', linewidth=1.2, label='Baseline Feature (0 m/s)')
    
    # Annotation: Positioned at top relative to axes to avoid clutter
    ax.text(0.5, 0.90, f'Robustness Bound: ||Δφ|| ≤ {max_deviation:.3f}', 
            transform=ax.transAxes, fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    ax.set_title("(a) Robustness Verification: Tight Tolerance Bound", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feature Magnitude ||φ||")
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig("final_plot_a_v6.png", dpi=300)

def plot_b_manifold_v6(phi_00, phi_XX, vel_XX):
    """(b) Physics Manifold: Cleaner, Sparse dots"""
    print("Generating (b) Manifold V6...")
    pca = PCA(n_components=2)
    combined = np.vstack([phi_00, phi_XX])
    pca.fit(combined)
    
    # Downsample for clarity (1/5 points)
    step = 5
    
    emb_00 = pca.transform(phi_00)[::step]
    emb_XX = pca.transform(phi_XX)[::step]
    vel_abs = np.abs(vel_XX[::step])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Base: Pale Green, very transparent, smaller dots
    ax.scatter(emb_00[:,0], emb_00[:,1], c='#98FB98', s=20, alpha=0.15, label='Baseline Manifold (0 m/s)')
    
    # Top: Strong Wind, Red Gradient, small dots but visible
    sc = ax.scatter(emb_XX[:,0], emb_XX[:,1], c=vel_abs, cmap='Reds', s=5, alpha=0.8, edgecolors='none', label='Perturbed State (6 m/s)')
    
    ax.set_title("(b) Physics Manifold Alignment", fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Velocity $v_x$ (m/s)', rotation=270, labelpad=15)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#98FB98', markersize=10, label='Baseline (0 m/s)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CB181D', markersize=6, label='Strong Wind (6 m/s)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("final_plot_b_v6.png", dpi=300)

def plot_c_components_v6(phi_seg):
    """(c) Latent Dynamics: Text moved up"""
    print("Generating (c) Latent Dynamics V6...")
    pca = PCA(n_components=3)
    emb = pca.fit_transform(phi_seg)
    t = np.arange(len(emb)) * 0.02
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, emb[:,0], color='#E63946', linewidth=2.5, label='Latent PC 1')
    ax.plot(t, emb[:,1], color='#457B9D', linewidth=2.5, label='Latent PC 2')
    ax.plot(t, emb[:,2], color='#2A9D8F', linewidth=2.5, label='Latent PC 3')
    
    ax.set_title("(c) Latent Space Dynamics Evaluation", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feature Amplitude")
    
    # MOVED TEXT: Use transAxes to place at top center, independent of data values
    ax.text(0.5, 1.02, "Lipschitz Continuous Evolution", transform=ax.transAxes,
            fontsize=10, color='#333', ha='center', va='bottom', fontweight='bold',
            style='italic')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', framealpha=0.9, shadow=True, ncol=3)
    
    plt.tight_layout()
    plt.savefig("final_plot_c_v6.png", dpi=300)

if __name__ == "__main__":
    model_path = "checkpoints/model_epoch_200.pth"
    if not os.path.exists(model_path): model_path = "checkpoints/best_clustering_model.pth"
    
    print("Processing V6...")
    p_f0, p_fX, p_s0, v_X = get_features(model_path, 
                                        "processed_data/processed_train_wind_00.csv",
                                        "processed_data/processed_train_wind_06.csv")
    
    plot_a_dynamic_corridor_v6(p_f0, p_fX)
    plot_b_manifold_v6(p_f0, p_fX, v_X)
    plot_c_components_v6(p_s0)
    
    print("\n✅ Generated final_plot_a_v6.png, final_plot_b_v6.png, final_plot_c_v6.png")