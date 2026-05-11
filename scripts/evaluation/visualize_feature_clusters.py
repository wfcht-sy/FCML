#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-dimensional feature manifold analysis (attitude-based clustering).

Generates Fig 4 (t-SNE) and Fig 5 (LDA/PCA) to demonstrate that the learned
feature representation is dominated by physical flight attitude rather than
wind conditions.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import warnings

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import OURS_MODEL_PATH as _OURS_MODEL_PATH, EVAL_RESULTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OURS_MODEL_PATH = _OURS_MODEL_PATH
RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

torch.set_default_dtype(torch.float64)

class PhiNetwork(nn.Module):
    def __init__(self, input_dim=11, basis_dim=8):
        super(PhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, basis_dim - 1)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        bias = torch.ones((out.shape[0], 1), device=out.device, dtype=out.dtype)
        return torch.cat([out, bias], dim=-1)

CSV_FILES = {
    '0 m/s (Calm)': os.path.join(RESULTS_DIR, 'eval_data_Ours_online_test_nowind.csv'),
    '8.5 m/s (Gust)': os.path.join(RESULTS_DIR, 'eval_data_Ours_online_test_70p20sint.csv'),
    '12.1 m/s (Extreme)': os.path.join(RESULTS_DIR, 'eval_data_Ours_online_test_100wind.csv')
}

def euler_from_quaternion(w, x, y, z):
    """Compute Euler angles (Roll, Pitch) from a quaternion. Returns radians."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return pitch, roll

def run():
    model = PhiNetwork()
    try:
        model.load_state_dict(torch.load(OURS_MODEL_PATH, map_location='cpu', weights_only=True)['model_state_dict'])
        model.eval()
    except Exception as e: 
        print(f"ERROR: Failed to load model: {e}"); return

    all_raw_data = []
    
    for wind_name, file_path in CSV_FILES.items():
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        df = df[df['time'] > 15.0] 
        
        states = df[['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']].values
        quats = df[['q_w', 'q_x', 'q_y', 'q_z']].values
        
        if len(states) == 0: continue

        with torch.no_grad(): 
            raw_feats = model(torch.tensor(states, dtype=torch.float64))
            dyn_feats = raw_feats[:, :-1].numpy() 
            
        for i in range(len(dyn_feats)):
            pitch_rad, roll_rad = euler_from_quaternion(quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3])
            all_raw_data.append({
                'feature': dyn_feats[i],
                'pitch': pitch_rad,
                'roll': roll_rad,
                'wind': wind_name
            })
            
    if not all_raw_data:
        print("ERROR: No evaluation data found. Please run the evaluation scripts first.")
        return

    # ================== Attitude-based KMeans Clustering ==================
    angles = np.array([[d['pitch'], d['roll']] for d in all_raw_data])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(angles)
    
    # Automatic quadrant naming (pitch>0 = nose-up, roll>0 = right-bank)
    centers = kmeans.cluster_centers_
    action_map = {}
    for i, center in enumerate(centers):
        p_str = "Nose-Up" if center[0] > 0 else "Nose-Down"
        r_str = "Right-Bank" if center[1] > 0 else "Left-Bank"
        action_map[i] = f"Attitude {i+1} ({p_str} & {r_str})"

    stratified_data = defaultdict(list)
    for idx, d in enumerate(all_raw_data):
        action = action_map[cluster_labels[idx]]
        stratified_data[(action, d['wind'])].append(d['feature'])

    # ================== Data Health Diagnostics ==================
    print("\n=== Flight Attitude Distribution Summary ===")
    action_counts = defaultdict(int)
    for (act, wnd), feats in stratified_data.items():
        action_counts[act] += len(feats)
        print(f"  - {act} | {wnd}: {len(feats)} frames")
    print("=" * 45 + "\n")

    # ================== Balanced Sampling ==================
    MAX_SAMPLES_PER_GROUP = 120 
    final_features, final_actions, final_winds = [], [], []
    
    for (act, wnd), feats in stratified_data.items():
        feats_arr = np.array(feats)
        if len(feats_arr) > MAX_SAMPLES_PER_GROUP:
            indices = np.random.choice(len(feats_arr), MAX_SAMPLES_PER_GROUP, replace=False)
            feats_sampled = feats_arr[indices]
        else:
            feats_sampled = feats_arr
            
        final_features.extend(feats_sampled)
        final_actions.extend([act] * len(feats_sampled))
        final_winds.extend([wnd] * len(feats_sampled))

    features_np = np.array(final_features)
    actions_np = np.array(final_actions)
    winds_np = np.array(final_winds)

    features_scaled = StandardScaler().fit_transform(features_np)

    print("Running t-SNE manifold embedding (unsupervised validation)...")
    try:
        tsne = TSNE(n_components=2, perplexity=35, max_iter=2000, init='pca', random_state=42)
        tsne_feats = tsne.fit_transform(features_scaled)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=35, n_iter=2000, init='pca', random_state=42)
        tsne_feats = tsne.fit_transform(features_scaled)

    print("Running supervised dimensionality reduction (LDA/PCA)...")
    unique_classes = len(np.unique(actions_np))
    
    if unique_classes >= 3:
        lda = LDA(n_components=2)
        lda_feats = lda.fit_transform(features_scaled, actions_np)
        fig5_title = "Fig 5: Supervised LDA Projection (Attitude-Dominant Feature Decoupling)"
    else:
        pca = PCA(n_components=2)
        lda_feats = pca.fit_transform(features_scaled)
        fig5_title = "Fig 5: PCA Projection (Fallback Due to Insufficient Attitude Classes)"

    # ================== Plotting ==================
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    action_color_map = {
        action_map[0]: '#d62728', 
        action_map[1]: '#1f77b4', 
        action_map[2]: '#2ca02c', 
        action_map[3]: '#ff7f0e'  
    }
    
    wind_marker_map = {
        '0 m/s (Calm)': 'o',       
        '8.5 m/s (Gust)': 's',     
        '12.1 m/s (Extreme)': '^'   
    }

    def plot_relation_proof(feats, title, filename):
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

        for action, color in action_color_map.items():
            for wind, marker in wind_marker_map.items():
                mask = (actions_np == action) & (winds_np == wind)
                if np.any(mask):
                    ax.scatter(feats[mask, 0], feats[mask, 1], 
                               c=color, marker=marker, 
                               alpha=0.85, edgecolors='white', linewidths=0.5, s=90, zorder=3)
            
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_xticks([]); ax.set_yticks([]) 
        
        action_legend = [mlines.Line2D([], [], color=c, marker='o', linestyle='None', markersize=10, label=l) for l, c in action_color_map.items()]
        wind_legend = [mlines.Line2D([], [], color='gray', marker=m, linestyle='None', markersize=10, label=l) for l, m in wind_marker_map.items()]
        
        legend1 = ax.legend(handles=action_legend, loc='upper left', title="Primary Factor: Flight Attitude (Color)", fontsize=12, title_fontsize=13, framealpha=0.9, edgecolor='black')
        ax.add_artist(legend1)
        ax.legend(handles=wind_legend, loc='upper right', title="Secondary Factor: Wind Condition (Marker)", fontsize=12, title_fontsize=13, framealpha=0.9, edgecolor='black')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {out_path}")

    plot_relation_proof(tsne_feats, "Fig 4: t-SNE Feature Clustering (Attitude-Dominant Representation)", 'fig4_tsne_action_dominant.png')
    plot_relation_proof(lda_feats, fig5_title, 'fig5_lda_action_dominant.png')

if __name__ == "__main__": run()