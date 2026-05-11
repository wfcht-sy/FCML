#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import TSNE_CKPT_DIR, CHECKPOINTS_DIR, DTW_CSV as _DTW_CSV, TSNE_RESULTS_DIR, FIGURES_DIR


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings("ignore")

from scripts.offline.models import PhiNetworkOurs, PhiNetwork as PhiNetworkNF

# ================= Configuration =================
MODEL_PATHS = {
    "Ours (Epoch 0)": os.path.join(TSNE_CKPT_DIR, "tsne_model_epoch_0.pth"),
    "Ours (Epoch Mid)": os.path.join(TSNE_CKPT_DIR, "tsne_model_epoch_mid.pth"),
    "Ours (Final)": os.path.join(TSNE_CKPT_DIR, "best_model.pth"),
    "Neural-Fly (Final)": os.path.join(CHECKPOINTS_DIR, "neural_fly_daiml_best.pth")
}

DATA_CSV = _DTW_CSV
OUTPUT_DIR = TSNE_RESULTS_DIR

WIND_CONDITIONS_CONFIG = {
    '0.0 m/s': '#1f77b4',  
    '4.2 m/s': '#2ca02c',  
    '8.5 m/s': '#ff7f0e',  
    '12.0 m/s': '#d62728'  
}
WIND_SPEEDS = list(WIND_CONDITIONS_CONFIG.keys())
NUM_WINDS = len(WIND_SPEEDS)

SAMPLES_PER_WIND = 350  
VIRTUAL_WINDOW = 60      
REG_LAMBDA = 5e-3        
FORCE_SCALE = 6.0

torch.set_default_dtype(torch.float64)

# ================= Core Feature Extraction =================
def generate_robust_a_stars(model, df):
    feat_cols = ['A_v_x', 'A_v_y', 'A_v_z', 'A_q_w', 'A_q_x', 'A_q_y', 'A_q_z', 'A_pwm_1', 'A_pwm_2', 'A_pwm_3', 'A_pwm_4']
    label_cols = ['A_f_x', 'A_f_y', 'A_f_z']
    
    forces = df[label_cols].values
    force_mags = np.linalg.norm(forces[:, :2], axis=1)
    
    print(f"   -> Extracting features for {NUM_WINDS} wind regimes...")
    kmeans = KMeans(n_clusters=NUM_WINDS, random_state=42, n_init=10).fit(force_mags.reshape(-1, 1))
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    
    sorted_idx = np.argsort(centers)
    wind_names = {
        sorted_idx[i]: WIND_SPEEDS[i] for i in range(NUM_WINDS)
    }

    a_stars_list = []
    final_labels = []
    
    model.eval()
    model.to('cpu')
    
    for cluster_id in range(NUM_WINDS):
        pool_indices = np.where(cluster_labels == cluster_id)[0]
        wind_name = wind_names[cluster_id]
        
        # Steady-state sampling: exclude extreme outliers
        pool_forces = force_mags[pool_indices]
        sorted_local_idx = np.argsort(pool_forces)
        start_idx = int(len(sorted_local_idx) * 0.15)
        end_idx = int(len(sorted_local_idx) * 0.85)
        steady_pool_indices = pool_indices[sorted_local_idx[start_idx:end_idx]]
        
        if len(steady_pool_indices) < VIRTUAL_WINDOW:
            continue
            
        for _ in range(SAMPLES_PER_WIND):
            batch_idx = np.random.choice(steady_pool_indices, size=VIRTUAL_WINDOW, replace=False)
            
            states_batch = torch.tensor(df.iloc[batch_idx][feat_cols].values, dtype=torch.float64)
            forces_batch = torch.tensor(forces[batch_idx], dtype=torch.float64) / FORCE_SCALE
            
            for i in range(7, 11):
                if states_batch[:, i].mean() > 100:
                    states_batch[:, i] = (states_batch[:, i] - 1500.0) / 500.0
                    
            with torch.no_grad():
                phi = model(states_batch)
                phi_t = phi.t()
                
                A = torch.matmul(phi_t, phi) + REG_LAMBDA * torch.eye(phi.shape[1], dtype=torch.float64)
                B = torch.matmul(phi_t, forces_batch)
                
                try:
                    a_star = torch.linalg.solve(A, B)
                except:
                    a_star = torch.linalg.lstsq(A, B).solution
                    
                a_stars_list.append(a_star.flatten().numpy())
                final_labels.append(wind_name)
                
    return np.array(a_stars_list), np.array(final_labels)

# ================= Main Plotting =================
def main():
    print(f"Loading dataset: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    for i, (model_name, pth_path) in enumerate(MODEL_PATHS.items()):
        print(f"\n[{i+1}/4] Analyzing stage: {model_name}")
        
        if "Neural-Fly" in model_name:
            model = PhiNetworkNF(input_dim=11, basis_dim=8)
        else:
            model = PhiNetworkOurs(input_dim=11, basis_dim=8)
            
        if os.path.exists(pth_path):
            ckpt = torch.load(pth_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        else:
            print(f"WARNING: Model file not found: {pth_path}")
            print(f"  Skipping '{model_name}'. Please ensure the model is trained first.")
            ax = axes[i]
            ax.text(0.5, 0.5, f"Model Not Found:\n{os.path.basename(pth_path)}",
                    ha='center', va='center', fontsize=12, color='red',
                    transform=ax.transAxes)
            ax.set_title(model_name, fontsize=16, fontweight='bold', pad=15)
            continue
            
        a_stars, labels = generate_robust_a_stars(model, df)
        if len(a_stars) == 0:
            continue
            
        # 1. Absolute value folding and initial scaling
        a_stars = np.abs(a_stars)
        
        # 2. Dynamic denoising per wind class
        valid_mask = np.zeros(len(a_stars), dtype=bool)
        for w in np.unique(labels):
            idx = np.where(labels == w)[0]
            class_norms = np.linalg.norm(a_stars[idx], axis=1)
            p_low, p_high = np.percentile(class_norms, 3), np.percentile(class_norms, 97)
            mask_w = (class_norms > p_low) & (class_norms < p_high)
            valid_mask[idx[mask_w]] = True
        
        a_stars_clean = a_stars[valid_mask]
        labels_clean = labels[valid_mask]
        
        # 3. Feature importance equalization (preserve magnitude, no L2 normalization)
        stds = np.std(a_stars_clean, axis=0) + 1e-8
        a_stars_input = a_stars_clean / stds 

        # 4. T-SNE embedding
        print("   -> Computing T-SNE manifold embedding...")
        tsne = TSNE(
            n_components=2, 
            perplexity=35,           
            early_exaggeration=15.0,
            metric='euclidean',      
            learning_rate='auto',
            n_iter=3000, 
            init='pca',              
            random_state=42          
        )
        a_tsne = tsne.fit_transform(a_stars_input)
        
        # 5. Rotation alignment: force 0-12 m/s axis to 45 degrees
        idx_min = (labels_clean == WIND_SPEEDS[0])
        idx_max = (labels_clean == WIND_SPEEDS[-1])
        
        if np.any(idx_min) and np.any(idx_max):
            c_low = np.mean(a_tsne[idx_min], axis=0)
            c_high = np.mean(a_tsne[idx_max], axis=0)
            
            vec = c_high - c_low
            current_angle = np.arctan2(vec[1], vec[0])
            target_angle = np.pi / 4.0
            
            theta = target_angle - current_angle
            center = np.mean(a_tsne, axis=0)
            
            a_tsne_centered = a_tsne - center
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            a_tsne = np.dot(a_tsne_centered, rot_mat.T) + center

        # 6. Scatter plot
        ax = axes[i]
        for w in WIND_SPEEDS:
            if w in np.unique(labels_clean):
                idx = (labels_clean == w)
                color = WIND_CONDITIONS_CONFIG.get(w, '#7f7f7f')
                ax.scatter(a_tsne[idx, 0], a_tsne[idx, 1], 
                           c=color, label=w, alpha=0.7, edgecolors='white', linewidths=0.3, s=35)
            
        ax.set_title(model_name, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        if i == 3:
            ax.legend(title="Wind Condition", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=13)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "tsne_evolution_final_v2.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")

if __name__ == "__main__":
    main()