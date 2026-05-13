#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import FIGURES_DIR, EVAL_RESULTS_DIR

# ---------------------------------------------------------------------------
# Controller display configuration
# ---------------------------------------------------------------------------
ALGOS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']
COLORS = {'Baseline': '#7f7f7f', 'INDI': '#1f77b4', 'L1': '#9467bd', 'Neural-Fly': '#ff7f0e', 'FCML': '#2ca02c'}
MARKERS = {'Baseline': 'o', 'INDI': 's', 'L1': '^', 'Neural-Fly': 'D', 'FCML': '*'}
LABELS = {'Baseline': 'Baseline (PID)', 'INDI': 'INDI', 'L1': 'L1 Adaptive', 'Neural-Fly': 'Neural-Fly (DAIML)', 'FCML': 'FCML'}

def plot_sweep(results, save_path=None):
    """Generate the Ki sweep figure from pre-computed results, with marked data points."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))

    for algo in ALGOS:
        if algo not in results:
            print(f"Skipping {algo}, not found in results.")
            continue
            
        data = results[algo]
        Ki_range = np.array(data['Ki_range'])
        rmses = data['rmses']
        best_Ki = data['best_Ki']
        best_rmse = data['best_rmse']

        # Plot trend line
        ax.plot(Ki_range, rmses, linewidth=2.0, color=COLORS[algo], label=LABELS[algo])
        
        # Plot ALL data points
        ax.scatter(Ki_range, rmses, color=COLORS[algo], marker=MARKERS[algo], 
                   s=40, alpha=0.7, edgecolors='none', zorder=4)

        # Highlight the optimal parameter point
        ax.scatter(best_Ki, best_rmse, color=COLORS[algo], marker=MARKERS[algo],
                   s=150, edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xlabel(r'Integral Gain ($K_i$)')
    ax.set_ylabel(r'Cross-Track RMSE (m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, 'Ki_sweep_optimization.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")

if __name__ == '__main__':
    json_path = os.path.join(EVAL_RESULTS_DIR, 'Ki_sweep_results.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run simulate_Ki_sweep.py first.")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    plot_sweep(results)
