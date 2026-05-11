#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ki Parameter Sweep Analysis (Simulation-based)

Separates experiment logic (simulate_for_rmse) from plotting logic (plot_sweep).
Can be imported as a module or run standalone.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import FIGURES_DIR, EVAL_RESULTS_DIR

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 0.01
T_END = 90.0
T_ARRAY = np.arange(0, T_END, DT)
N = len(T_ARRAY)

# Reference trajectory: X-axis projection of Figure-8 virtual waypoint path
OMEGA_REF = 10 * np.pi / 90.0
X_REF = 4.0 * np.sin(OMEGA_REF * T_ARRAY)
V_REF = 4.0 * OMEGA_REF * np.cos(OMEGA_REF * T_ARRAY)

# Wind disturbance: composite turbulence model
WIND = -3.5 + 1.5 * np.sin(0.8 * T_ARRAY) + 0.5 * np.sin(2.5 * T_ARRAY)

# ---------------------------------------------------------------------------
# Controller display configuration
# ---------------------------------------------------------------------------
ALGOS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'Ours']
COLORS = {'Baseline': '#7f7f7f', 'INDI': '#1f77b4', 'L1': '#9467bd', 'Neural-Fly': '#ff7f0e', 'Ours': '#2ca02c'}
MARKERS = {'Baseline': 'o', 'INDI': 's', 'L1': '^', 'Neural-Fly': 'D', 'Ours': '*'}
LABELS = {'Baseline': 'Baseline (PID)', 'INDI': 'INDI', 'L1': 'L1 Adaptive', 'Neural-Fly': 'Neural-Fly (DAIML)', 'Ours': 'FCML (Ours)'}


def simulate_for_rmse(algo='Ours', Ki=0.0):
    """Run a 1D tracking simulation and return cross-track RMSE."""
    x, v, integral_e = 0.0, 0.0, 0.0

    if algo in ['Ours', 'Neural-Fly']:
        Kp, Kd = 6.0, 4.0
    else:
        Kp, Kd = 3.5, 2.5

    d_hat, u_comp_ema = 0.0, 0.0
    gamma = 25.0 if algo == 'Ours' else 8.0
    intent_lambda = 1.5
    tau_indi = 0.4
    v_hat_L1, d_hat_L1 = 0.0, 0.0
    omega_L1, Gamma_L1, Am = 2 * np.pi * 0.3, 15.0, 5.0

    rmse_sum = 0.0

    for i in range(N):
        e = X_REF[i] - x
        de = V_REF[i] - v

        integral_e += e * DT
        integral_e = np.clip(integral_e, -10.0, 10.0)

        u_nom = Kp * e + Kd * de + Ki * integral_e
        d = WIND[i]
        u_comp = 0.0

        if algo == 'Baseline':
            u_comp = 0.0

        elif algo in ['Ours', 'Neural-Fly']:
            s = de + intent_lambda * e
            if algo == 'Neural-Fly':
                perceived_s = np.clip(s, -1.0, 1.0)
                d_hat_dot = 3.0 * perceived_s
                d_hat += d_hat_dot * DT
                d_hat = np.clip(d_hat, -3.0, 3.0)
            else:
                d_hat_dot = gamma * s
                d_hat += d_hat_dot * DT
                d_hat = np.clip(d_hat, -15, 15)

            u_comp = d_hat
            alpha_ema = 0.55 if algo == 'Ours' else 0.35
            u_comp_ema = (1 - alpha_ema) * u_comp_ema + alpha_ema * u_comp
            u_comp = u_comp_ema

        elif algo == 'INDI':
            alpha = DT / (tau_indi + DT)
            d_hat = (1 - alpha) * d_hat + alpha * (-d)
            u_comp = d_hat

        elif algo == 'L1':
            v_tilde = v_hat_L1 - v
            acc_known = u_nom + u_comp_ema
            v_hat_L1 += (acc_known + d_hat_L1 - Am * v_tilde) * DT
            d_hat_L1 += (-Gamma_L1 * v_tilde) * DT
            alpha_L1 = DT * omega_L1 / (1 + DT * omega_L1)
            u_comp_ema = (1 - alpha_L1) * u_comp_ema + alpha_L1 * d_hat_L1
            u_comp = -u_comp_ema

        u = u_nom + u_comp
        acc = u + d
        v += acc * DT
        x += v * DT

        rmse_sum += e**2

    return np.sqrt(rmse_sum / N)


def run_sweep(Ki_range=None):
    """Run parameter sweep for all controllers. Returns dict of results."""
    if Ki_range is None:
        Ki_range = np.linspace(0.0, 1.5, 40)

    results = {}
    for algo in ALGOS:
        rmses = [simulate_for_rmse(algo, Ki) for Ki in Ki_range]
        min_idx = np.argmin(rmses)
        results[algo] = {
            'Ki_range': Ki_range.tolist(),
            'rmses': rmses,
            'best_Ki': float(Ki_range[min_idx]),
            'best_rmse': float(rmses[min_idx])
        }
        print(f"[{algo}] Optimal Ki: {Ki_range[min_idx]:.2f}, Min RMSE: {rmses[min_idx]:.3f}")
    return results


def plot_sweep(results, save_path=None):
    """Generate the Ki sweep figure from pre-computed results."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))

    for algo in ALGOS:
        data = results[algo]
        Ki_range = np.array(data['Ki_range'])
        rmses = data['rmses']
        best_Ki = data['best_Ki']
        best_rmse = data['best_rmse']

        ax.plot(Ki_range, rmses, linewidth=2.0, color=COLORS[algo], label=LABELS[algo])
        ax.scatter(best_Ki, best_rmse, color=COLORS[algo], marker=MARKERS[algo],
                   s=120, edgecolors='black', zorder=5)

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
    print("Starting parameter sweep for Ki...")

    # 1. Run experiment
    results = run_sweep()

    # 2. Save raw results as JSON for reproducibility
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(EVAL_RESULTS_DIR, 'Ki_sweep_results.json')
    json_results = {algo: {k: v for k, v in data.items()} for algo, data in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw data saved to: {json_path}")

    # 3. Plot
    plot_sweep(results)
