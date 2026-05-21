#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Tracking Comparison: DTW-Triplet (FCML) vs MSE-only (FCML_NoTriplet).

This script completes the ablation study by comparing online flight tracking
performance between the two FCML variants that share the same backbone:
  - [D] FCML         : official Phi_Net + DTW-Triplet loss  (full method)
  - [B] FCML_NoTrip  : official Phi_Net + MSE-only loss     (triplet ablation)

Usage
-----
Step 1 – Generate MSE-only flight logs (skip if already done):
  Copy the MSE-only checkpoint to a known path, then run the online mission:

    cp training_results/backbone_ablation/run_ours_no_triplet/best_model.pth \\
       checkpoints/fcml_notriplet.pth

  Then for each wind condition, run:
    python scripts/missions/online_mission_compare.py \\
      --controller FCML_NoTriplet --wind <wind_tag>

Step 2 – Run this plot script:
    python scripts/evaluation/compare_online_triplet_vs_mse.py

Output: figures/fig_triplet_vs_mse_online.png
"""

import os
import sys
import glob
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import EVAL_RESULTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
CTRL_D   = "FCML"           # [D] DTW-Triplet
CTRL_B   = "FCML_NoTriplet" # [B] MSE-only

WIND_CONDITIONS = [
    ("nowind",       "0.0 m/s"),
    ("35wind",       "4.2 m/s"),
    ("70wind",       "8.5 m/s"),
    ("70p20sint",    "Sinusoidal"),
    ("100wind",      "12.1 m/s"),
]

COLORS = {
    CTRL_D: "#d62728",   # red  – our full method
    CTRL_B: "#ff7f0e",   # orange – MSE-only ablation
}
LABELS = {
    CTRL_D: "[D] FCML (DTW-Triplet)",
    CTRL_B: "[B] FCML w/o Triplet (MSE only)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_log(controller: str, wind_tag: str):
    """Return path to the eval CSV for (controller, wind_tag), or None."""
    pattern = os.path.join(
        EVAL_RESULTS_DIR,
        f"eval_data_VirtualMission_{controller}_{wind_tag}.csv"
    )
    matches = glob.glob(pattern)
    if not matches:
        pattern2 = os.path.join(
            EVAL_RESULTS_DIR,
            f"eval_data_VirtualMission_{controller}_online_test_{wind_tag}.csv"
        )
        matches = glob.glob(pattern2)
    return matches[0] if matches else None


def _load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pos_err_mag" not in df.columns:
        df["pos_err_mag"] = np.sqrt(
            df["pos_err_x"]**2 + df["pos_err_y"]**2 + df["pos_err_z"]**2
        )
    return df


def _rmse(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(series**2)))


# ── Main comparison ───────────────────────────────────────────────────────────

def compare_online():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # ── Collect data ──────────────────────────────────────────────────────
    data = {}
    missing = []
    for ctrl in [CTRL_D, CTRL_B]:
        data[ctrl] = {}
        for wind_tag, _ in WIND_CONDITIONS:
            path = _find_log(ctrl, wind_tag)
            if path:
                data[ctrl][wind_tag] = _load_log(path)
            else:
                missing.append(f"{ctrl} / {wind_tag}")

    if missing:
        print("\n  [WARNING] The following logs were not found:")
        for m in missing:
            print(f"    {m}")
        print("\n  Follow the Step 1 instructions in this script's docstring")
        print("  to generate the missing FCML_NoTriplet logs.")
        if not any(data[CTRL_B].values()):
            print("\n  [ABORT] No FCML_NoTriplet logs found at all. Cannot plot comparison.")
            _print_summary(data)
            return

    n_wind = len(WIND_CONDITIONS)
    fig, axes = plt.subplots(1, n_wind, figsize=(4 * n_wind, 4.5))
    fig.suptitle(
        "Online Tracking: DTW-Triplet vs MSE-only (unified backbone)",
        fontsize=14, fontweight="bold"
    )

    # ── Top row: time-series ───────────────────────────────────────────────
    for col, (wind_tag, wind_label) in enumerate(WIND_CONDITIONS):
        ax = axes[col]
        plotted_any = False
        for ctrl in [CTRL_D, CTRL_B]:
            if wind_tag in data[ctrl]:
                df = data[ctrl][wind_tag]
                ax.plot(
                    df["time"], df["pos_err_mag"],
                    color=COLORS[ctrl], linewidth=1.2, alpha=0.85,
                    label=LABELS[ctrl]
                )
                plotted_any = True
        ax.set_title(wind_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Position Error |e| (m)", fontsize=9)
        ax.grid(True, ls="--", alpha=0.45)
        ax.set_ylim(bottom=0)
        
        # Add RMSE annotations directly onto the time-series plot
        y_max = ax.get_ylim()[1]
        for idx_ctrl, ctrl in enumerate([CTRL_D, CTRL_B]):
            if wind_tag in data[ctrl]:
                df = data[ctrl][wind_tag]
                rmse_val = _rmse(df["pos_err_mag"])
                # Add horizontal line for average RMSE
                ax.axhline(y=rmse_val, color=COLORS[ctrl], linestyle="--", linewidth=1.5, alpha=0.6)
                # Add text annotation
                text_y = y_max * 0.9 - idx_ctrl * (y_max * 0.1)
                ax.text(
                    0.05, 0.95 - idx_ctrl * 0.08,
                    f"{ctrl} RMSE: {rmse_val:.3f} m",
                    transform=ax.transAxes,
                    color=COLORS[ctrl],
                    fontsize=9, fontweight="bold",
                    va="top", ha="left",
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

        if plotted_any and col == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_triplet_vs_mse_online.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")

    _print_summary(data)


def _print_summary(data: dict):
    print("\n  +----------------------+----------------+----------------+----------+")
    print(  "  | Wind Condition       | [D] DTW-Triplet| [B] MSE-only   |  Δ RMSE  |")
    print(  "  +----------------------+----------------+----------------+----------+")

    triplet_wins = 0
    for wind_tag, wind_label in WIND_CONDITIONS:
        rmse_d = rmse_b = float("nan")
        if wind_tag in data.get(CTRL_D, {}):
            rmse_d = _rmse(data[CTRL_D][wind_tag]["pos_err_mag"])
        if wind_tag in data.get(CTRL_B, {}):
            rmse_b = _rmse(data[CTRL_B][wind_tag]["pos_err_mag"])

        if not np.isnan(rmse_d) and not np.isnan(rmse_b):
            delta = rmse_d - rmse_b
            delta_str = f"{delta:+.4f}"
            if delta < 0:
                delta_str += " *"
                triplet_wins += 1
        else:
            delta_str = "  N/A   "

        d_str = f"{rmse_d:.5f}" if not np.isnan(rmse_d) else " MISSING"
        b_str = f"{rmse_b:.5f}" if not np.isnan(rmse_b) else " MISSING"
        print(f"  | {wind_label:<20s} |   {d_str}    |   {b_str}    | {delta_str:<8s} |")

    print(  "  +----------------------+----------------+----------------+----------+")
    print(  "  Δ RMSE = [D] - [B]  (negative = DTW-Triplet tracks better)")
    if triplet_wins > 0:
        total = sum(1 for t, _ in WIND_CONDITIONS
                    if t in data.get(CTRL_D, {}) and t in data.get(CTRL_B, {}))
        print(f"  DTW-Triplet wins in {triplet_wins}/{total} wind conditions.")


if __name__ == "__main__":
    compare_online()
