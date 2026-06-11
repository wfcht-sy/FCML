import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import EVAL_RESULTS_DIR, FIGURES_DIR
import warnings
warnings.filterwarnings("ignore")

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']

WIND_CONDITIONS = {
    'online_test_nowind': '0 m/s', 
    'online_test_35wind': '4.2 m/s',
    'online_test_70wind': '8.5 m/s', 
    'online_test_70p20sint': 'Sinusoidal',
    'online_test_100wind': '12.1 m/s'
}

def load_data(ctrl, wind):
    file_path = os.path.join(RESULTS_DIR, f"eval_data_VirtualMission_{ctrl}_{wind}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None


def calculate_cross_track_rmse(df_steady, ref_x, ref_y):
    import numpy as np
    pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
    ref_pts = np.vstack((ref_x, ref_y)).T
    distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
    return np.sqrt(np.mean(distances**2))

def plot_fig1_rmse_bar():
    print("  -> Computing Cross-Track RMSE and generating Fig 1 Bar chart...")
    import numpy as np
    import matplotlib.pyplot as plt
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
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    rmse = calculate_cross_track_rmse(df_steady, ref_x, ref_y)
                    rmse_data[ctrl].append(rmse * 100.0)
                else:
                    rmse_data[ctrl].append(np.nan)
            else: 
                rmse_data[ctrl].append(np.nan)

    x = np.arange(len(valid_winds))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    
    COLORS = {
        'Baseline': '#7f7f7f',
        'INDI': '#1f77b4',
        'L1': '#9467bd',
        'Neural-Fly': '#ff7f0e',
        'FCML': '#2ca02c'
    }
    
    LABELS = {
        'Baseline': 'Baseline PID',
        'INDI': 'INDI',
        'L1': 'L1 Adaptive',
        'Neural-Fly': 'Neural-Fly (DAIML)',
        'FCML': 'FCML (FCML)'
    }
    
    for i, ctrl in enumerate(CONTROLLERS):
        ax.bar(x + i*width - width*2, rmse_data[ctrl], width, label=LABELS.get(ctrl, ctrl), 
               color=COLORS.get(ctrl, '#000000'), edgecolor='black', linewidth=1.5, zorder=3)

    ax.set_ylabel('Cross-Track RMSE [cm]', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_winds)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Place legend at the bottom
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(CONTROLLERS), frameon=False, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_Mission_RMSE_Bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_fig2_trajectory_grid():
    winds = list(WIND_CONDITIONS.keys())
    fig, axes = plt.subplots(nrows=len(winds), ncols=len(CONTROLLERS), 
                             figsize=(16, 12), sharex=True, sharey=True)
                             
    theta_ref = np.linspace(0, 2 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)
    
    # Calculate global max error first to align colormap strictly with absolute MSE
    global_max_err = 0.0
    for w in winds:
        for c in CONTROLLERS:
            df_temp = load_data(c, w)
            if df_temp is not None and not df_temp.empty:
                df_s = df_temp[(df_temp['time'] >= 15.0) & (df_temp['time'] <= 85.0)]
                if not df_s.empty:
                    errs = np.linalg.norm(df_s[['pos_err_x', 'pos_err_y', 'pos_err_z']].values, axis=1) * 100.0
                    global_max_err = max(global_max_err, errs.max())
    
    import matplotlib.colors as colors
    # Use 'jet' colormap: low error (blue) to high error (red), matching academic standards
    cmap = plt.get_cmap('jet')
    norm = colors.LogNorm(vmin=1.0, vmax=50.0)
                             
    for i, wind_key in enumerate(winds):
        for j, controller in enumerate(CONTROLLERS):
            ax = axes[i, j]
            
            # Plot reference trajectory
            ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=1.5, alpha=0.9, zorder=1)
            
            df = load_data(controller, wind_key)
            if df is not None and not df.empty:
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    # Use Cross-Track Error (CTE) to match Neural-Fly paper visualization
                    pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
                    ref_pts = np.vstack((ref_x, ref_y)).T
                    distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
                    error_norm = distances * 100.0
                    
                    error_norm = np.clip(error_norm, 1.0, 50.0)
                    
                    # Construct line segments
                    points = np.array([df_steady['p_x'].values, df_steady['p_y'].values]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=2)
                    lc.set_array(error_norm[:-1])
                    lc.set_linewidth(2.5)
                    line = ax.add_collection(lc)

            if i == 0:
                ax.set_title(controller, fontsize=15, fontweight='bold', pad=15)
            if j == 0:
                ax.set_ylabel(f'{WIND_CONDITIONS[wind_key]}\nEast (Y) [m]', fontsize=12, fontweight='bold')
            if i == len(winds) - 1:
                ax.set_xlabel('North (X) [m]', fontsize=12, fontweight='bold')
                
            ax.set_aspect('equal', 'box')
            # Set axis limits based on true trajectory range
            ax.set_xlim(-6, 6)
            ax.set_ylim(-4, 4)
            ax.tick_params(labelsize=10)
            
    plt.tight_layout()
    # Leave top space for column titles, and right space for Colorbar
    fig.subplots_adjust(top=0.95, right=0.9)
    
    # Place Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[1, 3, 10, 30, 50])
    cbar.ax.set_yticklabels(['1', '3', '10', '30', '>50'])
    cbar.set_label('Tracking error [cm]', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Title removed as requested
    out_path = os.path.join(FIGURES_DIR, 'fig2_tracking_trajectory_grid.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_fig3_disturbance_estimation_grid(axis='x'):
    """Dense gray background + colored trend lines.
    Matches Neural-Fly paper aesthetic, saving independent images per axis.
    """
    winds = list(WIND_CONDITIONS.keys())

    STYLE = {
        'Baseline':   dict(color='#d62728', linestyle='--',  linewidth=1.6, alpha=0.92),
        'INDI':       dict(color='#1f77b4', linestyle='-',   linewidth=1.8, alpha=0.92),
        'L1':         dict(color='#9467bd', linestyle='-',   linewidth=1.8, alpha=0.93),
        'Neural-Fly': dict(color='#ff7f0e', linestyle='-',   linewidth=2.0, alpha=0.93),
        'FCML':       dict(color='#2ca02c', linestyle='-',   linewidth=2.2, alpha=0.95),
    }

    T_LOAD_START = 10.0   
    T_LOAD_END   = 80.0
    T_DISP_LEN   = 60.0    # Show overall trend
    
    WIND_T_MID = {
        'online_test_nowind': 45.0,
        'online_test_35wind': 45.0,
        'online_test_70wind': 45.0,
        'online_test_70p20sint': 45.0,
        'online_test_100wind': 45.0
    }

    fig3_ctrls = [c for c in CONTROLLERS if c != 'Baseline']

    fig, axes = plt.subplots(
        nrows=len(winds), ncols=len(fig3_ctrls),
        figsize=(22, 14), sharex='row', sharey='row')

    true_col = f'f_true_{axis}'
    est_col  = f'f_est_{axis}'
    
    for i, wind_key in enumerate(winds):
        # --- Pre-load reference gray line for the entire row to ensure PERFECT alignment ---
        ref_df = load_data('FCML', wind_key) 
        if ref_df is None or ref_df.empty or true_col not in ref_df.columns:
            ref_df = load_data('Baseline', wind_key) 
                
        if ref_df is not None and not ref_df.empty and true_col in ref_df.columns:
            ref_df_long = ref_df[(ref_df['time'] >= T_LOAD_START) & (ref_df['time'] <= T_LOAD_END)]
            ref_t_long = ref_df_long['time'].values
            ref_f_gray = ref_df_long[true_col].values
            
            T_MID = WIND_T_MID[wind_key]
            t0, t1 = T_MID - T_DISP_LEN / 2, T_MID + T_DISP_LEN / 2
            ref_mask = (ref_t_long >= t0) & (ref_t_long <= t1)
            ref_t_show = ref_t_long[ref_mask]
            ref_gray_show = ref_f_gray[ref_mask]
        else:
            ref_t_show, ref_gray_show = [], []

        for j, controller in enumerate(fig3_ctrls):
            ax = axes[i, j]
            df = load_data(controller, wind_key)

            if df is not None and not df.empty and est_col in df.columns:
                df_long = df[(df['time'] >= T_LOAD_START) & (df['time'] <= T_LOAD_END)]
                if df_long.empty: continue

                t_long = df_long['time'].values
                f_hat = df_long[est_col].values

                T_MID = WIND_T_MID[wind_key]
                t0, t1 = T_MID - T_DISP_LEN / 2, T_MID + T_DISP_LEN / 2
                mask   = (t_long >= t0) & (t_long <= t1)
                t_show     = t_long[mask]
                hat_show   = f_hat[mask]
                
                if len(t_show) == 0: continue

                # --- 1. Plot the SHARED reference gray line for the row ---
                if len(ref_t_show) > 0:
                    ax.plot(ref_t_show, ref_gray_show,
                            color='#b0b0b0', linewidth=1.2, alpha=1.0,
                            label=r'$f$ (measured)', zorder=1)
                    
                # --- 2. Plot the Actual Estimated Force Trend ---
                if controller == 'Baseline':
                    label_str = r'$K_i \int \tilde{p} dt$'
                else:
                    label_str = r'$\hat{f}$'

                ax.plot(t_show, hat_show, label=label_str, zorder=2, **STYLE[controller])

            if i == 0: axes[i, j].set_title(controller, fontsize=16, fontweight='bold', pad=15)
            if j == 0: axes[i, j].set_ylabel(f'{WIND_CONDITIONS[wind_key]}\nForce {axis.upper()} (N)', fontsize=14, fontweight='bold')
            axes[i, j].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
            
            # Add horizontal lines to match Neural-Fly paper aesthetic
            axes[i, j].axhline(0, color='gray', linestyle=':', linewidth=0.8, zorder=0)
            axes[i, j].tick_params(labelsize=12)

        # Fixed y-axis
        for j in range(len(fig3_ctrls)):
            axes[i, j].set_ylim(-10, 10)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(fig3_ctrls), fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(FIGURES_DIR, f'fig3_disturbance_grid_{axis}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Fig 1 RMSE Bar chart...")
    plot_fig1_rmse_bar()
    
    print("Generating Fig 2 Trajectory Grid layout with colormap...")
    plot_fig2_trajectory_grid()
    
    print("Generating Fig 3 Grids for XYZ independently...")
    for ax_str in ['x', 'y', 'z']:
        plot_fig3_disturbance_estimation_grid(ax_str)
        
    print(f"All Grid figures successfully re-generated and saved to {FIGURES_DIR}.")
