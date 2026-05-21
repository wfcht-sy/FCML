import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import butter, filtfilt, lfilter

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

def plot_fig2_trajectory_grid():
    winds = list(WIND_CONDITIONS.keys())
    fig, axes = plt.subplots(nrows=len(winds), ncols=len(CONTROLLERS), 
                             figsize=(16, 12), sharex=True, sharey=True)
                             
    theta_ref = np.linspace(0, 2 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)
    
    # 颜色映射统一范围 (统一转换为 cm 单位，上限缩小以提升色阶对比度)
    err_vmax = 80.0 # 设置上限为 80 cm，使得大部分主体误差落在中间绿色区间
    cmap = plt.get_cmap('jet') # 类似 Neural-Fly 采用的热力图色条
    norm = plt.Normalize(vmin=0, vmax=err_vmax)
                             
    for i, wind_key in enumerate(winds):
        for j, controller in enumerate(CONTROLLERS):
            ax = axes[i, j]
            
            # Plot reference trajectory
            ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=1.5, alpha=0.9, zorder=1)
            
            df = load_data(controller, wind_key)
            if df is not None and not df.empty:
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    # 计算误差映射作为颜色, 并转换为 cm 单位
                    error_norm = np.linalg.norm(df_steady[['pos_err_x', 'pos_err_y', 'pos_err_z']].values, axis=1) * 100.0
                    
                    # 构造线段
                    points = np.array([df_steady['p_x'].values, df_steady['p_y'].values]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=2)
                    lc.set_array(error_norm[:-1])
                    lc.set_linewidth(2.5)
                    line = ax.add_collection(lc)

            if i == 0:
                ax.set_title(controller, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{WIND_CONDITIONS[wind_key]}\nEast (Y) [m]', fontsize=12, fontweight='bold')
            if i == len(winds) - 1:
                ax.set_xlabel('North (X) [m]', fontsize=12, fontweight='bold')
                
            ax.set_aspect('equal', 'box')
            # 根据真实轨迹大小设定刻度范围
            ax.set_xlim(-6, 6)
            ax.set_ylim(-4, 4)
            ax.tick_params(labelsize=10)
            
    plt.tight_layout()
    # 预留顶部空间放置通用标题，以及右侧/右上部放置 Colorbar
    fig.subplots_adjust(top=0.92, right=0.9)
    
    # 放置 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Tracking Error (cm)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    fig.suptitle('Fig. 2: 2D Waypoint Tracking Trajectory Colored by Position Error', fontsize=18, fontweight='bold', x=0.45)
    out_path = os.path.join(FIGURES_DIR, 'fig2_tracking_trajectory_grid.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_fig3_disturbance_estimation_grid(axis='x'):
    """Dense gray background + colored trend lines (Neural-Fly Fig. S3 style).

    Gray  : f_true (Groundtruth)
    Colored (Raw unfiltered estimator output f_est):
               Baseline   → red
               INDI       → blue
               L1         → purple
               Neural-Fly → orange
               FCML       → green
    """
    winds = list(WIND_CONDITIONS.keys())

    STYLE = {
        'Baseline':   dict(color='#d62728', linestyle='--',  linewidth=1.6, alpha=0.92),
        'INDI':       dict(color='#1f77b4', linestyle='-',   linewidth=1.8, alpha=0.92),
        'L1':         dict(color='#9467bd', linestyle='-',   linewidth=1.8, alpha=0.93),
        'Neural-Fly': dict(color='#ff7f0e', linestyle='-',   linewidth=2.0, alpha=0.93),
        'FCML':       dict(color='#2ca02c', linestyle='-',   linewidth=2.2, alpha=0.95),
    }

    FS      = 47.0   # actual sampling rate

    T_LOAD_START = 10.0   # load from 10s for filter warmup
    T_LOAD_END   = 80.0
    T_DISP_LEN   = 20.0   # show 20 s (like FCML05183's 20-40s window)
    T_MID        = 30.0   # display [20, 40]s — covers dynamic tracking trends


    true_col = f'f_true_{axis}'

    fig, axes = plt.subplots(
        nrows=len(winds), ncols=len(CONTROLLERS),
        figsize=(22, 14), sharex=True, sharey='row')

    for i, wind_key in enumerate(winds):
        filt_vals = []
        for j, controller in enumerate(CONTROLLERS):
            ax = axes[i, j]
            df = load_data(controller, wind_key)

            if df is not None and not df.empty and true_col in df.columns:
                df_long = df[(df['time'] >= T_LOAD_START) &
                             (df['time'] <= T_LOAD_END)].copy()
                if df_long.empty:
                    continue

                t_long     = df_long['time'].values
                f_raw      = df_long[true_col].values

                # ── Dense gray background (RAW data for maximum density/area) ────────
                f_gray = f_raw

                # ── Raw unfiltered estimator output ──────────────────────────
                est_col = f'f_est_{axis}'
                if est_col in df_long.columns:
                    f_hat = df_long[est_col].values
                else:
                    f_hat = np.zeros_like(t_long)

                # ── Crop to display window ───────────────────────────────────
                t0, t1 = T_MID - T_DISP_LEN / 2, T_MID + T_DISP_LEN / 2
                mask   = (t_long >= t0) & (t_long <= t1)
                t_show     = t_long[mask]
                gray_show  = f_gray[mask]
                hat_show   = f_hat[mask]
                if len(t_show) == 0:
                    continue

                # ── Maximum density visible gray line (darker & fully opaque) ─────────
                ax.plot(t_show, gray_show,
                        color='#707070', linewidth=1.2, alpha=1.0,
                        label=r'$f$ (measured)', zorder=1)
                if controller == 'Baseline':
                    label_str = r'$K_i \int \tilde{p} dt$' + f' ({controller})'
                else:
                    label_str = r'$\hat{f}$' + f' ({controller})'

                ax.plot(t_show, hat_show,
                        label=label_str,
                        zorder=2, **STYLE[controller])

                # y-axis range based on gray only (trend lines stay within)
                filt_vals.extend(gray_show.tolist())

            if i == 0:
                ax.set_title(controller, fontsize=13, fontweight='bold')
            if j == 0:
                ax.set_ylabel(
                    f'{WIND_CONDITIONS[wind_key]}\nForce {axis.upper()} (N)',
                    fontsize=11, fontweight='bold')
            if i == len(winds) - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            ax.tick_params(labelsize=9)
            if i == 0 and j == len(CONTROLLERS) - 1:
                ax.legend(loc='upper right', fontsize=9,
                          framealpha=0.88, edgecolor='#aaaaaa')

        # Fixed y-axis: -10 to +10 N
        for j in range(len(CONTROLLERS)):
            axes[i, j].set_ylim(-10, 10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(
        f'Fig. S3: Measured Residual Force $f$ vs Adaptive Augmentation '
        f'$\\hat{{f}}$ (Axis: {axis.upper()})',
        fontsize=15, fontweight='bold')
    out_path = os.path.join(FIGURES_DIR, f'fig3_disturbance_grid_{axis}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Generating Fig 2 Trajectory Grid layout with colormap...")
    plot_fig2_trajectory_grid()
    
    print("Generating Fig 3 Grids for XYZ independently...")
    for ax_str in ['x', 'y', 'z']:
        plot_fig3_disturbance_estimation_grid(ax_str)
        
    print(f"All Grid figures successfully re-generated and saved to {FIGURES_DIR}.")
