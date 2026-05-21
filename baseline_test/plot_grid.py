import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.facecolor'] = 'white'

RESULTS_DIR = "./eval_results"
WIND_CONDITIONS = ['nowind', '35wind', '70wind', '100wind']
CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']
WIND_LABELS = ['No Wind', 'Wind 3.5 m/s', 'Wind 7.0 m/s', 'Wind 10.0 m/s']

def load_data(controller, wind):
    file_path = os.path.join(RESULTS_DIR, f"eval_data_VirtualMission_{controller}_{wind}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def plot_fig3_disturbance_estimation_grid(axis='x'):
    """
    Fig. 3 (Task 3): Disturbance Estimation in separate layout loops per Force axis
    Row: Wind Cond, Col: Controller
    """
    fig, axes = plt.subplots(nrows=len(WIND_CONDITIONS), ncols=len(CONTROLLERS), 
                             figsize=(16, 12), sharex=True, sharey=True)
    
    true_col = f'f_true_{axis}'
    est_col = f'f_est_{axis}'

    for i, wind in enumerate(WIND_CONDITIONS):
        for j, controller in enumerate(CONTROLLERS):
            ax = axes[i, j]
            df = load_data(controller, wind)
            
            if df is not None:
                df_plot = df[(df['time'] > 10) & (df['time'] < 40)]
                ax.plot(df_plot['time'], df_plot[true_col], label='Groundtruth', 
                        color='black', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.plot(df_plot['time'], df_plot[est_col], label=f'{controller} Est.', 
                        color='tab:blue', linewidth=1.8, alpha=0.9)
                
            if i == 0:
                ax.set_title(controller, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{WIND_LABELS[i]}\nForce {axis.upper()} (N)', fontsize=12)
            if i == len(WIND_CONDITIONS) - 1:
                ax.set_xlabel('Time (s)', fontsize=12)
            
            ax.tick_params(labelsize=10)
            if i == 0 and j == len(CONTROLLERS) - 1:
                ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f'Fig. 3: Force Disturbance Estimation ({axis.upper()}-Axis) vs Groundtruth', fontsize=18, fontweight='bold')
    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f'./figures/fig3_disturbance_grid_{axis}.png', dpi=300)
    plt.close()

def plot_fig2_position_error_grid():
    """
    Fig. 2: 3D Position Error L2 Norm Matrix Grid layout
    """
    fig, axes = plt.subplots(nrows=len(WIND_CONDITIONS), ncols=len(CONTROLLERS), 
                             figsize=(16, 12), sharex=True, sharey=True)
                             
    for i, wind in enumerate(WIND_CONDITIONS):
        for j, controller in enumerate(CONTROLLERS):
            ax = axes[i, j]
            df = load_data(controller, wind)
            
            if df is not None:
                error_norm = np.linalg.norm(df[['pos_err_x', 'pos_err_y', 'pos_err_z']].values, axis=1)
                ax.plot(df['time'], error_norm, color='tab:red', linewidth=1.5)
                ax.fill_between(df['time'], 0, error_norm, color='tab:red', alpha=0.15)

            if i == 0:
                ax.set_title(controller, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{WIND_LABELS[i]}\nPos Error Norm (m)', fontsize=12)
            if i == len(WIND_CONDITIONS) - 1:
                ax.set_xlabel('Time (s)', fontsize=12)
                
            ax.set_ylim(0, 2.5) # 控制在学术范围如 2.5 米以内
            ax.tick_params(labelsize=10)
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle('Fig. 2: 3D Tracking Error Normal over Mission Profile', fontsize=18, fontweight='bold')
    os.makedirs("./figures", exist_ok=True)
    plt.savefig('./figures/fig2_tracking_error_grid.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating Fig 2 Grid layout...")
    plot_fig2_position_error_grid()
    
    print("Generating Fig 3 Grids for XYZ independently...")
    for ax_str in ['x', 'y', 'z']:
        plot_fig3_disturbance_estimation_grid(ax_str)
        
    print("Charts successfully saved to ./figures directory.")