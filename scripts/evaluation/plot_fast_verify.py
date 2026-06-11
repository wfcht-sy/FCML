import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVAL_RESULTS_DIR = "eval_results"
FIGURES_DIR = "figures"

CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']
WIND = '100wind' # 12.1m/s

STYLE = {
    'Baseline':   dict(color='#1f77b4', linestyle='-',   linewidth=1.2, alpha=0.8),
    'INDI':       dict(color='#17becf', linestyle='-',   linewidth=1.2, alpha=0.8),
    'L1':         dict(color='#bcbd22', linestyle='-',   linewidth=1.2, alpha=0.8),
    'Neural-Fly': dict(color='#ff7f0e', linestyle='-',   linewidth=1.2, alpha=0.8),
    'FCML':       dict(color='#2ca02c', linestyle='-',   linewidth=1.2, alpha=0.8),
}

for axis, force_col in zip(['x', 'y', 'z'], ['f_ext_x', 'f_ext_y', 'f_ext_z']):
    fig, axes = plt.subplots(1, len(CONTROLLERS), figsize=(15, 3), sharey=True)
    
    for i, controller in enumerate(CONTROLLERS):
        ax = axes[i]
        csv_file = os.path.join(EVAL_RESULTS_DIR, f"fast_verify_{controller}_{WIND}.csv")
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        df_plot = df[(df['time'] >= 35.0) & (df['time'] <= 40.0)]
        
        t = df_plot['time'].values
        f_true = df_plot[force_col].values
        
        # Neural-Fly plots the baseline's integral response as Ki*integral(p_dt) which is f_est_x
        # f_est_x actually records the final estimated force for all methods.
        f_est = df_plot[f'f_est_{axis}'].values
        
        ax.plot(t, f_true, color='gray', alpha=0.5, linewidth=1.5, label='f (True)')
        ax.plot(t, f_est, **STYLE[controller], label='f_hat (Est)')
        
        ax.set_title(controller)
        if i == 0:
            ax.set_ylabel(f'Force {axis.upper()} (N)')
        ax.set_xlabel('Time (s)')
        
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(FIGURES_DIR, f'fast_verify_fig3_{axis}.png'), dpi=150)
    plt.close()
    
print("Fast verify plots generated.")
