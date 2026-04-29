#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟航点模式全量成图脚本 (基于空间 Cross-Track RMSE 的公平对比)
包含: RMSE全景柱状图、所有风速下的轨迹对比图、所有风速下的受力追踪对比图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'Ours']
RESULTS_DIR = "/home/zzx/testmodel/eval_results"
FIGURES_DIR = "/home/zzx/testmodel/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 所有的 5 种风况
WIND_CONDITIONS = {
    'online_test_nowind': '0 m/s (无风)', 
    'online_test_35wind': '4.2 m/s (恒风)',
    'online_test_70wind': '8.5 m/s (强风)', 
    'online_test_70p20sint': '动态阵风(8.5±2.4)', 
    'online_test_100wind': '12.1 m/s (极端风)'
}

LABEL_MAP = {
    'Baseline': '原生 PID (积分增强)', 
    'INDI': 'INDI 增量动态逆', 
    'L1': 'L1 自适应控制', 
    'Neural-Fly': '原版 Neural-Fly',
    'Ours': '我们的方案 (DTW-Triplet)'
}

COLORS = {'Baseline': '#7f7f7f', 'INDI': '#1f77b4', 'L1': '#9467bd', 'Neural-Fly': '#ff7f0e', 'Ours': '#2ca02c'}

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_cross_track_rmse(df_steady, ref_x, ref_y):
    """计算纯空间维度上的交叉轨迹误差"""
    pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
    ref_pts = np.vstack((ref_x, ref_y)).T
    distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
    return np.sqrt(np.mean(distances**2))

def load_data(ctrl, wind):
    file_path = os.path.join(RESULTS_DIR, f"eval_data_VirtualMission_{ctrl}_{wind}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def plot_cross_track_rmse_bar():
    print("  -> 正在计算 Cross-Track RMSE 并绘制全局柱状图...")
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
                # 截取 15秒 到 85秒的平稳段
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    rmse = calculate_cross_track_rmse(df_steady, ref_x, ref_y)
                    rmse_data[ctrl].append(rmse)
                else:
                    rmse_data[ctrl].append(np.nan)
            else: 
                rmse_data[ctrl].append(np.nan)

    x = np.arange(len(valid_winds))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, ctrl in enumerate(CONTROLLERS):
        ax.bar(x + i*width - width*2, rmse_data[ctrl], width, label=LABEL_MAP[ctrl], color=COLORS[ctrl], edgecolor='black')

    ax.set_ylabel('交叉轨迹误差 (Cross-Track RMSE) [米]', fontweight='bold')
    ax.set_title('虚拟航点模式下各方案的空间抗风追踪误差对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_winds)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_Mission_RMSE_Bar.png'), dpi=300)
    plt.close(fig)

def plot_all_trajectories():
    print("  -> 正在绘制所有风况下的 8 字轨迹对比图 (共 5 张)...")
    theta_ref = np.linspace(0, 2 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

    # 遍历每一种风场，生成单独的对比图
    for wind_key, wind_label in WIND_CONDITIONS.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=3, label='理想 8 字轨迹', zorder=10)

        has_data = False
        for ctrl in CONTROLLERS:
            df = load_data(ctrl, wind_key)
            if df is not None and not df.empty:
                df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
                if not df_steady.empty:
                    has_data = True
                    ax.plot(df_steady['p_x'], df_steady['p_y'], color=COLORS[ctrl], linewidth=2.5, label=LABEL_MAP[ctrl], alpha=0.8)

        if has_data:
            ax.set_xlabel('北向 (X) [米]', fontweight='bold', fontsize=12)
            ax.set_ylabel('东向 (Y) [米]', fontweight='bold', fontsize=12)
            ax.set_title(f'虚拟航点追踪轨迹表现 - {wind_label}', fontsize=16, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_aspect('equal', 'box')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'fig2_Mission_Trajectory_{wind_key}.png'), dpi=300)
        plt.close(fig)

def plot_all_force_tracking():
    print("  -> 正在绘制所有风况下的受力估算对比图 (共 5 张)...")
    
    # 遍历每一种风场，生成单独的受力估计图
    for wind_key, wind_label in WIND_CONDITIONS.items():
        fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'空气动力学残差 (风扰) 估算表现 - {wind_label}', fontsize=16, fontweight='bold', y=0.97)
        
        has_data = False
        for i, ctrl in enumerate(CONTROLLERS):
            df = load_data(ctrl, wind_key)
            if df is not None and not df.empty:
                # 选取 20s 到 40s 这个窗口来展示细节
                df_plot = df[(df['time'] >= 20.0) & (df['time'] <= 40.0)]
                if not df_plot.empty:
                    has_data = True
                    axs[i].plot(df_plot['time'], df_plot['f_true_x'], color='black', linestyle='--', linewidth=2, label='真实物理风扰 (X轴)', alpha=0.7)
                    axs[i].plot(df_plot['time'], df_plot['f_est_x'], color=COLORS[ctrl], linewidth=2.5, label=f'{LABEL_MAP[ctrl]} 算出的补偿力')
                    
                    rmse = np.sqrt(((df_plot['f_true_x'] - df_plot['f_est_x'])**2).mean())
                    axs[i].text(0.01, 0.85, f"受力估计 RMSE: {rmse:.2f} N", transform=axs[i].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            axs[i].set_ylabel('受力 [N]', fontweight='bold')
            axs[i].legend(loc='upper right')
            axs[i].grid(True, linestyle=':', alpha=0.6)
            
        if has_data:
            axs[-1].set_xlabel('时间 [s]', fontweight='bold', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(FIGURES_DIR, f'fig3_Mission_Force_Tracking_{wind_key}.png'), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    print("📈 开始生成评估图表矩阵...")
    plot_cross_track_rmse_bar()
    plot_all_trajectories()
    plot_all_force_tracking()
    print(f"✅ 图表已全部保存至 {FIGURES_DIR} 目录。")