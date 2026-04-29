#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
航点任务模式 (Mission Mode) 专属分图绘制脚本
功能：为每一个风况单独生成一张图片，避免轨迹互相遮挡，清晰展现 90s 长航时下的误差积累。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

RESULTS_DIR = "/home/zzx/testmodel/eval_results"
FIGURES_DIR = "/home/zzx/testmodel/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

WIND_CONDITIONS = {
    'online_mission_nowind': ('0 m/s (无风)', '#2ca02c'),       
    'online_mission_35wind': ('4.2 m/s (恒风)', '#1f77b4'),     
    'online_mission_70wind': ('8.5 m/s (强风)', '#ff7f0e'),     
    'online_mission_70p20sint': ('动态阵风(8.5±2.4)', '#9467bd'),
    'online_mission_100wind': ('12.1 m/s (极端风)', '#d62728')   
}

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_separate_mission_trajectories():
    # 绘制理论上的完美 4.0m 参考轨迹 (画两圈足够覆盖显示范围)
    theta_ref = np.linspace(0, 4 * np.pi, 500)
    ref_x = 4.0 * np.sin(theta_ref)
    ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

    # 循环为每一个风况单独生成一张图
    for wind_key, (wind_label, color) in WIND_CONDITIONS.items():
        file_path = os.path.join(RESULTS_DIR, f"eval_data_Mission_{wind_key}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # 剔除起降阶段，保留中间的 80 秒长航段
            df_steady = df[(df['time'] >= 5.0) & (df['time'] <= 85.0)]
            
            if not df_steady.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # 画参考线
                ax.plot(ref_x, ref_y, color='black', linestyle='--', linewidth=3, label='理想 8 字航线 (Reference)', zorder=10)
                
                # 画实际航点轨迹
                ax.plot(df_steady['p_x'], df_steady['p_y'], color=color, linewidth=2.0, label=f'实际航点飞行轨迹\n({wind_label})', alpha=0.85)
                
                # 计算交叉轨迹误差 (Cross-Track RMSE)
                pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
                ref_pts = np.vstack((ref_x, ref_y)).T
                distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
                rmse = np.sqrt(np.mean(distances**2))
                
                print(f"[{wind_label}] 生成完毕 -> Cross-Track RMSE: {rmse:.3f} m")

                # 图表装饰
                ax.set_xlabel('北向 (X) [米]', fontweight='bold', fontsize=12)
                ax.set_ylabel('东向 (Y) [米]', fontweight='bold', fontsize=12)
                ax.set_title(f'PX4 纯航点模式追踪能力评估 - {wind_label}\nCross-Track RMSE = {rmse:.3f} m', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=11)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_aspect('equal', 'box')
                
                # 单独保存每一张图片
                plt.tight_layout()
                out_path = os.path.join(FIGURES_DIR, f'fig_Mission_Baseline_{wind_key}.png')
                plt.savefig(out_path, dpi=300)
                plt.close(fig)

if __name__ == '__main__':
    print("📈 正在生成分图轨迹分析...")
    plot_separate_mission_trajectories()
    print(f"✅ 所有风况的图片已独立保存至 {FIGURES_DIR} 目录下。")