#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 训练过程基线对比图 (原版 DAIML vs 我们的 DTW-Triplet)
用途: 论证我们的方案在收敛速度和最终泛化性能上对原版算法的全面超越。
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# 屏蔽 Matplotlib 的字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 路径配置 =================
TRAIN_DIR = "/home/zzx/testmodel/training_results"
FIGURES_DIR = "/home/zzx/testmodel/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_training_curves():
    # 自动读取跑完消融实验后提取的 CSV
    csv_ours = os.path.join(TRAIN_DIR, "curve_ours.csv")
    csv_orig = os.path.join(TRAIN_DIR, "curve_original.csv")

    if not os.path.exists(csv_ours) or not os.path.exists(csv_orig):
        print(f"❌ 找不到训练数据 CSV。")
        print(f"请确保您已经先运行了: python3 run_ablations.py")
        return

    print("📊 正在生成: 训练收敛基线对比图 (fig0)...")

    # ================= 2. 数据清洗 =================
    # CSVLogger 会在 step 和 epoch 记录数据。验证集误差 (val_mse) 只有在 epoch 结束时才有
    # 所以我们通过 dropna 去掉那些只有 step 记录的空行，提取纯净的 epoch 曲线
    df_ours = pd.read_csv(csv_ours).dropna(subset=['val_mse'])
    df_orig = pd.read_csv(csv_orig).dropna(subset=['val_mse'])

    # ================= 3. 画图与中文字体设置 =================
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制原版 Neural-Fly (蓝色虚线)
    ax.plot(df_orig['epoch'], df_orig['val_mse'], 
            color='#1f77b4', linestyle='--', linewidth=2.5, 
            label='原版 Neural-Fly (DAIML 对抗博弈架构)')
    
    # 绘制我们的方案 (红色实线)
    ax.plot(df_ours['epoch'], df_ours['val_mse'], 
            color='#d62728', linestyle='-', linewidth=2.5, 
            label='我们的改进方案 (DTW-Triplet 特征流形对齐)')

    # ================= 4. 图表装饰 =================
    ax.set_title('验证集抗风追踪误差 (Validation MSE) 下降曲线对比', fontsize=16, fontweight='bold')
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='bold')
    ax.set_ylabel('验证集轨迹追踪均方误差 (MSE)', fontsize=13, fontweight='bold')
    
    # 【核心技巧】使用对数坐标系！这样能极其明显地放大后期收敛阶段的性能差距
    ax.set_yscale('log') 
    
    # 增加细密的网格线，提升学术图表的质感
    ax.grid(True, which="major", ls="-", alpha=0.6, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.4, color='gray')
    
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')

    # ================= 5. 保存输出 =================
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig0_Training_Convergence.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练收敛对比图已生成并保存至: {out_path}")

if __name__ == "__main__":
    plot_training_curves()