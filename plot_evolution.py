#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 特征演化可视化 (Evolution Plot)
功能: 自动读取不同 Epoch 的模型权重，绘制 t-SNE 演化过程，
证明模型是"学到了"特征解耦，而不是"碰巧"做到的。
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys
import glob
import re

# ================= 配置 =================
BASIS_DIM = 8
INPUT_DIM = 11
SAMPLE_LIMIT = 1000 # 每个风况采样的点数

# 要展示的关键 Epoch 节点 (根据您保存的文件名调整)
# 脚本会自动寻找最接近的文件
TARGET_EPOCHS = [20, 60, 120, 200] 

# ================= 模型定义 =================
class PhiNetwork(nn.Module):
    def __init__(self):
        super(PhiNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, BASIS_DIM))
        )
    def forward(self, x): return self.net(x)

def load_data(anchor_path, target_path):
    print("Loading data...")
    df_a = pd.read_csv(anchor_path)
    df_t = pd.read_csv(target_path)
    
    # 均匀采样
    idx_a = np.linspace(0, len(df_a)-1, SAMPLE_LIMIT, dtype=int)
    idx_t = np.linspace(0, len(df_t)-1, SAMPLE_LIMIT, dtype=int)
    
    cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
    
    tensor_a = torch.FloatTensor(df_a[cols].iloc[idx_a].values)
    tensor_t = torch.FloatTensor(df_t[cols].iloc[idx_t].values)
    
    return tensor_a, tensor_t

def find_checkpoint(epoch, checkpoint_dir="checkpoints"):
    # 寻找 model_epoch_X.pth
    pattern = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    if os.path.exists(pattern):
        return pattern
    # 如果找不到精确的，找最近的 (简单的容错)
    files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not files: return None
    
    # 解析所有文件的 epoch
    epochs = []
    for f in files:
        match = re.search(r'epoch_(\d+)', f)
        if match: epochs.append((int(match.group(1)), f))
    
    # 找差值最小的
    best_f = min(epochs, key=lambda x: abs(x[0] - epoch))
    print(f"  -> Requested Epoch {epoch}, using {best_f[1]} (Epoch {best_f[0]})")
    return best_f[1]

def main():
    anchor_csv = "processed_data/processed_train_wind_00.csv"
    target_csv = "processed_data/processed_train_wind_06.csv"
    
    if not os.path.exists(anchor_csv):
        print("Data not found.")
        return

    data_a, data_t = load_data(anchor_csv, target_csv)
    
    # 准备画布
    fig, axes = plt.subplots(1, len(TARGET_EPOCHS), figsize=(4 * len(TARGET_EPOCHS), 4), dpi=150)
    if len(TARGET_EPOCHS) == 1: axes = [axes]
    
    model = PhiNetwork()
    
    print(f"Generating evolution plot for epochs: {TARGET_EPOCHS}")
    
    for i, epoch in enumerate(TARGET_EPOCHS):
        ckpt_path = find_checkpoint(epoch)
        ax = axes[i]
        
        if ckpt_path is None:
            ax.text(0.5, 0.5, "Checkpoint Not Found", ha='center')
            continue
            
        # 加载模型
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 推理
        with torch.no_grad():
            phi_a = model(data_a).numpy()
            phi_t = model(data_t).numpy()
            
        # t-SNE
        X = np.vstack([phi_a, phi_t])
        # 使用 PCA 初始化以保持不同 Epoch 间的结构相似性，方便对比
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        X_emb = tsne.fit_transform(X)
        
        # 绘图
        n = len(phi_a)
        # 无风 (蓝)
        ax.scatter(X_emb[:n, 0], X_emb[:n, 1], c='royalblue', alpha=0.3, s=5, label='0 m/s')
        # 强风 (红)
        ax.scatter(X_emb[n:, 0], X_emb[n:, 1], c='crimson', alpha=0.3, s=5, label='6 m/s')
        
        ax.set_title(f"Epoch {epoch}", fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        # 给个简单的边框
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            
        # 只在第一个图显示图例
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle("Evolution of Feature Space (Wind Invariance)", fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = "evolution_plot.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Evolution plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()