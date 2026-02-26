#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 参数调优脚本: Step 2 - Basis Dimension
功能: 自动训练 [4, 8, 12, 16] 维度的模型，并画出 Loss 曲线寻找"肘部"拐点。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# ================= 配置 =================
FORCE_SCALE = 6.0       # 根据 Step 1 确定的最佳值
INPUT_DIM = 11
TEST_DIMS = [4, 8, 12, 16] # 要测试的维度列表
EPOCHS_PER_DIM = 100    # 每个维度训练的轮数 (由100轮足够看趋势)
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据集 & 模型定义 =================
# (为了保持脚本独立，我们将核心类直接包含在此)

class SimpleDataset(Dataset):
    def __init__(self, anchor_path, positive_paths):
        self.pairs = []
        feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
        label_cols = ['f_x', 'f_y', 'f_z']
        
        # 加载 Anchor
        df_anchor = pd.read_csv(anchor_path)
        
        for pos_path in positive_paths:
            df_pos = pd.read_csv(pos_path)
            min_len = min(len(df_anchor), len(df_pos))
            
            # 提取并归一化
            s_a = torch.FloatTensor(df_anchor.iloc[:min_len][feat_cols].values)
            f_a = torch.FloatTensor(df_anchor.iloc[:min_len][label_cols].values) / FORCE_SCALE
            s_p = torch.FloatTensor(df_pos.iloc[:min_len][feat_cols].values)
            f_p = torch.FloatTensor(df_pos.iloc[:min_len][label_cols].values) / FORCE_SCALE
            
            self.pairs.append({'sa': s_a, 'fa': f_a, 'sp': s_p, 'fp': f_p, 'len': min_len})
            
        self.total_len = sum(p['len'] for p in self.pairs)

    def __len__(self): return self.total_len

    def __getitem__(self, idx):
        curr = idx
        for p in self.pairs:
            if curr < p['len']:
                return p['sa'][curr], p['fa'][curr], p['sp'][curr], p['fp'][curr]
            curr -= p['len']
        raise IndexError

class DynamicPhiNet(nn.Module):
    def __init__(self, basis_dim):
        super(DynamicPhiNet, self).__init__()
        self.basis_dim = basis_dim
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, basis_dim)) # 动态维度
        )
    def forward(self, x): return self.net(x)

def batch_least_squares(phi, target, basis_dim):
    phi_t = phi.t()
    # 正则化系数 1e-4
    A = torch.matmul(phi_t, phi) + 1e-4 * torch.eye(basis_dim, device=DEVICE)
    B = torch.matmul(phi_t, target)
    try:
        return torch.linalg.solve(A, B)
    except:
        return torch.linalg.lstsq(A, B).solution

# ================= 训练核心 =================
def run_training(dim, loader):
    print(f"\n>>> Testing BASIS_DIM = {dim} ...")
    model = DynamicPhiNet(dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 保持和之前一致的 LR
    mse_fn = nn.MSELoss()
    triplet_fn = nn.TripletMarginLoss(margin=1.0)
    
    loss_history = []
    
    for epoch in range(EPOCHS_PER_DIM):
        epoch_loss = 0
        for sa, fa, sp, fp in loader:
            sa, fa, sp, fp = sa.to(DEVICE), fa.to(DEVICE), sp.to(DEVICE), fp.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Split
            split = sa.shape[0] // 2
            phi_a = model(sa); phi_p = model(sp)
            
            # Adaptation (Support)
            a_a = batch_least_squares(phi_a[:split], fa[:split], dim)
            a_p = batch_least_squares(phi_p[:split], fp[:split], dim)
            
            # Evaluation (Query)
            pred_a = phi_a[split:] @ a_a
            pred_p = phi_p[split:] @ a_p
            
            loss_task = mse_fn(pred_a, fa[split:]) + mse_fn(pred_p, fp[split:])
            
            # Triplet
            phi_n = torch.roll(phi_a[split:], 1, 0)
            loss_trip = triplet_fn(phi_a[split:], phi_p[split:], phi_n)
            
            loss = loss_task + 2.0 * loss_trip # Lambda=2.0
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        loss_history.append(avg_loss)
        
    return loss_history[-1] # 返回最终 Loss

# ================= 主程序 =================
if __name__ == "__main__":
    # 路径配置 (请确保路径正确)
    anchor_csv = "processed_data/processed_train_wind_00.csv"
    positive_csvs = [
        "processed_data/processed_train_wind_02.csv",
        "processed_data/processed_train_wind_04.csv",
        "processed_data/processed_train_wind_06.csv"
    ]
    
    if not os.path.exists(anchor_csv):
        print("错误: 找不到数据文件，请检查 processed_data 目录")
        exit()

    print("Loading Data...")
    dataset = SimpleDataset(anchor_csv, positive_csvs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    results = []
    for dim in TEST_DIMS:
        final_loss = run_training(dim, loader)
        results.append(final_loss)
        print(f"  -> Dim {dim} Final Loss: {final_loss:.4f}")
        
    # === 绘制肘部法则图 ===
    plt.figure(figsize=(8, 6))
    plt.plot(TEST_DIMS, results, 'o-', linewidth=2, markersize=8)
    plt.title(f"Elbow Method for Basis Dimension (Scale={FORCE_SCALE})")
    plt.xlabel("Basis Dimension")
    plt.ylabel("Final Training Loss")
    plt.grid(True)
    plt.xticks(TEST_DIMS)
    
    for x, y in zip(TEST_DIMS, results):
        plt.text(x, y + 0.001, f"{y:.4f}", ha='center')
        
    plt.savefig("basis_dim_selection.png")
    print("\n✅ 实验完成！结果已保存为 basis_dim_selection.png")
    plt.show()