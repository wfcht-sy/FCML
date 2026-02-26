#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 离线训练脚本 V3 (Meta-Learning Split + Evolution Logging)

功能:
1. 加载平行宇宙数据 (Anchor + Positive)
2. 实施 Split-Batch 训练策略 (Support Set -> a*, Query Set -> Loss)
3. 优化参数配置 (Basis=8, Scale=5.0)
4. 定期保存 Checkpoint 用于 t-SNE 进化分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import time

# ================= 核心配置 (Optimized) =================
BASIS_DIM = 8        # 增加特征维度，提升拟合能力
INPUT_DIM = 11       # v(3) + q(4) + pwm(4)
FORCE_SCALE = 6.0    # 力归一化因子 (关键! 防止 a* 过大) //均值 + 2倍标准差
REG_LAMBDA = 1e-4    # 降低正则化，减少欠拟合

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 数据集定义 =================
class MultiConditionDataset(Dataset):
    def __init__(self, anchor_path, positive_paths):
        print(f"[Dataset] Loading Anchor: {os.path.basename(anchor_path)}")
        self.df_anchor = pd.read_csv(anchor_path)
        self.pairs = [] 
        
        self.feat_cols = [
            'v_x', 'v_y', 'v_z', 
            'q_w', 'q_x', 'q_y', 'q_z', 
            'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4'
        ]
        self.label_cols = ['f_x', 'f_y', 'f_z']
        
        # 预处理函数
        def preprocess(df):
            for i in range(1, 5):
                col = f'pwm_{i}'
                if col in df.columns and df[col].mean() > 100: 
                    df[col] = (df[col] - 1500.0) / 500.0
            return df

        self.df_anchor = preprocess(self.df_anchor)
        
        for pos_path in positive_paths:
            print(f"[Dataset] Pairing with: {os.path.basename(pos_path)}")
            df_pos = pd.read_csv(pos_path)
            df_pos = preprocess(df_pos)
            
            # 长度对齐
            min_len = min(len(self.df_anchor), len(df_pos))
            
            state_a = torch.FloatTensor(self.df_anchor.iloc[:min_len][self.feat_cols].values)
            # [关键] 力归一化
            force_a = torch.FloatTensor(self.df_anchor.iloc[:min_len][self.label_cols].values) / FORCE_SCALE
            
            state_p = torch.FloatTensor(df_pos.iloc[:min_len][self.feat_cols].values)
            force_p = torch.FloatTensor(df_pos.iloc[:min_len][self.label_cols].values) / FORCE_SCALE
            
            self.pairs.append({
                'state_a': state_a, 'force_a': force_a,
                'state_p': state_p, 'force_p': force_p,
                'length': min_len
            })
            
        self.total_samples = sum(p['length'] for p in self.pairs)
        print(f"[Dataset] Total paired samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        current_idx = idx
        for pair in self.pairs:
            if current_idx < pair['length']:
                return pair['state_a'][current_idx], pair['force_a'][current_idx], \
                       pair['state_p'][current_idx], pair['force_p'][current_idx]
            current_idx -= pair['length']
        raise IndexError("Index out of range")

# ================= 2. 神经网络模型 =================
class PhiNetwork(nn.Module):
    def __init__(self):
        super(PhiNetwork, self).__init__()
        # 使用 Spectral Norm 保证 Lipschitz 连续性
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, BASIS_DIM))
        )
        
    def forward(self, x):
        return self.net(x)

# ================= 3. 可微分最小二乘 (DLS) =================
def batch_least_squares(phi, target, reg_lambda=REG_LAMBDA):
    """
    求解 a* = argmin || y - phi * a ||^2
    """
    phi_t = phi.t() 
    A = torch.matmul(phi_t, phi) + reg_lambda * torch.eye(BASIS_DIM, device=device) 
    B = torch.matmul(phi_t, target) 
    try:
        a_star = torch.linalg.solve(A, B)
    except:
        a_star = torch.linalg.lstsq(A, B).solution
    return a_star

# ================= 4. 训练流程 (Split-Batch Strategy) =================
def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 加载数据
    dataset = MultiConditionDataset(args.anchor_csv, args.positive_csvs)
    # [建议] Batch Size 设大一点 (256)，保证切分后 Support Set 依然有 128 个样本，足够计算稳定的 a*
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = PhiNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    mse_loss_fn = nn.MSELoss()
    
    print(f"Start Meta-Learning Training (Basis={BASIS_DIM}, Scale={FORCE_SCALE})...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        task_loss_acc = 0
        
        for state_a, force_a, state_p, force_p in loader:
            state_a, force_a = state_a.to(device), force_a.to(device)
            state_p, force_p = state_p.to(device), force_p.to(device)
            
            optimizer.zero_grad()
            
            # --- 1. 前向传播 (Full Batch Features) ---
            phi_a = model(state_a)
            phi_p = model(state_p)
            
            # --- 2. 切分 Support / Query Set ---
            split = args.batch_size // 2
            
            # Support Set (用于适应 a*)
            phi_a_supp, force_a_supp = phi_a[:split], force_a[:split]
            phi_p_supp, force_p_supp = phi_p[:split], force_p[:split]
            
            # Query Set (用于评估 Loss)
            phi_a_qry, force_a_qry = phi_a[split:], force_a[split:]
            phi_p_qry, force_p_qry = phi_p[split:], force_p[split:]
            
            # --- 3. 内层循环: 适应 (Adaptation) ---
            # 仅利用 Support Set 计算环境系数
            a_star_a = batch_least_squares(phi_a_supp, force_a_supp)
            a_star_p = batch_least_squares(phi_p_supp, force_p_supp)
            
            # --- 4. 外层循环: 评估 (Evaluation) ---
            # 利用 Support 算出的系数，预测 Query Set 的力
            pred_f_a_qry = torch.matmul(phi_a_qry, a_star_a)
            pred_f_p_qry = torch.matmul(phi_p_qry, a_star_p)
            
            # Task Loss (只在 Query Set 上计算)
            loss_task = mse_loss_fn(pred_f_a_qry, force_a_qry) + \
                        mse_loss_fn(pred_f_p_qry, force_p_qry)
            
            # Triplet Loss (只在 Query Set 上计算)
            # 负样本通过滚动生成
            phi_n_qry = torch.roll(phi_a_qry, shifts=1, dims=0)
            loss_triplet = triplet_loss_fn(phi_a_qry, phi_p_qry, phi_n_qry)
            
            # 总损失
            loss = loss_task + args.lambda_triplet * loss_triplet
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            task_loss_acc += loss_task.item()
        
        # 打印日志
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f} (Task: {task_loss_acc/len(loader):.4f})")
            
        # [核心] 定期保存模型，用于 t-SNE 进化分析 (如 Epoch 20, 40, 60...)
        if (epoch+1) % 20 == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'basis_dim': BASIS_DIM,
                'force_scale': FORCE_SCALE
            }, ckpt_path)
            
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "neural_fly_model_meta.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'basis_dim': BASIS_DIM,
        'force_scale': FORCE_SCALE
    }, final_path)
    
    print(f"Training finished. Final model saved to: {final_path}")
    print(f"Intermediate checkpoints saved in {args.output_dir} for t-SNE analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_csv", type=str, default="processed_data/processed_train_wind_00.csv")
    parser.add_argument("--positive_csvs", nargs='+', default=[
        "processed_data/processed_train_wind_02.csv",
        "processed_data/processed_train_wind_04.csv",
        "processed_data/processed_train_wind_06.csv"
    ])
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=160) # 稍微增加轮数以观察进化
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_triplet", type=float, default=0.5)
    
    args = parser.parse_args()
    train(args)