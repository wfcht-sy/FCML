#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 离线训练脚本 V8 (Final Production)
功能: 
1. 运行 200 轮以获取完整收敛曲线
2. 自动锁定并保存第 160 轮为最终模型 (final_model_160.pth)
3. 复合评分策略 (Score = Val_MSE + Val_Triplet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import sys
import csv 

# ================= 核心配置 =================
BASIS_DIM = 8        # [Test.py 确定的最佳值]
INPUT_DIM = 11       # v(3) + q(4) + pwm(4)
FORCE_SCALE = 6.0    # 力归一化因子
REG_LAMBDA = 1e-4    # 正则化参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 数据集定义 =================
class MultiConditionDataset(Dataset):
    def __init__(self, anchor_path, positive_paths):
        print(f"[Dataset] Loading Anchor: {os.path.basename(anchor_path)}")
        self.df_anchor = pd.read_csv(anchor_path)
        self.pairs = [] 
        
        self.feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
        self.label_cols = ['f_x', 'f_y', 'f_z']
        
        # 预处理: PWM 归一化
        def preprocess(df):
            for i in range(1, 5):
                col = f'pwm_{i}'
                if col in df.columns and df[col].mean() > 100: 
                    df[col] = (df[col] - 1500.0) / 500.0
            return df

        self.df_anchor = preprocess(self.df_anchor)
        anchor_vx = self.df_anchor['v_x'].values
        
        for idx, pos_path in enumerate(positive_paths):
            print(f"[Dataset] Pairing with: {os.path.basename(pos_path)}")
            df_pos = pd.read_csv(pos_path)
            df_pos = preprocess(df_pos)
            
            # --- 数据对齐自检 ---
            min_check = min(len(self.df_anchor), len(df_pos), 500)
            if min_check > 10:
                v_a_check = anchor_vx[:min_check]
                v_p_check = df_pos['v_x'].values[:min_check]
                corr = np.corrcoef(v_a_check, v_p_check)[0, 1]
                diff = np.mean(np.abs(v_a_check - v_p_check))
                
                print(f"  -> Alignment Check: Correlation={corr:.4f}, Mean Diff={diff:.4f}")
                if corr < 0.8: 
                    print(f"\033[91m  [WARNING] 严重警告! 数据未对齐 (Corr={corr:.4f})。\033[0m")

            min_len = min(len(self.df_anchor), len(df_pos))
            state_a = torch.FloatTensor(self.df_anchor.iloc[:min_len][self.feat_cols].values)
            force_a = torch.FloatTensor(self.df_anchor.iloc[:min_len][self.label_cols].values) / FORCE_SCALE
            state_p = torch.FloatTensor(df_pos.iloc[:min_len][self.feat_cols].values)
            force_p = torch.FloatTensor(df_pos.iloc[:min_len][self.label_cols].values) / FORCE_SCALE
            
            self.pairs.append({
                'state_a': state_a, 'force_a': force_a,
                'state_p': state_p, 'force_p': force_p,
                'length': min_len,
                'label_idx': idx + 1 
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
                       pair['state_p'][current_idx], pair['force_p'][current_idx], \
                       pair['label_idx'] 
            current_idx -= pair['length']
        raise IndexError("Index out of range")

# ================= 2. 网络结构 =================
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

# ================= 3. 工具函数 =================
def batch_least_squares(phi, target, reg_lambda=REG_LAMBDA):
    phi_t = phi.t() 
    A = torch.matmul(phi_t, phi) + reg_lambda * torch.eye(BASIS_DIM, device=device)
    B = torch.matmul(phi_t, target)
    try:
        return torch.linalg.solve(A, B)
    except:
        return torch.linalg.lstsq(A, B).solution

def validate_model(model, loader, triplet_fn, mse_fn):
    """ 计算复合分数 = MSE + Triplet """
    model.eval()
    total_trip = 0
    total_mse = 0
    
    with torch.no_grad():
        for state_a, force_a, state_p, force_p, _ in loader:
            state_a, force_a = state_a.to(device), force_a.to(device)
            state_p, force_p = state_p.to(device), force_p.to(device)
            
            phi_a = model(state_a)
            phi_p = model(state_p)
            
            # 验证集 Triplet
            phi_n = torch.roll(phi_a, shifts=1, dims=0)
            trip_loss = triplet_fn(phi_a, phi_p, phi_n)
            
            # 验证集 MSE (简单回代)
            a_star_a = batch_least_squares(phi_a, force_a)
            a_star_p = batch_least_squares(phi_p, force_p)
            pred_a = torch.matmul(phi_a, a_star_a)
            pred_p = torch.matmul(phi_p, a_star_p)
            mse_loss = mse_fn(pred_a, force_a) + mse_fn(pred_p, force_p)
            
            total_trip += trip_loss.item()
            total_mse += mse_loss.item()
            
    avg_trip = total_trip / len(loader)
    avg_mse = total_mse / len(loader)
    return avg_mse + avg_trip, avg_mse, avg_trip

# ================= 4. 训练主流程 =================
def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # CSV 日志
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    log_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Epoch', 'Train_Total', 'Train_Task', 'Train_Trip', 'Val_MSE', 'Val_Trip', 'Val_Composite_Score'])
    print(f"Logging metrics to: {csv_path}")

    # 加载数据
    dataset = MultiConditionDataset(args.anchor_csv, args.positive_csvs)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    model = PhiNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2) 
    mse_loss_fn = nn.MSELoss()
    
    print(f"Start Training (Epochs={args.epochs}, Basis={BASIS_DIM})...")
    
    best_score = float('inf') 
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        t_total, t_task, t_trip = 0, 0, 0
        
        for state_a, force_a, state_p, force_p, _ in train_loader:
            state_a, force_a = state_a.to(device), force_a.to(device)
            state_p, force_p = state_p.to(device), force_p.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            phi_a = model(state_a)
            phi_p = model(state_p)
            
            # Split
            split = args.batch_size // 2
            phi_a_sup, force_a_sup = phi_a[:split], force_a[:split]
            phi_p_sup, force_p_sup = phi_p[:split], force_p[:split]
            phi_a_qry, force_a_qry = phi_a[split:], force_a[split:]
            phi_p_qry, force_p_qry = phi_p[split:], force_p[split:]
            
            # Meta-Learning Step
            a_star_a = batch_least_squares(phi_a_sup, force_a_sup)
            a_star_p = batch_least_squares(phi_p_sup, force_p_sup)
            
            pred_a = torch.matmul(phi_a_qry, a_star_a)
            pred_p = torch.matmul(phi_p_qry, a_star_p)
            
            loss_task = mse_loss_fn(pred_a, force_a_qry) + mse_loss_fn(pred_p, force_p_qry)
            
            phi_n = torch.roll(phi_a_qry, shifts=1, dims=0)
            loss_triplet = triplet_loss_fn(phi_a_qry, phi_p_qry, phi_n)
            
            loss = loss_task + args.lambda_triplet * loss_triplet
            loss.backward()
            optimizer.step()
            
            t_total += loss.item()
            t_task += loss_task.item()
            t_trip += loss_triplet.item()
            
        # --- 评估 ---
        avg_train_total = t_total / len(train_loader)
        avg_train_task = t_task / len(train_loader)
        avg_train_trip = t_trip / len(train_loader)
        val_comp_score, val_mse, val_trip = validate_model(model, val_loader, triplet_loss_fn, mse_loss_fn)
        
        # 记录
        csv_writer.writerow([epoch+1, f"{avg_train_total:.6f}", f"{avg_train_task:.6f}", f"{avg_train_trip:.6f}", f"{val_mse:.6f}", f"{val_trip:.6f}", f"{val_comp_score:.6f}"])
        log_file.flush() 
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Task: {avg_train_task:.4f} | Val Composite: {val_comp_score:.4f}")
        
        # 保存 Score 最好的 
        if val_comp_score < best_score:
            best_score = val_comp_score
            best_epoch = epoch + 1
            save_path = os.path.join(args.output_dir, "best_clustering_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'basis_dim': BASIS_DIM,
                'force_scale': FORCE_SCALE,
                'epoch': best_epoch,
                'score': best_score
            }, save_path)

        # --- 保存第 160 轮为最终指定模型 ---
        if epoch + 1 == 160:
            final_path = os.path.join(args.output_dir, "final_model_160.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'basis_dim': BASIS_DIM,
                'force_scale': FORCE_SCALE,
                'epoch': 160,
                'score': val_comp_score
            }, final_path)
            print(f"  >>> [Auto-Save] Epoch 160 已锁定并保存为: {final_path}")

        # 定期备份
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({'model_state_dict': model.state_dict(), 'basis_dim': BASIS_DIM, 'force_scale': FORCE_SCALE}, ckpt_path)
            
    log_file.close()
    print(f"\nTraining Finished.")
    print(f"Best Composite Score was at Epoch {best_epoch}. ")
    print(f"Target Model (Epoch 160) saved at: {os.path.join(args.output_dir, 'final_model_160.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_csv", type=str, default="processed_data/processed_train_wind_00.csv")
    parser.add_argument("--positive_csvs", nargs='+', default=[
        "processed_data/processed_train_wind_02.csv",
        "processed_data/processed_train_wind_04.csv",
        "processed_data/processed_train_wind_06.csv"
    ])
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    # [修改] 保持 200 轮
    parser.add_argument("--epochs", type=int, default=200) 
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00005) 
    parser.add_argument("--lambda_triplet", type=float, default=2.0)
    
    args = parser.parse_args()
    train(args)