#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ================= 动态配置 (将由模型文件覆盖) =================
INPUT_DIM = 11
BASIS_DIM = 8        # 默认值，会被 checkpoint 覆盖
FORCE_SCALE = 1.0    # 默认值，会被 checkpoint 覆盖

device = torch.device("cpu") # 验证通常用 CPU 即可

# ================= 模型定义 (必须与训练脚本完全一致) =================
# 注意：为了代码独立运行，这里重新定义一遍 PhiNetwork
# 实际工程中建议将网络定义单独放在 model.py 中 import
class PhiNetwork(nn.Module):
    def __init__(self, basis_dim): # [修改] 接收动态 basis_dim
        super(PhiNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            # 使用传入的维度
            nn.utils.spectral_norm(nn.Linear(64, basis_dim))
        )
        
    def forward(self, x):
        return self.net(x)

def load_data(path):
    print(f"Loading test data: {path}")
    df = pd.read_csv(path)
    
    feat_cols = [
        'v_x', 'v_y', 'v_z', 
        'q_w', 'q_x', 'q_y', 'q_z', 
        'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4'
    ]
    label_cols = ['f_x', 'f_y', 'f_z']
    
    # 同样的预处理 (PWM 归一化)
    for i in range(1, 5):
        col = f'pwm_{i}'
        if col in df.columns and df[col].mean() > 100: 
            df[col] = (df[col] - 1500.0) / 500.0
            
    x = torch.FloatTensor(df[feat_cols].values).to(device)
    y = torch.FloatTensor(df[label_cols].values).to(device)
    timestamp = df['timestamp'].values
    return x, y, timestamp

def verify(args):
    # 1. 加载模型与配置
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    try:
        # 加载 checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # [核心] 自动读取训练配置
        loaded_basis_dim = checkpoint.get('basis_dim', 4)
        loaded_force_scale = checkpoint.get('force_scale', 1.0)
        
        print(f"Model Configuration Loaded:")
        print(f"  - Basis Dim: {loaded_basis_dim}")
        print(f"  - Force Scale: {loaded_force_scale}")
        
        # 初始化网络
        model = PhiNetwork(basis_dim=loaded_basis_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model weights loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: 确保使用的是 train_offline_all_v2.py 训练出的模型")
        return
    
    # 2. 加载测试数据
    if not os.path.exists(args.test_csv):
        print(f"Error: Test data not found at {args.test_csv}")
        return
        
    x_test, y_test, t = load_data(args.test_csv)
    
    # 3. 推理特征 Phi
    with torch.no_grad():
        phi = model(x_test) # [N, Basis]
    
    # 4. 模拟适应过程 (Global Least Squares)
    # 注意：这里的 y_test 是真实物理值（未归一化），
    # 而 phi 是基于归一化训练出来的。
    # 为了求解正确的 a*，我们需要让两边量纲一致。
    # 方法A: 将 y_test 归一化后求 a* (得到归一化下的 a*)
    # 方法B: 直接求 (Phi * Scale) * a = y_test
    
    # 这里采用方法 A，因为数值稳定性更好
    y_test_scaled = y_test / loaded_force_scale
    
    reg = 1e-4 # 与训练保持一致
    phi_t = phi.t()
    A = torch.matmul(phi_t, phi) + reg * torch.eye(loaded_basis_dim).to(device)
    B = torch.matmul(phi_t, y_test_scaled)
    
    try:
        a_star = torch.linalg.solve(A, B)
    except:
        a_star = torch.linalg.lstsq(A, B).solution
        
    print("\nEstimated Wind Coefficients (a*) [Normalized Domain]:")
    print(a_star.numpy())
    
    # 5. 预测力 (反归一化)
    # y_pred = (phi * a*) * scale
    y_pred_scaled = torch.matmul(phi, a_star)
    y_pred = y_pred_scaled * loaded_force_scale
    
    # 6. 计算误差 metrics
    # 对比的是真实的物理力 (Newtons)
    mse = torch.mean((y_pred - y_test)**2, dim=0)
    rmse = torch.sqrt(mse)
    
    print("\n=== Validation Results (Newtons) ===")
    print(f"RMSE Force X: {rmse[0]:.4f} N")
    print(f"RMSE Force Y: {rmse[1]:.4f} N  <-- 侧风预测精度")
    print(f"RMSE Force Z: {rmse[2]:.4f} N")
    
    # 7. 绘图
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    
    # F_x
    plt.subplot(3, 1, 1)
    plt.plot(t, y_test_np[:, 0], 'k-', label='Ground Truth', alpha=0.6, linewidth=1.5)
    plt.plot(t, y_pred_np[:, 0], 'r--', label='Neural-Fly Pred', linewidth=2.0)
    plt.title(f'Force X Prediction (RMSE={rmse[0]:.3f} N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F_y (侧向力 - 最重要)
    plt.subplot(3, 1, 2)
    plt.plot(t, y_test_np[:, 1], 'k-', label='Ground Truth', alpha=0.6, linewidth=1.5)
    plt.plot(t, y_pred_np[:, 1], 'b--', label='Neural-Fly Pred', linewidth=2.0)
    plt.title(f'Force Y (Side Force) Prediction (RMSE={rmse[1]:.3f} N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F_z
    plt.subplot(3, 1, 3)
    plt.plot(t, y_test_np[:, 2], 'k-', label='Ground Truth', alpha=0.6, linewidth=1.5)
    plt.plot(t, y_pred_np[:, 2], 'g--', label='Neural-Fly Pred', linewidth=2.0)
    plt.title(f'Force Z Prediction (RMSE={rmse[2]:.3f} N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_file = 'verification_result_optimized.png'
    plt.savefig(save_file)
    print(f"\nResult plot saved to '{save_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认使用 8.5m/s 测试集
    parser.add_argument("--test_csv", type=str, default="processed_data/processed_test_wind_08.csv")
    # 默认使用 meta 模型
    parser.add_argument("--model_path", type=str, default="checkpoints/neural_fly_model_meta.pth")
    
    args = parser.parse_args()
    verify(args)