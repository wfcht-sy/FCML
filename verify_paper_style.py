#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ================= 配置 =================
device = torch.device("cpu")

# ================= RLS 算法 (模拟在线适应) =================
class OnlineRLS:
    def __init__(self, n_features, lambda_factor=0.98, p_init=10.0):
        self.theta = np.zeros((n_features, 3)) # [Basis, 3]
        self.P = np.eye(n_features) * p_init
        self.lambda_factor = lambda_factor
        
    def update(self, phi, y):
        # phi: [1, Basis], y: [1, 3]
        # P_new = (P - K * phi * P) / lambda
        # theta_new = theta + K * error
        
        P_phi = np.dot(self.P, phi.T) # [Basis, 1]
        phi_P = np.dot(phi, self.P)   # [1, Basis]
        denom = self.lambda_factor + np.dot(phi_P, phi.T)
        K = P_phi / denom # [Basis, 1]
        
        y_pred = np.dot(phi, self.theta)
        error = y - y_pred
        
        self.theta += np.dot(K, error)
        self.P = (self.P - np.dot(K, phi_P)) / self.lambda_factor
        
        return y_pred.flatten(), self.theta.copy()

# ================= 模型定义 =================
class PhiNetwork(nn.Module):
    def __init__(self, basis_dim):
        super(PhiNetwork, self).__init__()
        # 必须与训练时的结构完全一致 (11 -> 64 -> 64 -> basis_dim)
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(11, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, basis_dim))
        )
        
    def forward(self, x):
        return self.net(x)

def load_data(path):
    print(f"Loading data: {path}")
    df = pd.read_csv(path)
    
    feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
    label_cols = ['f_x', 'f_y', 'f_z']
    
    # 归一化 PWM
    for i in range(1, 5):
        col = f'pwm_{i}'
        if col in df.columns and df[col].mean() > 100: 
            df[col] = (df[col] - 1500.0) / 500.0
            
    x = torch.FloatTensor(df[feat_cols].values).to(device)
    y = torch.FloatTensor(df[label_cols].values).to(device)
    t = df['timestamp'].values
    return x, y, t

def verify_paper_style(args):
    # 1. 加载模型
    if not os.path.exists(args.model_path):
        print("Model not found!")
        return
        
    checkpoint = torch.load(args.model_path, map_location=device)
    basis_dim = checkpoint.get('basis_dim', 8)
    force_scale = checkpoint.get('force_scale', 5.0)
    
    print(f"Config: Basis={basis_dim}, Scale={force_scale}")
    
    model = PhiNetwork(basis_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. 加载测试数据 (8.5m/s 或 12m/s)
    x_test, y_test_raw, t = load_data(args.test_csv)
    
    # 3. 提取特征 (Phi)
    with torch.no_grad():
        phi = model(x_test).numpy() # [N, Basis]
    
    # 真实力 (Ground Truth)
    f_gt = y_test_raw.numpy() # [N, 3] (未归一化，真实的牛顿)
    
    # 归一化的目标力 (用于 RLS 输入)
    f_target_norm = f_gt / force_scale
    
    # ================= 验证 A: 全局最优适应 (离线极限性能) =================
    # 计算 Global Least Squares a*
    # (Phi^T Phi + reg I)^-1 Phi^T Y
    reg = 1e-4
    A = np.dot(phi.T, phi) + reg * np.eye(basis_dim)
    B = np.dot(phi.T, f_target_norm)
    a_star = np.linalg.solve(A, B)
    
    f_pred_global = np.dot(phi, a_star) * force_scale
    
    # 计算全局 RMSE
    rmse_global = np.sqrt(np.mean((f_gt - f_pred_global)**2, axis=0))
    print(f"\n[Global Adaptation] RMSE: X={rmse_global[0]:.3f}, Y={rmse_global[1]:.3f}, Z={rmse_global[2]:.3f}")

    # ================= 验证 B: 模拟在线 RLS (真实飞行性能) =================
    rls = OnlineRLS(n_features=basis_dim, lambda_factor=0.98, p_init=10.0)
    
    f_pred_rls = []
    a_history = []
    
    print("Simulating Online Adaptation...")
    for i in range(len(f_gt)):
        # 模拟每一步: 输入当前 phi 和 真实力，更新 a，输出预测力
        curr_phi = phi[i:i+1] # [1, Basis]
        curr_y = f_target_norm[i:i+1] # [1, 3]
        
        # RLS Update
        # 注意：这里我们用"后验"误差来画图，或者用"先验"误差
        # 在控制中，我们用上一时刻的 a 预测当前，然后更新 a
        # 这里为了展示收敛后的效果，我们记录更新后的预测
        pred_norm, theta_updated = rls.update(curr_phi, curr_y)
        
        f_pred_rls.append(pred_norm * force_scale)
        a_history.append(theta_updated.flatten())
        
    f_pred_rls = np.array(f_pred_rls)
    a_history = np.array(a_history) # [N, Basis*3]

    # ================= 绘图 (Paper Style) =================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. 侧向力时域对比 (Time Domain Force Y)
    ax = axes[0, 0]
    ax.plot(t, f_gt[:, 1], 'k-', alpha=0.5, label='Ground Truth (Wind Disturbance)')
    ax.plot(t, f_pred_global[:, 1], 'r--', linewidth=1.5, label='Global Adaptation (Best)')
    ax.plot(t, f_pred_rls[:, 1], 'b:', linewidth=1.5, label='Online RLS (Simulated)')
    ax.set_title(f'Lateral Force Prediction (8.5m/s Wind)\nGlobal RMSE={rmse_global[1]:.3f} N')
    ax.set_ylabel('Force Y (N)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 误差分布直方图 (Error Histogram)
    ax = axes[0, 1]
    error_base = f_gt[:, 1] # 无控制时的误差就是风力本身
    error_nf = f_gt[:, 1] - f_pred_global[:, 1] # 控制后的残差
    
    ax.hist(error_base, bins=50, alpha=0.5, color='gray', label='No Adaptation (Raw Wind)')
    ax.hist(error_nf, bins=50, alpha=0.7, color='green', label='Neural-Fly Residual')
    ax.set_title('Error Distribution (Lateral)')
    ax.set_xlabel('Force Error (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 适应系数收敛过程 (Adaptation Coefficients)
    ax = axes[1, 0]
    # 只画前 4 个系数的变化
    for i in range(4):
        ax.plot(t, a_history[:, i], label=f'a_{i}')
    ax.set_title('Adaptation Coefficients Convergence (First 4)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coefficient Value')
    ax.grid(True)
    
    # 4. 纵向力对比 (Force X)
    ax = axes[1, 1]
    ax.plot(t, f_gt[:, 0], 'k-', alpha=0.5, label='GT')
    ax.plot(t, f_pred_global[:, 0], 'r--', label='Pred')
    ax.set_title(f'Longitudinal Force X (RMSE={rmse_global[0]:.3f} N)')
    ax.grid(True)
    
    # 5. RLS 瞬时误差收敛 (Online Error)
    ax = axes[2, 0]
    err_rls = np.abs(f_gt[:, 1] - f_pred_rls[:, 1])
    # 移动平均平滑一下以便观察趋势
    err_smooth = pd.Series(err_rls).rolling(window=50).mean()
    ax.plot(t, err_smooth, 'b-')
    ax.set_title('Online Adaptation Error (Moving Avg)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Abs Error Y (N)')
    ax.grid(True)
    
    # 6. 垂向力对比 (Force Z)
    ax = axes[2, 1]
    ax.plot(t, f_gt[:, 2], 'k-', alpha=0.5, label='GT')
    ax.plot(t, f_pred_global[:, 2], 'r--', label='Pred')
    ax.set_title(f'Vertical Force Z (RMSE={rmse_global[2]:.3f} N)')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('paper_verification.png')
    print("\n[Result] Verification plot saved to 'paper_verification.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="processed_data/processed_test_wind_08.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/neural_fly_model_meta.pth")
    args = parser.parse_args()
    verify_paper_style(args)