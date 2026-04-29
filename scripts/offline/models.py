#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心特征提取网络库 (双架构统一管理)
分离出单独文件，确保训练端与在线部署端的结构绝对对齐！
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== [架构 1]: 原版 Neural-Fly 网络 (保持原名，兼容 DAIML 训练) ==================
class PhiNetwork(nn.Module):
    def __init__(self, input_dim=11, basis_dim=8):
        super(PhiNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, basis_dim))
        )
    def forward(self, x): 
        return self.net(x)

# ================== [架构 2]: 我们方案 (DTW-Triplet) 网络 ==================
class PhiNetworkOurs(nn.Module):
    def __init__(self, input_dim=11, basis_dim=8):
        super(PhiNetworkOurs, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, basis_dim - 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        # Ours 专有: 拼接常数物理偏置 1.0
        bias = torch.ones((out.shape[0], 1), device=out.device, dtype=out.dtype)
        return torch.cat([out, bias], dim=-1)