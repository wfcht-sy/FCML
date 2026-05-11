#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== Architecture 1: Original Neural-Fly Network ==================
# Retains original name for backward compatibility with DAIML training checkpoints.
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

# ================== Architecture 2: FCML Network (DTW-Triplet) ==================
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
        # Append constant bias term (1.0) as the base drag offset
        bias = torch.ones((out.shape[0], 1), device=out.device, dtype=out.dtype)
        return torch.cat([out, bias], dim=-1)