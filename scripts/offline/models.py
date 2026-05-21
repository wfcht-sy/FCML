#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified model definitions for FCML and Neural-Fly.

Architecture source:
  PhiNetworkFCML  -> Phi_Net  in Neural-Fly official code (aerorobotics/neural-fly, mlmodel.py)
                     This IS the official Neural-Fly feature extractor.
                     4 layers: 11->50->60->50->(basis_dim-1), ReLU, constant bias appended.
  DomainDiscriminator -> H_Net_CrossEntropy in Neural-Fly official code (mlmodel.py L44-53)
                     2 layers: basis_dim->20->num_classes, ReLU.

Note on the removed "PhiNetwork" (2-layer, 64-unit, Spectral Norm):
  That structure does NOT appear anywhere in the Neural-Fly paper or official codebase.
  It was an incorrect approximation introduced earlier and has been removed.
  All experiments (NF baseline and FCML) now use PhiNetworkFCML as the shared backbone,
  which is the legitimate architecture cited from the official Neural-Fly implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal Layer (from Neural-Fly DAIML training) ─────────────────
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


# ── Phi Network (Official Neural-Fly Architecture) ────────────────────────────
# Exactly mirrors Phi_Net in aerorobotics/neural-fly/mlmodel.py
# Input : [v_x, v_y, v_z, q_w, q_x, q_y, q_z, pwm_1, pwm_2, pwm_3, pwm_4] (dim=11)
# Output: basis functions phi(x) in R^{basis_dim}, with last dim fixed to 1.0
class PhiNetworkFCML(nn.Module):
    """
    Official Neural-Fly Phi_Net architecture (4-layer MLP with constant bias).
    Used as the shared backbone for both the NF-DAIML baseline and FCML method.
    """
    def __init__(self, input_dim=11, basis_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, basis_dim - 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        # Append constant physical bias 1.0 (directly from official mlmodel.py L39-42)
        if out.dim() == 1:
            return torch.cat([out, torch.ones(1, device=out.device, dtype=out.dtype)])
        return torch.cat([out, torch.ones(out.shape[0], 1, device=out.device, dtype=out.dtype)], dim=-1)


# ── Domain Discriminator (Official Neural-Fly Architecture) ───────────────────
# Exactly mirrors H_Net_CrossEntropy in aerorobotics/neural-fly/mlmodel.py L44-53
# Input : phi(x) in R^{basis_dim}
# Output: logits over num_domains classes (CrossEntropy loss, no Softmax here)
class DomainDiscriminator(nn.Module):
    """
    Official Neural-Fly H_Net_CrossEntropy architecture.
    2-layer MLP: basis_dim -> 20 -> num_domains, ReLU activation, no Spectral Norm.
    """
    def __init__(self, basis_dim=8, num_domains=6):
        super().__init__()
        self.fc1 = nn.Linear(basis_dim, 20)
        self.fc2 = nn.Linear(20, num_domains)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PhiNetwork(nn.Module):
    """
    Original Neural-Fly approximated network.
    3-layer MLP with spectral normalization.
    """
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

# ── Backward-compatible alias ─────────────────────────────────────────────────
# Note: PhiNetwork is explicitly defined above because old checkpoints use spectral_norm.