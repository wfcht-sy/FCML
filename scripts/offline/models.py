#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified model definitions.

Architecture source:
  PhiNet -> Phi_Net in official code (aerorobotics/neural-fly, mlmodel.py)
            This is the official feature extractor.
            4 layers: 11->50->60->50->(basis_dim-1), ReLU, constant bias appended.
  DomainDiscriminator -> H_Net_CrossEntropy in official code (mlmodel.py L44-53)
                     2 layers: basis_dim->20->num_classes, ReLU.
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


# ── Phi Network (Official Architecture) ────────────────────────────
# Exactly mirrors Phi_Net in aerorobotics/neural-fly/mlmodel.py
# Input : [v_x, v_y, v_z, q_w, q_x, q_y, q_z, pwm_1, pwm_2, pwm_3, pwm_4] (dim=11)
# Output: basis functions phi(x) in R^{basis_dim}, with last dim fixed to 1.0
class PhiNet(nn.Module):
    """
    Official Phi_Net architecture (4-layer MLP with constant bias).
    Used as the shared backbone.
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


# ── Backward-compatible alias ─────────────────────────────────────────────────
# Any code that imports PhiNetwork will continue to work and receive PhiNet.
# This alias should be removed after all call-sites are updated.
PhiNetwork = PhiNet
PhiNet = PhiNet