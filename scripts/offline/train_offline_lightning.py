#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 离线训练脚本 (DTW-Triplet 极限俯冲与深海探底版)
核心突破: 
1. [起步俯冲] 保持 Orthogonal 正交初始化 + Cosine 满血起步 + 0.85 极速退火，确立前期下降的绝对优势。
2. [维度隔离探底] 仅对前 7 维气动特征进行流形约束 ([:, :-1])，彻底解放第 8 维常数偏置，消除后期内耗。
3. [精度与防抖探底] 全局 FP64 双精度贯通 + AdamW(1e-5) 微弱正则化，在 reg_lambda=1e-5 的极限状态下稳稳扎入最低 MSE。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Dataset, DataLoader

# ================= [探底核心 1]: 全局强制 FP64 双精度 =================
# 彻底消除网络前向与反向传播的 FP32 截断误差，抹平最后一点精度劣势
torch.set_default_dtype(torch.float64)

BASIS_DIM, INPUT_DIM, FORCE_SCALE = 8, 11, 6.0    

class DTWTripletDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
        self.label_cols = ['f_x', 'f_y', 'f_z']
        
        for prefix in ['A_', 'P_', 'N_']:
            for i in range(1, 5):
                col = f'{prefix}pwm_{i}'
                if col in self.df.columns and self.df[col].mean() > 100: 
                    self.df[col] = (self.df[col] - 1500.0) / 500.0

        # 确保数据集以 FP64 加入内存
        self.state_a = torch.tensor(self.df[[f'A_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_a = torch.tensor(self.df[[f'A_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE
        self.state_p = torch.tensor(self.df[[f'P_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_p = torch.tensor(self.df[[f'P_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE
        self.state_n = torch.tensor(self.df[[f'N_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_n = torch.tensor(self.df[[f'N_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE

    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return self.state_a[idx], self.force_a[idx], self.state_p[idx], self.force_p[idx], self.state_n[idx], self.force_n[idx]

# 严格对齐官方架构
class PhiNetwork(nn.Module):
    def __init__(self, input_dim=11, basis_dim=8):
        super(PhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, basis_dim - 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        # 官方精髓: 拼接常数 1 作为基础阻力偏置
        bias = torch.ones((out.shape[0], 1), device=out.device, dtype=out.dtype)
        return torch.cat([out, bias], dim=-1)

class NeuralFlyLightning(pl.LightningModule):
    def __init__(self, lr=3e-3, lambda_triplet=1.0, reg_lambda=1e-5, epochs=300):
        super().__init__()
        self.save_hyperparameters()
        self.phi_net = PhiNetwork(INPUT_DIM, BASIS_DIM)
        self.mse_fn = nn.MSELoss()
        
        # 极其微小的 Margin (0.05)
        self.triplet_fn = nn.TripletMarginLoss(margin=0.05, p=2, swap=True) 
        
        # 正交初始化，确保起步的极速响应
        for m in self.phi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def batch_least_squares(self, phi, target):
        # 已经全局 FP64，安全运行 1e-5 的极小正则化
        phi_t = phi.t() 
        A = torch.matmul(phi_t, phi) + self.hparams.reg_lambda * torch.eye(BASIS_DIM, device=self.device, dtype=torch.float64)
        B = torch.matmul(phi_t, target)
        
        try: 
            a_star = torch.linalg.solve(A, B)
        except: 
            a_star = torch.linalg.lstsq(A, B).solution
            
        return a_star

    def shared_step(self, batch, mode="train"):
        state_a, force_a, state_p, force_p, state_n, _ = batch
        
        phi_a = self.phi_net(state_a)
        phi_p = self.phi_net(state_p)
        phi_n = self.phi_net(state_n)
        
        split = state_a.shape[0] // 2
        a_star_a = self.batch_least_squares(phi_a[:split], force_a[:split])
        a_star_p = self.batch_least_squares(phi_p[:split], force_p[:split])
        
        pred_a = torch.matmul(phi_a[split:], a_star_a)
        pred_p = torch.matmul(phi_p[split:], a_star_p)
        
        loss_task = self.mse_fn(pred_a, force_a[split:]) + self.mse_fn(pred_p, force_p[split:])
        
        # ================= [探底核心 2]: 维度污染隔离 =================
        # 切片 [:, :-1] 彻底保护第 8 维的常数 1.0 不被归一化破坏！
        # 让 Triplet 只在动态特征空间里发挥作用，后期 MSE 探底毫无阻力。
        phi_a_norm = F.normalize(phi_a[:, :-1], p=2, dim=1)
        phi_p_norm = F.normalize(phi_p[:, :-1], p=2, dim=1)
        phi_n_norm = F.normalize(phi_n[:, :-1], p=2, dim=1)
        loss_triplet = self.triplet_fn(phi_a_norm, phi_p_norm, phi_n_norm)
        
        # 极速褪去的先验: Decay 0.85，后期隐形
        decay_rate = 0.85
        current_lambda = self.hparams.lambda_triplet * (decay_rate ** self.current_epoch)
        
        total_loss = loss_task + current_lambda * loss_triplet
        
        self.log(f'{mode}_mse', loss_task, on_step=False, on_epoch=True, prog_bar=(mode=="train"))
        self.log(f'{mode}_trip', loss_triplet, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_lambda', current_lambda, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_total', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        if mode == "val": 
            self.log('val_score', loss_task, on_step=False, on_epoch=True, prog_bar=True)
            
        return total_loss

    def training_step(self, batch, batch_idx): return self.shared_step(batch, mode="train")
    def validation_step(self, batch, batch_idx): self.shared_step(batch, mode="val")
    
    def configure_optimizers(self): 
        # ================= [探底核心 3]: AdamW 微弱权重衰减 =================
        # 配合 reg_lambda=1e-5，加入 1e-5 的 weight_decay 可以收紧网络后期的发散，让最低点更稳更低
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        # Cosine 退火平滑到底
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main(args):
    dataset = DTWTripletDataset(args.dtw_csv)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    
    model = NeuralFlyLightning(lr=args.lr, lambda_triplet=args.lambda_triplet, epochs=args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tb_logger = TensorBoardLogger(args.output_dir, name="lightning_logs")
    csv_logger = CSVLogger(args.output_dir, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(monitor='val_score', dirpath=args.output_dir, filename='neural_fly_best', save_top_k=1, mode='min')
    
    # 梯度裁剪 0.5 作为护城河，防住满血起步的任何微小震荡
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        logger=[tb_logger, csv_logger], 
        callbacks=[checkpoint_callback], 
        accelerator="auto", 
        devices=1,
        gradient_clip_val=0.5
    )
    trainer.fit(model, train_loader, val_loader)
    
    best_pl_model = NeuralFlyLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save({'model_state_dict': best_pl_model.phi_net.state_dict(), 'basis_dim': BASIS_DIM}, os.path.join(args.output_dir, "best_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtw_csv", type=str, default="/home/zzx/testmodel/dtw_triplets_data/dtw_triplet_combined_all.csv")
    parser.add_argument("--output_dir", type=str, default="/home/zzx/testmodel/checkpoints")
    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=512) 
    # 保持 3e-3 提供最强劲的初速度
    parser.add_argument("--lr", type=float, default=3e-3) 
    parser.add_argument("--lambda_triplet", type=float, default=1.0) 
    main(parser.parse_args())