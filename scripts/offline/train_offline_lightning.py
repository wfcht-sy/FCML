#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import os
import sys
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DTW_CSV, CHECKPOINTS_DIR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Dataset, DataLoader

# Global FP64 precision
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

        self.state_a = torch.tensor(self.df[[f'A_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_a = torch.tensor(self.df[[f'A_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE
        self.state_p = torch.tensor(self.df[[f'P_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_p = torch.tensor(self.df[[f'P_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE
        self.state_n = torch.tensor(self.df[[f'N_{c}' for c in self.feat_cols]].values, dtype=torch.float64)
        self.force_n = torch.tensor(self.df[[f'N_{c}' for c in self.label_cols]].values, dtype=torch.float64) / FORCE_SCALE

    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return self.state_a[idx], self.force_a[idx], self.state_p[idx], self.force_p[idx], self.state_n[idx], self.force_n[idx]

class PhiNetwork(nn.Module):
    """Feature extraction network (basis function phi)."""
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
        # Append constant bias term (1.0) as the base drag offset
        bias = torch.ones((out.shape[0], 1), device=out.device, dtype=out.dtype)
        return torch.cat([out, bias], dim=-1)

class NeuralFlyLightning(pl.LightningModule):
    def __init__(self, lr=3e-3, lambda_triplet=1.0, reg_lambda=1e-5, epochs=300):
        super().__init__()
        self.save_hyperparameters()
        self.phi_net = PhiNetwork(INPUT_DIM, BASIS_DIM)
        self.mse_fn = nn.MSELoss()
        self.triplet_fn = nn.TripletMarginLoss(margin=0.05, p=2, swap=True) 
        
        # Orthogonal initialization for fast convergence
        for m in self.phi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def batch_least_squares(self, phi, target):
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
        
        # Triplet loss on dynamic features only (exclude constant bias dimension)
        phi_a_norm = F.normalize(phi_a[:, :-1], p=2, dim=1)
        phi_p_norm = F.normalize(phi_p[:, :-1], p=2, dim=1)
        phi_n_norm = F.normalize(phi_n[:, :-1], p=2, dim=1)
        loss_triplet = self.triplet_fn(phi_a_norm, phi_p_norm, phi_n_norm)
        
        # Exponential decay of triplet weight
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
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
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
    parser.add_argument("--dtw_csv", type=str, default=DTW_CSV)
    parser.add_argument("--output_dir", type=str, default=CHECKPOINTS_DIR)
    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=512) 
    parser.add_argument("--lr", type=float, default=3e-3) 
    parser.add_argument("--lambda_triplet", type=float, default=1.0) 
    main(parser.parse_args())