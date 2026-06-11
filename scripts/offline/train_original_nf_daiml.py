#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PROCESSED_DIR, CHECKPOINTS_DIR

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import os
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Dataset, DataLoader
from scripts.offline.models import PhiNet, DomainDiscriminator, grad_reverse

# Global FP64 — must match the float64 tensors in DAIMLDataset and the online controller
torch.set_default_dtype(torch.float64)

BASIS_DIM, INPUT_DIM, FORCE_SCALE = 8, 11, 6.0

# ================= 1. DAIML Dataset Adapter =================
class DAIMLDataset(Dataset):
    def __init__(self, processed_dir):
        # Read per-wind-condition CSVs and assign domain labels
        csv_files = sorted(glob.glob(os.path.join(processed_dir, "processed_train_*wind.csv")))

        # Clear error if processed_data directory is missing or empty
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"\n[ERROR] No 'processed_train_*wind.csv' files found in: {processed_dir}\n"
                f"  The processed_data directory must contain pre-processed training CSVs.\n"
                f"  Solution: Copy from testmodel1:\n"
                f"    cp -r ~/testmodel1/testmodel/processed_data ~/FCML/processed_data\n"
                f"  Or run the preprocessing pipeline first."
            )

        self.num_domains = len(csv_files)
        print(f"  [DAIML] Found {self.num_domains} domain CSV files: {[os.path.basename(f) for f in csv_files]}")
        
        all_states, all_forces, all_labels = [], [], []
        feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
        label_cols = ['f_x', 'f_y', 'f_z']

        for domain_idx, f in enumerate(csv_files):
            df = pd.read_csv(f)
            for i in range(1, 5):
                col = f'pwm_{i}'
                if col in df.columns and df[col].mean() > 100:
                    df[col] = (df[col] - 1500.0) / 500.0
            all_states.append(df[feat_cols].values)
            all_forces.append(df[label_cols].values / FORCE_SCALE)
            all_labels.extend([domain_idx] * len(df))

        # Use float64 to match online controller's torch.set_default_dtype(torch.float64)
        self.states = torch.tensor(np.vstack(all_states), dtype=torch.float64)
        self.forces = torch.tensor(np.vstack(all_forces), dtype=torch.float64)
        self.labels = torch.LongTensor(all_labels)

    def __len__(self): return len(self.states)
    def __getitem__(self, idx): return self.states[idx], self.forces[idx], self.labels[idx]

# ================= 3. PyTorch Lightning Module =================
class OriginalNeuralFlyDAIML(pl.LightningModule):
    def __init__(self, num_domains, lr=1e-3, lambda_adv=0.1, reg_lambda=1e-4, epochs=300):
        super().__init__()
        self.save_hyperparameters()
        self.phi_net = PhiNet(INPUT_DIM, BASIS_DIM)
        self.discriminator = DomainDiscriminator(BASIS_DIM, num_domains)
        
        self.mse_fn = nn.MSELoss()
        self.ce_fn = nn.CrossEntropyLoss()

    def batch_least_squares(self, phi, target):
        phi_t = phi.t() 
        A = torch.matmul(phi_t, phi) + self.hparams.reg_lambda * torch.eye(BASIS_DIM, device=self.device)
        B = torch.matmul(phi_t, target)
        try: return torch.linalg.solve(A, B)
        except: return torch.linalg.lstsq(A, B).solution

    def shared_step(self, batch, mode="train"):
        states, forces, domain_labels = batch
        phi = self.phi_net(states)
        
        # 1. Domain adversarial loss (GRL forces domain-invariant features)
        phi_rev = grad_reverse(phi, alpha=1.0)
        domain_preds = self.discriminator(phi_rev)
        loss_adv = self.ce_fn(domain_preds, domain_labels)
        
        # 2. Meta-learning task loss (support/query split per Neural-Fly paper)
        split = states.shape[0] // 2
        a_star = self.batch_least_squares(phi[:split], forces[:split])
        pred_forces = torch.matmul(phi[split:], a_star)
        loss_task = self.mse_fn(pred_forces, forces[split:])
        
        # 3. Total loss
        total_loss = loss_task + self.hparams.lambda_adv * loss_adv
        
        self.log(f'{mode}_mse', loss_task, on_step=False, on_epoch=True, prog_bar=(mode=="train"))
        self.log(f'{mode}_adv', loss_adv, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_total', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # Primary evaluation metric: generalization MSE
        if mode == "val": self.log('val_score', loss_task, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx): return self.shared_step(batch, mode="train")
    def validation_step(self, batch, batch_idx): self.shared_step(batch, mode="val")
    def configure_optimizers(self): 
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main(args):
    dataset = DAIMLDataset(args.processed_dir)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    
    model = OriginalNeuralFlyDAIML(num_domains=dataset.num_domains, lr=args.lr, lambda_adv=args.lambda_adv, epochs=args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tb_logger = TensorBoardLogger(args.output_dir, name="lightning_logs")
    csv_logger = CSVLogger(args.output_dir, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(monitor='val_score', dirpath=args.output_dir, filename='neural_fly_daiml_best', save_top_k=1, mode='min')
    
    trainer = pl.Trainer(max_epochs=args.epochs, logger=[tb_logger, csv_logger], callbacks=[checkpoint_callback], accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)
    
    # Extract weights for deployment compatibility
    best_pl_model = OriginalNeuralFlyDAIML.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save({'model_state_dict': best_pl_model.phi_net.state_dict(), 'basis_dim': BASIS_DIM}, os.path.join(args.output_dir, "neural_fly_daiml_best.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--output_dir", type=str, default=CHECKPOINTS_DIR)
    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    main(parser.parse_args())