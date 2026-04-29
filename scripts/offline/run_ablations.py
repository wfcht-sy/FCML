#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import glob
import shutil

BASE_DIR = "/home/zzx/testmodel"
TRAIN_DIR = os.path.join(BASE_DIR, "training_results")
os.makedirs(TRAIN_DIR, exist_ok=True)

def run_training_and_extract(lambda_val, scheme_name):
    out_dir = os.path.join(TRAIN_DIR, f"run_{scheme_name}")
    print(f"\n{'='*50}\n🚀 开始训练: {scheme_name} (Triplet Lambda = {lambda_val})\n{'='*50}")
    
    # 调用更新后的训练脚本
    cmd = f"python3 train_offline_lightning.py --lambda_triplet {lambda_val} --output_dir {out_dir}"
    subprocess.run(cmd, shell=True, check=True)

    csv_files = glob.glob(os.path.join(out_dir, "lightning_logs", "version_*", "metrics.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        dest_csv = os.path.join(TRAIN_DIR, f"curve_{scheme_name}.csv")
        shutil.copy(latest_csv, dest_csv)
        print(f"\n✅ [{scheme_name}] 训练曲线数据已自动提取: {dest_csv}")
        
        if scheme_name == "ours":
            main_ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
            os.makedirs(main_ckpt_dir, exist_ok=True)
            shutil.copy(os.path.join(out_dir, "best_model.pth"), os.path.join(main_ckpt_dir, "best_model.pth"))

if __name__ == "__main__":
    # 1. 原版 Neural-Fly (纯 MSE)
    run_training_and_extract(0.0, "original")
    # 2. 我们的方案 (动态衰减 Triplet)
    run_training_and_extract(1.0, "ours")
    print("\n🎉 全部消融实验结束！您可以运行 python3 plot_training_curve.py 查看对比了。")