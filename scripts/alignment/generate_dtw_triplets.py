#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly DTW 三元组生成器 (50Hz 满血精度版)
"""

import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from pathlib import Path
import random
import time

# ================= 核心超参数 (适配 50Hz) =================
DOWNSAMPLE_STEP = 1   # 50Hz数据量极度精简(约8000帧)，废除降采样，开启 1:1 满血精度对齐！
DTW_RADIUS = 50       # 搜索半径 (50Hz下，50帧 = 1.0秒的允许相位差，足够覆盖被风吹偏的滞后)
NEG_MARGIN = 150      # 负样本随机区间 (50Hz下，150帧 = 3.0秒外的动作，确保动态特征完全不同)
# ==========================================================

def smooth_and_normalize_3d(v_3d, window=11, poly=3):
    """对 3D 速度序列进行平滑与归一化 (适配50Hz的窗口大小)"""
    v_smooth = np.zeros_like(v_3d)
    for i in range(3):
        w = min(window, len(v_3d) if len(v_3d) % 2 != 0 else len(v_3d) - 1)
        if w > 3:
            v_smooth[:, i] = savgol_filter(v_3d[:, i], window_length=w, polyorder=poly)
        else:
            v_smooth[:, i] = v_3d[:, i]
    # 归一化，消除绝对数值差异，只对齐波形趋势
    return (v_smooth - np.mean(v_smooth, axis=0)) / (np.std(v_smooth, axis=0) + 1e-8)

def evaluate_alignment(v_a, v_p):
    """自我评估模块"""
    corr = np.corrcoef(v_a, v_p)[0, 1]
    mse = np.mean((v_a - v_p)**2)
    return corr, mse

def generate_triplets(csv_0, csv_w, output_path):
    print(f"\n🚀 正在极速对齐目标文件: {Path(csv_w).name}")
    
    df_0 = pd.read_csv(csv_0)
    df_w = pd.read_csv(csv_w)

    # 1. 提取全量数据
    v0_3d = df_0[['v_x', 'v_y', 'v_z']].values
    vw_3d = df_w[['v_x', 'v_y', 'v_z']].values
    
    # 2. 生成平滑引导序列 (按 DOWNSAMPLE_STEP 提取)
    v0_guide = smooth_and_normalize_3d(v0_3d)[::DOWNSAMPLE_STEP]
    vw_guide = smooth_and_normalize_3d(vw_3d)[::DOWNSAMPLE_STEP]

    # 3. 极速执行 FastDTW
    print(f"   [计算] 启动 FastDTW (精度: 满血 {len(v0_guide)} 帧, 搜索半径: {DTW_RADIUS})...")
    start_time = time.time()
    distance, path_sub = fastdtw(v0_guide, vw_guide, radius=DTW_RADIUS, dist=euclidean)
    print(f"   [完成] 耗时: {time.time()-start_time:.2f} 秒 | 规整距离: {distance:.2f}")

    # 4. 构建三元组 (并进行自我验证)
    triplet_data = []
    max_idx_0_sub = len(v0_guide) - 1
    
    feat_cols = ['v_x', 'v_y', 'v_z', 'q_w', 'q_x', 'q_y', 'q_z', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']
    label_cols = ['f_x', 'f_y', 'f_z']
    
    # 用于最终验证的抽样序列
    aligned_v_a, aligned_v_p = [], []

    for i_sub, j_sub in path_sub:
        # 将降采样的索引还原回原始 DataFrame 行索引
        idx_0_real = i_sub * DOWNSAMPLE_STEP
        idx_w_real = j_sub * DOWNSAMPLE_STEP
        
        # 边界保护
        if idx_0_real >= len(df_0) or idx_w_real >= len(df_w): continue
        
        row_A = df_0.iloc[idx_0_real]
        row_P = df_w.iloc[idx_w_real]
        
        # 负样本随机采样 (基于当前无风坐标进行安全偏移)
        idx_N_sub = random.randint(0, max_idx_0_sub)
        while abs(idx_N_sub - i_sub) < NEG_MARGIN:
            idx_N_sub = random.randint(0, max_idx_0_sub)
        idx_N_real = idx_N_sub * DOWNSAMPLE_STEP
        row_N = df_0.iloc[idx_N_real]

        # 组装一条数据 (A: Anchor无风, P: Positive有风对齐, N: Negative无风偏移)
        record = {'t_A': row_A['timestamp'], 't_P': row_P['timestamp'], 't_N': row_N['timestamp']}
        for c in feat_cols: record[f'A_{c}'] = row_A[c]; record[f'P_{c}'] = row_P[c]; record[f'N_{c}'] = row_N[c]
        for c in label_cols: record[f'A_{c}'] = row_A[c]; record[f'P_{c}'] = row_P[c]; record[f'N_{c}'] = row_N[c]
            
        triplet_data.append(record)
        
        # 保存用于验证的真实速度值 (以 X 轴速度为例)
        aligned_v_a.append(row_A['v_x'])
        aligned_v_p.append(row_P['v_x'])

    # === 自我验证环节 ===
    aligned_v_a = np.array(aligned_v_a)
    aligned_v_p = np.array(aligned_v_p)
    norm_v_a = (aligned_v_a - np.mean(aligned_v_a)) / (np.std(aligned_v_a) + 1e-8)
    norm_v_p = (aligned_v_p - np.mean(aligned_v_p)) / (np.std(aligned_v_p) + 1e-8)
    
    corr, mse = evaluate_alignment(norm_v_a, norm_v_p)
    print(f"   📊 [评估报告] Pearson r: {corr:.4f} | 规整后 MSE: {mse:.4f}")
    if corr >= 0.95: print("      ✅ [验证通过] 波峰波谷严格锁定！动作已完美对齐。")
    elif corr >= 0.85: print("      ⚠️ [验证警告] 对齐质量尚可，局部因狂风存在必然畸变。")
    else: print("      ❌ [验证失败] 波形对齐严重错位！")

    return pd.DataFrame(triplet_data)

def main():
    INPUT_DIR = Path("/home/zzx/testmodel/processed_data")
    OUTPUT_DIR = Path("/home/zzx/testmodel/dtw_triplets_data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 锚定无风基准测试集
    base_file = INPUT_DIR / "processed_train_nowind.csv"
    if not base_file.exists():
        print(f"❌ 找不到基准文件: {base_file}，请检查数据处理步骤。")
        return

    # 获取所有有风的训练集
    target_files = sorted(INPUT_DIR.glob("processed_train_*wind.csv"))
    target_files = [f for f in target_files if f.name != base_file.name]
    
    if not target_files:
        print("❌ 找不到需要对齐的有风数据。")
        return

    all_triplets_dfs = []
    for target_file in target_files:
        # 提取风况名称，例如从 'processed_train_10wind' 中提取 '10wind'
        suffix = target_file.stem.split('_')[-1]
        out_name = OUTPUT_DIR / f"dtw_triplet_nowind_to_{suffix}.csv"
        
        df_triplet = generate_triplets(base_file, target_file, out_name)
        df_triplet.to_csv(out_name, index=False)
        print(f"   💾 保存单风况数据集: {out_name.name} (有效特征组: {len(df_triplet)})\n")
        all_triplets_dfs.append(df_triplet)

    if all_triplets_dfs:
        combined_df = pd.concat(all_triplets_dfs, ignore_index=True)
        combined_out = OUTPUT_DIR / "dtw_triplet_combined_all.csv"
        combined_df.to_csv(combined_out, index=False)
        print(f"🎉 大功告成！所有风况 ({len(target_files)}组) 已合并为总训练集: {combined_out.name}")
        print(f"   总数据量: {len(combined_df)} 组 Triplet。即将可以喂给神经网络！")

if __name__ == "__main__":
    main()