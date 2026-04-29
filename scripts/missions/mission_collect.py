#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly 数据集生成器 V5 (完美对齐 50Hz 论文版)
- 核心改进:
  1. 降频至标准的 50Hz，剔除高频振动噪声，完美对标 Neural-Fly 论文。
  2. 针对新的 train/test 命名规范优化输出提示。
  3. 引入按时间裁剪逻辑，安全剔除起飞前 3 秒的瞬态数据。
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from pathlib import Path
import os
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==================== 1. 物理配置 ====================
G_VAL = 9.8066
MASS = 1.5
HOVER_THR = 0.705810

# [关键修改] 目标频率改为 50 Hz (对标论文控制频率)
TARGET_FREQ = 50.0
DT_TARGET = 1.0 / TARGET_FREQ

# 滤波参数 (50Hz下的窗口需要缩小，11个点约等于0.22秒的平滑窗口)
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

# 路径配置
ROOT_DIR = "/home/zzx/testmodel/raw_logs"
OUTPUT_DIR = "/home/zzx/testmodel/processed_data"

# ==================== 2. 辅助函数 ====================

def normalize_column_names(df):
    if df is None: return None
    new_columns = {}
    for col in df.columns:
        new_col = re.sub(r'_(\d+)$', r'[\1]', col)
        if new_col != col:
            new_columns[col] = new_col
    if new_columns:
        df.rename(columns=new_columns, inplace=True)
    return df

def read_csv_smart(path):
    if not path or not Path(path).exists(): return None
    try:
        df = pd.read_csv(path)
    except: return None

    df = normalize_column_names(df)
    t_cols = [c for c in df.columns if 'timestamp' in c]
    if not t_cols: return None
    t_col = t_cols[0]

    # 转换为秒
    df['t'] = pd.to_numeric(df[t_col], errors='coerce') * 1e-6
    df = df.dropna(subset=['t']).sort_values('t').drop_duplicates('t', keep='last')
    return df.set_index('t')

def get_exact_file(case_dir, topic_name):
    pattern = f"*{topic_name}*.csv"
    files = list(Path(case_dir).glob(pattern))
    if not files: return None
    files.sort(key=lambda x: len(x.name))
    return str(files[0])

# ==================== 3. 核心处理逻辑 ====================

def process_one_case(case_dir, output_path):
    case_name = Path(case_dir).name
    print(f"正在处理: [{case_name}] ...")

    path_pos = get_exact_file(case_dir, "vehicle_local_position")
    path_att = get_exact_file(case_dir, "vehicle_attitude")
    path_out = get_exact_file(case_dir, "actuator_outputs")

    df_pos  = read_csv_smart(path_pos)
    df_att  = read_csv_smart(path_att)
    df_out  = read_csv_smart(path_out)

    if df_pos is None or df_att is None or df_out is None:
        print(f"  [跳过] 数据文件不完整 (缺少 pos/att/out)")
        return False

    # === 物理事件对齐 (Physical Alignment) ===
    # 寻找起飞点：高度 z 首次超过 0.3m (NED 下 z < -0.3)
    takeoff_mask = df_pos['z'] < -0.3
    if not takeoff_mask.any():
        print("  [失败] 未检测到起飞动作 (高度始终未超过 0.3m)")
        return False

    t_zero_epoch = df_pos[takeoff_mask].index[0]
    t_end_epoch = min(df_pos.index[-1], df_att.index[-1], df_out.index[-1])

    if t_end_epoch - t_zero_epoch < 5.0:
        print("  [失败] 有效飞行时间太短 (< 5s)")
        return False

    print(f"  -> 锁定起飞时刻: {t_zero_epoch:.2f}s, 有效时长: {t_end_epoch - t_zero_epoch:.2f}s")

    # 构建 50Hz 相对时间轴
    new_t = np.arange(0, t_end_epoch - t_zero_epoch, DT_TARGET)
    df = pd.DataFrame(index=new_t)
    df.index.name = 't'

    # 重采样函数
    def resample_channel(source_df, cols, prefix=""):
        if source_df is None: return
        t_source_relative = source_df.index - t_zero_epoch
        available_cols = [c for c in cols if c in source_df.columns]
        for c in available_cols:
            interpolated_data = np.interp(new_t, t_source_relative, source_df[c].values)
            df[f"{prefix}{c}"] = interpolated_data

    # 执行重采样
    resample_channel(df_pos, ['vx', 'vy', 'vz'])
    resample_channel(df_att, ['q[0]', 'q[1]', 'q[2]', 'q[3]'], prefix="att_")
    resample_channel(df_out, ['output[0]', 'output[1]', 'output[2]', 'output[3]'], prefix="pwm_")

    if 'vx' not in df.columns or 'att_q[0]' not in df.columns:
        print("  [失败] 缺少速度或姿态关键列")
        return False

    # === 物理计算 ===
    # 1. 重新计算加速度 (50Hz 下求一阶导并平滑)
    for axis in ['x', 'y', 'z']:
        df[f'acc_{axis}'] = savgol_filter(df[f'v{axis}'], SAVGOL_WINDOW, SAVGOL_POLY, deriv=1, delta=DT_TARGET)

    # 2. 推力计算 (适配 PWM 到牛顿的映射)
    pwm_cols = [f'pwm_output[{i}]' for i in range(4)]
    pwm_mean = df[pwm_cols].mean(axis=1)

    if pwm_mean.mean() > 100:
        thrust_norm = (pwm_mean - 1000.0) / 1000.0
    else:
        thrust_norm = pwm_mean

    thrust_newton = (thrust_norm / HOVER_THR) * MASS * G_VAL

    # 3. 残差力 fa 计算: f_residual = m(a - g) - R * T_body
    vec_ma = np.vstack((df['acc_x'], df['acc_y'], df['acc_z'])).T * MASS
    vec_mg = np.array([0, 0, MASS * G_VAL])

    quats = df[['att_q[1]', 'att_q[2]', 'att_q[3]', 'att_q[0]']].to_numpy() # x,y,z,w
    rot_mats = R.from_quat(quats).as_matrix()

    vec_thrust_body = np.zeros_like(vec_ma)
    vec_thrust_body[:, 2] = -thrust_newton # 推力在机体坐标系向上 (Z负)

    vec_thrust_world = np.einsum('ijk,ik->ij', rot_mats, vec_thrust_body)
    vec_fa = vec_ma - vec_mg - vec_thrust_world

    # === 导出与裁剪 ===
    export_df = pd.DataFrame({
        'timestamp': df.index,
        'v_x': df['vx'], 'v_y': df['vy'], 'v_z': df['vz'],
        'q_w': df['att_q[0]'], 'q_x': df['att_q[1]'], 'q_y': df['att_q[2]'], 'q_z': df['att_q[3]'],
        'pwm_1': df['pwm_output[0]'], 'pwm_2': df['pwm_output[1]'],
        'pwm_3': df['pwm_output[2]'], 'pwm_4': df['pwm_output[3]'],
        'f_x': vec_fa[:, 0], 'f_y': vec_fa[:, 1], 'f_z': vec_fa[:, 2]
    })

    # [关键裁剪] 丢弃前 3.0 秒的数据，避开起飞爬升段的剧烈非线性震荡
    # 同时丢弃最后 2.0 秒的数据，避开降落指令下发后的减速段
    export_df = export_df[(export_df['timestamp'] >= 3.0) & (export_df['timestamp'] <= df.index[-1] - 2.0)]

    output_path = str(Path(output_path).with_suffix('.csv'))
    export_df.to_csv(output_path, index=False)
    print(f"  [完成] -> 保存为: {Path(output_path).name} (共 {len(export_df)} 帧 50Hz 数据)")
    return True

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"错误: 找不到原始日志目录 {ROOT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取所有的子文件夹 (如 train_nowind, test_100wind 等)
    subdirs = [f for f in Path(ROOT_DIR).iterdir() if f.is_dir()]
    subdirs.sort()

    print("\n" + "="*50)
    print(f"开始处理日志，总计发现 {len(subdirs)} 个风况工况")
    print("="*50 + "\n")

    success_count = 0
    for subdir in subdirs:
        # 直接用原本的文件名，例如 processed_train_nowind.csv
        output_name = f"processed_{subdir.name}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        if process_one_case(str(subdir), output_path):
            success_count += 1

    print("\n" + "="*50)
    print(f"全部处理完毕! 成功生成高质量 50Hz 数据: {success_count}/{len(subdirs)}")
    print("您现在可以使用这些 CSV 文件进行 DTW 三元组对齐和网络训练了。")
    print("="*50)

if __name__ == '__main__':
    main()
