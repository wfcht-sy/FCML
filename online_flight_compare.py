#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly Comparison Experiment (Auto Plot) - Fixed Architecture
修复: 还原网络结构以匹配训练好的 checkpoint (final_model_160.pth)
"""

import asyncio
import numpy as np
import math
import torch
import torch.nn as nn
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw, AccelerationNed)
import time
import matplotlib.pyplot as plt  # 绘图库

# ================= 配置区域 =================
MODEL_PATH = "checkpoints/final_model_160.pth"

# 物理参数
MASS = 1.5          
G_VAL = 9.8066      
HOVER_THR = 0.70581 
FORCE_SCALE = 6.0   

# 自适应参数
LAMBDA = 0.995      
R_GAIN = 5.0        
Q_GAIN = 0.001      
P_INIT = 1.0        

INPUT_DIM = 11      
BASIS_DIM = 8       
CTRL_FREQ = 50.0    
DT = 1.0 / CTRL_FREQ

device = torch.device("cpu")

# ================= 1. 轨迹生成器 (直线往返) =================
class TrajectoryGenerator:
    def __init__(self):
        print("=== 轨迹生成器 (Linear Oscillation) ===")
        self.amplitude = 2.5    # 振幅 (米)
        self.period = 10.0      # 周期 (秒)
        self.altitude = -1.5    # 高度
        self.omega = 2 * math.pi / self.period 

    def get_setpoint(self, t):
        w = self.omega
        A = self.amplitude
        
        # 沿 X 轴 (北向) 运动
        px = A * math.sin(w * t)
        py = 0.0 
        pz = self.altitude

        vx = A * w * math.cos(w * t)
        vy = 0.0
        vz = 0.0

        ax = -A * w**2 * math.sin(w * t)
        ay = 0.0
        az = 0.0

        return np.array([px, py, pz]), np.array([vx, vy, vz]), np.array([ax, ay, az])

# ================= 2. 神经网络 (已修复) =================
class PhiNetwork(nn.Module):
    def __init__(self):
        super(PhiNetwork, self).__init__()
        # [修复] 这里去掉了多余的一层，使其与训练时的结构一致
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)), # Layer 0
            nn.ReLU(),                                        # Layer 1
            nn.utils.spectral_norm(nn.Linear(64, 64)),        # Layer 2
            nn.ReLU(),                                        # Layer 3
            nn.utils.spectral_norm(nn.Linear(64, BASIS_DIM))  # Layer 4 (Output)
        )
    def forward(self, x): return self.net(x)

# ================= 3. 控制器逻辑 =================
class NeuralFlyController:
    def __init__(self):
        self.model = PhiNetwork().to(device)
        self.load_model()
        self.model.eval()
        
        self.a_hat = torch.zeros(BASIS_DIM, 3).to(device) 
        self.P = torch.eye(BASIS_DIM).to(device) * P_INIT 
        self.last_pwm = np.array([HOVER_THR]*4) 

    def load_model(self):
        try:
            # 如果 PyTorch 版本报错 weights_only，请删除该参数
            checkpoint = torch.load(MODEL_PATH, map_location=device) 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[NeuralFly] 模型加载成功: {MODEL_PATH}")
        except Exception as e:
            print(f"加载失败: {e}")
            print("建议检查网络结构定义是否与 checkpoint 匹配。")
            exit(1)

    def preprocess_state(self, v, q, pwm):
        pwm_arr = np.array(pwm)
        if np.min(pwm_arr) < -0.9: pwm_norm = pwm_arr
        else: pwm_norm = pwm_arr * 2.0 - 1.0
        state = np.concatenate([v, q, pwm_norm])
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def update_and_predict(self, v, q, pwm, acc_meas, s_err):
        state_tensor = self.preprocess_state(v, q, self.last_pwm)
        with torch.no_grad():
            phi = self.model(state_tensor) 
        self.last_pwm = pwm 

        pwm_mean = np.mean(pwm)
        if pwm_mean > 1.0: pwm_mean = 1.0
        thrust_mag = (pwm_mean / HOVER_THR) * MASS * G_VAL
        
        r_mat = self._quat_to_rot_matrix(q)
        t_body = np.array([0, 0, -thrust_mag])
        t_world = r_mat @ t_body
        
        y_force = MASS * acc_meas - (np.array([0, 0, MASS * G_VAL]) + t_world)
        y_norm = torch.FloatTensor(y_force).unsqueeze(0).to(device) / FORCE_SCALE
        s_tensor = torch.FloatTensor(s_err).unsqueeze(0).to(device)

        phi_t = phi.t()
        pred_y = torch.mm(phi, self.a_hat)
        pred_error = y_norm - pred_y 
        
        term_pred = torch.mm(self.P, phi_t) * (1.0/R_GAIN) * pred_error
        term_track = torch.mm(self.P, phi_t) * s_tensor * 0.1 
        
        self.a_hat = self.a_hat + (term_pred + term_track) * DT
        
        comp_force_norm = torch.mm(phi, self.a_hat)
        comp_force = comp_force_norm.cpu().numpy()[0] * FORCE_SCALE
        comp_acc = comp_force / MASS
        
        acc_norm = np.linalg.norm(comp_acc)
        MAX_ACC = 4.0
        if acc_norm > MAX_ACC:
            comp_acc = comp_acc * (MAX_ACC / acc_norm)
        
        return comp_acc

    def _quat_to_rot_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

# ================= 4. 主程序 =================
async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    print("Connecting...")
    async for state in drone.core.connection_state():
        if state.is_connected: break
    print("Done.")

    nf_ctrl = NeuralFlyController()
    traj_gen = TrajectoryGenerator()
    
    state_cache = {
        'v': np.zeros(3),
        'p': np.zeros(3),
        'q': np.array([1.0, 0.0, 0.0, 0.0]),
        'pwm': np.array([HOVER_THR]*4)
    }

    # 数据记录容器
    data_log = {
        'off_x': [], 'off_y': [], # 第一阶段：关
        'on_x': [],  'on_y': [],  # 第二阶段：开
        'ref_x': [], 'ref_y': []  # 参考轨迹
    }

    # 后台协程
    async def update_pos_vel():
        try:
            async for packet in drone.telemetry.position_velocity_ned():
                state_cache['p'] = np.array([packet.position.north_m, packet.position.east_m, packet.position.down_m])
                state_cache['v'] = np.array([packet.velocity.north_m_s, packet.velocity.east_m_s, packet.velocity.down_m_s])
        except Exception: pass
    
    async def update_attitude():
        try:
            async for q in drone.telemetry.attitude_quaternion():
                state_cache['q'] = np.array([q.w, q.x, q.y, q.z])
        except Exception: pass

    async def update_throttle():
        try:
            async for hud in drone.telemetry.vfr_hud():
                thr_01 = hud.throttle / 100.0
                state_cache['pwm'] = np.array([thr_01]*4)
        except Exception: pass 

    asyncio.ensure_future(update_pos_vel())
    asyncio.ensure_future(update_attitude())
    asyncio.ensure_future(update_throttle())

    print("Checking GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok: break
        await asyncio.sleep(1)
    
    print("Arming & Takeoff...")
    p0, v0, a0 = traj_gen.get_setpoint(0.0)
    await drone.offboard.set_position_ned(PositionNedYaw(p0[0], p0[1], p0[2], 0.0))

    try:
        await drone.offboard.start()
        await drone.action.arm()
    except Exception as e:
        print(f"Start failed: {e}"); return

    await asyncio.sleep(5) # 等待起飞稳定
    
    start_time_traj = time.time()
    last_time = time.time()
    last_vel = np.zeros(3)
    
    print("\n>>> EXPERIMENT START (60s Total) <<<")
    print("   0-30s: OFF (Blue Line)")
    print("  30-60s: ON  (Red Line)")
    print("--------------------------------------")
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 计时与阶段控制
            now = time.time()
            t_current = now - start_time_traj
            
            enable_nf = False
            phase_name = "WAIT"

            if t_current < 30.0:
                enable_nf = False
                phase_name = "PHASE 1 (OFF)"
                # 记录第一阶段数据
                data_log['off_x'].append(state_cache['p'][0])
                data_log['off_y'].append(state_cache['p'][1])
            elif t_current < 60.0:
                enable_nf = True
                phase_name = "PHASE 2 (ON) "
                # 记录第二阶段数据
                data_log['on_x'].append(state_cache['p'][0])
                data_log['on_y'].append(state_cache['p'][1])
            else:
                print("\n>>> EXPERIMENT FINISHED! <<<")
                break # 结束实验

            # 2. 轨迹生成
            ref_p, ref_v, ref_a = traj_gen.get_setpoint(t_current)
            
            # 记录参考轨迹
            data_log['ref_x'].append(ref_p[0])
            data_log['ref_y'].append(ref_p[1])

            # 3. 状态获取
            curr_v = state_cache['v']
            curr_p = state_cache['p']
            curr_q = state_cache['q']
            curr_pwm = state_cache['pwm']

            # 4. 微分加速度
            dt = time.time() - last_time
            if dt > 0.001: acc_meas = (curr_v - last_vel) / dt
            else: acc_meas = np.zeros(3)
            last_vel = curr_v
            last_time = time.time()
            
            # 5. 误差
            Lambda = 2.0
            s = (curr_v - ref_v) + Lambda * (curr_p - ref_p)
            
            # 6. Neural-Fly 计算
            comp_acc = nf_ctrl.update_and_predict(curr_v, curr_q, curr_pwm, acc_meas, s)
            
            # === 开关应用 ===
            if not enable_nf:
                comp_acc = np.zeros(3)
            # ===============
            
            # 7. 打印
            print(f"\r [{phase_name}] Time:{t_current:4.1f}s | TrajX:{ref_p[0]:5.2f} | Y-Err:{curr_p[1]-ref_p[1]:5.2f}", end="")
            
            # 8. 控制
            ff_acc_x = ref_a[0] - comp_acc[0]
            ff_acc_y = ref_a[1] - comp_acc[1]
            ff_acc_z = ref_a[2] - comp_acc[2]

            await drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(ref_p[0], ref_p[1], ref_p[2], 0.0),
                VelocityNedYaw(ref_v[0], ref_v[1], ref_v[2], 0.0),
                AccelerationNed(ff_acc_x, ff_acc_y, ff_acc_z)
            )
            
            elapsed = time.time() - loop_start
            if elapsed < DT:
                await asyncio.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # 结束后的处理
    print("\nLanding...")
    try:
        await drone.action.land()
    except: pass
    
    # === 绘图逻辑 ===
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # 画参考轨迹 (黑色虚线)
    plt.plot(data_log['ref_x'], data_log['ref_y'], 'k--', label='Reference', alpha=0.5)
    
    # 画 Phase 1 (关闭抗风) - 蓝色
    plt.plot(data_log['off_x'], data_log['off_y'], 'b-', label='Baseline (OFF)', linewidth=2, alpha=0.7)
    
    # 画 Phase 2 (开启抗风) - 红色
    plt.plot(data_log['on_x'], data_log['on_y'], 'r-', label='Neural-Fly (ON)', linewidth=2)
    
    plt.title('Neural-Fly Adaptation Comparison (Linear Trajectory)')
    plt.xlabel('North Position (X) [m]')
    plt.ylabel('East Position (Y) - Drift [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    
    filename = "flight_comparison.png"
    plt.savefig(filename)
    print(f"Graph saved to: {filename}")
    print("Use file manager to view the image.")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt: pass