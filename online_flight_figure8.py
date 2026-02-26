#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly Online Flight V7 (Global Switch)
功能: 
1. 8字形动态轨迹跟踪
2. 包含全局抗风开关 (ENABLE_NEURAL_FLY)
3. 包含所有稳定性修复 (GPS等待, API容错)
"""

import asyncio
import numpy as np
import math
import torch
import torch.nn as nn
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw, AccelerationNed)
import time

# ================= 配置区域 =================
MODEL_PATH = "checkpoints/final_model_160.pth"

# === [核心开关] ===
# True  = 开启 Neural-Fly 抗风 (Active)
# False = 关闭抗风 (Baseline, 只有 PID + 轨迹前馈)
ENABLE_NEURAL_FLY = False  

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

# ================= 1. 轨迹生成器 =================
class TrajectoryGenerator:
    def __init__(self):
        print("=== 轨迹生成器 (Figure-8) ===")
        self.radius = 1.5       # 半径
        self.period = 12.0      # 周期(秒)
        self.altitude = -1.5    # 高度
        self.omega = 2 * math.pi / self.period 

    def get_setpoint(self, t):
        w = self.omega
        r = self.radius
        
        # 位置
        px = r * math.sin(w * t)
        py = r * math.sin(2 * w * t) / 2.0 
        pz = self.altitude

        # 速度
        vx = r * w * math.cos(w * t)
        vy = (r * 2 * w * math.cos(2 * w * t)) / 2.0
        vz = 0.0

        # 加速度 (轨迹前馈)
        ax = -r * w**2 * math.sin(w * t)
        ay = -(r * (2 * w)**2 * math.sin(2 * w * t)) / 2.0
        az = 0.0

        return np.array([px, py, pz]), np.array([vx, vy, vz]), np.array([ax, ay, az])

# ================= 2. 神经网络 =================
class PhiNetwork(nn.Module):
    def __init__(self):
        super(PhiNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(INPUT_DIM, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, BASIS_DIM))
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
            checkpoint = torch.load(MODEL_PATH, map_location=device) 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[NeuralFly] 模型加载成功: {MODEL_PATH}")
        except Exception as e:
            print(f"加载失败: {e}"); exit(1)

    def preprocess_state(self, v, q, pwm):
        pwm_arr = np.array(pwm)
        if np.min(pwm_arr) < -0.9: pwm_norm = pwm_arr
        else: pwm_norm = pwm_arr * 2.0 - 1.0
        state = np.concatenate([v, q, pwm_norm])
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def update_and_predict(self, v, q, pwm, acc_meas, s_err):
        # 1. 推理
        state_tensor = self.preprocess_state(v, q, self.last_pwm)
        with torch.no_grad():
            phi = self.model(state_tensor) 
        self.last_pwm = pwm 

        # 2. 观测
        pwm_mean = np.mean(pwm)
        if pwm_mean > 1.0: pwm_mean = 1.0
        thrust_mag = (pwm_mean / HOVER_THR) * MASS * G_VAL
        
        r_mat = self._quat_to_rot_matrix(q)
        t_body = np.array([0, 0, -thrust_mag])
        t_world = r_mat @ t_body
        
        y_force = MASS * acc_meas - (np.array([0, 0, MASS * G_VAL]) + t_world)
        y_norm = torch.FloatTensor(y_force).unsqueeze(0).to(device) / FORCE_SCALE
        s_tensor = torch.FloatTensor(s_err).unsqueeze(0).to(device)

        # 3. 更新
        phi_t = phi.t()
        pred_y = torch.mm(phi, self.a_hat)
        pred_error = y_norm - pred_y 
        
        term_pred = torch.mm(self.P, phi_t) * (1.0/R_GAIN) * pred_error
        term_track = torch.mm(self.P, phi_t) * s_tensor * 0.1 
        
        self.a_hat = self.a_hat + (term_pred + term_track) * DT
        
        # 4. 预测补偿
        comp_force_norm = torch.mm(phi, self.a_hat)
        comp_force = comp_force_norm.cpu().numpy()[0] * FORCE_SCALE
        comp_acc = comp_force / MASS
        
        # 限幅
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
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected: break
    print("Done.")

    # 显示当前模式
    mode_str = "ON (Neural-Fly)" if ENABLE_NEURAL_FLY else "OFF (Baseline)"
    print(f"========================================")
    print(f"   Current Mode: {mode_str}")
    print(f"========================================")

    nf_ctrl = NeuralFlyController()
    traj_gen = TrajectoryGenerator()
    
    state_cache = {
        'v': np.zeros(3),
        'p': np.zeros(3),
        'q': np.array([1.0, 0.0, 0.0, 0.0]),
        'pwm': np.array([HOVER_THR]*4)
    }

    # 后台数据协程
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

    # 起飞前检查
    print("Waiting for GPS Lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- GPS Ready")
            break
        print(".", end="", flush=True)
        await asyncio.sleep(1)
    
    # 启动
    print("\n-- Setting Initial Setpoint")
    p0, v0, a0 = traj_gen.get_setpoint(0.0)
    await drone.offboard.set_position_ned(PositionNedYaw(p0[0], p0[1], p0[2], 0.0))

    print("-- Starting Offboard & Arming")
    try:
        await drone.offboard.start()
        await drone.action.arm()
    except Exception as e:
        print(f"Start failed: {e}"); return

    print("Taking off (5s)...")
    await asyncio.sleep(5)
    
    start_time_traj = time.time()
    last_time = time.time()
    last_vel = np.zeros(3)
    
    print(f">>> Flight Started | Mode: {mode_str} <<<")
    
    while True:
        loop_start = time.time()
        
        # 1. 轨迹
        now = time.time()
        t_current = now - start_time_traj
        ref_p, ref_v, ref_a = traj_gen.get_setpoint(t_current)
        
        # 2. 状态
        curr_v = state_cache['v']
        curr_p = state_cache['p']
        curr_q = state_cache['q']
        curr_pwm = state_cache['pwm']

        # 3. 测量加速度
        dt = time.time() - last_time
        if dt > 0.001: acc_meas = (curr_v - last_vel) / dt
        else: acc_meas = np.zeros(3)
        last_vel = curr_v
        last_time = time.time()
        
        # 4. 误差 s
        Lambda = 2.0
        s = (curr_v - ref_v) + Lambda * (curr_p - ref_p)
        
        # 5. Neural-Fly 计算
        # 注意：即使开关关闭，我们也让网络跑一下(更新a_hat)，以便随时可以切回来
        # 或者为了彻底的对比，关闭时也不更新参数(取决于你想测什么)
        # 这里选择：始终计算，但只在开关开启时应用补偿
        comp_acc = nf_ctrl.update_and_predict(curr_v, curr_q, curr_pwm, acc_meas, s)
        
        # === [核心开关逻辑] ===
        if not ENABLE_NEURAL_FLY:
            comp_acc = np.zeros(3) # 强制清零，不给飞控发送抗风指令
            status_text = "OFF"
        else:
            status_text = "ON "
        # ====================
        
        # 6. 计算总前馈 (轨迹前馈 - 抗风补偿)
        ff_acc_x = ref_a[0] - comp_acc[0]
        ff_acc_y = ref_a[1] - comp_acc[1]
        ff_acc_z = ref_a[2] - comp_acc[2]
        
        # 7. 打印
        acc_mag = np.linalg.norm(comp_acc[:2]) 
        print(f"\r [{status_text}] TrajX: {ref_p[0]:5.2f} | CompX: {comp_acc[0]:5.2f} | Total: {acc_mag:5.2f}", end="")
        
        # 8. 发送
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(ref_p[0], ref_p[1], ref_p[2], 0.0),
            VelocityNedYaw(ref_v[0], ref_v[1], ref_v[2], 0.0),
            AccelerationNed(ff_acc_x, ff_acc_y, ff_acc_z)
        )
        
        elapsed = time.time() - loop_start
        if elapsed < DT:
            await asyncio.sleep(DT - elapsed)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt: pass