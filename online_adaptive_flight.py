#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-Fly Online Controller V5 (Preflight Check)
修复: 
1. 增加起飞前健康检查 (Wait for Global Position)
2. 优化 Offboard 启动顺序 (Set -> Start -> Arm)
3. 保持 V4 的全部稳定性修复 (vfr_hud 容错, API 修复)
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityNedYaw, AccelerationNed)
import time

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

# ================= 网络定义 =================
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

# ================= 控制器类 =================
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
            print(f"加载失败: {e}"); exit(1)

    def preprocess_state(self, v, q, pwm):
        pwm_arr = np.array(pwm)
        if np.min(pwm_arr) < -0.9: pwm_norm = pwm_arr
        else: pwm_norm = pwm_arr * 2.0 - 1.0
        state = np.concatenate([v, q, pwm_norm])
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def update_and_predict(self, v, q, pwm, acc_meas, s_err):
        # 1. 网络推理
        state_tensor = self.preprocess_state(v, q, self.last_pwm)
        with torch.no_grad():
            phi = self.model(state_tensor) 
        self.last_pwm = pwm 

        # 2. 计算残差力
        pwm_mean = np.mean(pwm)
        if pwm_mean > 1.0: pwm_mean = 1.0
        thrust_mag = (pwm_mean / HOVER_THR) * MASS * G_VAL
        
        r_mat = self._quat_to_rot_matrix(q)
        t_body = np.array([0, 0, -thrust_mag])
        t_world = r_mat @ t_body
        
        y_force = MASS * acc_meas - (np.array([0, 0, MASS * G_VAL]) + t_world)
        y_norm = torch.FloatTensor(y_force).unsqueeze(0).to(device) / FORCE_SCALE
        s_tensor = torch.FloatTensor(s_err).unsqueeze(0).to(device)

        # 3. 自适应更新
        phi_t = phi.t()
        pred_y = torch.mm(phi, self.a_hat)
        pred_error = y_norm - pred_y 
        
        term_pred = torch.mm(self.P, phi_t) * (1.0/R_GAIN) * pred_error
        term_track = torch.mm(self.P, phi_t) * s_tensor * 0.1 
        
        self.a_hat = self.a_hat + (term_pred + term_track) * DT
        
        # 4. 计算输出
        comp_force_norm = torch.mm(phi, self.a_hat)
        comp_force = comp_force_norm.cpu().numpy()[0] * FORCE_SCALE
        comp_acc = comp_force / MASS
        
        # 全向限幅
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

# ================= 主循环 =================
async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected: break
    print("Done.")

    nf_ctrl = NeuralFlyController()
    
    # 状态缓存
    state_cache = {
        'v': np.zeros(3),
        'p': np.zeros(3),
        'q': np.array([1.0, 0.0, 0.0, 0.0]),
        'pwm': np.array([HOVER_THR]*4)
    }

    # 协程: Position/Velocity
    async def update_pos_vel():
        try:
            async for packet in drone.telemetry.position_velocity_ned():
                state_cache['p'] = np.array([packet.position.north_m, packet.position.east_m, packet.position.down_m])
                state_cache['v'] = np.array([packet.velocity.north_m_s, packet.velocity.east_m_s, packet.velocity.down_m_s])
        except Exception as e: print(f"Pos error: {e}")
    
    # 协程: Attitude
    async def update_attitude():
        try:
            async for q in drone.telemetry.attitude_quaternion():
                state_cache['q'] = np.array([q.w, q.x, q.y, q.z])
        except Exception as e: print(f"Att error: {e}")

    # 协程: Throttle
    async def update_throttle():
        try:
            async for hud in drone.telemetry.vfr_hud():
                thr_01 = hud.throttle / 100.0
                state_cache['pwm'] = np.array([thr_01]*4)
        except Exception as e:
            # 容错：只打印一次警告，不刷屏
            pass 

    asyncio.ensure_future(update_pos_vel())
    asyncio.ensure_future(update_attitude())
    asyncio.ensure_future(update_throttle())

    # === [关键新增] 等待无人机 GPS 就绪 ===
    print("Waiting for Global Position Estimate (GPS Lock)...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            print("-- Home position OK")
            break
        # 打印一点提示让用户知道没卡死
        print(".", end="", flush=True)
        await asyncio.sleep(1)
    print("\nDrone is Ready!")

    # 设定初始设定点 (必须在 Start Offboard 之前发送)
    print("-- Setting Initial Setpoint")
    setpoint_pos = np.array([0.0, 0.0, -1.5])
    await drone.offboard.set_position_ned(PositionNedYaw(*setpoint_pos, 0.0))

    # 启动 Offboard
    print("-- Starting Offboard Mode")
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        return

    # 最后解锁 (Arming)
    print("-- Arming")
    try:
        await drone.action.arm()
    except Exception as e:
        print(f"Arming failed: {e}")
        print("Tip: Check PX4 console for 'Preflight Fail' messages.")
        return

    print("Taking off (5s)...")
    await asyncio.sleep(5)
    
    # 变量初始化
    last_time = time.time()
    last_vel = np.zeros(3)
    
    print(">>> 3D Omnidirectional Adaptive Control Started <<<")
    
    while True:
        loop_start = time.time()
        
        curr_v = state_cache['v']
        curr_p = state_cache['p']
        curr_q = state_cache['q']
        curr_pwm = state_cache['pwm']

        dt = time.time() - last_time
        if dt > 0.001: 
            acc_meas = (curr_v - last_vel) / dt
        else: acc_meas = np.zeros(3)
        last_vel = curr_v
        last_time = time.time()
        
        s = (curr_v - 0) + 2.0 * (curr_p - setpoint_pos)
        
        comp_acc = nf_ctrl.update_and_predict(curr_v, curr_q, curr_pwm, acc_meas, s)
        
        acc_mag = np.linalg.norm(comp_acc[:2]) 
        print(f"\r Comp_N: {comp_acc[0]:5.2f} | Comp_E: {comp_acc[1]:5.2f} | Total: {acc_mag:5.2f}", end="")
        
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(*setpoint_pos, 0.0),
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0),
            AccelerationNed(-comp_acc[0], -comp_acc[1], -comp_acc[2]) 
        )
        
        elapsed = time.time() - loop_start
        if elapsed < DT:
            await asyncio.sleep(DT - elapsed)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt: pass