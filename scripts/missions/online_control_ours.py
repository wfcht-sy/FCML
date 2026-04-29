#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
import torch
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, Attitude
import time
import warnings

from scripts.offline.models import PhiNetworkOurs
from virtual_navigator import VirtualWaypointNavigator
from kinematic_smoother import KinematicSmoother

warnings.filterwarnings("ignore", category=UserWarning)

OURS_MODEL_PATH = "/home/zzx/testmodel/checkpoints/best_model.pth"
RESULTS_DIR = "/home/zzx/testmodel/eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.set_default_dtype(torch.float64)

# ================== 核心在线自适应控制器 ==================
class OffboardControl:
    def __init__(self, state_cache, wind_condition='0'):
        self.wind_condition = wind_condition
        self.state_cache = state_cache
        self.device = torch.device("cpu")
        self.results_dir = RESULTS_DIR
        
        # 物理与控制基础参数
        self.dt, self.MASS, self.g, self.HOVER_THR, self.FORCE_SCALE = 0.02, 1.5, 9.8066, 0.70581, 6.0
        self.prev_vel, self.acc_filt, self.EMA_ALPHA = np.zeros(3), np.zeros(3), 0.2
        
        # 标称底层控制器 (PD) 参数
        self.Kp = np.array([6.0, 6.0, 6.0])
        self.Kd = np.array([4.0, 4.0, 4.0])

        # 神经网络与自适应律超参数
        self.num_basis = 8
        self.model = PhiNetworkOurs(input_dim=11, basis_dim=self.num_basis).to(self.device)
        self.a_hat = torch.zeros(self.num_basis, 3, dtype=torch.float64).to(self.device)
        self.P_cov = torch.eye(self.num_basis, dtype=torch.float64).to(self.device) * 1.0
        self.last_pwm = np.array([self.HOVER_THR] * 4)
        
        self.a_ext_wind_target = np.zeros(3) 
        self.a_comp_nn_ema = np.zeros(3)     
        
        self.R_GAIN = 4.0            
        self.INTENT_LAMBDA = 1.5     
        self.TRACK_WEIGHT = 0.55     
        self.TARGET_ALPHA = 0.25     
        self.OUTPUT_ALPHA = 0.55     
        self.LAMBDA_DAMP = 0.05      
        
        self.load_model()
        self.log_data = []
        self.printed_times = set()
        
        self.navigator = VirtualWaypointNavigator(num_points=150, acceptance_radius=0.45, total_time=90.0)
        self.smoother = KinematicSmoother(p_init=np.array([0.0, 0.0, -1.5]))
        self.is_finished = False

    def load_model(self):
        if os.path.exists(OURS_MODEL_PATH):
            ckpt = torch.load(OURS_MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            self.model.eval()

    def get_acc(self, cv):
        self.acc_filt = (1.0 - self.EMA_ALPHA) * self.acc_filt + self.EMA_ALPHA * ((cv - self.prev_vel) / self.dt)
        self.prev_vel = cv.copy()
        return self.acc_filt

    def get_attitude_thrust(self, a_des, yaw_des=0.0):
        f_sp = a_des - np.array([0.0, 0.0, self.g])
        T = np.linalg.norm(f_sp)
        if T < 0.01: return 0.0, 0.0, yaw_des, 0.0
        
        z_b = -f_sp / T
        x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
        y_b = np.cross(z_b, x_c)
        if np.linalg.norm(y_b) < 1e-6: y_b = np.array([-math.sin(yaw_des), math.cos(yaw_des), 0.0])
        else: y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        
        pitch = math.asin(np.clip(-x_b[2], -1.0, 1.0))
        roll = math.atan2(y_b[2], z_b[2])
        yaw = math.atan2(x_b[1], x_b[0])
        thrust_norm = np.clip((T / self.g) * self.HOVER_THR, 0.0, 1.0)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw), thrust_norm

    def compute(self, t):
        cp, cv, cq, cpwm = self.state_cache['p'], self.state_cache['v'], self.state_cache['q'], self.state_cache['pwm']
        
        # 1. 轨迹生成与平滑
        raw_target, self.is_finished = self.navigator.get_raw_waypoint(cp, t)
        pref, vref, aref = self.smoother.update(raw_target, self.dt)
        
        pe = pref - cp
        ve = vref - cv
        
        # 2. 气动力预测与真实力估计
        ac_meas = self.get_acc(cv)
        pwm_mean = np.clip(np.mean(cpwm), 0.0, 1.0)
        thrust_mag = (pwm_mean / self.HOVER_THR) * self.MASS * self.g
        w, x, y, z = cq
        t_world = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w], 
                            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w], 
                            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]) @ np.array([0, 0, -thrust_mag])
        
        a_thrust_kinematic = t_world / self.MASS + np.array([0, 0, self.g])
        a_ext_wind = ac_meas - a_thrust_kinematic
        
        # 3. 标称反馈控制律 (PD)
        a_pid = self.Kp * pe + self.Kd * ve
            
        # 4. 神经网络自适应前馈补偿
        real_pwm = self.state_cache['real_pwm']
        pwm_arr = np.array(self.last_pwm)
        pwm_norm = pwm_arr if np.min(pwm_arr) < -0.9 else pwm_arr * 2.0 - 1.0
        state_tensor = torch.tensor(np.concatenate([cv, cq, pwm_norm]), dtype=torch.float64).unsqueeze(0).to(self.device)
        
        with torch.no_grad(): phi = self.model(state_tensor)
        self.last_pwm = real_pwm
        
        self.a_ext_wind_target = (1.0 - self.TARGET_ALPHA) * self.a_ext_wind_target + self.TARGET_ALPHA * a_ext_wind
        y_f = self.MASS * self.a_ext_wind_target
        y_n = torch.tensor(y_f / self.FORCE_SCALE, dtype=torch.float64).unsqueeze(0).to(self.device)
        
        s_err = ve + self.INTENT_LAMBDA * pe
        s_tensor = torch.tensor(np.array([s_err[0], s_err[1], s_err[2]]), dtype=torch.float64).unsqueeze(0).to(self.device)
        
        phi_t = phi.t()
        pred_error = y_n - torch.mm(phi, self.a_hat)
        term_pred = torch.mm(self.P_cov, phi_t) * (1.0 / self.R_GAIN) * pred_error
        term_track = torch.mm(self.P_cov, phi_t) * s_tensor * self.TRACK_WEIGHT 
        
        # 特征归一化与阻尼泄露约束
        phi_norm_sq = torch.sum(phi**2).item()
        norm_factor = 1.0 / (1.0 + 0.05 * phi_norm_sq)
        adaptive_rate = (term_pred + term_track) * norm_factor - self.LAMBDA_DAMP * self.a_hat
        
        if pwm_mean > 0.85: adaptive_rate = torch.zeros_like(adaptive_rate)
        self.a_hat = self.a_hat + adaptive_rate * self.dt
        
        # 参数投影算子与输出截断
        norm_a_hat = torch.norm(self.a_hat).item()
        if norm_a_hat > 30.0: 
            self.a_hat = self.a_hat * (30.0 / norm_a_hat)
            
        a_comp_raw = (torch.mm(phi, self.a_hat).cpu().numpy()[0] * self.FORCE_SCALE) / self.MASS
        acc_norm = np.linalg.norm(a_comp_raw)
        if acc_norm > 15.0: a_comp_raw = a_comp_raw * (15.0 / acc_norm)

        # 一阶低通滤波
        self.a_comp_nn_ema = (1.0 - self.OUTPUT_ALPHA) * self.a_comp_nn_ema + self.OUTPUT_ALPHA * a_comp_raw
        a_comp = self.a_comp_nn_ema

        # 5. 总加速度指令与姿态解算
        a_des = a_pid + aref - a_comp  
        roll_d, pitch_d, yaw_d, thrust_norm = self.get_attitude_thrust(a_des, yaw_des=0.0)
        
        self.state_cache['pwm'] = np.array([thrust_norm]*4)
        
        # 数据记录 (仅保留必要特征)
        self.log_data.append({
            'time': t, 'p_x': cp[0], 'p_y': cp[1], 'p_z': cp[2], 
            'pos_err_x': pe[0], 'pos_err_y': pe[1], 'pos_err_z': pe[2],
            'f_true_x': a_ext_wind[0]*self.MASS, 'f_est_x': a_comp[0]*self.MASS,
            'v_x': cv[0], 'v_y': cv[1], 'v_z': cv[2],
            'q_w': w, 'q_x': x, 'q_y': y, 'q_z': z,
            'pwm_1': cpwm[0], 'pwm_2': cpwm[1], 'pwm_3': cpwm[2], 'pwm_4': cpwm[3]
        })
        
        elapsed = int(t)
        if elapsed % 15 == 0 and elapsed not in self.printed_times:
            print(f"    ... 追踪执行中 (已完成 {elapsed} / 90 秒) ...")
            self.printed_times.add(elapsed)
            
        return roll_d, pitch_d, yaw_d, thrust_norm

    def save_data(self):
        if self.log_data: pd.DataFrame(self.log_data).to_csv(os.path.join(self.results_dir, f"eval_data_Ours_{self.wind_condition}.csv"), index=False)

async def run(args):
    drone = System()
    try: await asyncio.wait_for(drone.connect(system_address="udpin://0.0.0.0:14540"), timeout=10.0)
    except: os._exit(1)

    sc = {'v': np.zeros(3), 'p': np.zeros(3), 'q': np.array([1.0, 0, 0, 0]), 'pwm': np.array([0.70581]*4), 'real_pwm': np.array([0.70581]*4)}
    async def u_pv():
        async for x in drone.telemetry.position_velocity_ned(): sc['p'], sc['v'] = np.array([x.position.north_m, x.position.east_m, x.position.down_m]), np.array([x.velocity.north_m_s, x.velocity.east_m_s, x.velocity.down_m_s])
    async def u_at():
        async for x in drone.telemetry.attitude_quaternion(): sc['q'] = np.array([x.w, x.x, x.y, x.z])
    async def u_throttle():
        try:
            async for hud in drone.telemetry.vfr_hud(): sc['real_pwm'] = np.array([hud.throttle / 100.0] * 4)
        except Exception: pass

    asyncio.ensure_future(u_pv()); asyncio.ensure_future(u_at()); asyncio.ensure_future(u_throttle())

    print("  [系统] 正在等待 GPS 与 Home 点锁定...")
    try:
        async def wait_gps():
            async for h in drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok: return
        await asyncio.wait_for(wait_gps(), timeout=30.0)
    except asyncio.TimeoutError:
        print("  [警告] GPS 锁定超时，系统退出。")
        os._exit(1) 

    try:
        await drone.telemetry.set_rate_position_velocity_ned(50.0); await drone.telemetry.set_rate_attitude(50.0)
    except: pass

    ctrl = None
    try:
        for _ in range(10):
            await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
            await asyncio.sleep(0.05)
            
        await drone.offboard.start()
        for attempt in range(15):
            try:
                await drone.action.arm(); break
            except:
                await asyncio.sleep(0.4)
                await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
        await asyncio.sleep(3) 
        
        print(f"  [系统] 自适应控制算法测试启动...")
        sc['p'] = np.array([0.0, 0.0, -1.5])
        
        for _ in range(5):
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.70581))
            await asyncio.sleep(0.02)
        
        ctrl = OffboardControl(sc, args.wind)
        st = time.time()
        
        while (time.time()-st) < 90.0:
            now = time.time()
            roll_d, pitch_d, yaw_d, thrust_norm = ctrl.compute(now - st)
            
            if ctrl.is_finished:
                print("  [任务] 虚拟航点序列追踪完成。")
                break
                
            try: await drone.offboard.set_attitude(Attitude(roll_d, pitch_d, yaw_d, thrust_norm))
            except: pass
            if (time.time() - now) < 0.02: await asyncio.sleep(0.02 - (time.time() - now))
            
    except Exception as e: 
        print(f"\n  [异常] 运行时错误: {e}")
        os._exit(1)
    finally:
        if ctrl is not None: ctrl.save_data()
        try: await drone.action.land()
        except: pass
        os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=str, default='0', help='Wind condition label for logging')
    asyncio.run(run(parser.parse_args()))
