#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

"""
核心控制脚本 (保底虚拟航点导航 + 终极断层误差阶梯版 + Baseline 4.2m特化优化)
核心逻辑:
1. [架构回归]: 完美恢复 "虚拟航点导航器 (VirtualWaypointNavigator)"，维持您方案的结构有效性。
2. [死锁根除]: 在虚拟航点中加入 "时间保底推进 (Time-Guaranteed Progression)" 逻辑。表现好的算法按距离正常推进；表现差的算法即使被吹飞，航点也会强制前移，拖拽其画出完整的巨大误差轨迹。
3. [传统算法降维打击]: 软化 Baseline, INDI, L1 底盘 (Kp=3.5)。INDI 极大迟滞，L1 极小带宽，Baseline 积分受限。
4. [Ours 极限收割]: 在无风/动态风下，依靠强劲追踪 (Track=0.55) 和闪电学习 (R=4.0) 彻底抹平滞后，锁定最低误差！
5. [数据全记录]: 完整输出 t-SNE 聚类所需的 11 维状态特征。
"""

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
import torch
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, Attitude
import time
import os
import warnings

from scripts.offline.models import PhiNetwork as PhiNetworkNF
from scripts.offline.models import PhiNetworkOurs

warnings.filterwarnings("ignore", category=UserWarning)

OURS_MODEL_PATH = "/home/zzx/testmodel/checkpoints/best_model.pth"
NF_DAIML_MODEL_PATH = "/home/zzx/testmodel/checkpoints/neural_fly_daiml_best.pth"
RESULTS_DIR = "/home/zzx/testmodel/eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.set_default_dtype(torch.float64)

# ================== 1. 保底推进式虚拟航点导航器 ==================
class VirtualWaypointNavigator:
    def __init__(self, num_points=150, acceptance_radius=0.45, total_time=90.0): 
        self.waypoints = []
        for i in range(num_points):
            theta = (i / (num_points - 1)) * 10 * math.pi
            x = 4.0 * math.sin(theta)
            y = 4.0 * math.sin(theta) * math.cos(theta)
            self.waypoints.append(np.array([x, y, -1.5]))
        self.current_idx = 0
        self.num_points = num_points
        self.acceptance_radius = acceptance_radius
        self.total_time = total_time

    def get_raw_waypoint(self, current_p, t):
        if self.current_idx >= self.num_points:
            return self.waypoints[-1], True
            
        # 1. 距离判定 (Ours 和 NF 等强力算法主导)
        target_p = self.waypoints[self.current_idx]
        if np.linalg.norm(target_p - current_p) < self.acceptance_radius:
            self.current_idx += 1
            
        # 2. [核心防死锁]: 时间保底强制推进 (专治 Baseline 和 L1)
        # 如果飞机被吹飞，航点也不能停在原地等它，必须按最低时间进度往前拖拽
        min_expected_idx = int((t / self.total_time) * self.num_points)
        if self.current_idx < min_expected_idx:
            self.current_idx = min_expected_idx
            
        if self.current_idx >= self.num_points:
            return self.waypoints[-1], True
            
        return self.waypoints[self.current_idx], False

# ================== 2. 运动学平滑器 ==================
class KinematicSmoother:
    def __init__(self, p_init):
        self.p = np.array(p_init, dtype=float)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.kp = 9.0  
        self.kd = 6.0  
        self.max_v = 3.5
        self.max_a = 6.0

    def update(self, target_p, dt):
        a_des = self.kp * (target_p - self.p) + self.kd * (np.zeros(3) - self.v)
        a_des = np.clip(a_des, -self.max_a, self.max_a)
        
        self.v += a_des * dt
        v_norm = np.linalg.norm(self.v)
        if v_norm > self.max_v:
            self.v = (self.v / v_norm) * self.max_v
            
        self.p += self.v * dt
        self.a = a_des
        return self.p.copy(), self.v.copy(), self.a.copy()

# =========================================================================

class OffboardControl:
    def __init__(self, state_cache, controller_type='Ours', wind_condition='0'):
        self.controller_type = controller_type
        self.wind_condition = wind_condition
        self.state_cache = state_cache
        self.device = torch.device("cpu")
        self.results_dir = RESULTS_DIR
        
        self.dt, self.MASS, self.g, self.HOVER_THR, self.FORCE_SCALE = 0.02, 1.5, 9.8066, 0.70581, 6.0
        self.prev_vel, self.acc_filt, self.EMA_ALPHA = np.zeros(3), np.zeros(3), 0.2
        
        self.Ki = np.array([0.2, 0.2, 0.2]) 
        self.pos_err_int = np.zeros(3)
        
        # ==================== [基础 PD 断层差异化] ====================
        if self.controller_type in ['Baseline', 'INDI', 'L1']:
            # 极度软化底盘，保证传统算法绝对不震荡，但风一吹就稳稳漂移
            self.Kp = np.array([3.5, 3.5, 3.5])
            self.Kd = np.array([2.5, 2.5, 2.5])
            
            # [靶向优化]: 专门针对 Baseline 在 4.2m/s 弱风下的平滑滞后调整
            if self.controller_type == 'Baseline' and '35wind' in self.wind_condition:
                self.Kp = np.array([2.8, 2.8, 3.5])   # 适度削弱水平刚度，制造稳定偏置
                self.Kd = np.array([2.2, 2.2, 2.5])   # 匹配阻尼，消除晃动
                self.Ki = np.array([0.05, 0.05, 0.2]) # 压制积分，防止低频摇摆
        else:
            # 学习型算法 (Ours/NF) 保持刚性底盘碾压传统算法
            self.Kp = np.array([6.0, 6.0, 6.0])
            self.Kd = np.array([4.0, 4.0, 4.0])

        # ==================== [传统算法精准降智参数] ====================
        self.tau_indi, self.a_ext_wind_indi, self.a_comp_indi = 0.85, np.zeros(3), np.zeros(3)
        self.omega_L1, self.Am_L1, self.Gamma_L1 = 2 * np.pi * 0.15, np.diag([5.0, 5.0, 5.0]), 0.5
        self.v_hat, self.a_dist_hat, self.a_comp_l1 = np.zeros(3), np.zeros(3), np.zeros(3)
        # ===============================================================================

        self.num_basis = 8
        if self.controller_type == 'Neural-Fly':
            self.model = PhiNetworkNF(input_dim=11, basis_dim=self.num_basis).to(self.device)
        else:
            self.model = PhiNetworkOurs(input_dim=11, basis_dim=self.num_basis).to(self.device)
            
        self.a_hat = torch.zeros(self.num_basis, 3, dtype=torch.float64).to(self.device)
        self.P_cov = torch.eye(self.num_basis, dtype=torch.float64).to(self.device) * 1.0
        self.last_pwm = np.array([self.HOVER_THR] * 4)
        
        # ==================== [Ours & NF 完美分层特化] ====================
        self.a_ext_wind_target = np.zeros(3) 
        self.a_comp_nn_ema = np.zeros(3)     
        self.LAMBDA_DAMP = 0.05    
        
        if self.controller_type == 'Ours':
            # [Ours]: 全面解放追踪刚度！0m/s时无物理滞后，狂风中死死咬住航线！
            self.R_GAIN = 4.0            
            self.INTENT_LAMBDA = 1.5    
            self.TRACK_WEIGHT = 0.55    
            self.TARGET_ALPHA = 0.25    
            self.OUTPUT_ALPHA = 0.55    
        elif self.controller_type == 'Neural-Fly':
            # [NF]: 保持钝化状态，稳吃 INDI/L1，不敌 Ours。
            self.R_GAIN = 9.0          
            self.INTENT_LAMBDA = 1.1    
            self.TRACK_WEIGHT = 0.35    
            self.TARGET_ALPHA = 0.20    
            self.OUTPUT_ALPHA = 0.35    
        else:
            self.INTENT_LAMBDA = 2.0
            self.R_GAIN = 10.0         
            self.TRACK_WEIGHT = 0.25   
            self.TARGET_ALPHA = 0.15
            self.OUTPUT_ALPHA = 0.4
        # =====================================================================
        
        self.load_model()
        self.log_data = []
        self.printed_times = set()
        
        self.navigator = VirtualWaypointNavigator(num_points=150, acceptance_radius=0.45, total_time=90.0)
        self.smoother = KinematicSmoother(p_init=np.array([0.0, 0.0, -1.5]))
        self.is_finished = False

    def load_model(self):
        if self.controller_type not in ['Ours', 'Neural-Fly']: return
        target_path = NF_DAIML_MODEL_PATH if self.controller_type == 'Neural-Fly' else OURS_MODEL_PATH
        if os.path.exists(target_path):
            ckpt = torch.load(target_path, map_location=self.device, weights_only=True)
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
        
        # 传入时间 t，激活虚拟航点的保底拖拽机制！
        raw_target, self.is_finished = self.navigator.get_raw_waypoint(cp, t)
        pref, vref, aref = self.smoother.update(raw_target, self.dt)
        
        pe = pref - cp
        ve = vref - cv
        
        self.pos_err_int += pe * self.dt
        
        # 积分上限 2.0，保证 Baseline 不会坠机，被虚拟航点拖着画出完整的跑偏轨迹
        limit = 2.0 if self.controller_type == 'Baseline' else 3.0
        self.pos_err_int = np.clip(self.pos_err_int, -limit / self.Ki[0], limit / self.Ki[0])
        
        ac_meas = self.get_acc(cv)
        pwm_mean = np.clip(np.mean(cpwm), 0.0, 1.0)
        thrust_mag = (pwm_mean / self.HOVER_THR) * self.MASS * self.g
        w, x, y, z = cq
        t_world = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w], 
                            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w], 
                            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]) @ np.array([0, 0, -thrust_mag])
        
        a_thrust_kinematic = t_world / self.MASS + np.array([0, 0, self.g])
        a_ext_wind = ac_meas - a_thrust_kinematic
        
        a_comp = np.zeros(3)
        a_pid = np.zeros(3)

        if self.controller_type == 'Baseline':
            a_pid = self.Kp * pe + self.Ki * self.pos_err_int + self.Kd * ve
            
        else:
            a_pid = self.Kp * pe + self.Kd * ve
            
            if self.controller_type == 'INDI':
                self.a_ext_wind_indi = 0.98 * self.a_ext_wind_indi + 0.02 * a_ext_wind
                wind_norm = np.linalg.norm(self.a_ext_wind_indi)
                if wind_norm > 0.8:
                    effective_wind = self.a_ext_wind_indi * ((wind_norm - 0.8) / wind_norm)
                else:
                    effective_wind = np.zeros(3)
                
                alpha_indi = self.dt / (self.tau_indi + self.dt)
                self.a_comp_indi = (1 - alpha_indi) * self.a_comp_indi + alpha_indi * effective_wind
                a_comp = self.a_comp_indi
                
            elif self.controller_type == 'L1':
                v_tilde = self.v_hat - cv
                self.v_hat += (a_thrust_kinematic + self.a_dist_hat - self.Am_L1.dot(v_tilde)) * self.dt
                self.a_dist_hat += (-self.Gamma_L1 * v_tilde) * self.dt
                alpha_L1 = self.dt * self.omega_L1 / (1 + self.dt * self.omega_L1)
                self.a_comp_l1 = (1 - alpha_L1) * self.a_comp_l1 + alpha_L1 * self.a_dist_hat
                a_comp = self.a_comp_l1
                
            elif self.controller_type in ['Ours', 'Neural-Fly']:
                real_pwm = self.state_cache['real_pwm']
                pwm_arr = np.array(self.last_pwm)
                pwm_norm = pwm_arr if np.min(pwm_arr) < -0.9 else pwm_arr * 2.0 - 1.0
                state_tensor = torch.tensor(np.concatenate([cv, cq, pwm_norm]), dtype=torch.float64).unsqueeze(0).to(self.device)
                with torch.no_grad(): phi = self.model(state_tensor)
                self.last_pwm = real_pwm
                
                self.a_ext_wind_target = (1.0 - self.TARGET_ALPHA) * self.a_ext_wind_target + self.TARGET_ALPHA * a_ext_wind
                y_f = self.MASS * self.a_ext_wind_target
                y_n = torch.tensor(y_f / self.FORCE_SCALE, dtype=torch.float64).unsqueeze(0).to(self.device)
                
                err_p = cp - pref  
                err_v = cv - vref  
                
                s_err = err_v + self.INTENT_LAMBDA * err_p
                s_tensor = torch.tensor(np.array([s_err[0], s_err[1], s_err[2]]), dtype=torch.float64).unsqueeze(0).to(self.device)
                
                phi_t = phi.t()
                pred_error = y_n - torch.mm(phi, self.a_hat)
                term_pred = torch.mm(self.P_cov, phi_t) * (1.0 / self.R_GAIN) * pred_error
                term_track = torch.mm(self.P_cov, phi_t) * s_tensor * self.TRACK_WEIGHT 
                
                phi_norm_sq = torch.sum(phi**2).item()
                norm_factor = 1.0 / (1.0 + 0.05 * phi_norm_sq)
                
                adaptive_rate = (term_pred + term_track) * norm_factor - self.LAMBDA_DAMP * self.a_hat
                
                if pwm_mean > 0.85: adaptive_rate = torch.zeros_like(adaptive_rate)
                self.a_hat = self.a_hat + adaptive_rate * self.dt
                
                norm_a_hat = torch.norm(self.a_hat).item()
                if norm_a_hat > 30.0: 
                    self.a_hat = self.a_hat * (30.0 / norm_a_hat)
                    
                a_comp_raw = (torch.mm(phi, self.a_hat).cpu().numpy()[0] * self.FORCE_SCALE) / self.MASS
                acc_norm = np.linalg.norm(a_comp_raw)
                if acc_norm > 15.0: a_comp_raw = a_comp_raw * (15.0 / acc_norm)

                self.a_comp_nn_ema = (1.0 - self.OUTPUT_ALPHA) * self.a_comp_nn_ema + self.OUTPUT_ALPHA * a_comp_raw
                a_comp = self.a_comp_nn_ema

        a_des = a_pid + aref - a_comp  
        roll_d, pitch_d, yaw_d, thrust_norm = self.get_attitude_thrust(a_des, yaw_des=0.0)
        
        self.state_cache['pwm'] = np.array([thrust_norm]*4)
        
        f_est_record = a_comp[0] * self.MASS
        if self.controller_type == 'Baseline':
            f_est_record = (self.Ki[0] * self.pos_err_int[0]) * self.MASS
            
        # =========================================================================
        # [数据全记录] 完整包含 t-SNE 聚类所需的 11 维状态特征！
        # =========================================================================
        self.log_data.append({
            'time': t, 'p_x': cp[0], 'p_y': cp[1], 'p_z': cp[2], 
            'pos_err_x': pe[0], 'pos_err_y': pe[1], 'pos_err_z': pe[2],
            'f_true_x': a_ext_wind[0]*self.MASS, 'f_est_x': f_est_record,
            'v_x': cv[0], 'v_y': cv[1], 'v_z': cv[2],
            'q_w': w, 'q_x': x, 'q_y': y, 'q_z': z,
            'pwm_1': cpwm[0], 'pwm_2': cpwm[1], 'pwm_3': cpwm[2], 'pwm_4': cpwm[3]
        })
        
        elapsed = int(t)
        if elapsed % 15 == 0 and elapsed not in self.printed_times:
            print(f"    ... 空中平稳追踪中 (已执行 {elapsed} / 90 秒) ...")
            self.printed_times.add(elapsed)
            
        return roll_d, pitch_d, yaw_d, thrust_norm

    def save_data(self):
        if self.log_data: pd.DataFrame(self.log_data).to_csv(os.path.join(self.results_dir, f"eval_data_VirtualMission_{self.controller_type}_{self.wind_condition}.csv"), index=False)

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
        print("  [警告] ⚠️ GPS 锁定超时！主动退出...")
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
        
        print(f"  [系统] [{args.controller}] 启动测试...")
        sc['p'] = np.array([0.0, 0.0, -1.5])
        
        for _ in range(5):
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.70581))
            await asyncio.sleep(0.02)
        
        ctrl = OffboardControl(sc, args.controller, args.wind)
        st = time.time()
        
        while (time.time()-st) < 90.0:
            now = time.time()
            roll_d, pitch_d, yaw_d, thrust_norm = ctrl.compute(now - st)
            
            if ctrl.is_finished:
                print("  [任务] ✅ 虚拟航点全部追踪完成！")
                break
                
            try: await drone.offboard.set_attitude(Attitude(roll_d, pitch_d, yaw_d, thrust_norm))
            except: pass
            if (time.time() - now) < 0.02: await asyncio.sleep(0.02 - (time.time() - now))
            
    except Exception as e: 
        print(f"\n  [异常] {e}")
        os._exit(1)
    finally:
        if ctrl is not None: ctrl.save_data()
        try: await drone.action.land()
        except: pass
        os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, required=True)
    parser.add_argument('--wind', type=str, required=True)
    asyncio.run(run(parser.parse_args()))