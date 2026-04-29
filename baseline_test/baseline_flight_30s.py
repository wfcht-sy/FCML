#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯原始 PID (Baseline) 30秒性能测试脚本 - 外环几何姿态注入版
核心功能: 
1. 废除位置环：不再使用 PVA 接口，彻底绕开 PX4 的位置与速度平滑滤波器。
2. 几何追踪：Python 端直接运算 PD + 运动学前馈，算出期望加速度并转为姿态 (Attitude) 注入。
3. 验证极限：无风状态下将达到物理追踪极限 (近乎零误差)。
"""

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
from mavsdk import System
# [核心修改]: 引入 Attitude 接口
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, Attitude
import time
import os
import traceback

RESULTS_DIR = "/home/zzx/testmodel/eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class BaselineControl:
    def __init__(self, state_cache, wind_condition='0'):
        self.wind_condition = wind_condition
        self.state_cache = state_cache
        self.results_dir = RESULTS_DIR
        self.log_data = []

        # 物理常数与悬停推力
        self.g = 9.8066
        self.HOVER_THR = 0.70581
        
        # 既然绕开了飞控自带的臃肿滤波器，Python 端可以放心使用高刚度！
        self.Kp = np.array([5.0, 5.0, 5.0])
        self.Kd = np.array([3.5, 3.5, 3.5])

    # 4.0m 大轨迹数学方程 (P, V, A 严格对齐)
    def get_target_position(self, t):
        return np.array([4.0 * math.sin(0.5 * t), 4.0 * math.sin(0.5 * t) * math.cos(0.5 * t), -1.5])
    def get_target_velocity(self, t):
        return np.array([2.0 * math.cos(0.5 * t), 2.0 * (math.cos(0.5 * t)**2 - math.sin(0.5 * t)**2), 0.0])
    def get_target_acceleration(self, t):
        return np.array([-1.0 * math.sin(0.5 * t), -2.0 * math.sin(t), 0.0])

    # ================== [核心：期望加速度 -> 期望姿态反解] ==================
    def get_attitude_thrust(self, a_des, yaw_des=0.0):
        """将物理期望加速度转化为直接驱动飞控的欧拉角与归一化推力"""
        f_sp = a_des - np.array([0.0, 0.0, self.g])
        T = np.linalg.norm(f_sp)
        
        if T < 0.01: return 0.0, 0.0, yaw_des, 0.0
        
        z_b = -f_sp / T
        x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
        
        y_b = np.cross(z_b, x_c)
        if np.linalg.norm(y_b) < 1e-6:
            y_b = np.array([-math.sin(yaw_des), math.cos(yaw_des), 0.0])
        else:
            y_b /= np.linalg.norm(y_b)
            
        x_b = np.cross(y_b, z_b)
        
        pitch = math.asin(np.clip(-x_b[2], -1.0, 1.0))
        roll = math.atan2(y_b[2], z_b[2])
        yaw = math.atan2(x_b[1], x_b[0])
        
        thrust_norm = np.clip((T / self.g) * self.HOVER_THR, 0.0, 1.0)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw), thrust_norm

    def compute(self, t):
        cp = self.state_cache['p']
        cv = self.state_cache['v']  # 引入速度反馈
        
        pref = self.get_target_position(t)
        vref = self.get_target_velocity(t) 
        aref = self.get_target_acceleration(t)
        
        pe = pref - cp
        ve = vref - cv
        
        # 记录真实位置与误差 (兼容现有的画图脚本)
        self.log_data.append({'time': t, 'pos_err_x': pe[0], 'pos_err_y': pe[1], 'pos_err_z': pe[2], 'p_x': cp[0], 'p_y': cp[1], 'p_z': cp[2]})
        
        # 纯原始 PID 控制算出的期望加速度
        a_pd = self.Kp * pe + self.Kd * ve
        a_des = a_pd + aref
        
        # 转化为飞机姿态指令
        roll_d, pitch_d, yaw_d, thrust_norm = self.get_attitude_thrust(a_des, 0.0)
        return roll_d, pitch_d, yaw_d, thrust_norm

    def save_data(self):
        if self.log_data: 
            pd.DataFrame(self.log_data).to_csv(os.path.join(self.results_dir, f"eval_data_Baseline_{self.wind_condition}.csv"), index=False)

async def run(args):
    drone = System()
    try: await asyncio.wait_for(drone.connect(system_address="udpin://0.0.0.0:14540"), timeout=10.0)
    except: os._exit(1)

    # 增加速度缓存
    sc = {'p': np.zeros(3), 'v': np.zeros(3)}
    
    async def u_pv():
        async for x in drone.telemetry.position_velocity_ned(): 
            sc['p'] = np.array([x.position.north_m, x.position.east_m, x.position.down_m])
            sc['v'] = np.array([x.velocity.north_m_s, x.velocity.east_m_s, x.velocity.down_m_s])

    asyncio.ensure_future(u_pv())

    print("  [飞行器] 正在等待 GPS 锁定...")
    try:
        async for h in drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok: break
        
        # 强行拉升 MAVSDK 遥测反馈至 50Hz
        try:
            await drone.telemetry.set_rate_position_velocity_ned(50.0)
            await drone.telemetry.set_rate_attitude(50.0)
        except: pass
        
        print("  [飞行器] ✅ GPS 已锁定，就绪。")
    except Exception: os._exit(1)

    ctrl = None
    try:
        print("  [飞行器] 准备起飞 (先用 Position 模式悬停)...")
        for _ in range(10):
            await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
            await asyncio.sleep(0.05)
            
        await drone.offboard.start()
        
        for _ in range(15):
            try:
                await drone.action.arm()
                print("  [飞行器] ✅ 解锁成功！")
                break
            except: 
                await asyncio.sleep(0.4)
                await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
                
        await asyncio.sleep(3) 
        
        print("  [系统] ⚡ 切入直接姿态控制 (Attitude Injection)...")
        for _ in range(5):
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.70581))
            await asyncio.sleep(0.02)
        
        ctrl = BaselineControl(sc, args.wind)
        st = time.time()
        
        # [限制为 30 秒]
        while (time.time()-st) < 30.0:
            now = time.time()
            
            # [核心落实] 下发姿态，绕过位置环阻碍！
            roll_d, pitch_d, yaw_d, thrust_norm = ctrl.compute(now - st)
            try: 
                await drone.offboard.set_attitude(Attitude(roll_d, pitch_d, yaw_d, thrust_norm))
            except: pass
            
            if (time.time() - now) < 0.02: await asyncio.sleep(0.02 - (time.time() - now))
            
    except Exception as e: 
        print(f"\n  [异常] {e}")
        traceback.print_exc()
        os._exit(1)
    finally:
        if ctrl is not None: ctrl.save_data()
        try: await drone.action.land()
        except: pass
        os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=str, required=True)
    asyncio.run(run(parser.parse_args()))