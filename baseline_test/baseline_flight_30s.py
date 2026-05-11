#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Original PID (Baseline) 30s Performance Test Script - Outer-loop Geometric Attitude Injection
Core functions:
1. Bypass position loop: No longer use PVA interface, completely bypass PX4's position and velocity smoothing filters.
2. Geometric tracking: Calculate PD + kinematic feedforward directly in Python, compute expected acceleration and convert to Attitude injection.
3. Verify limits: Should approach physical tracking limits (near-zero error) in no-wind conditions.
"""

import asyncio
import numpy as np
import pandas as pd
import argparse
import math
from mavsdk import System
# Core modification: Introduce Attitude interface
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, AccelerationNed, Attitude
import time
import os
import traceback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EVAL_RESULTS_DIR
RESULTS_DIR = EVAL_RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

class BaselineControl:
    def __init__(self, state_cache, wind_condition='0'):
        self.wind_condition = wind_condition
        self.state_cache = state_cache
        self.results_dir = RESULTS_DIR
        self.log_data = []

        # Physical constants and hover thrust
        self.g = 9.8066
        self.HOVER_THR = 0.70581
        
        # Since we bypassed the bloated flight controller filters, we can use high stiffness in Python!
        self.Kp = np.array([5.0, 5.0, 5.0])
        self.Kd = np.array([3.5, 3.5, 3.5])

    # 4.0m large trajectory mathematical equation (P, V, A strictly aligned)
    def get_target_position(self, t):
        return np.array([4.0 * math.sin(0.5 * t), 4.0 * math.sin(0.5 * t) * math.cos(0.5 * t), -1.5])
    def get_target_velocity(self, t):
        return np.array([2.0 * math.cos(0.5 * t), 2.0 * (math.cos(0.5 * t)**2 - math.sin(0.5 * t)**2), 0.0])
    def get_target_acceleration(self, t):
        return np.array([-1.0 * math.sin(0.5 * t), -2.0 * math.sin(t), 0.0])

    # ================== [Core: Expected acceleration -> Expected attitude inverse kinematics] ==================
    def get_attitude_thrust(self, a_des, yaw_des=0.0):
        """Convert physical expected acceleration to Euler angles and normalized thrust to directly drive the flight controller"""
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
        cv = self.state_cache['v']  # Introduce velocity feedback
        
        pref = self.get_target_position(t)
        vref = self.get_target_velocity(t) 
        aref = self.get_target_acceleration(t)
        
        pe = pref - cp
        ve = vref - cv
        
        # Log real position and error (compatible with existing plotting scripts)
        self.log_data.append({'time': t, 'pos_err_x': pe[0], 'pos_err_y': pe[1], 'pos_err_z': pe[2], 'p_x': cp[0], 'p_y': cp[1], 'p_z': cp[2]})
        
        # Expected acceleration computed by pure original PID control
        a_pd = self.Kp * pe + self.Kd * ve
        a_des = a_pd + aref
        
        # Convert to aircraft attitude commands
        roll_d, pitch_d, yaw_d, thrust_norm = self.get_attitude_thrust(a_des, 0.0)
        return roll_d, pitch_d, yaw_d, thrust_norm

    def save_data(self):
        if self.log_data: 
            pd.DataFrame(self.log_data).to_csv(os.path.join(self.results_dir, f"eval_data_Baseline_{self.wind_condition}.csv"), index=False)

async def run(args):
    drone = System()
    try: await asyncio.wait_for(drone.connect(system_address="udpin://0.0.0.0:14540"), timeout=10.0)
    except: os._exit(1)

    # Add velocity cache
    sc = {'p': np.zeros(3), 'v': np.zeros(3)}
    
    async def u_pv():
        async for x in drone.telemetry.position_velocity_ned(): 
            sc['p'] = np.array([x.position.north_m, x.position.east_m, x.position.down_m])
            sc['v'] = np.array([x.velocity.north_m_s, x.velocity.east_m_s, x.velocity.down_m_s])

    asyncio.ensure_future(u_pv())

    print("  [System] Waiting for GPS lock...")
    try:
        async for h in drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok: break
        
        # Force MAVSDK telemetry feedback to 50Hz
        try:
            await drone.telemetry.set_rate_position_velocity_ned(50.0)
            await drone.telemetry.set_rate_attitude(50.0)
        except: pass
        
        print("  [System] GPS locked and ready.")
    except Exception: os._exit(1)

    ctrl = None
    try:
        print("  [System] Preparing for takeoff (hovering in Position mode first)...")
        for _ in range(10):
            await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
            await asyncio.sleep(0.05)
            
        await drone.offboard.start()
        
        for _ in range(15):
            try:
                await drone.action.arm()
                print("  [System] Arming successful!")
                break
            except: 
                await asyncio.sleep(0.4)
                await drone.offboard.set_position_velocity_acceleration_ned(PositionNedYaw(0.0, 0.0, -1.5, 0.0), VelocityNedYaw(0.0, 0.0, 0.0, 0.0), AccelerationNed(0,0,0))
                
        await asyncio.sleep(3) 
        
        print("  [System] Switching to direct Attitude Control injection...")
        for _ in range(5):
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.70581))
            await asyncio.sleep(0.02)
        
        ctrl = BaselineControl(sc, args.wind)
        st = time.time()
        
        # Limit to 30 seconds
        while (time.time()-st) < 30.0:
            now = time.time()
            
            # [Core execution] Issue attitude commands, bypassing position loop obstruction!
            roll_d, pitch_d, yaw_d, thrust_norm = ctrl.compute(now - st)
            try: 
                await drone.offboard.set_attitude(Attitude(roll_d, pitch_d, yaw_d, thrust_norm))
            except: pass
            
            if (time.time() - now) < 0.02: await asyncio.sleep(0.02 - (time.time() - now))
            
    except Exception as e: 
        print(f"\n  [Exception] {e}")
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