#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import math
import random
import subprocess
import sys
import os
import time
from typing import Tuple
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

FLIGHT_DURATION_SEC = 150
CONTROL_UPDATE_RATE_HZ = 50
BOUNDARY_LIMITS = {'x_min': -1.5, 'x_max': 1.5, 'y_min': -1.0, 'y_max': 1.0, 'z_min': -2.5, 'z_max': -1.0}
MAXIMUM_VELOCITY = 2.0

class Random2SplineTrajectory:
    def __init__(self, random_seed: int, start_pos: np.ndarray):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.current_start_pos = start_pos
        self.current_target_pos = start_pos
        self.segment_start_time = 0.0
        self.segment_duration = 0.1
        self._pick_next_waypoint(0.0)

    def _pick_next_waypoint(self, current_time_sec: float) -> None:
        self.current_start_pos = self.current_target_pos.copy()
        self.current_target_pos = np.array([
            random.uniform(BOUNDARY_LIMITS['x_min'], BOUNDARY_LIMITS['x_max']),
            random.uniform(BOUNDARY_LIMITS['y_min'], BOUNDARY_LIMITS['y_max']),
            random.uniform(BOUNDARY_LIMITS['z_min'], BOUNDARY_LIMITS['z_max'])
        ])
        distance_meters = float(np.linalg.norm(self.current_target_pos - self.current_start_pos))
        self.segment_duration = max(distance_meters / MAXIMUM_VELOCITY, 1.5)
        self.segment_start_time = current_time_sec

    def get_target_pose(self, current_time_sec: float) -> Tuple[float, float, float, float]:
        if current_time_sec >= self.segment_start_time + self.segment_duration:
            self._pick_next_waypoint(current_time_sec)
        tau = np.clip((current_time_sec - self.segment_start_time) / self.segment_duration, 0.0, 1.0)
        alpha = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
        interpolated_position = self.current_start_pos + (self.current_target_pos - self.current_start_pos) * alpha
        position_difference = self.current_target_pos - self.current_start_pos
        target_yaw_degrees = math.atan2(position_difference[1], position_difference[0]) * 180.0 / math.pi
        return (float(interpolated_position[0]), float(interpolated_position[1]), float(interpolated_position[2]), float(target_yaw_degrees))

async def execute_flight_mission(wind_parameters: list, random_seed: int) -> None:
    drone = System()
    print("  [飞行器] 正在连接 MAVSDK (udpin://0.0.0.0:14540)...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for state in drone.core.connection_state():
        if state.is_connected: break

    print("  [飞行器] 正在等待 GPS 锁定...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok: break
        await asyncio.sleep(1)

    subprocess.run(["bash", "./set_wind.sh"] + wind_parameters, check=False)

    print("  [飞行器] 解锁并起飞...")
    try: await drone.action.arm()
    except Exception: return

    await drone.action.set_takeoff_altitude(1.5)
    await drone.action.takeoff()

    async for position in drone.telemetry.position():
        if position.relative_altitude_m > 1.4: break
    await asyncio.sleep(2)

    initial_pos = np.array([0.0, 0.0, -1.5])
    trajectory_generator = Random2SplineTrajectory(random_seed, initial_pos)
    await drone.offboard.set_position_ned(PositionNedYaw(initial_pos[0], initial_pos[1], initial_pos[2], 0.0))

    try: await drone.offboard.start()
    except OffboardError: return

    print(f"  [飞行器] 开始 Random2 纯航点飞行 ({FLIGHT_DURATION_SEC} 秒)...")
    mission_start_time = time.time()
    time_step_sec = 1.0 / CONTROL_UPDATE_RATE_HZ
    last_print_time = 0

    while time.time() - mission_start_time < FLIGHT_DURATION_SEC:
        elapsed = time.time() - mission_start_time
        if time.time() - last_print_time > 5.0:
            print(f"    -> 飞行进度: {elapsed:.1f} / {FLIGHT_DURATION_SEC} 秒")
            last_print_time = time.time()

        target_x, target_y, target_z, target_yaw = trajectory_generator.get_target_pose(elapsed)
        await drone.offboard.set_position_ned(PositionNedYaw(target_x, target_y, target_z, target_yaw))
        await asyncio.sleep(time_step_sec)

    print("  [飞行器] 时间到，正在降落...")
    try: await drone.offboard.stop()
    except Exception: pass
    await drone.action.land()

    # === 修复死锁的关键区域 ===
    print("  [飞行器] 等待触地并自动上锁...")
    async for is_armed in drone.telemetry.armed():
        if not is_armed:
            print("  [飞行器] 已成功上锁！安全退出。")
            break

    os._exit(0) # 强行抹杀进程，防止 gRPC 线程死锁

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind", nargs=6, required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    try: asyncio.run(execute_flight_mission(arguments.wind, arguments.seed))
    except KeyboardInterrupt: pass
